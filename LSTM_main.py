import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, Masking
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# 修改重點（相較原本 LSTM_main.py）
# ------------------------------------------------------------
# 1. 移除單一 train/test split（GroupShuffleSplit 一次切分）。
#    原本的切分下，測試集 23 人只有 2 個 IHCA 陽性病人，驗證集
#    14 人只有 1 個陽性病人 —— 這種規模下 AUC/Recall/Precision
#    的數字幾乎完全取決於這 1-2 個病人有沒有被抓到，換一個
#    random_state 結果可能天差地遠，不足以作為論文報告的依據。
#
# 2. 改用 StratifiedKFold（5 折）交叉驗證作為主要評估方式。
#    因為資料在聚合階段已經是「每位病人一筆序列」，所以病人分組
#    與樣本本身是一對一的，不需要再用 GroupKFold，直接對病人
#    層級做分層 K-fold 即可保證各折都有正負案例。
#    每一折都會產生一組獨立的 test 結果，最後回報
#    mean ± std，並額外回報 pooled out-of-fold AUC
#    （把 5 折的 test 預測全部合併起來算一次 AUC，
#      這是小樣本、正例稀少時比較穩定的整體效能估計）。
#
# 3. 特徵重要性改為「重複多次打亂取平均」，而不是原本只打亂一次。
#    單次打亂的 AUC 掉幅本身有隨機性，重複 20 次取平均與標準差
#    才具有參考價值。
#
# 4. 移除原本結尾的「壓力測試」區塊：它訓練的是另一個更簡化的模型、
#    且同樣是單次隨機切分，本質上跟主流程的問題一樣（只是換個模型
#    重複同樣的錯誤），現在用正式的 K-fold CV 取代，資訊量更完整。
#
# 5. 尚未解決、需要你用領域知識確認的問題（本程式無法從資料本身
#    判斷）：
#    - CSV 沒有時間戳記欄位，只能假設同一位病人的資料列是依照時間
#      先後排列。程式取每位病人「最後 MAX_LEN 筆」資料作為序列，
#      但這「最後幾筆」距離 IHCA 發生的時間點有多近，資料本身看
#      不出來。如果這些資料包含了急救當下或急救後量測的數值，
#      模型學到的會是「病人已經在惡化」而不是「提前預警」，效能
#      會被嚴重高估。這件事必須由你（或熟悉資料來源的臨床端）
#      確認並在論文方法段寫清楚 prediction horizon 的定義。
#    - Cleaned CSV 中 Na/K/Cl/Urea/Creatinine 缺失比例超過 55%，
#      填補方法未知。程式保留 *_is_missing 欄位供你選用（見下方
#      INCLUDE_MISSING_FLAGS），但實際填補方式是否合理，需要你
#      回頭確認 CardiacPatientData_Cleaned.csv 是怎麼產生的。
# ============================================================

# ============================================================
# 常數定義
# ============================================================
FEATURES = ['SBP', 'DBP', 'HR', 'RR', 'BT', 'SpO2', 'Age', 'Gender', 'GCS',
            'Na', 'K', 'Cl', 'Urea', 'Ceratinine']  # 欄位名稱與 CSV 一致

# 是否把缺失指標也當作特徵放入模型（見上方第 5 點說明）。
# 預設關閉，維持與原始模型相同的特徵集；你確認填補方式合理、
# 且想測試「缺失本身是否有資訊量」時可打開。
INCLUDE_MISSING_FLAGS = False
if INCLUDE_MISSING_FLAGS:
    FEATURES = FEATURES + ['Urea_is_missing', 'Ceratinine_is_missing', 'Cl_is_missing',
                            'Na_is_missing', 'K_is_missing']

MAX_LEN = 10
N_FOLDS = 5          # 5-fold：每折測試集約 22-23 人、3-4 個陽性案例
N_IMPORTANCE_REPEATS = 20  # 特徵重要性每個特徵重複打亂的次數

# ------------------------------------------------------------
# 以下超參數來自 LSTM_tuning.py 的 Optuna 搜尋結果
# （200 組 trial、3-fold 平均 val_loss，最佳 val_loss = 0.2344）。
# 這些數值不是人工挑選，論文方法段請註明來源是 Optuna 搜尋。
#
# 超參數重要性排名（供論文討論參考）：
#   learning_rate  0.445（影響最大）
#   cw_multiplier  0.435（影響次之，兩者遠高於其餘四個）
#   lstm_units     0.039
#   dropout        0.033
#   dense_units    0.031
#   batch_size     0.018
# ------------------------------------------------------------
LSTM_UNITS    = 96
DROPOUT       = 0.4
DENSE_UNITS   = 16
LEARNING_RATE = 0.009862
CW_MULTIPLIER = 1.5
BATCH_SIZE    = 8

# ============================================================
# 1. 載入資料、標籤反轉（1 = IHCA，0 = 正常）
# ============================================================
df = pd.read_csv('CardiacPatientData_Cleaned.csv')
df['Outcome'] = df['Outcome'].map({0: 1, 1: 0})

# ============================================================
# 2. 依病人 ID 聚合成序列（每位病人一筆樣本）
# ============================================================
X_list, y_list, patient_ids = [], [], []
for pid, group in df.groupby('ID'):
    X_list.append(group[FEATURES].values[-MAX_LEN:])
    y_list.append(group['Outcome'].iloc[-1])
    patient_ids.append(pid)

y_all = np.array(y_list)
patient_ids = np.array(patient_ids)

print(f"病人總數：{len(y_all)}，IHCA 陽性：{y_all.sum()}（{y_all.mean():.1%}）")


def make_sequences(X_raw_list, scaler, fit_scaler=False):
    """先標準化（依序列個別做 transform），再 padding。"""
    if fit_scaler:
        scaler.fit(np.vstack(X_raw_list))
    X_scaled = [scaler.transform(x) for x in X_raw_list]
    return pad_sequences(X_scaled, maxlen=MAX_LEN, dtype='float32', padding='pre')


def build_model():
    model = Sequential([
        Input(shape=(MAX_LEN, len(FEATURES))),
        Masking(mask_value=0.0),
        LSTM(units=LSTM_UNITS, return_sequences=True),
        Dropout(DROPOUT),
        LSTM(units=LSTM_UNITS, return_sequences=False),
        Dropout(DROPOUT),
        Dense(units=DENSE_UNITS, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]  # Recall/Precision 改在 CV 迴圈外用 sklearn 算，門檻可控
    )
    return model


# ============================================================
# 3. Stratified K-Fold 交叉驗證（主要評估流程）
# ============================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_metrics = []
oof_y_true, oof_y_proba = [], []
importance_records = []

print(f"\n========== {N_FOLDS}-Fold 交叉驗證開始 ==========")

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all)):
    print(f"\n>>> Fold {fold_idx + 1}/{N_FOLDS}")

    X_train_raw = [X_list[i] for i in train_idx]
    y_train_all = y_all[train_idx]
    X_test_raw = [X_list[i] for i in test_idx]
    y_test = y_all[test_idx]

    # 從本折訓練資料再切出驗證集（供 EarlyStopping 使用），
    # 一樣做 stratify，避免驗證集正例數過少。
    sub_pos, val_pos = train_test_split(
        np.arange(len(X_train_raw)), test_size=0.15,
        stratify=y_train_all, random_state=100 + fold_idx
    )
    X_sub_raw = [X_train_raw[i] for i in sub_pos]
    y_sub = y_train_all[sub_pos]
    X_val_raw = [X_train_raw[i] for i in val_pos]
    y_val = y_train_all[val_pos]

    scaler = StandardScaler()
    X_sub = make_sequences(X_sub_raw, scaler, fit_scaler=True)
    X_val = make_sequences(X_val_raw, scaler, fit_scaler=False)
    X_test = make_sequences(X_test_raw, scaler, fit_scaler=False)

    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_sub)
    class_weights = {0: weights[0], 1: weights[1] * CW_MULTIPLIER}

    tf.keras.backend.clear_session()
    tf.random.set_seed(fold_idx)
    np.random.seed(fold_idx)

    model = build_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True)

    model.fit(
        X_sub, y_sub,
        epochs=50, batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=0
    )

    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    fold_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
    fold_recall = recall_score(y_test, y_pred, zero_division=0)
    fold_precision = precision_score(y_test, y_pred, zero_division=0)
    fold_f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"  test n={len(y_test)}（陽性 {int(y_test.sum())}）"
          f" | AUC={fold_auc:.4f} Recall={fold_recall:.4f} "
          f"Precision={fold_precision:.4f} F1={fold_f1:.4f}")

    fold_metrics.append({
        'fold': fold_idx + 1, 'n_test': len(y_test), 'n_pos_test': int(y_test.sum()),
        'auc': fold_auc, 'recall': fold_recall, 'precision': fold_precision, 'f1': fold_f1
    })
    oof_y_true.extend(y_test.tolist())
    oof_y_proba.extend(y_proba.tolist())

    # --- 特徵重要性（本折內，重複多次打亂取平均） ---
    baseline_auc = fold_auc
    for i, col in enumerate(FEATURES):
        rng = np.random.default_rng(fold_idx * 1000 + i)
        drops = []
        for _ in range(N_IMPORTANCE_REPEATS):
            X_perturbed = X_test.copy()
            flat = X_perturbed[:, :, i].flatten()
            rng.shuffle(flat)
            X_perturbed[:, :, i] = flat.reshape(X_test.shape[0], X_test.shape[1])
            proba_p = model.predict(X_perturbed, verbose=0).ravel()
            auc_p = roc_auc_score(y_test, proba_p) if len(np.unique(y_test)) > 1 else np.nan
            drops.append(baseline_auc - auc_p)
        importance_records.append({
            'fold': fold_idx + 1, 'feature': col,
            'mean_auc_drop': np.nanmean(drops), 'std_auc_drop': np.nanstd(drops)
        })

print("\n========== 交叉驗證結束 ==========")

# ============================================================
# 4. 彙整每折結果
# ============================================================
metrics_df = pd.DataFrame(fold_metrics)
print("\n--- 各折結果 ---")
print(metrics_df.to_string(index=False))

summary = metrics_df[['auc', 'recall', 'precision', 'f1']].agg(['mean', 'std'])
print("\n--- 跨折彙整（mean ± std，這是論文應該報告的主要數字） ---")
for metric in ['auc', 'recall', 'precision', 'f1']:
    print(f"{metric.upper():<10}: {summary.loc['mean', metric]:.4f} ± {summary.loc['std', metric]:.4f}")

# Pooled out-of-fold AUC：把 5 折的測試預測全部合併算一次，
# 在正例稀少時比逐折平均更不受單一折運氣影響。
pooled_auc = roc_auc_score(oof_y_true, oof_y_proba)
print(f"\nPooled out-of-fold AUC（5 折測試預測合併計算）: {pooled_auc:.4f}")
print("※ 論文中建議同時報告『跨折 mean ± std』與『pooled OOF AUC』兩個數字，"
      "並在方法段說明樣本量小、每折陽性案例僅個位數，估計值的信賴區間較寬。")

# ============================================================
# 5. 特徵重要性彙整（跨折平均）
# ============================================================
importance_df = pd.DataFrame(importance_records)
importance_summary = (
    importance_df.groupby('feature')['mean_auc_drop']
    .agg(['mean', 'std'])
    .sort_values('mean', ascending=False)
)
print("\n--- 特徵重要性（跨 5 折、每折重複 {} 次打亂取平均） ---".format(N_IMPORTANCE_REPEATS))
print(importance_summary.to_string())

top_feature = importance_summary.index[0]
top_drop = importance_summary.iloc[0]['mean']
if top_drop > 0.4:
    print(f"\n⚠ 警告：發現潛在作弊特徵！")
    print(f"當 [{top_feature}] 被打亂時，AUC 平均劇降了 {top_drop:.4f}。")
    print(f"請檢查 [{top_feature}] 是否包含『事後資訊』（例如發病後才測得的數值），"
          f"或是否與 IHCA 事件時間點過於接近（見檔案開頭第 5 點說明）。")