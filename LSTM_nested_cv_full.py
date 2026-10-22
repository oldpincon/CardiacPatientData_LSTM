# -*- coding: utf-8 -*-
"""
LSTM_nested_cv_full.py
========================
真正的 nested cross-validation 主程式：外層 5-fold 評估，內層 Optuna（每折各自獨立搜尋）
選超參數，測試資料完全不參與調參。同時輸出：

  - results_lstm_nested.csv          LSTM（nested CV）逐折結果
  - results_lr_baseline.csv          Logistic Regression baseline（最後一筆觀測值）
  - results_rf_baseline.csv          Random Forest baseline（最後一筆觀測值）
  - results_news_baseline.csv        簡化版早期預警評分（規則式，不訓練）
  - feature_importance_single.csv    單一特徵重要性（permutation，跨折平均）
  - feature_importance_group.csv     分組特徵重要性（跨折平均，用於偵測聯合洩漏）
  - truncation_sweep.csv             截斷測試（洩漏間接檢查）

========================================================================
誠實揭露：這支程式是回頭重建的版本
========================================================================
2026/09/17 執行 nested CV 時所用的原始腳本沒有被存進版本控制（用 git diff 對照後發現，
這個檔名在 repo 裡先前是空的），只留下了終端機輸出紀錄和對應的 results_*.csv／
feature_importance_*.csv／truncation_sweep.csv。本檔案依照：
  1) 終端機輸出紀錄裡每折實際選到的超參數（units / dropout / dense_units / lr / cw_multiplier）
  2) LSTM_main.py（stratified 5-fold 版本）與 LSTM_tuning.py（Optuna 目標函式）既有的邏輯
  3) results_*.csv 的欄位結構與檔名
重建而成。外層切分方式（StratifiedKFold, n_splits=5, shuffle=True, random_state=42）與舊版
完全一致，重新執行應能得到非常接近的數字，但不保證逐位小數點對齊——這是 nested CV／深度
學習訓練本身可預期的隨機性（TPE 搜尋路徑、TensorFlow 內部運算順序等），論文方法段建議寫
「同一套 nested CV 流程重跑一次得到的結果」，而不是強調對齊到小數點後第 4 位。

Baseline 部分（Logistic Regression／Random Forest／簡化版早期預警評分）的原始程式碼也一併
遺失，本檔案用最直觀、符合檔名與欄位結構的方式重建：
  - LR／RF：用每位病人「最後一筆觀測值」（不是整段序列）當特徵，只在各外層 fold 的訓練
    病人上訓練、在該 fold 的測試病人上評估，不會跨到測試病人。
  - 簡化版早期預警評分：純規則式評分（不訓練），仿照早期預警評分「異常程度分段給分」的
    概念，用 HR／RR／SBP／SpO2／BT／GCS 的異常程度加總成一個分數，直接當作風險分數算
    AUC。這不是官方驗證過的 NEWS score，論文與報告都要清楚註明這點（先前已經在投影片和
    準備手冊修正過這個稱呼）。

========================================================================
效能與時間備註（因為你說有時間壓力，先寫在這裡）
========================================================================
上一次執行（100 trials/fold，內層 3-fold）實測：
  Fold1 ~29 分鐘、Fold2 ~51 分鐘、Fold3 ~1h19m、Fold4 ~1h43m、Fold5 ~13h14m，總計約 17-18 小時。
  Fold1→4 呈現逐折變慢（17.5s/trial → 62.1s/trial），Fold5 更暴增到 476.5s/trial。這個「越跑
  越慢、最後暴增」的模式很像「熱節流（thermal throttling）」或背景負載干擾，不太像是演算法
  本身的成本上升。建議重跑前：筆電接電源、切到高效能電源模式、關掉其他吃資源的程式（瀏覽器
  分頁、掃毒排程）、確認睡眠/螢幕關閉不會中斷執行（Windows 設定裡把「電腦睡眠」設成「永不」）。

  本檔案預設把 N_TRIALS 從 100 調降到 40（下面常數區塊可以再調），並加入：
    - SQLite 儲存 Optuna study（每折一個 study，可中斷續跑，不用整折重來）
    - 每折跑完立刻把所有結果 CSV 存一次（不會因為中途當機而全部結果報銷）
  預期整體時間：樂觀情況（硬體穩定、不重演 Fold5 那種暴增）大約 2-4 小時；保守估計（照
  Fold1-4 那種逐折變慢的比例抓）大約 6-9 小時。無法保證更精確，因為原本的異常很可能是
  硬體／系統層級的問題，不是這支程式能控制的。如果你的時間真的只剩 3-4 小時，可以把最下面
  N_TRIALS 改成 15-20，跑出來的 nested CV 依然是「真的」nested CV（只是搜尋沒那麼窮盡），
  記得在論文方法段誠實寫上實際用的 trial 數。
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, Masking
from tensorflow.keras.callbacks import EarlyStopping

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# 常數（時間壓力大的話，先調這幾個）
# ============================================================
FEATURES = ['SBP', 'DBP', 'HR', 'RR', 'BT', 'SpO2', 'Age', 'Gender', 'GCS',
            'Na', 'K', 'Cl', 'Urea', 'Ceratinine']  # 欄位名稱與 CSV 一致
FEATURE_GROUPS = {
    'circulatory_resp': ['SBP', 'DBP', 'HR', 'RR', 'SpO2'],
    'demographics': ['Age', 'Gender'],
    'labs': ['Na', 'K', 'Cl', 'Urea', 'Ceratinine'],
    'consciousness': ['GCS'],
    'temperature': ['BT'],
}

MAX_LEN = 10
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3           # 內層調參用幾折；原版就是 3
N_TRIALS = 40               # 原始執行是 100，這裡先調降以控制時間；時間更緊迫可再調到 15-20
N_IMPORTANCE_REPEATS = 20
RANDOM_STATE = 42
OPTUNA_STORAGE = 'sqlite:///optuna_nested_cv.db'   # 讓每折的 study 可以中斷續跑

# ============================================================
# 1. 載入資料、標籤反轉（1 = IHCA，0 = 正常）
# ============================================================
df = pd.read_csv('CardiacPatientData_Cleaned.csv')
df['Outcome'] = df['Outcome'].map({0: 1, 1: 0})

X_list, y_list, patient_ids, last_obs_list = [], [], [], []
for pid, group in df.groupby('ID'):
    X_list.append(group[FEATURES].values[-MAX_LEN:])
    y_list.append(group['Outcome'].iloc[-1])
    patient_ids.append(pid)
    last_obs_list.append(group[FEATURES].iloc[-1].values)

y_all = np.array(y_list)
patient_ids = np.array(patient_ids)
X_last_obs = np.array(last_obs_list, dtype=float)  # 給 LR / RF baseline 用

print(f"病人總數：{len(y_all)}，IHCA 陽性：{y_all.sum()}（{y_all.mean():.1%}）")


def make_sequences(X_raw_list, scaler, fit_scaler=False):
    """先標準化（依序列個別做 transform），再 padding。"""
    if fit_scaler:
        scaler.fit(np.vstack(X_raw_list))
    X_scaled = [scaler.transform(x) if len(x) > 0 else np.zeros((0, len(FEATURES))) for x in X_raw_list]
    return pad_sequences(X_scaled, maxlen=MAX_LEN, dtype='float32', padding='pre')


def build_model(lstm_units, dropout, dense_units, learning_rate):
    model = Sequential([
        Input(shape=(MAX_LEN, len(FEATURES))),
        Masking(mask_value=0.0),
        LSTM(units=lstm_units, return_sequences=True),
        Dropout(dropout),
        LSTM(units=lstm_units, return_sequences=False),
        Dropout(dropout),
        Dense(units=dense_units, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return model


# ============================================================
# 簡化版早期預警評分（規則式，不訓練）—— 重建版，非官方 NEWS score
# ============================================================
def simplified_ews(row):
    score = 0
    hr = row['HR']
    if hr <= 40 or hr >= 131:
        score += 3
    elif 111 <= hr <= 130:
        score += 2
    elif (41 <= hr <= 50) or (91 <= hr <= 110):
        score += 1

    rr = row['RR']
    if rr <= 8 or rr >= 25:
        score += 3
    elif 21 <= rr <= 24:
        score += 2
    elif 9 <= rr <= 11:
        score += 1

    sbp = row['SBP']
    if sbp <= 90 or sbp >= 220:
        score += 3
    elif 91 <= sbp <= 100:
        score += 2
    elif 101 <= sbp <= 110:
        score += 1

    spo2 = row['SpO2']
    if spo2 <= 91:
        score += 3
    elif 92 <= spo2 <= 93:
        score += 2
    elif 94 <= spo2 <= 95:
        score += 1

    bt = row['BT']
    if bt <= 35.0:
        score += 3
    elif bt >= 39.1:
        score += 2
    elif (35.1 <= bt <= 36.0) or (38.1 <= bt <= 39.0):
        score += 1

    gcs = row['GCS']
    if gcs < 15:
        score += 3

    return score


# ============================================================
# 內層 Optuna 目標函式：只用外層訓練資料，做 N_INNER_FOLDS 折驗證
# ============================================================
def make_objective(X_train_raw, y_train_all, outer_fold_idx):
    inner_skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_splits = list(inner_skf.split(np.zeros(len(y_train_all)), y_train_all))

    def objective(trial):
        lstm_units    = trial.suggest_categorical('lstm_units', [32, 64, 96, 128])
        dropout       = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        dense_units   = trial.suggest_categorical('dense_units', [16, 32, 64])
        lr            = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        cw_multiplier = trial.suggest_float('cw_multiplier', 1.0, 2.0, step=0.25)
        batch_size    = trial.suggest_categorical('batch_size', [8, 16, 32])

        fold_val_losses = []
        for inner_idx, (tr_idx, val_idx) in enumerate(inner_splits):
            X_tr_raw = [X_train_raw[i] for i in tr_idx]
            y_tr = y_train_all[tr_idx]
            X_val_raw = [X_train_raw[i] for i in val_idx]
            y_val = y_train_all[val_idx]

            scaler = StandardScaler()
            X_tr = make_sequences(X_tr_raw, scaler, fit_scaler=True)
            X_val = make_sequences(X_val_raw, scaler, fit_scaler=False)

            classes = np.array([0, 1])
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
            class_weights = {0: weights[0], 1: weights[1] * cw_multiplier}

            tf.keras.backend.clear_session()
            tf.random.set_seed(outer_fold_idx * 100 + inner_idx)

            model = build_model(lstm_units, dropout, dense_units, lr)
            early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

            history = model.fit(
                X_tr, y_tr, epochs=30, batch_size=batch_size,
                validation_data=(X_val, y_val), class_weight=class_weights,
                callbacks=[early_stop], verbose=0
            )

            fold_val_losses.append(min(history.history['val_loss']))
            trial.report(float(np.mean(fold_val_losses)), step=inner_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_val_losses))

    return objective


# ============================================================
# 外層 5-fold nested CV 主迴圈
# ============================================================
outer_skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

lstm_rows, lr_rows, rf_rows, news_rows = [], [], [], []
importance_records, truncation_records = [], []

t_start_all = time.time()

for fold_idx, (train_idx, test_idx) in enumerate(outer_skf.split(np.zeros(len(y_all)), y_all)):
    fold_no = fold_idx + 1
    print(f"\n===== 外層 Fold {fold_no}/{N_OUTER_FOLDS} =====")
    t_fold_start = time.time()

    X_train_raw = [X_list[i] for i in train_idx]
    y_train_all = y_all[train_idx]
    X_test_raw = [X_list[i] for i in test_idx]
    y_test = y_all[test_idx]

    # ---------- 內層 Optuna 搜尋（只看外層訓練資料，可中斷續跑） ----------
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=RANDOM_STATE + fold_idx),
        pruner=MedianPruner(n_warmup_steps=1),
        study_name=f'ihca_nested_fold{fold_no}',
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
    )
    remaining = max(0, N_TRIALS - len(study.trials))
    if remaining > 0:
        study.optimize(make_objective(X_train_raw, y_train_all, fold_idx),
                        n_trials=remaining, show_progress_bar=True)
    best = study.best_params
    print(f"  本折最佳內層參數: {best}")

    # ---------- 用最佳超參數，在外層訓練資料上重新訓練，測外層測試資料 ----------
    scaler = StandardScaler()
    sub_pos, val_pos = train_test_split(
        np.arange(len(X_train_raw)), test_size=0.15,
        stratify=y_train_all, random_state=100 + fold_idx
    )
    X_sub_raw = [X_train_raw[i] for i in sub_pos]
    y_sub = y_train_all[sub_pos]
    X_val_raw = [X_train_raw[i] for i in val_pos]
    y_val = y_train_all[val_pos]

    X_sub = make_sequences(X_sub_raw, scaler, fit_scaler=True)
    X_val = make_sequences(X_val_raw, scaler, fit_scaler=False)
    X_test = make_sequences(X_test_raw, scaler, fit_scaler=False)

    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_sub)
    class_weights = {0: weights[0], 1: weights[1] * best['cw_multiplier']}

    tf.keras.backend.clear_session()
    tf.random.set_seed(fold_idx)
    np.random.seed(fold_idx)

    model = build_model(best['lstm_units'], best['dropout'], best['dense_units'], best['learning_rate'])
    early_stop = EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True)
    model.fit(X_sub, y_sub, epochs=50, batch_size=best['batch_size'],
              validation_data=(X_val, y_val), class_weight=class_weights,
              callbacks=[early_stop], verbose=0)

    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    fold_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
    fold_recall = recall_score(y_test, y_pred, zero_division=0)
    fold_precision = precision_score(y_test, y_pred, zero_division=0)
    fold_f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"  LSTM (nested) -> AUC={fold_auc:.4f} Recall={fold_recall:.4f} "
          f"Precision={fold_precision:.4f} F1={fold_f1:.4f}")

    lstm_rows.append({'fold': fold_no, 'n_test': len(y_test), 'n_pos_test': int(y_test.sum()),
                       'auc': fold_auc, 'recall': fold_recall, 'precision': fold_precision, 'f1': fold_f1})

    # ---------- 特徵重要性（本折內，重複多次打亂取平均） ----------
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
        importance_records.append({'fold': fold_no, 'feature': col,
                                     'mean_auc_drop': np.nanmean(drops), 'std_auc_drop': np.nanstd(drops)})

    # ---------- 截斷測試（洩漏間接檢查） ----------
    for trunc_n in [1, 2, 3, 5]:
        X_test_trunc_raw = [x[:-trunc_n] if len(x) > trunc_n else x[0:0] for x in X_test_raw]
        X_test_trunc = make_sequences(X_test_trunc_raw, scaler, fit_scaler=False)
        proba_t = model.predict(X_test_trunc, verbose=0).ravel()
        auc_t = roc_auc_score(y_test, proba_t) if len(np.unique(y_test)) > 1 else np.nan
        print(f"    截斷 {trunc_n} 筆 -> AUC={auc_t:.4f}")
        truncation_records.append({'fold': fold_no, 'truncate_n': trunc_n, 'auc': auc_t})

    # ---------- Baselines：以「最後一筆觀測值」為特徵，只用該折的訓練/測試病人 ----------
    X_lo_train, y_lo_train = X_last_obs[train_idx], y_all[train_idx]
    X_lo_test, y_lo_test = X_last_obs[test_idx], y_all[test_idx]

    lr_clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=fold_idx)
    lr_clf.fit(X_lo_train, y_lo_train)
    lr_proba = lr_clf.predict_proba(X_lo_test)[:, 1]
    lr_pred = (lr_proba >= 0.5).astype(int)
    lr_auc = roc_auc_score(y_lo_test, lr_proba) if len(np.unique(y_lo_test)) > 1 else np.nan
    print(f"  Logistic Regression baseline -> AUC={lr_auc:.4f}")
    lr_rows.append({'fold': fold_no, 'auc': lr_auc,
                     'recall': recall_score(y_lo_test, lr_pred, zero_division=0),
                     'precision': precision_score(y_lo_test, lr_pred, zero_division=0),
                     'f1': f1_score(y_lo_test, lr_pred, zero_division=0)})

    rf_clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=fold_idx)
    rf_clf.fit(X_lo_train, y_lo_train)
    rf_proba = rf_clf.predict_proba(X_lo_test)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
    rf_auc = roc_auc_score(y_lo_test, rf_proba) if len(np.unique(y_lo_test)) > 1 else np.nan
    print(f"  Random Forest baseline -> AUC={rf_auc:.4f}")
    rf_rows.append({'fold': fold_no, 'auc': rf_auc,
                     'recall': recall_score(y_lo_test, rf_pred, zero_division=0),
                     'precision': precision_score(y_lo_test, rf_pred, zero_division=0),
                     'f1': f1_score(y_lo_test, rf_pred, zero_division=0)})

    news_df_test = df[df['ID'].isin(patient_ids[test_idx])].groupby('ID').last()
    news_scores = news_df_test.apply(simplified_ews, axis=1).values
    news_auc = roc_auc_score(y_lo_test, news_scores) if len(np.unique(y_lo_test)) > 1 else np.nan
    print(f"  Simplified early-warning score baseline -> AUC={news_auc:.4f}")
    news_rows.append({'fold': fold_no, 'auc': news_auc})

    # ---------- 每折跑完立刻存檔，避免中途中斷全部報銷 ----------
    pd.DataFrame(lstm_rows).to_csv('results_lstm_nested.csv', index=False)
    pd.DataFrame(lr_rows).to_csv('results_lr_baseline.csv', index=False)
    pd.DataFrame(rf_rows).to_csv('results_rf_baseline.csv', index=False)
    pd.DataFrame(news_rows).to_csv('results_news_baseline.csv', index=False)
    pd.DataFrame(importance_records).to_csv('feature_importance_by_fold.csv', index=False)
    pd.DataFrame(truncation_records).to_csv('truncation_sweep_by_fold.csv', index=False)

    print(f"  本折耗時：{(time.time() - t_fold_start) / 60:.1f} 分鐘")

print(f"\n全部 {N_OUTER_FOLDS} 折總耗時：{(time.time() - t_start_all) / 3600:.2f} 小時")

# ============================================================
# 彙整輸出（跨折平均）
# ============================================================
importance_df = pd.DataFrame(importance_records)
single_summary = importance_df.groupby('feature')['mean_auc_drop'].agg(['mean', 'std'])
single_summary.to_csv('feature_importance_single.csv')

feat_to_group = {f: g for g, feats in FEATURE_GROUPS.items() for f in feats}
importance_df['feature_group'] = importance_df['feature'].map(feat_to_group)
group_summary = importance_df.groupby('feature_group')['mean_auc_drop'].agg(['mean', 'std'])
group_summary.to_csv('feature_importance_group.csv')

trunc_df = pd.DataFrame(truncation_records)
trunc_summary = trunc_df.groupby('truncate_n')['auc'].agg(['mean', 'std'])
lstm_df = pd.DataFrame(lstm_rows)
trunc_summary.loc[0] = [lstm_df['auc'].mean(), lstm_df['auc'].std()]
trunc_summary = trunc_summary.sort_index()
trunc_summary.to_csv('truncation_sweep.csv')

print("\n========================= 總結 =========================")
print("\n--- LSTM（Nested CV，正式結果，應寫入論文 Results） ---")
print(lstm_df.to_string(index=False))
print(f"Mean AUC: {lstm_df['auc'].mean():.4f} ± {lstm_df['auc'].std():.4f}")
print("\n完成。所有結果已輸出成 CSV，檔名開頭為 results_ / feature_importance_ / truncation_sweep")
