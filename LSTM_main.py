import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, Masking
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# 常數定義（只定義一次，全程式共用）
# ============================================================
FEATURES = ['SBP', 'DBP', 'HR', 'RR', 'BT', 'SpO2', 'Age', 'Gender', 'GCS',
            'Na', 'K', 'Cl', 'Urea', 'Ceratinine']  # 欄位名稱與 CSV 一致
MAX_LEN = 10

# ============================================================
# 1. 載入資料
# ============================================================
df = pd.read_csv('CardiacPatientData_Cleaned.csv')

# 2. 標籤反轉：讓 1 代表 IHCA（原本是 0），讓 0 代表正常（原本是 1）
df['Outcome'] = df['Outcome'].map({0: 1, 1: 0})

# ============================================================
# 3. 依照 ID 分組，建立病人序列（先做，padding 之前標準化）
# ============================================================
X_list = []
y_list = []
patient_ids = []

for pid, group in df.groupby('ID'):
    data = group[FEATURES].values
    outcome = group['Outcome'].iloc[-1]
    X_list.append(data[-MAX_LEN:])
    y_list.append(outcome)
    patient_ids.append(pid)

y_final = np.array(y_list)
patient_ids = np.array(patient_ids)

# ============================================================
# 4. 按病人 ID 切分訓練集 / 測試集
# ============================================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_list, y_final, groups=patient_ids))

X_train_raw = [X_list[i] for i in train_idx]
X_test_raw  = [X_list[i] for i in test_idx]
y_train_all = y_final[train_idx]
y_test      = y_final[test_idx]
train_patient_ids = patient_ids[train_idx]

# ============================================================
# 5. 資料標準化（先標準化原始序列，再 padding）
# ============================================================
scaler = StandardScaler()
X_train_flat = np.vstack(X_train_raw)
scaler.fit(X_train_flat)

X_train_scaled = [scaler.transform(x) for x in X_train_raw]
X_test_scaled  = [scaler.transform(x) for x in X_test_raw]

X_train_all = pad_sequences(X_train_scaled, maxlen=MAX_LEN,
                            dtype='float32', padding='pre')
X_test      = pad_sequences(X_test_scaled,  maxlen=MAX_LEN,
                            dtype='float32', padding='pre')

# ============================================================
# 6. 從訓練集切出驗證集
#    修正：test_size 從 0.1 → 0.15，讓驗證集有更多病人，val_auc 才可信
# ============================================================
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=99)
sub_train_idx, val_idx = next(
    gss_val.split(X_train_all, y_train_all, groups=train_patient_ids)
)

X_train = X_train_all[sub_train_idx]
y_train = y_train_all[sub_train_idx]
X_val   = X_train_all[val_idx]
y_val   = y_train_all[val_idx]

print(f"訓練集：{len(X_train)} 人，驗證集：{len(X_val)} 人，測試集：{len(X_test)} 人")
print(f"驗證集 IHCA 比例：{y_val.mean():.2f}（若接近0或1代表驗證集仍太小）")

# ============================================================
# 7. 根據實際類別比例計算 class weight
#    修正：倍數從 * 2 → * 1.5，在 Recall 和 Precision 之間取得平衡
# ============================================================
classes = np.array([0, 1])
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {0: weights[0], 1: weights[1] * 1.75}
print(f"自動計算的 Class Weights: {class_weights}")

# ============================================================
# 8. 建立 LSTM 模型
# ============================================================
model = Sequential([
    Input(shape=(MAX_LEN, len(FEATURES))),
    Masking(mask_value=0.0),
    LSTM(units=96, return_sequences=True),
    Dropout(0.3),
    LSTM(units=96, return_sequences=False),
    Dropout(0.3),
    Dense(units=64, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

# ============================================================
# 9. 編譯模型
# ============================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002666),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# ============================================================
# 10. 訓練模型
#     修正：監控 val_loss 取代 val_auc（val_auc 因驗證集小而不穩定）
#           patience 從 5 → 8，給模型更多時間學習
# ============================================================
early_stop = EarlyStopping(
    monitor='val_loss',   # 改監控 val_loss，比 val_auc 在小驗證集上更穩定
    patience=8,           # 給模型更多機會，不要太早停
    mode='min',
    restore_best_weights=True
)

print("開始訓練 LSTM 模型...")
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# 11. 評估與輸出結果
# ============================================================
loss, recall, precision, auc = model.evaluate(X_test, y_test, verbose=0)

print("\n--- 測試集評估結果 ---")
print(f"IHCA 抓漏率 (Recall):    {recall:.4f}")
print(f"預測精準度 (Precision):  {precision:.4f}")
print(f"模型辨識力 (AUC):        {auc:.4f}")

# F1 score 手動計算（Recall 和 Precision 的調和平均，綜合指標）
if (precision + recall) > 0:
    f1 = 2 * precision * recall / (precision + recall)
    print(f"綜合指標   (F1 Score):   {f1:.4f}")

# ============================================================
# 12. 特徵重要性分析
# ============================================================
baseline_auc = auc
results = []

print("\n正在分析特徵重要性...")

for i, col in enumerate(FEATURES):
    X_test_perturbed = X_test.copy()
    flat = X_test_perturbed[:, :, i].flatten()
    np.random.shuffle(flat)
    X_test_perturbed[:, :, i] = flat.reshape(X_test.shape[0], X_test.shape[1])

    _, _, _, auc_p = model.evaluate(X_test_perturbed, y_test, verbose=0)
    delta_auc = baseline_auc - auc_p
    results.append({'Feature': col, 'AUC_Drop': delta_auc, 'New_AUC': auc_p})

importance_df = pd.DataFrame(results).sort_values(by='AUC_Drop', ascending=False)

print("\n--- 特徵影響力排名 ---")
print(importance_df.to_string(index=False))

top_feature = importance_df.iloc[0]['Feature']
if importance_df.iloc[0]['AUC_Drop'] > 0.4:
    print(f"\n⚠ 警告：發現潛在作弊特徵！")
    print(f"當 [{top_feature}] 被打亂時，AUC 劇降了 {importance_df.iloc[0]['AUC_Drop']:.4f}。")
    print(f"請檢查 [{top_feature}] 是否包含『事後資訊』（例如發病後才測得的數值）。")

# ============================================================
# 壓力測試
# ============================================================

def create_sequences(input_df, scaler_obj, fit_scaler=False):
    """建立病人序列：先標準化，再 padding。"""
    X_raw, y = [], []
    for pid, group in input_df.groupby('ID'):
        X_raw.append(group[FEATURES].values[-MAX_LEN:])
        y.append(group['Outcome'].iloc[-1])

    if fit_scaler:
        scaler_obj.fit(np.vstack(X_raw))

    X_scaled = [scaler_obj.transform(x) for x in X_raw]
    X_padded = pad_sequences(X_scaled, maxlen=MAX_LEN,
                             dtype='float32', padding='pre')
    return X_padded, np.array(y)


print("\n\n========== 壓力測試開始 ==========")

for i in range(5):
    print(f"\n>>> 進行第 {i+1} 次隨機獨立實驗...")

    tf.random.set_seed(i * 10)
    np.random.seed(i * 10)

    gss_stress = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=i * 10)
    train_idx_s, test_idx_s = next(
        gss_stress.split(df, df['Outcome'], groups=df['ID'])
    )
    train_df = df.iloc[train_idx_s]
    test_df  = df.iloc[test_idx_s]

    stress_scaler = StandardScaler()
    X_train_s, y_train_s = create_sequences(train_df, stress_scaler, fit_scaler=True)
    X_test_s,  y_test_s  = create_sequences(test_df,  stress_scaler, fit_scaler=False)

    # 修正：印出每次測試集大小，方便診斷 AUC=1.0 的原因
    print(f"  訓練集：{len(y_train_s)} 人，測試集：{len(y_test_s)} 人，"
          f"測試集 IHCA 比例：{y_test_s.mean():.2f}")

    stress_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_LEN, len(FEATURES))),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    stress_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )

    stress_model.fit(X_train_s, y_train_s, epochs=10, batch_size=32, verbose=0)
    _, test_auc_s = stress_model.evaluate(X_test_s, y_test_s, verbose=0)
    print(f"  本次隨機實驗測試集 AUC: {test_auc_s:.4f}")

print("\n========== 壓力測試結束 ==========")
print("若5次 AUC 均持續偏高（>0.85），建議深入檢查是否有事後資訊洩漏。")