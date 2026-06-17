import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # 關閉 oneDNN，避免 DLL 載入衝突
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # 關掉 TensorFlow 的 C++ 警告訊息

"""
LSTM_tuning.py
==============
用 Optuna 搜尋最佳超參數，搜尋完畢後印出建議數值，
手動填入 LSTM_main.py 頂部的 BEST_* 常數即可。

使用方式：
    pip install optuna
    python LSTM_tuning.py

搜尋的超參數：
    - lstm_units       : LSTM 單元數
    - dropout          : Dropout 比率
    - dense_units      : Dense 層大小
    - learning_rate    : Adam learning rate
    - cw_multiplier    : IHCA class weight 倍數
    - batch_size       : Batch size
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, Masking
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# 常數（與 LSTM_main.py 保持一致）
# ============================================================
FEATURES = ['SBP', 'DBP', 'HR', 'RR', 'BT', 'SpO2', 'Age', 'Gender', 'GCS',
            'Na', 'K', 'Cl', 'Urea', 'Ceratinine']
MAX_LEN  = 10
N_TRIALS = 200  # 搜尋幾組組合（資料小跑得快，200組約20分鐘）

# ============================================================
# 資料準備（只做一次，所有 trial 共用）
# ============================================================
print("載入並準備資料...")

df = pd.read_csv('CardiacPatientData_Cleaned.csv')
df['Outcome'] = df['Outcome'].map({0: 1, 1: 0})

X_list, y_list, patient_ids = [], [], []
for pid, group in df.groupby('ID'):
    X_list.append(group[FEATURES].values[-MAX_LEN:])
    y_list.append(group['Outcome'].iloc[-1])
    patient_ids.append(pid)

y_final     = np.array(y_list)
patient_ids = np.array(patient_ids)

# 訓練 / 測試切分（固定，確保每次 trial 評估在同一個測試集）
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_list, y_final, groups=patient_ids))

X_train_raw       = [X_list[i] for i in train_idx]
X_test_raw        = [X_list[i] for i in test_idx]
y_train_all       = y_final[train_idx]
y_test            = y_final[test_idx]
train_patient_ids = patient_ids[train_idx]

# 標準化（先 fit，所有 trial 共用同一個 scaler）
scaler = StandardScaler()
scaler.fit(np.vstack(X_train_raw))

X_train_scaled = [scaler.transform(x) for x in X_train_raw]
X_test_scaled  = [scaler.transform(x) for x in X_test_raw]

X_train_all = pad_sequences(X_train_scaled, maxlen=MAX_LEN,
                            dtype='float32', padding='pre')
X_test      = pad_sequences(X_test_scaled,  maxlen=MAX_LEN,
                            dtype='float32', padding='pre')

# 驗證集切分（固定）
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=99)
sub_train_idx, val_idx = next(
    gss_val.split(X_train_all, y_train_all, groups=train_patient_ids)
)

X_train = X_train_all[sub_train_idx]
y_train = y_train_all[sub_train_idx]
X_val   = X_train_all[val_idx]
y_val   = y_train_all[val_idx]

print(f"訓練集：{len(X_train)} 人，驗證集：{len(X_val)} 人，測試集：{len(X_test)} 人")
print(f"開始搜尋超參數（共 {N_TRIALS} 組）...\n")

# ============================================================
# Objective 函數：Optuna 每次 trial 都會呼叫這個函數
#
# 底層邏輯：
#   1. trial.suggest_* 從搜尋空間裡選一組超參數
#   2. 用這組參數建立並訓練模型
#   3. 回傳一個分數（這裡用 val_loss，越小越好）
#   4. Optuna 的 TPE 演算法根據歷史結果，推測下一組最有希望的參數
# ============================================================
def objective(trial):

    # --- 1. 讓 Optuna 選這次要試的超參數 ---
    lstm_units    = trial.suggest_categorical('lstm_units', [32, 64, 96, 128])
    dropout       = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    dense_units   = trial.suggest_categorical('dense_units', [16, 32, 64])
    lr            = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    # log=True：在對數尺度上搜尋，因為 lr 的有效範圍跨越好幾個數量級
    cw_multiplier = trial.suggest_float('cw_multiplier', 1.0, 3.0, step=0.25)
    batch_size    = trial.suggest_categorical('batch_size', [8, 16, 32])

    # --- 2. 計算 class weight ---
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight='balanced',
                                   classes=classes, y=y_train)
    class_weights = {0: weights[0], 1: weights[1] * cw_multiplier}

    # --- 3. 建立模型 ---
    # 每次 trial 都要重新建立，避免上一次的權重影響這次
    tf.keras.backend.clear_session()

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )

    # --- 4. 訓練（epoch 數少一點，讓搜尋快一點）---
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,        # tuning 時 patience 設小一點，加快速度
        mode='min',
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=30,         # tuning 時 epoch 上限設小一點
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=0          # 關掉訓練輸出，避免畫面被刷爆
    )

    # --- 5. 回傳分數（val_loss 最小值）---
    # Optuna 預設是最小化這個值
    best_val_loss = min(history.history['val_loss'])
    return best_val_loss


# ============================================================
# 建立 Optuna Study 並開始搜尋
#
# TPESampler：Tree-structured Parzen Estimator
#   - 把歷史 trial 分成「好的」和「差的」兩組
#   - 分別建立機率分布 l(x) 和 g(x)
#   - 選使 l(x)/g(x) 最大的下一組參數（期望改進最大）
#   - 比純隨機更聰明，但不需要複雜的 Gaussian Process
# ============================================================
sampler = TPESampler(seed=42)  # 固定種子確保重現性
study = optuna.create_study(
    direction='minimize',      # 最小化 val_loss
    sampler=sampler,
    study_name='LSTM_IHCA_tuning'
)

# 讓 Optuna 安靜一點，只印重要資訊
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    interrupted = False
except KeyboardInterrupt:
    interrupted = True
    print("\n\n⚠ 偵測到 Ctrl+C，提前結束搜尋，印出目前最佳結果...")

# ============================================================
# 輸出結果（不管是跑完還是 Ctrl+C 中斷，都會印）
# ============================================================
if len(study.trials) == 0:
    print("尚未完成任何 trial，無法輸出結果。")
    exit()

completed = [t for t in study.trials if t.value is not None]
print(f"\n已完成 {len(completed)} / {N_TRIALS} 組試驗")

best = study.best_params
best_val_loss = study.best_value

print("\n" + "="*55)
print("  搜尋完成！最佳超參數如下")
print("="*55)
print(f"  最佳 val_loss        : {best_val_loss:.4f}")
print(f"  lstm_units           : {best['lstm_units']}")
print(f"  dropout              : {best['dropout']}")
print(f"  dense_units          : {best['dense_units']}")
print(f"  learning_rate        : {best['learning_rate']:.6f}")
print(f"  cw_multiplier        : {best['cw_multiplier']}")
print(f"  batch_size           : {best['batch_size']}")
print("="*55)

print("\n請將以下數值複製到 LSTM_main.py 頂部的 BEST_* 常數：\n")
print(f"BEST_LSTM_UNITS    = {best['lstm_units']}")
print(f"BEST_DROPOUT       = {best['dropout']}")
print(f"BEST_DENSE_UNITS   = {best['dense_units']}")
print(f"BEST_LR            = {best['learning_rate']:.6f}")
print(f"BEST_CW_MULTIPLIER = {best['cw_multiplier']}")
print(f"BEST_BATCH_SIZE    = {best['batch_size']}")

# ============================================================
# 參數重要性分析（哪個超參數對結果影響最大）
# ============================================================
try:
    print("\n--- 超參數重要性排名 ---")
    importance = optuna.importance.get_param_importances(study)
    for param, score in importance.items():
        bar = '█' * int(score * 30)
        print(f"  {param:<20} {bar} {score:.3f}")
except Exception:
    pass  # trial 數不夠時 importance 可能無法計算，跳過即可