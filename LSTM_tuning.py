import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
LSTM_tuning.py
==============
用 Optuna 搜尋最佳超參數。

修改重點（相較原本版本）：
    原本的 objective() 只用「一組固定的 train/val split」評分，
    而這個 val 集只有 14 人、1 個 IHCA 陽性病人 —— 用單一病人的
    結果去引導 200 組 trial 的搜尋方向，選出來的參數很可能只是
    剛好適合那一個病人，換一批資料就不成立。

    現在改成：每個 trial 都跑 3-fold 分層交叉驗證，
    objective 回傳「跨折平均 val_loss」，讓超參數的選擇基於多組
    不同的驗證資料，而不是單一組小到只有 1-2 個陽性案例的驗證集。

    這裡用 3 折（而非 main.py 的 5 折）是為了讓 200 組 trial 的
    搜尋時間不要過長；folds 越多，單一 trial 越慢。如果你不趕時間，
    可以把 N_FOLDS 提高到 5 以更貼近最終評估流程。

使用方式：
    pip install optuna
    python LSTM_tuning.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, train_test_split
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
N_TRIALS = 200
N_FOLDS  = 3   # tuning 用的內層折數，可視時間調整（main.py 最終評估用 5 折）

# ============================================================
# 資料準備（只做一次，所有 trial 共用）
# ============================================================
print("載入並準備資料...")

df = pd.read_csv('CardiacPatientData_Cleaned.csv')
df['Outcome'] = df['Outcome'].map({0: 1, 1: 0})

X_list, y_list = [], []
for pid, group in df.groupby('ID'):
    X_list.append(group[FEATURES].values[-MAX_LEN:])
    y_list.append(group['Outcome'].iloc[-1])

y_all = np.array(y_list)
print(f"病人總數：{len(y_all)}，IHCA 陽性：{y_all.sum()}（{y_all.mean():.1%}）")
print(f"開始搜尋超參數（共 {N_TRIALS} 組，每組跑 {N_FOLDS} 折內部驗證）...\n")


def make_sequences(X_raw_list, scaler, fit_scaler=False):
    if fit_scaler:
        scaler.fit(np.vstack(X_raw_list))
    X_scaled = [scaler.transform(x) for x in X_raw_list]
    return pad_sequences(X_scaled, maxlen=MAX_LEN, dtype='float32', padding='pre')


# 固定的內層 CV 切分：所有 trial 共用同一組折，確保 trial 之間可比較
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
cv_splits = list(skf.split(np.zeros(len(y_all)), y_all))


# ============================================================
# Objective 函數：每個 trial 跑 N_FOLDS 折，回傳平均 val_loss
# ============================================================
def objective(trial):
    lstm_units    = trial.suggest_categorical('lstm_units', [32, 64, 96, 128])
    dropout       = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    dense_units   = trial.suggest_categorical('dense_units', [16, 32, 64])
    lr            = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    cw_multiplier = trial.suggest_float('cw_multiplier', 1.0, 3.0, step=0.25)
    batch_size    = trial.suggest_categorical('batch_size', [8, 16, 32])

    fold_val_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_train_raw = [X_list[i] for i in train_idx]
        y_train     = y_all[train_idx]
        X_val_raw   = [X_list[i] for i in val_idx]
        y_val       = y_all[val_idx]

        scaler = StandardScaler()
        X_train = make_sequences(X_train_raw, scaler, fit_scaler=True)
        X_val   = make_sequences(X_val_raw, scaler, fit_scaler=False)

        classes = np.array([0, 1])
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = {0: weights[0], 1: weights[1] * cw_multiplier}

        tf.keras.backend.clear_session()
        tf.random.set_seed(fold_idx)

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

        early_stop = EarlyStopping(
            monitor='val_loss', patience=5, mode='min', restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0
        )

        fold_val_losses.append(min(history.history['val_loss']))

        # Optuna pruning：如果前幾折就明顯很差，提早結束這個 trial
        trial.report(np.mean(fold_val_losses), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_val_losses))


# ============================================================
# 建立 Optuna Study 並開始搜尋
# ============================================================
sampler = TPESampler(seed=42)
pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
study = optuna.create_study(
    direction='minimize',
    sampler=sampler,
    pruner=pruner,
    study_name='LSTM_IHCA_tuning'
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
except KeyboardInterrupt:
    print("\n\n⚠ 偵測到 Ctrl+C，提前結束搜尋，印出目前最佳結果...")

# ============================================================
# 輸出結果
# ============================================================
completed = [t for t in study.trials if t.value is not None]
if len(completed) == 0:
    print("尚未完成任何 trial，無法輸出結果。")
    raise SystemExit

best = study.best_params
best_val_loss = study.best_value

print(f"\n已完成 {len(completed)} / {N_TRIALS} 組試驗")
print("\n" + "=" * 55)
print(f"  搜尋完成！最佳超參數如下（{N_FOLDS} 折平均 val_loss）")
print("=" * 55)
print(f"  最佳平均 val_loss    : {best_val_loss:.4f}")
print(f"  lstm_units           : {best['lstm_units']}")
print(f"  dropout              : {best['dropout']}")
print(f"  dense_units          : {best['dense_units']}")
print(f"  learning_rate        : {best['learning_rate']:.6f}")
print(f"  cw_multiplier        : {best['cw_multiplier']}")
print(f"  batch_size           : {best['batch_size']}")
print("=" * 55)

print("\n請將以下數值複製到 LSTM_main.py 頂部的常數（CW_MULTIPLIER 等），"
      "並在論文方法段註明這些數值來自 Optuna 搜尋，而非人工挑選：\n")
print(f"BEST_LSTM_UNITS    = {best['lstm_units']}")
print(f"BEST_DROPOUT       = {best['dropout']}")
print(f"BEST_DENSE_UNITS   = {best['dense_units']}")
print(f"BEST_LR            = {best['learning_rate']:.6f}")
print(f"BEST_CW_MULTIPLIER = {best['cw_multiplier']}")
print(f"BEST_BATCH_SIZE    = {best['batch_size']}")

try:
    print("\n--- 超參數重要性排名 ---")
    importance = optuna.importance.get_param_importances(study)
    for param, score in importance.items():
        bar = '█' * int(score * 30)
        print(f"  {param:<20} {bar} {score:.3f}")
except Exception:
    pass