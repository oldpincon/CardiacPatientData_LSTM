from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input

# 1. 載入補值後的資料
df = pd.read_csv('CardiacPatientData_Cleaned.csv')

# 2. 標籤反轉：讓 1 代表 IHCA (原本是 0)，讓 0 代表正常 (原本是 1)
# 這樣 Recall 才會正確計算 IHCA 的抓漏率
df['Outcome'] = df['Outcome'].map({0: 1, 1: 0})

# 定義特徵欄位
features = ['SBP', 'DBP', 'HR', 'RR', 'BT', 'SpO2', 'Age', 'Gender', 'GCS',
            'Na', 'K', 'Cl', 'Urea', 'Ceratinine']

X_list = []
y_list = []
patient_ids = []

# 設定序列長度 (max_len)，根據 count_data_per_id 的分布，我們選擇 10 作為合理的截斷長度
max_len = 10

# 3. 依照 ID 分組建立病人序列資料
for pid, group in df.groupby('ID'):
    data = group[features].values
    # 取最後一筆紀錄的 Outcome 作為該病人的標籤[cite: 1]
    outcome = group['Outcome'].iloc[-1]

    X_list.append(data[-max_len:])
    y_list.append(outcome)
    patient_ids.append(pid)

# 補齊序列長度[cite: 3]
X_padded = pad_sequences(X_list, maxlen = max_len, dtype = 'float32', padding = 'pre')
y_final = np.array(y_list)
patient_ids = np.array(patient_ids)

#檢視code到這裡------------------------------------------------------------------------------------

# 4. 嚴謹切分：按病人 ID 切分，確保測試集是模型沒見過的病人[cite: 1]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_padded, y_final, groups=patient_ids))

X_train_raw, X_test_raw = X_padded[train_idx], X_padded[test_idx]
y_train, y_test = y_final[train_idx], y_final[test_idx]

# 5. 資料標準化 (StandardScaler)[cite: 1, 3]
scaler = StandardScaler()
# 先攤平再標準化，最後轉回 3D
X_train = scaler.fit_transform(
    X_train_raw.reshape(-1, len(features))).reshape(-1, max_len, len(features))
X_test = scaler.transform(
    X_test_raw.reshape(-1, len(features))).reshape(-1, max_len, len(features))

# 6. 建立 LSTM 模型架構 (參考 LSTM_2.py)[cite: 3]
model = Sequential([
    Input(shape=(max_len, len(features))),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=32, activation='relu'),
    Dense(units=1, activation='sigmoid')  # 二分類輸出[cite: 1]
])

# 7. 編譯模型：沿用 deepLearning.py 的醫療指標[cite: 1]
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.Recall(name='recall'),  # 現在代表 IHCA 抓漏率
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# 8. 訓練模型：設定 Class Weight (此時 1 是 IHCA 少數，給予較高權重)[cite: 1]
class_weights = {1: 2.0, 0: 1.0}

print("開始訓練 LSTM 模型...")
model.fit(X_train, y_train, epochs=30, batch_size=16,
          class_weight=class_weights, verbose=1)

# 9. 評估與輸出結果[cite: 1]
loss, recall, precision, auc = model.evaluate(X_test, y_test, verbose=0)

print("\n--- 測試集評估結果 ---")
print(f"IHCA 抓漏率 (Recall): {recall:.4f}")
print(f"預測精準度 (Precision): {precision:.4f}")
print(f"模型辨識力 (AUC): {auc:.4f}")


# 1. 定義一個封裝函數，讓 sklearn 能讀取 LSTM 的預測結果

def model_predict(X):
    # 因為 permutation_importance 會傳入 2D 或 3D 數組，我們需要確保格式正確
    return (model.predict(X, verbose=0) > 0.5).astype(int)


# 2. 計算排列重要性 (Permutation Importance)
# 我們打亂測試集的特徵，看誰對結果影響最大
results = []
baseline_auc = auc  # 剛才得到的 1.0

print("正在分析特徵重要性，尋找 AUC 1.0 的原因...")

# 針對每一個特徵進行打亂測試
for i, col in enumerate(features):
    X_test_perturbed = X_test.copy()
    # 打亂該特徵在所有時間步中的數值
    np.random.shuffle(X_test_perturbed[:, :, i])

    # 計算打亂後的表現
    loss_p, recall_p, prec_p, auc_p = model.evaluate(
        X_test_perturbed, y_test, verbose=0)
    delta_auc = baseline_auc - auc_p
    results.append({'Feature': col, 'AUC_Drop': delta_auc, 'New_AUC': auc_p})

# 3. 排序並顯示結果
importance_df = pd.DataFrame(results).sort_values(
    by='AUC_Drop', ascending=False)

print("\n--- 特徵影響力排名 ---")
print(importance_df)

# 4. 自動檢查
top_feature = importance_df.iloc[0]['Feature']
if importance_df.iloc[0]['AUC_Drop'] > 0.4:
    print(f"\n警告：發現潛在作弊特徵！")
    print(f"當 [{top_feature}] 被打亂時，AUC 劇降了 {importance_df.iloc[0]['AUC_Drop']:.4f}。")
    print(f"請檢查 [{top_feature}] 是否包含『事後資訊』（例如發病後才測得的數值）。")

    import numpy as np

# 載入資料
df = pd.read_csv('CardiacPatientData_Cleaned.csv')
df['Outcome'] = df['Outcome'].map({0: 1, 1: 0})  # 反轉標籤：1為IHCA

features = ['SBP', 'DBP', 'HR', 'RR', 'BT', 'SpO2', 'Age',
            'Gender', 'GCS', 'Na', 'K', 'Cl', 'Urea', 'Ceratinine']

# --- 壓力測試開始 ---
for i in range(5):
    print(f"\n>>> 進行第 {i+1} 次隨機獨立實驗...")

    # 重新隨機切分 ID (確保每次切分的人完全不同)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=i*10)
    unique_ids = df['ID'].unique()
    train_ids, test_ids = next(gss.split(unique_ids, groups=unique_ids))

    train_df = df[df['ID'].isin(unique_ids[train_ids])]
    test_df = df[df['ID'].isin(unique_ids[test_ids])]

    # 嚴格的資料隔離：先切分，再標準化 (避免資訊污染)
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()

    train_df_scaled[features] = scaler.fit_transform(train_df[features])
    test_df_scaled[features] = scaler.transform(
        test_df[features])  # 測試集只能用訓練集的標準

    # 建立序列函數
    def create_sequences(input_df):
        X, y = [], []
        for pid, group in input_df.groupby('ID'):
            X.append(group[features].values[-10:])
            y.append(group['Outcome'].iloc[-1])
        return pad_sequences(X, maxlen=10, dtype='float32', padding='post'), np.array(y)

    X_train, y_train = create_sequences(train_df_scaled)
    X_test, y_test = create_sequences(test_df_scaled)

    # 簡化模型 (排除過度擬合)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10, len(features))),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['AUC'])

    # 只訓練 10 個 Epoch，看模型是否秒懂答案
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    _, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"本次隨機實驗測試集 AUC: {test_auc:.4f}")
