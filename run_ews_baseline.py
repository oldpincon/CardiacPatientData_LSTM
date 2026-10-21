"""
run_ews_baseline.py
在 CardiacPatientData_Cleaned.csv 上計算正式 MEWS / NEWS / NEWS2 基準，
外層切分與 LSTM_main.py 完全相同：
  - 病人依 df.groupby('ID') 的排序（ID 遞增）
  - 標籤反轉：Outcome 0 → IHCA = 1
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - 評估視窗：每位病人最後 10 筆（MAX_LEN = 10）

評分系統不需訓練，因此沒有內層調參；用外層折計算是為了能與 LSTM / RF / LR 逐折對照。

輸出：
  results_ews_per_fold.csv      主分析（清除不合理值）各折結果
  results_ews_summary.csv       主分析 + 敏感度分析（保留原值）的 AUC 摘要
  results_ews_thresholds.csv    臨床切點 recall / precision
"""

import pandas as pd

from early_warning_scores import clean_vitals, patient_scores, report_missing
from evaluate_ews_baseline import evaluate

DATA = "CardiacPatientData_Cleaned.csv"
SEED = 42
MAX_LEN = 10


def run(drop_implausible: bool):
    raw = pd.read_csv(DATA)
    # 穩定排序：病人順序 = groupby('ID')；同一病人內保留原始列順序（視為時間順序）
    raw = raw.sort_values("ID", kind="mergesort").reset_index(drop=True)
    raw["IHCA"] = 1 - raw["Outcome"]
    df, removed = clean_vitals(raw, temp_unit="F", drop_implausible=drop_implausible)
    sc = patient_scores(df, id_col="ID", last_n=MAX_LEN)
    y = df.groupby("ID", sort=False)["IHCA"].last().loc[sc.index].values.astype(int)
    assert len(y) == 112 and y.sum() == 19
    per_fold, summary, thr = evaluate(sc, y, seed=SEED)
    left = report_missing(df, id_col="ID", last_n=MAX_LEN)
    return per_fold, summary, thr, removed, left


if __name__ == "__main__":
    pf, sm, th, removed, left = run(drop_implausible=True)
    pf_s, sm_s, th_s, _, _ = run(drop_implausible=False)

    print("清除的不合理值筆數（全資料）：\n", removed.to_string())
    print("\n評估視窗內 LOCF 後仍缺失（計 0 分）：\n", left.to_string())
    print("\n各折陽性數：", pf["n_pos"].tolist(), "（應與 results_lstm_nested.csv 的 4,4,3,4,4 相同）")
    print("\n=== 主分析：AUC ===\n", sm.round(3).to_string(index=False))
    print("\n=== 主分析：臨床切點 ===\n", th.round(3).to_string(index=False))
    print("\n=== 敏感度分析（保留原值）：AUC ===\n", sm_s.round(3).to_string(index=False))

    pf.to_csv("results_ews_per_fold.csv", index=False)
    sm.insert(0, "analysis", "primary_implausible_removed")
    sm_s.insert(0, "analysis", "sensitivity_raw_values")
    pd.concat([sm, sm_s]).to_csv("results_ews_summary.csv", index=False)
    th.to_csv("results_ews_thresholds.csv", index=False)
