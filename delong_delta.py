"""
delong_delta.py — 變化量特徵給 RF / LR 的配對 DeLong 檢定（見 變化量特徵_RF_預先指定.md）
每個分析、每個 repeat（1–3）：RF_delta vs RF、LR_delta vs LR、v4G vs RF_delta、v4G vs LR_delta，
每個分析 12 組 Holm 校正。

需要先跑完 baselines_nested_v2.py --delta（三個分析、--repeats 3）。
執行：python delong_delta.py
輸出：results_delong_delta.csv
"""

import pandas as pd

from delong_v4 import paired
from delong_v2 import holm
from LSTM_nested_cv_v2 import MAX_LEN, load_patients

ANALYSES = ["primary", "no_labs", "min10"]
PAIRS = [("RF_delta", "RF"), ("LR_delta", "LR"), ("v4G", "RF_delta"), ("v4G", "LR_delta")]


def compare(analysis):
    min_records = MAX_LEN if analysis == "min10" else 1
    y, ids = load_patients(analysis != "no_labs", min_records)[1:3]
    base = pd.read_csv(f"results_baselines_v2_{analysis}_oof.csv")
    delta = pd.read_csv(f"results_baselines_v2_delta_{analysis}_oof.csv")
    g = pd.read_csv(f"results_v4G_{analysis}_oof.csv")
    rows = []
    for r in [1, 2, 3]:
        wide = pd.DataFrame(index=ids)
        for m in ["RF", "LR"]:
            wide[m] = base[(base.model == m) & (base.repeat == r)].set_index("ID").proba
            wide[f"{m}_delta"] = delta[(delta.model == m) & (delta.repeat == r)].set_index("ID").proba
        wide["v4G"] = g[g.repeat == r].set_index("ID").proba
        rows += paired(y, wide, PAIRS, {"analysis": analysis, "repeat": r})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.values)
    return out


def main():
    df = pd.concat([compare(a) for a in ANALYSES], ignore_index=True)
    df.to_csv("results_delong_delta.csv", index=False)
    cols = ["repeat", "variant", "reference", "auc_variant", "auc_reference", "diff",
            "diff_ci_low", "diff_ci_high", "p", "p_holm"]
    for a, g in df.groupby("analysis", sort=False):
        print(f"\n===== {a}（n={g.n_patients.iloc[0]}，IHCA {g.n_ihca.iloc[0]}）=====")
        print(g[cols].round(3).to_string(index=False))
    print("\n輸出：results_delong_delta.csv")


if __name__ == "__main__":
    main()
