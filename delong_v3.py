"""
delong_v3.py — v3 各版本 LSTM 的配對 DeLong 檢定
================================================
預先指定（見 LSTM_v3_預先指定.md）：每個版本（A–E）分別和 v2 LSTM、RF、LR、NEWS2_last 比較，
每個分析內 5 × 4 = 20 組，Holm 校正。用 repeat 1 的 OOF 預測。

需要先跑完 LSTM_v3.py 全部版本。
執行：python delong_v3.py
輸出：results_delong_v3.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from delong_v2 import collect, delong_cov, holm

VARIANTS = ["A", "B", "C", "D", "E"]
REFS = ["LSTM", "RF", "LR", "NEWS2_last"]      # LSTM = v2


def compare(analysis):
    y, wide = collect(analysis, average_repeats=False)
    wide = wide[REFS].copy()
    for v in VARIANTS:
        o = pd.read_csv(f"results_v3{v}_{analysis}_oof.csv")
        wide[f"v3{v}"] = o[o.repeat == 1].set_index("ID").proba.reindex(wide.index)
    assert wide.notna().all().all(), "有病人缺少某版本的預測"
    names = list(wide.columns)
    preds = wide.T.values.astype(float)
    aucs, cov = delong_cov(y, preds)
    for k, n in enumerate(names):
        assert abs(aucs[k] - roc_auc_score(y, preds[k])) < 1e-9, n

    rows = []
    for v in VARIANTS:
        i = names.index(f"v3{v}")
        for ref in REFS:
            j = names.index(ref)
            diff = aucs[i] - aucs[j]
            se = np.sqrt(max(cov[i, i] + cov[j, j] - 2 * cov[i, j], 1e-12))
            z = diff / se
            rows.append({"analysis": analysis, "n_patients": len(y), "n_ihca": int(y.sum()),
                         "variant": f"v3{v}", "reference": "LSTM_v2" if ref == "LSTM" else ref,
                         "auc_variant": aucs[i], "auc_reference": aucs[j], "diff": diff,
                         "diff_ci_low": diff - 1.96 * se, "diff_ci_high": diff + 1.96 * se,
                         "z": z, "p": 2 * norm.sf(abs(z))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.values)
    return out


def main():
    df = pd.concat([compare(a) for a in ["primary", "no_labs", "min10"]], ignore_index=True)
    df.to_csv("results_delong_v3.csv", index=False)
    cols = ["variant", "reference", "auc_variant", "auc_reference", "diff",
            "diff_ci_low", "diff_ci_high", "p", "p_holm"]
    for a, g in df.groupby("analysis", sort=False):
        print(f"\n===== {a}（n={g.n_patients.iloc[0]}，IHCA {g.n_ihca.iloc[0]}）=====")
        print(g[cols].round(3).to_string(index=False))
    print("\n輸出：results_delong_v3.csv")


if __name__ == "__main__":
    main()
