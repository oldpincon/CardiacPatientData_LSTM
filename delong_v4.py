"""
delong_v4.py — v4 的配對 DeLong 檢定（見 LSTM_v4_預先指定.md）
=================================================================
第一階段（repeat 1）：F、G、H 各自 vs v2 LSTM、v3B、RF、LR、NEWS2_last，每個分析 15 組，Holm 校正
第二階段（確認）：被選中的版本在 repeat 2、3（新切分）vs RF、LR、NEWS2_last，
                  每個分析 2 個 repeat × 3 = 6 組，Holm 校正

需要先跑完 F、G、H（repeat 1）、select_v4.py，以及被選中版本的 --repeats 3。
執行：python delong_v4.py
輸出：results_delong_v4.csv
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from baselines_nested_v2 import ews_scores
from delong_v2 import collect, delong_cov, holm
from LSTM_nested_cv_v2 import MAX_LEN, load_patients

ANALYSES = ["primary", "no_labs", "min10"]


def oof_repeat(path, repeat, index):
    o = pd.read_csv(path)
    return o[o.repeat == repeat].set_index("ID").proba.reindex(index)


def paired(y, wide, pairs, extra):
    """wide：病人 × 方法；pairs：[(方法 a, 方法 b)]。回傳每組的差、CI、p。"""
    assert wide.notna().all().all(), "有病人缺少某方法的預測"
    names = list(wide.columns)
    preds = wide.T.values.astype(float)
    aucs, cov = delong_cov(y, preds)
    for k, n in enumerate(names):
        assert abs(aucs[k] - roc_auc_score(y, preds[k])) < 1e-9, n
    rows = []
    for a, b in pairs:
        i, j = names.index(a), names.index(b)
        diff = aucs[i] - aucs[j]
        se = np.sqrt(max(cov[i, i] + cov[j, j] - 2 * cov[i, j], 1e-12))
        z = diff / se
        rows.append({**extra, "n_patients": len(y), "n_ihca": int(y.sum()),
                     "variant": a, "reference": b, "auc_variant": aucs[i],
                     "auc_reference": aucs[j], "diff": diff,
                     "diff_ci_low": diff - 1.96 * se, "diff_ci_high": diff + 1.96 * se,
                     "z": z, "p": 2 * norm.sf(abs(z))})
    return rows


def stage1(analysis):
    y, wide = collect(analysis, average_repeats=False)
    wide = wide[["LSTM", "RF", "LR", "NEWS2_last"]].rename(columns={"LSTM": "LSTM_v2"})
    wide["v3B"] = oof_repeat(f"results_v3B_{analysis}_oof.csv", 1, wide.index)
    for v in "FGH":
        wide[f"v4{v}"] = oof_repeat(f"results_v4{v}_{analysis}_oof.csv", 1, wide.index)
    pairs = [(f"v4{v}", r) for v in "FGH" for r in ["LSTM_v2", "v3B", "RF", "LR", "NEWS2_last"]]
    out = pd.DataFrame(paired(y, wide, pairs, {"stage": "1: repeat 1", "analysis": analysis,
                                               "repeat": 1}))
    out["p_holm"] = holm(out.p.values)
    return out


def stage2(analysis, chosen):
    use_labs = analysis != "no_labs"
    min_records = MAX_LEN if analysis == "min10" else 1
    X_list, y, ids, _, _, cols, _ = load_patients(use_labs, min_records)
    news2 = pd.Series(ews_scores(X_list, cols)["NEWS2_last"], index=ids)
    base = pd.read_csv(f"results_baselines_v2_{analysis}_oof.csv")
    rows = []
    for r in [2, 3]:
        wide = pd.DataFrame(index=ids)
        wide[f"v4{chosen}"] = oof_repeat(f"results_v4{chosen}_{analysis}_oof.csv", r, ids)
        for m in ["RF", "LR"]:
            b = base[(base.model == m) & (base.repeat == r)]
            wide[m] = b.set_index("ID").proba.reindex(ids)
        wide["NEWS2_last"] = news2
        rows += paired(y, wide, [(f"v4{chosen}", m) for m in ["RF", "LR", "NEWS2_last"]],
                       {"stage": "2: 確認（新切分）", "analysis": analysis, "repeat": r})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.values)
    return out


def main():
    with open("results_v4_selection.json", encoding="utf-8") as fh:
        chosen = json.load(fh)["chosen"]
    res = [stage1(a) for a in ANALYSES] + [stage2(a, chosen) for a in ANALYSES]
    df = pd.concat(res, ignore_index=True)
    df.to_csv("results_delong_v4.csv", index=False)
    cols = ["repeat", "variant", "reference", "auc_variant", "auc_reference", "diff",
            "diff_ci_low", "diff_ci_high", "p", "p_holm"]
    for (s, a), g in df.groupby(["stage", "analysis"], sort=False):
        print(f"\n===== {s} · {a}（n={g.n_patients.iloc[0]}，IHCA {g.n_ihca.iloc[0]}）=====")
        print(g[cols].round(3).to_string(index=False))
    print(f"\n被選中的版本：v4{chosen}；輸出：results_delong_v4.csv")


if __name__ == "__main__":
    main()
