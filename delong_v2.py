"""
delong_v2.py — v2 各方法 AUC 的配對 DeLong 檢定
================================================
所有方法都在同一批病人、同一組外層折上產生 OOF 預測，因此用配對的 DeLong 檢定
（DeLong et al., 1988；快速演算法 Sun & Xu, 2014）比較 AUC 差異。

預先指定（看結果前固定）
- 方法：LSTM、RF、LR、NEWS2_last、MEWS_last、MEWS_max，兩兩比較共 15 組
- 主分析：repeat 1 的 OOF 預測（與報告的 pooled AUC 一致），三個分析各自檢定
- 敏感度：primary 用 3 次重複的平均預測（LSTM/RF/LR；EWS 不需訓練，各次相同）
- 多重比較：每個分析內 15 組比較做 Holm 校正

需要先跑完 LSTM_nested_cv_v2.py 與 baselines_nested_v2.py。
執行：python delong_v2.py
輸出：results_delong_v2.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from baselines_nested_v2 import EWS, ews_scores
from LSTM_nested_cv_v2 import MAX_LEN, load_patients

METHODS = ["LSTM", "RF", "LR", "NEWS2_last", "MEWS_last", "MEWS_max"]


# ============================================================
# 1. 快速 DeLong（Sun & Xu, 2014）
# ============================================================
def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1        # 同分取平均名次
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_cov(y, preds):
    """preds: (k 個方法, n 位病人)。回傳各方法 AUC 與其共變異數矩陣。"""
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(p) for p in pos])
    ty = np.array([_midrank(p) for p in neg])
    tz = np.array([_midrank(p) for p in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# ============================================================
# 2. 整理各方法的 OOF 預測（同一批病人、同一順序）
# ============================================================
def collect(analysis, average_repeats):
    use_labs = analysis != "no_labs"
    min_records = MAX_LEN if analysis == "min10" else 1
    X_list, y, ids, _, _, cols, _ = load_patients(use_labs, min_records)
    truth = pd.Series(y, index=ids)

    lstm = pd.read_csv(f"results_v2_{analysis}_oof.csv").assign(model="LSTM")
    base = pd.read_csv(f"results_baselines_v2_{analysis}_oof.csv")
    oof = pd.concat([lstm, base])
    if not average_repeats:
        oof = oof[oof.repeat == 1]
    wide = oof.groupby(["ID", "model"]).proba.mean().unstack()

    scores = ews_scores(X_list, cols)
    for name in EWS:
        wide[name] = pd.Series(scores[name], index=ids)

    wide = wide.loc[ids, METHODS]
    assert wide.notna().all().all(), "有病人缺少某方法的預測"
    return truth.values, wide


def compare(analysis, average_repeats):
    y, wide = collect(analysis, average_repeats)
    preds = wide.T.values.astype(float)
    aucs, cov = delong_cov(y, preds)
    for k, mth in enumerate(METHODS):           # 自我檢查：DeLong AUC 應與 sklearn 相同
        assert abs(aucs[k] - roc_auc_score(y, preds[k])) < 1e-9, mth

    rows = []
    for i in range(len(METHODS)):
        for j in range(i + 1, len(METHODS)):
            diff = aucs[i] - aucs[j]
            se = np.sqrt(max(cov[i, i] + cov[j, j] - 2 * cov[i, j], 1e-12))
            z = diff / se
            rows.append({"analysis": analysis,
                         "predictions": "mean of 3 repeats" if average_repeats else "repeat 1",
                         "n_patients": len(y), "n_ihca": int(y.sum()),
                         "method_a": METHODS[i], "method_b": METHODS[j],
                         "auc_a": aucs[i], "auc_b": aucs[j], "diff": diff,
                         "diff_ci_low": diff - 1.96 * se, "diff_ci_high": diff + 1.96 * se,
                         "z": z, "p": 2 * norm.sf(abs(z))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.values)
    return out, aucs, cov, y, preds


def main():
    res = []
    for analysis in ["primary", "no_labs", "min10"]:
        out, aucs, cov, y, preds = compare(analysis, average_repeats=False)
        res.append(out)
        # 自我檢查：DeLong 的單一 AUC 標準誤應接近 bootstrap 標準誤
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                boot.append(roc_auc_score(y[i], preds[0][i]))
        print(f"[{analysis}] LSTM AUC 標準誤：DeLong {np.sqrt(cov[0, 0]):.4f} vs "
              f"bootstrap {np.std(boot):.4f}")
    res.append(compare("primary", average_repeats=True)[0])

    df = pd.concat(res, ignore_index=True)
    df.to_csv("results_delong_v2.csv", index=False)
    show = df.copy()
    show["pair"] = show.method_a + " vs " + show.method_b
    cols = ["pair", "auc_a", "auc_b", "diff", "diff_ci_low", "diff_ci_high", "p", "p_holm"]
    for (a, src), g in show.groupby(["analysis", "predictions"], sort=False):
        print(f"\n===== {a}（{src}，n={g.n_patients.iloc[0]}，IHCA {g.n_ihca.iloc[0]}）=====")
        print(g[cols].round(3).to_string(index=False))
    print("\n輸出：results_delong_v2.csv")


if __name__ == "__main__":
    main()
