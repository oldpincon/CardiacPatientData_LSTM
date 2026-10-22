"""
evaluate_ews_baseline.py
以與 LSTM nested CV 完全相同的外層 StratifiedKFold 切分，評估 MEWS / NEWS / NEWS2。

評分系統不需訓練，因此不需內層調參；仍按外層折計算，
使 mean ± SD 可與 LSTM、RF、LR 直接比較。另報 pooled AUC 與 bootstrap 95% CI。

使用方式（請把 SEED、N_SPLITS、資料讀取改成與 LSTM_nested_cv_full.py 一致）：
    python evaluate_ews_baseline.py --data <你的資料.csv> --seed <LSTM 用的 seed>
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from early_warning_scores import patient_scores, report_missing

# 預先指定的臨床切點（RCP 2017；Subbe 2001）
THRESHOLDS = {
    "NEWS2_last": [5, 7],   # 5 = 緊急回應門檻，7 = 高風險
    "MEWS_last": [5],       # Subbe 2001 的 critical score
}
SCORES = ["NEWS2_last", "NEWS2_max", "MEWS_last", "MEWS_max"]


def recall_precision(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    rec = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    return rec, prec


def bootstrap_auc(y, s, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def evaluate(scores: pd.DataFrame, y: np.ndarray, seed: int, n_splits: int = 5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (_, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        r = {"fold": fold, "n_test": len(te), "n_pos": int(y[te].sum())}
        for s in SCORES:
            r[f"{s}_AUC"] = roc_auc_score(y[te], scores[s].values[te])
        for s, cuts in THRESHOLDS.items():
            for c in cuts:
                rec, prec = recall_precision(y[te], (scores[s].values[te] >= c).astype(int))
                r[f"{s}>={c}_recall"] = rec
                r[f"{s}>={c}_precision"] = prec
        rows.append(r)
    per_fold = pd.DataFrame(rows)

    summary = []
    for s in SCORES:
        lo, hi = bootstrap_auc(y, scores[s].values)
        summary.append({
            "score": s,
            "AUC_mean": per_fold[f"{s}_AUC"].mean(),
            "AUC_sd": per_fold[f"{s}_AUC"].std(ddof=1),
            "pooled_AUC": roc_auc_score(y, scores[s].values),
            "pooled_95CI_low": lo, "pooled_95CI_high": hi,
        })
    thr = []
    for s, cuts in THRESHOLDS.items():
        for c in cuts:
            k = f"{s}>={c}"
            pr, pp = recall_precision(y, (scores[s].values >= c).astype(int))
            thr.append({
                "rule": k,
                "recall_mean": per_fold[f"{k}_recall"].mean(),
                "recall_sd": per_fold[f"{k}_recall"].std(ddof=1),
                "precision_mean": per_fold[f"{k}_precision"].mean(),
                "precision_sd": per_fold[f"{k}_precision"].std(ddof=1),
                "precision_n_folds_defined": int(per_fold[f"{k}_precision"].notna().sum()),
                "pooled_recall": pr, "pooled_precision": pp,
                "n_flagged": int((scores[s].values >= c).sum()),
            })
    return per_fold, pd.DataFrame(summary), pd.DataFrame(thr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="必須與 LSTM 外層 StratifiedKFold 的 random_state 相同")
    ap.add_argument("--id-col", default="patient_id")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--last-n", type=int, default=10)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    # 注意：病人順序必須與 LSTM 程式中建立 y 的順序一致，外層折才會相同
    sc = patient_scores(df, a.id_col, a.last_n)
    y = df.groupby(a.id_col, sort=False)[a.label_col].max().loc[sc.index].values.astype(int)

    print("LOCF 後仍缺失、計 0 分的筆數：\n", report_missing(df, a.id_col, a.last_n))
    per_fold, summary, thr = evaluate(sc, y, a.seed)
    print(per_fold.round(3).to_string(index=False))
    print(summary.round(3).to_string(index=False))
    print(thr.round(3).to_string(index=False))
    per_fold.to_csv("results_ews_per_fold.csv", index=False)
    summary.to_csv("results_ews_summary.csv", index=False)
    thr.to_csv("results_ews_thresholds.csv", index=False)
