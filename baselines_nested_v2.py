"""
baselines_nested_v2.py — 與 LSTM_nested_cv_v2.py 相同前處理、相同外層折的 RF / LR 基準
==========================================================================================
舊的 results_rf_baseline.csv / results_lr_baseline.csv 用的是 Cleaned CSV（全資料常數填補），
有同樣的缺失捷徑問題，不能直接跟 v2 的 LSTM 比。

- 資料：load_patients()（不合理值清除 + 病人內 LOCF），與 LSTM v2 完全相同
- 特徵：每位病人最後 10 筆的摘要（生命徵象：最後值、平均、最小、最大、標準差、斜率；
        靜態：Age、Gender；檢驗值：最後值）
- 填補：每一折只用訓練病人的中位數
- 內層：3 折 GridSearchCV（以 AUC 選參數）；門檻＝內層 OOF 預測的 Youden 指數
- 外層：StratifiedKFold(5, shuffle=True, random_state=42+repeat)，與 LSTM v2 相同

執行：python baselines_nested_v2.py --analysis primary --repeats 3
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from LSTM_nested_cv_v2 import (BASE_SEED, LABS, MAX_LEN, N_INNER, N_OUTER, STATIC,
                               VITALS, bootstrap_ci, load_patients, youden_threshold)


def summarize(X_list, cols):
    feats, names = [], []
    t = np.arange(MAX_LEN, dtype=float)
    for x in X_list:
        row = []
        for c in cols:
            v = x[:, cols.index(c)]
            if c in VITALS:
                ok = ~np.isnan(v)
                vv = v[ok]
                if len(vv) == 0:
                    row += [np.nan] * 6
                    continue
                slope = np.polyfit(t[:len(v)][ok], vv, 1)[0] if len(vv) >= 2 else 0.0
                row += [vv[-1], vv.mean(), vv.min(), vv.max(),
                        vv.std() if len(vv) > 1 else 0.0, slope]
            else:                      # 靜態與檢驗值：取最後值
                row.append(v[-1])
        feats.append(row)
    for c in cols:
        if c in VITALS:
            names += [f"{c}_{s}" for s in ["last", "mean", "min", "max", "std", "slope"]]
        else:
            names.append(f"{c}_last")
    return np.array(feats, dtype=float), names


MODELS = {
    "LR": (Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", LogisticRegression(class_weight="balanced", max_iter=5000))]),
           {"m__C": [0.01, 0.03, 0.1, 0.3, 1.0]}),
    "RF": (Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                                  n_jobs=-1))]),
           {"m__max_depth": [3, 5, None], "m__min_samples_leaf": [1, 3, 5]}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", choices=["primary", "no_labs", "min10"], default="primary")
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    use_labs = a.analysis != "no_labs"
    min_records = MAX_LEN if a.analysis == "min10" else 1
    X_list, y, ids, nrec, _, cols, _ = load_patients(use_labs, min_records)
    X, names = summarize(X_list, cols)
    print(f"[baselines_v2_{a.analysis}] 病人 {len(y)}，IHCA {y.sum()}，摘要特徵 {X.shape[1]} 個")

    fold_rows, oof_rows = [], []
    for r in range(a.repeats):
        seed = BASE_SEED + r
        outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=seed)
        for f, (tr, te) in enumerate(outer.split(X, y), start=1):
            inner = StratifiedKFold(N_INNER, shuffle=True, random_state=seed * 10 + f)
            for name, (pipe, grid) in MODELS.items():
                pipe.set_params(**({"m__random_state": seed * 10 + f} if name == "RF" else {}))
                gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1)
                gs.fit(X[tr], y[tr])
                inner_oof = cross_val_predict(gs.best_estimator_, X[tr], y[tr], cv=inner,
                                              method="predict_proba")[:, 1]
                thr = youden_threshold(y[tr], inner_oof)
                p = gs.best_estimator_.predict_proba(X[te])[:, 1]
                pred = (p >= thr).astype(int)
                fold_rows.append({"model": name, "repeat": r + 1, "fold": f,
                                  "auc": roc_auc_score(y[te], p), "threshold": thr,
                                  "recall": recall_score(y[te], pred, zero_division=0),
                                  "precision": precision_score(y[te], pred, zero_division=0),
                                  "best_params": json.dumps(gs.best_params_)})
                for i, pi in zip(te, p):
                    oof_rows.append({"model": name, "repeat": r + 1, "fold": f, "ID": ids[i],
                                     "y": y[i], "proba": pi, "pred": int(pi >= thr)})

    folds, oof = pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)
    tag = f"baselines_v2_{a.analysis}"
    folds.to_csv(f"results_{tag}_folds.csv", index=False)
    oof.to_csv(f"results_{tag}_oof.csv", index=False)
    out = []
    for name, g in folds.groupby("model"):
        o = oof[(oof.model == name) & (oof.repeat == 1)]
        lo, hi = bootstrap_ci(o.y.values, o.proba.values)
        out.append({"model": name, "auc_mean": g.auc.mean(), "auc_sd": g.auc.std(ddof=1),
                    "pooled_auc_r1": roc_auc_score(o.y, o.proba), "ci_low": lo, "ci_high": hi,
                    "recall_mean": g.recall.mean(), "precision_mean": g.precision.mean()})
    s = pd.DataFrame(out)
    s.to_csv(f"results_{tag}_summary.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
