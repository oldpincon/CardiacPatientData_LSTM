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

from early_warning_scores import mews_row, news2_row
from LSTM_nested_cv_v2 import (BASE_SEED, LABS, MAX_LEN, N_INNER, N_OUTER, STATIC,
                               TRUNC_KS, VITALS, bootstrap_ci, load_patients, shifted_window,
                               truncation_summary, youden_threshold)

# 臨床評分（不需訓練）：名稱 → (逐筆計分函式, 視窗彙總方式, 臨床切點)
EWS = {"NEWS2_last": (news2_row, "last", 5),
       "MEWS_last": (mews_row, "last", 5),
       "MEWS_max": (mews_row, "max", 5)}


def ews_scores(x_list, cols):
    """每位病人視窗的 EWS 分數（BT 由 °F 轉 °C）。截斷測試的臨床評分對照。"""
    out = {name: [] for name in EWS}
    for x in x_list:
        rows = [dict(zip(cols, row.astype(float))) for row in x]
        for r in rows:
            r["BT"] = (r["BT"] - 32.0) * 5.0 / 9.0
        for name, (fn, agg, _) in EWS.items():
            s = [fn(r) for r in rows]
            out[name].append(s[-1] if agg == "last" else max(s))
    return {name: np.array(v, dtype=float) for name, v in out.items()}


DELTA_STATS = ["delta_last", "delta_min", "delta_max", "delta_absmean"]


def summarize(X_list, cols, delta=False):
    """delta=True 時每項生命徵象另加 4 個「與上一筆的差」特徵（見 變化量特徵_RF_預先指定.md）。"""
    feats, names = [], []
    t = np.arange(MAX_LEN, dtype=float)
    n_vit = 6 + (len(DELTA_STATS) if delta else 0)
    for x in X_list:
        row = []
        for c in cols:
            v = x[:, cols.index(c)]
            if c in VITALS:
                ok = ~np.isnan(v)
                vv = v[ok]
                if len(vv) == 0:
                    row += [np.nan] * n_vit
                    continue
                slope = np.polyfit(t[:len(v)][ok], vv, 1)[0] if len(vv) >= 2 else 0.0
                row += [vv[-1], vv.mean(), vv.min(), vv.max(),
                        vv.std() if len(vv) > 1 else 0.0, slope]
                if delta:
                    d = np.diff(vv)
                    row += ([d[-1], d.min(), d.max(), np.abs(d).mean()] if len(d)
                            else [0.0] * len(DELTA_STATS))
            else:                      # 靜態與檢驗值：取最後值
                row.append(v[-1])
        feats.append(row)
    for c in cols:
        if c in VITALS:
            stats = ["last", "mean", "min", "max", "std", "slope"] + (DELTA_STATS if delta else [])
            names += [f"{c}_{s}" for s in stats]
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
    ap.add_argument("--delta", action="store_true",
                    help="加入生命徵象變化量特徵（見 變化量特徵_RF_預先指定.md）")
    a = ap.parse_args()
    use_labs = a.analysis != "no_labs"
    min_records = MAX_LEN if a.analysis == "min10" else 1
    X_list, y, ids, nrec, _, cols, _, X_full = load_patients(use_labs, min_records, full=True)
    X, names = summarize(X_list, cols, a.delta)
    print(f"[baselines_v2_{a.analysis}] 病人 {len(y)}，IHCA {y.sum()}，摘要特徵 {X.shape[1]} 個")

    # 截斷測試用：各 k 下可往前移的病人，及其往前移後的摘要特徵與 EWS 分數
    shifted = {}
    for k in TRUNC_KS:
        keep = np.array([i for i in range(len(y)) if shifted_window(X_full[i], k) is not None])
        wins = [shifted_window(X_full[i], k) for i in keep]
        shifted[k] = (keep, summarize(wins, cols, a.delta)[0], ews_scores(wins, cols))
    ews0 = ews_scores(X_list, cols)

    fold_rows, oof_rows, trunc_rows = [], [], []
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
                for k, (keep, Xk, _) in shifted.items():
                    sel = np.isin(keep, te)
                    pk = gs.best_estimator_.predict_proba(Xk[sel])[:, 1]
                    for i, pi in zip(keep[sel], pk):
                        trunc_rows.append({"model": name, "repeat": r + 1, "fold": f, "k": k,
                                           "ID": ids[i], "y": y[i], "proba": pi})
            # EWS 不需訓練，但照相同外層折記錄，才能和模型逐折比較
            for name, (_, _, cut) in EWS.items():
                for i in te:
                    oof_rows.append({"model": name, "repeat": r + 1, "fold": f, "ID": ids[i],
                                     "y": y[i], "proba": ews0[name][i],
                                     "pred": int(ews0[name][i] >= cut)})
                for k, (keep, _, ek) in shifted.items():
                    sel = np.isin(keep, te)
                    for i, pi in zip(keep[sel], ek[name][sel]):
                        trunc_rows.append({"model": name, "repeat": r + 1, "fold": f, "k": k,
                                           "ID": ids[i], "y": y[i], "proba": pi})

    folds, oof = pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)
    trunc = pd.DataFrame(trunc_rows)
    tag = f"baselines_v2{'_delta' if a.delta else ''}_{a.analysis}"
    folds.to_csv(f"results_{tag}_folds.csv", index=False)
    oof[~oof.model.isin(EWS)].to_csv(f"results_{tag}_oof.csv", index=False)
    trunc.to_csv(f"results_{tag}_truncation_oof.csv", index=False)
    ts = pd.concat([truncation_summary(oof[oof.model == m], trunc[trunc.model == m])
                    .assign(model=m) for m in oof.model.unique()])
    ts = ts[["model"] + [c for c in ts.columns if c != "model"]]
    ts.to_csv(f"results_{tag}_truncation.csv", index=False)
    print("截斷測試（視窗往前移 k 筆）：")
    print(ts.round(3).to_string(index=False))
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
