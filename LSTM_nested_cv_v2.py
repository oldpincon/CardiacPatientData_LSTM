import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

"""
LSTM_nested_cv_v2.py — 修補漏洞版（2026-10-14）
=================================================
目的：只修正評估流程的漏洞，「不改模型架構與搜尋空間」，
      讓結果的變化可以歸因到修正本身，而不是模型變強或變弱。

相較舊版的修正
--------------
1. 缺失值填補移到每一折內部（不再使用 Cleaned CSV 的全資料常數填補）
   - 讀原始 CardiacPatientData.csv
   - 同一病人內先以前值延續（LOCF，只用過去資訊）＝「最近一次已知的檢驗值」
   - 仍缺失者用「該折訓練病人」的中位數填補
   - 舊 Cleaned CSV 把每一筆沒驗的列都填成同一常數（Na=139…），等於讓模型看到
     「這一列有沒有抽血」——IHCA 病人 95% 的列沒驗、非 IHCA 只有 48%，這是捷徑。
     LOCF 後模型只看到檢驗值本身，看不到抽血頻率。
     （實測：每位病人在視窗前都至少驗過一次，LOCF 後視窗內無任何檢驗缺失，
       因此不另加缺失指標——那些指標會全為 0。）
2. 生理不合理值（範圍事先指定，與 EWS 基準相同）設為缺失，再走第 1 點流程
3. 內層 3 折 CV 同時決定：超參數、訓練 epoch 數、分類門檻
   - 外層訓練不再切出只有約 2 個陽性的 15% validation 做 early stopping；
     改用內層各折最佳 epoch 的中位數，在整個外層訓練集上訓練
   - 分類門檻改用內層 OOF 預測的 Youden 指數決定，不再固定 0.5
4. 儲存每位病人的外層 OOF 預測 → 可算 pooled AUC、bootstrap CI、DeLong
5. 修正 permutation importance：只在「真實時間步」之間打亂，
   不再把 padding 的 0 混進真實資料、或讓被 mask 的步驟變成未 mask
6. 捷徑探針：在相同外層折下，評估「只用紀錄筆數」和「只用缺失比例」
   的預測力，作為模型是否靠捷徑的參照
7. 固定所有 seed；可用 --repeats 做 repeated nested CV

預先指定的分析（請三個都跑、全部報告，不要挑好看的）
--------------------------------------------------
  --analysis primary   主分析：全部 112 人，含檢驗值與缺失指標
  --analysis no_labs   敏感度：完全不用檢驗值（排除缺失捷徑）
  --analysis min10     敏感度：只用紀錄 ≥ 10 筆的病人（排除 padding／序列長度捷徑）

建議執行（Windows，在 repo 資料夾、啟用 .venv 後）
--------------------------------------------------
  python LSTM_nested_cv_v2.py --analysis primary --repeats 3 --trials 40
  python LSTM_nested_cv_v2.py --analysis no_labs --repeats 1 --trials 40
  python LSTM_nested_cv_v2.py --analysis min10   --repeats 1 --trials 40
  快速檢查程式能跑：加 --trials 2 --quick

注意：LSTM_tuning.py 是在全部 112 人上調參，LSTM_main.py 用它的參數跑 5 折，
      兩者都「不是」nested，結果不可報告。
"""

import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Masking
from tensorflow.keras.models import Sequential

# ============================================================
# 常數（架構與搜尋空間沿用舊版，刻意不改）
# ============================================================
DATA_RAW = "CardiacPatientData.csv"
VITALS = ["SBP", "DBP", "HR", "RR", "BT", "SpO2", "GCS"]
STATIC = ["Age", "Gender"]
LABS = ["Na", "K", "Cl", "Urea", "Ceratinine"]
MAX_LEN = 10
N_OUTER = 5
N_INNER = 3
BASE_SEED = 42                 # repeat 0 的外層切分與舊版相同（random_state=42）

# 預先指定的生理合理範圍（與 early_warning_scores.py 相同；BT 為 °F）
PLAUSIBLE = {"SBP": (40, 300), "DBP": (20, 200), "HR": (20, 250), "RR": (4, 60),
             "SpO2": (50, 100), "BT": (86, 113), "GCS": (3, 15)}

FEATURE_GROUPS = {
    "circulatory_resp": ["SBP", "DBP", "HR", "RR", "SpO2"],
    "consciousness": ["GCS"],
    "temperature": ["BT"],
    "demographics": STATIC,
    "labs": LABS,
}


# ============================================================
# 1. 資料載入與「不需 fit 的」前處理（規則式，與切分無關，不會洩漏）
# ============================================================
def load_patients(use_labs: bool, min_records: int):
    df = pd.read_csv(DATA_RAW)
    df = df.sort_values("ID", kind="mergesort").reset_index(drop=True)  # 病人內保留原列順序
    df["IHCA"] = 1 - df["Outcome"]                                      # 0 → IHCA

    n_removed = {}
    for c, (lo, hi) in PLAUSIBLE.items():
        m = (df[c] < lo) | (df[c] > hi)
        n_removed[c] = int(m.sum())
        df.loc[m, c] = np.nan

    cols = VITALS + STATIC
    if use_labs:
        cols = cols + LABS
    # 捷徑探針用：LOCF「之前」每一列是否實際有抽血（模型看不到這個）
    df["_lab_measured"] = df[LABS].notna().any(axis=1).astype(float)
    # 病人內 LOCF（只用過去資訊）
    df[cols] = df.groupby("ID", sort=False)[cols].ffill()

    n_records = df.groupby("ID", sort=False).size()
    X, y, ids, nrec, lab_meas = [], [], [], [], []
    for pid, g in df.groupby("ID", sort=False):
        if len(g) < min_records:
            continue
        X.append(g[cols].values[-MAX_LEN:].astype("float32"))  # 尚未填補，可能含 NaN
        y.append(int(g["IHCA"].iloc[-1]))
        ids.append(pid)
        nrec.append(int(n_records[pid]))
        lab_meas.append(float(g["_lab_measured"].values[-MAX_LEN:].mean()))
    return (X, np.array(y), np.array(ids), np.array(nrec), np.array(lab_meas),
            cols, n_removed)


# ============================================================
# 2. 需要 fit 的前處理：只用訓練病人 fit（中位數填補 + 標準化）
# ============================================================
class FoldPreprocessor:
    def fit(self, X_list):
        stacked = np.vstack(X_list)
        self.medians = np.nanmedian(stacked, axis=0)
        self.medians = np.where(np.isnan(self.medians), 0.0, self.medians)
        filled = np.where(np.isnan(stacked), self.medians, stacked)
        self.scaler = StandardScaler().fit(filled)
        return self

    def transform(self, X_list):
        out = np.zeros((len(X_list), MAX_LEN, len(self.medians)), dtype="float32")
        mask = np.zeros((len(X_list), MAX_LEN), dtype=bool)
        for i, x in enumerate(X_list):
            x = np.where(np.isnan(x), self.medians, x)
            x = self.scaler.transform(x).astype("float32")
            # 避免真實時間步剛好整列為 0 而被 Masking 誤當 padding
            x[np.all(x == 0, axis=1)] = 1e-6
            out[i, MAX_LEN - len(x):] = x           # padding='pre'
            mask[i, MAX_LEN - len(x):] = True
        return out, mask


# ============================================================
# 3. 模型（與舊版相同架構）
# ============================================================
def build_model(n_feat, p):
    model = Sequential([
        Input(shape=(MAX_LEN, n_feat)),
        Masking(mask_value=0.0),
        LSTM(p["lstm_units"], return_sequences=True),
        Dropout(p["dropout"]),
        LSTM(p["lstm_units"]),
        Dropout(p["dropout"]),
        Dense(p["dense_units"], activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p["learning_rate"]),
                  loss="binary_crossentropy")
    return model


def class_weights(y, mult):
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return {0: w[0], 1: w[1] * mult}


def set_seed(s):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(s))


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    t = thr[int(np.argmax(j))]
    return float(min(max(t, 1e-6), 1 - 1e-6))


# ============================================================
# 4. 內層：Optuna + 3 折 CV（回傳最佳參數、epoch 數、門檻）
# ============================================================
def inner_search(X_tr, y_tr, n_trials, seed, max_epochs, patience):
    inner = list(StratifiedKFold(N_INNER, shuffle=True, random_state=seed)
                 .split(np.zeros(len(y_tr)), y_tr))
    prepared = []
    for itr, iva in inner:
        pp = FoldPreprocessor().fit([X_tr[i] for i in itr])
        Xa, _ = pp.transform([X_tr[i] for i in itr])
        Xb, _ = pp.transform([X_tr[i] for i in iva])
        prepared.append((itr, iva, Xa, Xb))

    def objective(trial):
        p = {
            "lstm_units": trial.suggest_categorical("lstm_units", [32, 64, 96, 128]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
            "dense_units": trial.suggest_categorical("dense_units", [16, 32, 64]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "cw_multiplier": trial.suggest_float("cw_multiplier", 1.0, 3.0, step=0.25),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        }
        losses, epochs = [], []
        oof = np.full(len(y_tr), np.nan)
        for k, (itr, iva, Xa, Xb) in enumerate(prepared):
            set_seed(seed * 100 + k)
            m = build_model(Xa.shape[2], p)
            es = EarlyStopping(monitor="val_loss", patience=patience,
                               mode="min", restore_best_weights=True)
            h = m.fit(Xa, y_tr[itr], epochs=max_epochs, batch_size=p["batch_size"],
                      validation_data=(Xb, y_tr[iva]),
                      class_weight=class_weights(y_tr[itr], p["cw_multiplier"]),
                      callbacks=[es], verbose=0)
            vl = h.history["val_loss"]
            losses.append(float(np.min(vl)))
            epochs.append(int(np.argmin(vl)) + 1)
            oof[iva] = m.predict(Xb, verbose=0).ravel()
            trial.report(float(np.mean(losses)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("epochs", epochs)
        trial.set_user_attr("oof", oof.tolist())
        return float(np.mean(losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    n_epochs = int(np.median(best.user_attrs["epochs"]))
    thr = youden_threshold(y_tr, np.array(best.user_attrs["oof"]))
    return dict(best.params), n_epochs, thr, float(best.value)


# ============================================================
# 5. 修正版 permutation importance（只在真實時間步間打亂）
# ============================================================
def permutation_importance(model, X, mask, y, cols, base_auc, rng, n_rep):
    rows = []
    real = np.where(mask)                         # (patient_idx, time_idx) of real steps
    groups = {c: [c] for c in cols}
    groups.update({f"[group] {g}": [c for c in cs if c in cols]
                   for g, cs in FEATURE_GROUPS.items()})
    for name, members in groups.items():
        if not members:
            continue
        idx = [cols.index(c) for c in members]
        drops = []
        for _ in range(n_rep):
            Xp = X.copy()
            perm = rng.permutation(len(real[0]))
            for j in idx:
                vals = X[real[0], real[1], j]
                Xp[real[0], real[1], j] = vals[perm]  # 群組內各欄同一組 permutation
            p = model.predict(Xp, verbose=0).ravel()
            drops.append(base_auc - roc_auc_score(y, p))
        rows.append({"feature": name, "mean_auc_drop": float(np.mean(drops)),
                     "std_auc_drop": float(np.std(drops))})
    return rows


# ============================================================
# 6. 捷徑探針（相同外層折；模型只看單一捷徑變數）
# ============================================================
def shortcut_probe(feature, y, tr, te):
    m = LogisticRegression().fit(feature[tr, None], y[tr])
    return roc_auc_score(y[te], m.predict_proba(feature[te, None])[:, 1])


def bootstrap_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], p[i]))
    return np.percentile(a, [2.5, 97.5])


# ============================================================
# 7. 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", choices=["primary", "no_labs", "min10"], default="primary")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--importance-repeats", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="只用來測試程式能否執行")
    a = ap.parse_args()

    use_labs = a.analysis != "no_labs"
    min_records = MAX_LEN if a.analysis == "min10" else 1
    max_epochs, patience = (3, 1) if a.quick else (50, 8)

    X, y, ids, nrec, lab_meas, cols, n_removed = load_patients(use_labs, min_records)
    tag = f"v2_{a.analysis}"
    print(f"[{tag}] 病人 {len(y)}，IHCA {y.sum()}（{y.mean():.1%}）；特徵 {len(cols)} 個")
    print(f"不合理值清除筆數：{n_removed}")
    lab_missing = 1.0 - lab_meas   # 視窗內「沒抽血」列的比例（只給探針用）
    left = int(sum(np.isnan(x).sum() for x in X))
    print(f"LOCF 後視窗內仍缺失的格數（將以訓練折中位數填補）：{left}")

    fold_rows, oof_rows, imp_rows, param_rows = [], [], [], []
    t0 = time.time()
    for r in range(a.repeats):
        seed = BASE_SEED + r
        outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=seed)
        for f, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y), start=1):
            print(f"\n>>> repeat {r + 1}/{a.repeats} · fold {f}/{N_OUTER} "
                  f"(test n={len(te)}, 陽性 {y[te].sum()})  已用 {(time.time() - t0) / 60:.1f} 分")
            X_tr = [X[i] for i in tr]
            params, n_epochs, thr, inner_loss = inner_search(
                X_tr, y[tr], a.trials, seed * 10 + f, max_epochs, patience)
            print(f"    內層最佳 {params} | epochs={n_epochs} | 門檻={thr:.3f}")

            pp = FoldPreprocessor().fit(X_tr)
            Xtr, _ = pp.transform(X_tr)
            Xte, mte = pp.transform([X[i] for i in te])
            set_seed(seed * 1000 + f)
            model = build_model(Xtr.shape[2], params)
            model.fit(Xtr, y[tr], epochs=n_epochs, batch_size=params["batch_size"],
                      class_weight=class_weights(y[tr], params["cw_multiplier"]), verbose=0)
            proba = model.predict(Xte, verbose=0).ravel()
            pred = (proba >= thr).astype(int)
            auc = roc_auc_score(y[te], proba)

            row = {"repeat": r + 1, "fold": f, "n_test": len(te), "n_pos": int(y[te].sum()),
                   "auc": auc, "threshold": thr,
                   "recall": recall_score(y[te], pred, zero_division=0),
                   "precision": precision_score(y[te], pred, zero_division=0),
                   "f1": f1_score(y[te], pred, zero_division=0),
                   "probe_n_records_auc": shortcut_probe(-nrec.astype(float), y, tr, te),
                   "n_epochs": n_epochs, "inner_val_loss": inner_loss}
            row["probe_lab_missing_auc"] = shortcut_probe(lab_missing, y, tr, te)
            fold_rows.append(row)
            print(f"    AUC={auc:.3f}  recall={row['recall']:.3f}  precision={row['precision']:.3f}"
                  f"  | 捷徑探針（紀錄筆數）AUC={row['probe_n_records_auc']:.3f}")
            param_rows.append({"repeat": r + 1, "fold": f, **params,
                               "n_epochs": n_epochs, "threshold": thr})
            for i, p in zip(te, proba):
                oof_rows.append({"repeat": r + 1, "fold": f, "ID": ids[i], "y": y[i],
                                 "proba": p, "pred": int(p >= thr), "n_records": nrec[i]})
            rng = np.random.default_rng(seed * 1000 + f)
            for ir in permutation_importance(model, Xte, mte, y[te], cols, auc, rng,
                                             2 if a.quick else a.importance_repeats):
                imp_rows.append({"repeat": r + 1, "fold": f, **ir})

    folds = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)
    imp = pd.DataFrame(imp_rows)
    folds.to_csv(f"results_{tag}_folds.csv", index=False)
    oof.to_csv(f"results_{tag}_oof.csv", index=False)
    imp.to_csv(f"results_{tag}_importance.csv", index=False)
    pd.DataFrame(param_rows).to_csv(f"results_{tag}_params.csv", index=False)

    # ---------- 摘要 ----------
    summ = {"analysis": a.analysis, "n_patients": int(len(y)), "n_ihca": int(y.sum()),
            "repeats": a.repeats, "trials": a.trials, "quick_test": a.quick}
    for m in ["auc", "recall", "precision", "f1", "probe_n_records_auc",
              "probe_lab_missing_auc"]:
        summ[f"{m}_mean"] = float(folds[m].mean())
        summ[f"{m}_sd"] = float(folds[m].std(ddof=1))
    pooled = []
    for r, g in oof.groupby("repeat"):
        pooled.append(roc_auc_score(g.y, g.proba))
    summ["pooled_auc_per_repeat"] = pooled
    g0 = oof[oof.repeat == 1]
    lo, hi = bootstrap_ci(g0.y.values, g0.proba.values)
    summ["pooled_auc_repeat1_95ci"] = [float(lo), float(hi)]
    if a.repeats > 1:
        avg = oof.groupby("ID").agg(y=("y", "first"), proba=("proba", "mean"))
        summ["pooled_auc_repeat_averaged"] = float(roc_auc_score(avg.y, avg.proba))
    with open(f"results_{tag}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summ, fh, ensure_ascii=False, indent=2)

    imp_s = (imp.groupby("feature")["mean_auc_drop"].agg(["mean", "std"])
             .sort_values("mean", ascending=False))
    print("\n================ 摘要 ================")
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    print("\n特徵重要性（跨折平均 AUC 下降）：")
    print(imp_s.round(4).to_string())
    print(f"\n總耗時 {(time.time() - t0) / 60:.1f} 分鐘；輸出檔 results_{tag}_*.csv / .json")


if __name__ == "__main__":
    main()
