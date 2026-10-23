import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

"""
LSTM_v3.py — 預先指定的 LSTM 改善版本（2026-10-22）
=====================================================
版本內容與比較方式見 LSTM_v3_預先指定.md（跑之前寫定，全部報告）。

  A  多視窗：訓練時每位病人取結尾在最後 1–5 筆的視窗，每位病人總權重相同
  B  小模型：單層 LSTM（8–32 units）+ L2
  C  混合：B 的 LSTM 輸出 + RF 用的摘要特徵
  D  種子集成：沿用 v2 每折最佳參數與 epoch 數，外層 5 個種子平均（不重新調參）
  E  組合：A + B + D

v4 版本（見 LSTM_v4_預先指定.md，輸出 results_v4<版本>_*）：
  F  B + 外層 5 個種子平均
  G  F + 生命徵象變化量特徵
  H  F，但 Optuna 以內層驗證 AUC 挑參數

資料前處理、外層／內層切分、epoch 數與門檻的決定方式、截斷測試都與 v2 相同
（直接匯入 LSTM_nested_cv_v2）。測試時每位病人只用最後 10 筆。
v3 不做 permutation importance 與捷徑探針（v2 已做過，前處理相同）。

執行：
  python LSTM_v3.py --variant B --analysis primary --trials 40
  快速檢查程式能跑：加 --trials 2 --quick（輸出檔名會加 _quick，不會蓋掉正式結果）
"""

import argparse
import gc
import json
import pickle
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from optuna.samplers import TPESampler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Concatenate, Dense, Dropout, Input, Masking

from baselines_nested_v2 import summarize
from LSTM_nested_cv_v2 import (BASE_SEED, MAX_LEN, N_INNER, N_OUTER, TRUNC_KS, VITALS,
                               FoldPreprocessor, bootstrap_ci, build_model as build_v2,
                               class_weights, load_patients, set_seed, shifted_window,
                               truncation_summary, youden_threshold)

N_WINDOWS = 5          # A、E：每位病人最多取幾個訓練視窗（結尾在最後 1–5 筆）
N_SEEDS = 5            # D、E：外層模型的種子數

VARIANTS = {           # 多視窗、模型（v2 / small）、混合、種子數
    "A": {"multi": True, "model": "v2", "hybrid": False, "seeds": 1},
    "B": {"multi": False, "model": "small", "hybrid": False, "seeds": 1},
    "C": {"multi": False, "model": "small", "hybrid": True, "seeds": 1},
    "D": {"multi": False, "model": "v2", "hybrid": False, "seeds": N_SEEDS},
    "E": {"multi": True, "model": "small", "hybrid": False, "seeds": N_SEEDS},
    # v4（見 LSTM_v4_預先指定.md）：以 B 為基礎
    "F": {"multi": False, "model": "small", "hybrid": False, "seeds": N_SEEDS},
    "G": {"multi": False, "model": "small", "hybrid": False, "seeds": N_SEEDS, "delta": True},
    "H": {"multi": False, "model": "small", "hybrid": False, "seeds": N_SEEDS,
          "objective": "auc"},
}
V4 = {"F", "G", "H"}


def add_delta(x, cols):
    """G：每項生命徵象加上「與上一筆的差」，視窗第一筆為 0（NaN 留給中位數填補）。"""
    v = x[:, [cols.index(c) for c in VITALS]]
    d = np.vstack([np.zeros((1, v.shape[1]), dtype=v.dtype), np.diff(v, axis=0)])
    return np.hstack([x, d])


# ============================================================
# 1. 訓練視窗
# ============================================================
def training_windows(X_full, idx, y, multi):
    """回傳 (視窗清單, 標籤, 視窗權重)。multi=False 時只有最後一個視窗，權重 1。"""
    wins, ys, ws = [], [], []
    for i in idx:
        if multi:
            w = [x for x in (shifted_window(X_full[i], j) if j else X_full[i][-MAX_LEN:]
                             for j in range(N_WINDOWS)) if x is not None]
        else:
            w = [X_full[i][-MAX_LEN:]]
        wins += w
        ys += [y[i]] * len(w)
        ws += [1.0 / len(w)] * len(w)
    return wins, np.array(ys), np.array(ws)


class Encoder:
    """序列用 v2 的 FoldPreprocessor；混合版另外把摘要特徵做訓練折中位數填補與標準化。
    兩者都只用訓練病人的最後一個視窗 fit（與 v2 相同）。"""

    def __init__(self, hybrid, cols, delta=False):
        self.hybrid, self.cols, self.delta = hybrid, cols, delta

    def _seq(self, X_list):
        return [add_delta(x, self.cols) for x in X_list] if self.delta else X_list

    def fit(self, X_list):
        self.pp = FoldPreprocessor().fit(self._seq(X_list))
        if self.hybrid:
            S = summarize(X_list, self.cols)[0]
            self.imp = SimpleImputer(strategy="median").fit(S)
            self.sc = StandardScaler().fit(self.imp.transform(S))
        return self

    def transform(self, X_list):
        seq = self.pp.transform(self._seq(X_list))[0]
        if not self.hybrid:
            return seq
        S = summarize(X_list, self.cols)[0]
        return [seq, self.sc.transform(self.imp.transform(S)).astype("float32")]


# ============================================================
# 2. 模型與搜尋空間
# ============================================================
def build_small(n_feat, p, n_sum=None):
    reg = regularizers.l2(p["l2"])
    seq_in = Input(shape=(MAX_LEN, n_feat))
    h = Masking(mask_value=0.0)(seq_in)
    h = LSTM(p["lstm_units"], kernel_regularizer=reg)(h)
    h = Dropout(p["dropout"])(h)
    inputs = [seq_in]
    if n_sum:
        sum_in = Input(shape=(n_sum,))
        h = Concatenate()([h, sum_in])
        inputs.append(sum_in)
    if p["dense_units"]:
        h = Dense(p["dense_units"], activation="relu", kernel_regularizer=reg)(h)
    out = Dense(1, activation="sigmoid", kernel_regularizer=reg)(h)
    model = Model(inputs if n_sum else seq_in, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p["learning_rate"]),
                  loss="binary_crossentropy")
    return model


def make_model(cfg, Xa, p):
    n_feat = (Xa[0] if cfg["hybrid"] else Xa).shape[2]
    if cfg["model"] == "v2":
        return build_v2(n_feat, p)
    return build_small(n_feat, p, Xa[1].shape[1] if cfg["hybrid"] else None)


def suggest(trial, cfg):
    if cfg["model"] == "v2":                 # 與 v2 相同的搜尋空間
        p = {"lstm_units": trial.suggest_categorical("lstm_units", [32, 64, 96, 128]),
             "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
             "dense_units": trial.suggest_categorical("dense_units", [16, 32, 64])}
    else:
        p = {"lstm_units": trial.suggest_categorical("lstm_units", [8, 16, 32]),
             "dropout": trial.suggest_float("dropout", 0.2, 0.5, step=0.1),
             "l2": trial.suggest_float("l2", 1e-4, 1e-2, log=True),
             "dense_units": trial.suggest_categorical("dense_units", [0, 8, 16])}
    p["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    p["cw_multiplier"] = trial.suggest_float("cw_multiplier", 1.0, 3.0, step=0.25)
    p["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32])
    return p


def sample_weights(y_patients, y_win, w_win, mult):
    cw = class_weights(y_patients, mult)     # 類別權重以病人計算
    return np.array([cw[int(t)] for t in y_win]) * w_win


# ============================================================
# 3. 內層：Optuna + 3 折（以病人為單位切分）
# ============================================================
def inner_search(cfg, cols, X_full, tr, y, n_trials, seed, max_epochs, patience):
    y_tr = y[tr]
    inner = list(StratifiedKFold(N_INNER, shuffle=True, random_state=seed)
                 .split(np.zeros(len(tr)), y_tr))
    prepared = []
    for itr, iva in inner:
        enc = Encoder(cfg["hybrid"], cols, cfg.get("delta", False)).fit(
            [X_full[tr[i]][-MAX_LEN:] for i in itr])
        wins, yw, ww = training_windows(X_full, tr[itr], y, cfg["multi"])
        Xa = enc.transform(wins)
        Xb = enc.transform([X_full[tr[i]][-MAX_LEN:] for i in iva])
        prepared.append((itr, iva, Xa, yw, ww, Xb))

    def objective(trial):
        p = suggest(trial, cfg)
        losses, epochs = [], []
        oof = np.full(len(y_tr), np.nan)
        for k, (itr, iva, Xa, yw, ww, Xb) in enumerate(prepared):
            set_seed(seed * 100 + k)
            m = make_model(cfg, Xa, p)
            es = EarlyStopping(monitor="val_loss", patience=patience,
                               mode="min", restore_best_weights=True)
            h = m.fit(Xa, yw, epochs=max_epochs, batch_size=p["batch_size"],
                      validation_data=(Xb, y_tr[iva]),
                      sample_weight=sample_weights(y_tr[itr], yw, ww, p["cw_multiplier"]),
                      callbacks=[es], verbose=0)
            vl = h.history["val_loss"]
            epochs.append(int(np.argmin(vl)) + 1)
            oof[iva] = m.predict(Xb, verbose=0).ravel()
            if cfg.get("objective") == "auc":    # H：以內層驗證 AUC 挑參數（取負值最小化）
                losses.append(-roc_auc_score(y_tr[iva], oof[iva]))
            else:
                losses.append(float(np.min(vl)))
            del m, h
            gc.collect()
            trial.report(float(np.mean(losses)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("epochs", epochs)
        trial.set_user_attr("oof", oof.tolist())
        return float(np.mean(losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    n_epochs = int(np.median(best.user_attrs["epochs"]))
    thr = youden_threshold(y_tr, np.array(best.user_attrs["oof"]))
    return dict(best.params), n_epochs, thr


def v2_params(analysis, repeat, fold):
    """D：讀 v2 該折的最佳參數、epoch 數與門檻（v2 的內層結果，不重新調參）。"""
    df = pd.read_csv(f"results_v2_{analysis}_params.csv")
    row = df[(df.repeat == repeat) & (df.fold == fold)].iloc[0]
    p = {"lstm_units": int(row.lstm_units), "dropout": float(row.dropout),
         "dense_units": int(row.dense_units), "learning_rate": float(row.learning_rate),
         "cw_multiplier": float(row.cw_multiplier), "batch_size": int(row.batch_size)}
    return p, int(row.n_epochs), float(row.threshold)


# ============================================================
# 4. 主程式
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--analysis", choices=["primary", "no_labs", "min10"], default="primary")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--quick", action="store_true", help="只用來測試程式能否執行")
    ap.add_argument("--one-fold", action="store_true",
                    help="只跑一個尚未完成的折就結束（exit code 3），由外部迴圈重複呼叫")
    a = ap.parse_args()
    cfg = VARIANTS[a.variant]

    use_labs = a.analysis != "no_labs"
    min_records = MAX_LEN if a.analysis == "min10" else 1
    max_epochs, patience = (3, 1) if a.quick else (50, 8)
    n_seeds = 2 if (a.quick and cfg["seeds"] > 1) else cfg["seeds"]

    X, y, ids, nrec, _, cols, _, X_full = load_patients(use_labs, min_records, full=True)
    tag = (f"v{4 if a.variant in V4 else 3}{a.variant}_{a.analysis}"
           f"{'_quick' if a.quick else ''}")
    print(f"[{tag}] 病人 {len(y)}，IHCA {y.sum()}；特徵 {len(cols)} 個；設定 {cfg}")

    ckpt_dir = f"ckpt_{tag}_t{a.trials}"
    os.makedirs(ckpt_dir, exist_ok=True)
    fold_rows, oof_rows, param_rows, trunc_rows = [], [], [], []
    t0 = time.time()
    for r in range(a.repeats):
        seed = BASE_SEED + r
        outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=seed)
        for f, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y), start=1):
            ckpt = os.path.join(ckpt_dir, f"r{r + 1}_f{f}.pkl")
            if os.path.exists(ckpt):
                with open(ckpt, "rb") as fh:
                    saved = pickle.load(fh)
                fold_rows += saved["fold"]
                param_rows += saved["param"]
                oof_rows += saved["oof"]
                trunc_rows += saved["trunc"]
                print(f"\n>>> repeat {r + 1} · fold {f} 已完成，讀取存檔 {ckpt}"
                      f"（AUC={saved['fold'][0]['auc']:.3f}）")
                continue
            n0 = (len(fold_rows), len(param_rows), len(oof_rows), len(trunc_rows))
            print(f"\n>>> repeat {r + 1}/{a.repeats} · fold {f}/{N_OUTER} "
                  f"(test n={len(te)}, 陽性 {y[te].sum()})  已用 {(time.time() - t0) / 60:.1f} 分")
            if a.variant == "D":
                params, n_epochs, thr = v2_params(a.analysis, r + 1, f)
            else:
                params, n_epochs, thr = inner_search(cfg, cols, X_full, tr, y, a.trials,
                                                     seed * 10 + f, max_epochs, patience)
            print(f"    參數 {params} | epochs={n_epochs} | 門檻={thr:.3f}")

            enc = Encoder(cfg["hybrid"], cols, cfg.get("delta", False)).fit([X[i] for i in tr])
            wins, yw, ww = training_windows(X_full, tr, y, cfg["multi"])
            Xtr = enc.transform(wins)
            sw = sample_weights(y[tr], yw, ww, params["cw_multiplier"])
            models = []
            for s in range(n_seeds):          # s=0 與 v2 的外層種子相同
                set_seed(seed * 1000 + f + s * 100000)
                m = make_model(cfg, Xtr, params)
                m.fit(Xtr, yw, epochs=n_epochs, batch_size=params["batch_size"],
                      sample_weight=sw, verbose=0)
                models.append(m)

            def predict(win_list):
                Z = enc.transform(win_list)
                return np.mean([m.predict(Z, verbose=0).ravel() for m in models], axis=0)

            proba = predict([X[i] for i in te])
            pred = (proba >= thr).astype(int)
            auc = roc_auc_score(y[te], proba)
            fold_rows.append({"repeat": r + 1, "fold": f, "n_test": len(te),
                              "n_pos": int(y[te].sum()), "auc": auc, "threshold": thr,
                              "recall": recall_score(y[te], pred, zero_division=0),
                              "precision": precision_score(y[te], pred, zero_division=0),
                              "f1": f1_score(y[te], pred, zero_division=0),
                              "n_epochs": n_epochs, "n_train_windows": len(wins),
                              "n_seeds": n_seeds})
            print(f"    AUC={auc:.3f}  recall={fold_rows[-1]['recall']:.3f}  "
                  f"precision={fold_rows[-1]['precision']:.3f}  訓練視窗 {len(wins)} 個")
            param_rows.append({"repeat": r + 1, "fold": f, **params,
                               "n_epochs": n_epochs, "threshold": thr})
            for i, p in zip(te, proba):
                oof_rows.append({"repeat": r + 1, "fold": f, "ID": ids[i], "y": y[i],
                                 "proba": p, "pred": int(p >= thr), "n_records": nrec[i]})

            trunc_msg = []
            for k in TRUNC_KS:
                keep = [i for i in te if shifted_window(X_full[i], k) is not None]
                pk = predict([shifted_window(X_full[i], k) for i in keep])
                for i, p in zip(keep, pk):
                    trunc_rows.append({"repeat": r + 1, "fold": f, "k": k, "ID": ids[i],
                                       "y": y[i], "proba": p})
                if len(np.unique(y[keep])) == 2:
                    trunc_msg.append(f"k={k}:{roc_auc_score(y[keep], pk):.3f}")
            print(f"    截斷測試 AUC  {'  '.join(trunc_msg)}")

            with open(ckpt + ".tmp", "wb") as fh:
                pickle.dump({"fold": fold_rows[n0[0]:], "param": param_rows[n0[1]:],
                             "oof": oof_rows[n0[2]:], "trunc": trunc_rows[n0[3]:]}, fh)
            os.replace(ckpt + ".tmp", ckpt)
            if a.one_fold:
                print(f"    已存檔 {ckpt}；--one-fold 模式，結束程序以釋放記憶體")
                raise SystemExit(3)
            del models
            tf.keras.backend.clear_session()
            gc.collect()

    folds, oof, trunc = pd.DataFrame(fold_rows), pd.DataFrame(oof_rows), pd.DataFrame(trunc_rows)
    folds.to_csv(f"results_{tag}_folds.csv", index=False)
    oof.to_csv(f"results_{tag}_oof.csv", index=False)
    pd.DataFrame(param_rows).to_csv(f"results_{tag}_params.csv", index=False)
    trunc.to_csv(f"results_{tag}_truncation_oof.csv", index=False)
    trunc_s = truncation_summary(oof, trunc)
    trunc_s.to_csv(f"results_{tag}_truncation.csv", index=False)

    summ = {"variant": a.variant, "config": cfg, "analysis": a.analysis,
            "n_patients": int(len(y)), "n_ihca": int(y.sum()), "repeats": a.repeats,
            "trials": a.trials, "quick_test": a.quick}
    for m in ["auc", "recall", "precision", "f1"]:
        summ[f"{m}_mean"] = float(folds[m].mean())
        summ[f"{m}_sd"] = float(folds[m].std(ddof=1))
    summ["pooled_auc_per_repeat"] = [roc_auc_score(g.y, g.proba) for _, g in oof.groupby("repeat")]
    g0 = oof[oof.repeat == 1]
    summ["pooled_auc_repeat1_95ci"] = [float(v) for v in bootstrap_ci(g0.y.values, g0.proba.values)]
    with open(f"results_{tag}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summ, fh, ensure_ascii=False, indent=2)
    print("\n================ 摘要 ================")
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    print("\n截斷測試（視窗往前移 k 筆）：")
    print(trunc_s.round(3).to_string(index=False))
    print(f"\n總耗時 {(time.time() - t0) / 60:.1f} 分鐘；輸出檔 results_{tag}_*")


if __name__ == "__main__":
    main()
