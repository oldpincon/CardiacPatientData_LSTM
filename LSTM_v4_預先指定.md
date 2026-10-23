# LSTM v4：預先指定的改善版本與確認步驟（2026-10-23，跑之前寫定）

v3 的結果：沒有任何版本顯著優於 v2 LSTM 或 RF；三個分析平均最高的是 B 小模型（0.840）。
v4 以 B 為基礎再試三個版本，並加上**確認步驟**，避免挑到「碰巧符合 repeat 1 切分」的版本。

## 與 v3 相同、不可更動的部分
資料前處理、外層／內層切分、Optuna 40 trials、epoch 數與門檻的決定方式、截斷測試、測試時只用最後 10 筆，全部與 v3 相同。
B 的搜尋空間：單層 LSTM units ∈ {8, 16, 32}、dropout 0.2–0.5、L2 ∈ [1e-4, 1e-2]、dense ∈ {0, 8, 16}，學習率、class weight、batch size 同 v2。

## 三個版本
| 代號 | 內容 |
|---|---|
| F | B + 外層 5 個種子平均 |
| G | F + 變化量特徵：7 項生命徵象各加一個「與上一筆的差」（視窗第一筆為 0），與原特徵一起做訓練折中位數填補與標準化 |
| H | F，但 Optuna 以內層 3 折驗證 AUC 的平均（取負值最小化）挑參數；epoch 數仍以內層 val_loss 的 early stopping 決定 |

## 第一階段：repeat 1（與 v2、v3 相同的外層折）
F、G、H 各跑 primary、no_labs、min10。

## 挑選規則（事先固定）
三個分析的 repeat 1 pooled AUC **平均最高**者為 v4 最終版本；平手（差 < 0.001）時依 F → G → H 的順序取較簡單者。由 `select_v4.py` 自動挑選，寫入 `results_v4_selection.json`。

## 第二階段：確認
只有被選中的版本，以 `--repeats 3` 再跑 repeat 2、3（外層 `random_state` 43、44，是 v4 挑選時沒看過的新切分）。
確認結果 = repeat 2、3 的 pooled AUC，以及每個 repeat 內與 RF、LR、NEWS2_last 的 DeLong 比較（3 個分析 × 2 個 repeat，每個分析內 6 組 Holm 校正）。
注意：病人還是同一批 112 人，新切分只能排除「切分運氣」，不能取代外部驗證。

## 比較與報告（`delong_v4.py` → `results_delong_v4.csv`）
- 第一階段：F、G、H 各自 vs v2 LSTM、v3B、RF、LR、NEWS2_last，每個分析 15 組，Holm 校正
- 第二階段：如上
- 全部版本、全部分析都報告

## 輸出
`LSTM_v3.py --variant {F,G,H}`（v4 版本加在同一支程式），輸出 `results_v4<variant>_<analysis>_*`。
