# 變化量特徵給 RF / LR：預先指定（2026-10-28，跑之前寫定）

## 要回答的問題
v4G（LSTM + 每一筆「與上一筆的差」）比 v4F（沒有變化量）高。進步是來自 LSTM 能利用變化量，
還是變化量這個資訊本身就有用？把同樣的資訊給 RF / LR，看它們會不會也進步。

## 做法
`baselines_nested_v2.py --delta`：原本的摘要特徵不變（生命徵象最後值、平均、最小、最大、標準差、斜率；
Age、Gender；檢驗值最後值），每項生命徵象**另外加 4 個**由「與上一筆的差」算出的特徵：

| 特徵 | 意義 |
|---|---|
| `delta_last` | 最後一筆與前一筆的差 |
| `delta_min` | 視窗內最大的單次下降 |
| `delta_max` | 視窗內最大的單次上升 |
| `delta_absmean` | 單次變化絕對值的平均（波動程度） |

7 項生命徵象 × 4 = 28 個新特徵。視窗內有效值少於 2 筆時這 4 個特徵為 0（與 v4G 視窗第一筆為 0 一致）。
其他全部與原本的 RF / LR 相同：前處理、外層折（3 個 repeat）、內層 GridSearch 與搜尋範圍、門檻、截斷測試。

## 比較（`delong_delta.py` → `results_delong_delta.csv`）
每個分析、每個 repeat（1–3）比 4 組，共 12 組，Holm 校正：
1. RF_delta vs RF
2. LR_delta vs LR
3. v4G vs RF_delta
4. v4G vs LR_delta

## 事先說好的解讀
- RF_delta / LR_delta 沒有比原本好 → 支持「LSTM 比較會利用變化量」
- RF_delta / LR_delta 也進步 → 進步主要來自變化量特徵本身，不是 LSTM 架構
- 三個分析都要報告

## 輸出
`results_baselines_v2_delta_<analysis>_*`
