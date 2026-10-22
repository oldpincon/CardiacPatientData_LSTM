# 正式早期預警評分基準（MEWS / NEWS / NEWS2）：方法與結果

> 研討會／論文用文字。數字來自 `run_ews_baseline.py`（2026-10-13 執行）。

## 方法

**比較基準。** 本研究以三種已發表的早期預警評分作為臨床比較基準，取代先前自訂的簡化規則式評分：Modified Early Warning Score（MEWS；Subbe 等，2001）、National Early Warning Score（NEWS；Royal College of Physicians, 2012）及 NEWS2（Royal College of Physicians, 2017）。各評分依原始文獻計分表逐項計分（MEWS 0–14 分；NEWS/NEWS2 0–20 分），程式實作已通過所有閾值邊界值的單元測試。

**對應本資料集的計分假設。** (1) 資料未記錄是否使用氧氣，故一律視為室內空氣（氧氣項 0 分）。(2) 無高碳酸血症呼吸衰竭資訊，故 NEWS2 一律使用 SpO₂ Scale 1。(3) 資料僅有 GCS，故以近似對應轉為 AVPU：GCS 15 對應 A，13–14 對應 V，9–12 對應 P，≤8 對應 U。NEWS2 的「新發生意識混亂」無法與既有意識障礙區分，GCS < 15 一律計為非清醒。在上述條件下，NEWS 與 NEWS2 的計分完全相同，以下合併以 NEWS/NEWS2 報告。(4) 體溫原始單位為 °F，計分前轉換為 °C。(5) 生理上不可能的數值於檢視結果前預先定義為缺失值，範圍為：SBP 40–300 mmHg、HR 20–250 /min、RR 4–60 /min、SpO₂ 50–100%、GCS 3–15。此規則清除 SBP 8 筆、HR 14 筆、RR 14 筆、SpO₂ 10 筆、GCS 48 筆，其中 GCS 欄的 76–100 疑為 SpO₂ 誤植。清除後以同一病人前值延續（LOCF）補值，仍缺失者該項計 0 分；評估視窗內僅 3 筆 SpO₂ 屬此情形。

**評估方式。** 每位病人取最後 10 筆紀錄，與 LSTM 輸入視窗相同。主分析採視窗內最後一筆的分數，敏感度分析採視窗內最高分。評分系統不需訓練，因此無內層調參；為與 LSTM、隨機森林（RF）及邏輯斯迴歸（LR）逐折比較，沿用完全相同的外層分層 5 折切分（StratifiedKFold，random_state = 42；各折陽性數為 4、4、3、4、4）。報告各折 AUC 的平均 ± 標準差，以及合併 112 名病人計算的 pooled AUC 與 bootstrap 95% 信賴區間（2,000 次）。另以預先指定的臨床切點評估 recall 與 precision：NEWS2 ≥ 5（緊急回應門檻）、NEWS2 ≥ 7（高風險）、MEWS ≥ 5（Subbe 的 critical score）。

## 結果

**表 1. 外層 5 折 AUC（n = 112，IHCA 陽性 19 人）**

| 模型 | 各折 AUC 平均 ± SD | Pooled AUC（95% CI） |
|---|---|---|
| NEWS / NEWS2（最後一筆） | 0.857 ± 0.133 | 0.855（0.727–0.955） |
| NEWS / NEWS2（視窗最高分） | 0.825 ± 0.138 | 0.827（0.697–0.935） |
| MEWS（最後一筆） | 0.807 ± 0.154 | 0.815（0.691–0.917） |
| MEWS（視窗最高分） | 0.848 ± 0.202 | 0.839（0.711–0.946） |
| LSTM（nested CV） | 0.888 ± 0.089 | — † |
| 隨機森林 | 0.917 ± 0.105 | — † |
| 邏輯斯迴歸 | 0.812 ± 0.130 | — † |

† 機器學習模型目前未存 out-of-fold 預測值，尚無法計算 pooled AUC 及進行配對比較。

**表 2. 臨床切點表現（合併 5 折）**

| 規則 | 觸發人數 | Recall | Precision |
|---|---|---|---|
| NEWS2 ≥ 5 | 11 | 0.526（10/19） | 0.909（10/11） |
| NEWS2 ≥ 7 | 4 | 0.211（4/19） | 1.000（4/4） |
| MEWS ≥ 5 | 6 | 0.316（6/19） | 1.000（6/6） |

敏感度分析保留不合理原始值時，結果幾乎不變：NEWS/NEWS2 最後一筆 pooled AUC 0.858，MEWS 最後一筆 0.815。

## 結果解讀

- 各模型 AUC 的信賴區間大幅重疊。LSTM（0.888）與 NEWS2（0.857）的差異遠小於單折的變異（SD 約 0.09–0.15），本資料量**不足以宣稱 LSTM 優於傳統評分**。
- 在臨床常用門檻 NEWS2 ≥ 5 下，約半數 IHCA 病人未被觸發（recall 0.53）。這是傳統評分在本族群的實際限制，也是機器學習模型可能補足之處；但仍需更大樣本才能驗證。

## 研究限制（須於報告中揭露）

1. 樣本數小：112 人，IHCA 19 人，每折測試集僅 3–4 名陽性病人，各折 AUC 變異大。
2. 資料無時間戳記：「最後 10 筆」距離 IHCA 發生的時間無法確認。不合理數值（如 SBP 5 mmHg、HR 7 /min）集中出現在 IHCA 病人的最後紀錄，部分可能是急救當下測得。若是如此，所有模型（含傳統評分）的效能都會被高估。
3. 缺少給氧與 AVPU 資料：NEWS2 的氧氣項無法計分，意識項由 GCS 近似，NEWS2 可能被低估。

## 參考文獻

- Royal College of Physicians. *National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in the NHS.* London: RCP; 2017.
- Royal College of Physicians. *National Early Warning Score (NEWS): Standardising the assessment of acute-illness severity in the NHS.* London: RCP; 2012.
- Subbe CP, Kruger M, Rutherford P, Gemmel L. Validation of a modified Early Warning Score in medical admissions. *QJM.* 2001;94(10):521–526.
