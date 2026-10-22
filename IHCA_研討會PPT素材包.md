# IHCA 預測研究：研討會 PPT 素材包（2026-10-13 版）

> **給生成 PPT 的 Claude**：本文件是唯一的數字來源，請勿自行補數字或改寫結論強度。投影片要用正式學術語氣（繁體中文），speaker notes 可以用口語朗讀的講稿語氣。附帶兩張圖檔：`fig1_AUC比較.png`、`fig2_臨床門檻recall.png`。
>
> **相較 2026-09-18 版的主要更新**：原本的「簡化版早期預警評分（規則式）」已全面替換為正式的 **MEWS、NEWS、NEWS2**。舊版投影片和文件中所有「簡化版評分 0.858」都要改成本文件的數字。

---

## 1. 研究題目與動機

- **題目**：以長短期記憶網路（LSTM）預測院內心臟驟停（in-hospital cardiac arrest, IHCA）
- **動機**：IHCA 預後差，發生前常有可偵測的生命徵象惡化 [1–3]。臨床常用 MEWS／NEWS2 等規則式評分，但其敏感度有限 [4, 5]。本研究評估 LSTM 能否利用生命徵象序列提供額外資訊。

## 2. 資料

| 項目 | 內容 |
|---|---|
| 來源 | 公開資料集 Rajapaksha et al. (2023) [6]，Zenodo record 7603772；斯里蘭卡 Galle 教學醫院，2018/08–2020/02 |
| 病人數 | 112 人，IHCA 陽性 19 人（17%） |
| 紀錄數 | 5,906 筆，每人 1–266 筆（中位數約 50 筆） |
| 特徵 | SBP、DBP、HR、RR、BT、SpO₂、Age、Gender、GCS、Na、K、Cl、Urea、Creatinine |
| 序列 | 每人取最後 10 筆紀錄 |
| 資料特性 | 無時間戳記；電解質缺失率 > 55%；BT 以 °F 記錄；GCS 欄有 48 筆數值 76–100，疑為 SpO₂ 誤植 |

## 3. 方法

### 3.1 LSTM（nested cross-validation）
- 外層：StratifiedKFold 5 折（random_state = 42），每折測試集 22–23 人、陽性 3–4 人
- 內層：每個外層折獨立以 Optuna [8] 搜尋超參數（100 trials），測試資料不參與調參
- 架構：雙層 LSTM [7] → Dropout → Dense → Sigmoid；使用 Masking 處理 padding、class weight 處理不平衡
- 各折選出的超參數不同：

| Fold | lstm_units | dropout | dense_units | learning_rate | cw_multiplier | batch_size |
|---|---|---|---|---|---|---|
| 1 | 96 | 0.20 | 64 | 0.005852 | 1.75 | 8 |
| 2 | 64 | 0.40 | 16 | 0.000544 | 1.00 | 8 |
| 3 | 128 | 0.10 | 64 | 0.007870 | 1.50 | 8 |
| 4 | 32 | 0.10 | 64 | 0.001638 | 1.25 | 8 |
| 5 | 64 | 0.50 | 64 | 0.007905 | 1.50 | 32 |

### 3.2 比較模型
- 機器學習基準：隨機森林、邏輯斯迴歸，使用相同外層 5 折
- **臨床基準（本次更新）**：MEWS [9]、NEWS [10]、NEWS2 [11]，依原始文獻計分表計分，程式已通過所有閾值邊界測試
  - 評分不需訓練，因此無內層調參；沿用相同外層 5 折，以便逐折比較
  - 主分析採最後 10 筆中「最後一筆」的分數
  - 預先指定的臨床切點：NEWS2 ≥ 5（緊急回應）、NEWS2 ≥ 7（高風險）、MEWS ≥ 5（critical score）

### 3.3 臨床評分的計分假設（須在方法頁或附註揭露）
1. 資料未記錄是否給氧，一律視為室內空氣（氧氣項 0 分）。
2. 無高碳酸血症資訊，NEWS2 一律使用 SpO₂ Scale 1。
3. 無 AVPU，以 GCS 近似對應：15 → A；13–14 → V；9–12 → P；≤ 8 → U。
4. 在上述條件下，NEWS 與 NEWS2 的分數完全相同，合併報告為「NEWS/NEWS2」。
5. BT 由 °F 轉換為 °C。
6. 不合理數值事先定義為缺失值並以前值補值（SBP 40–300、HR 20–250、RR 4–60、SpO₂ 50–100、GCS 3–15）。保留原值的敏感度分析結果幾乎相同（NEWS2 pooled AUC 0.858 vs 0.855）。

### 3.4 評估指標
- 各折 AUC 的平均 ± SD
- 臨床評分另報 pooled AUC（合併 112 人計算）及 bootstrap 95% CI（2,000 次）
- 在固定切點下計算 recall 與 precision

## 4. 結果

### 表 1. AUC 比較（外層 5 折）→ 對應 `fig1_AUC比較.png`

| 模型 | 各折 AUC 平均 ± SD | 各折 AUC | Pooled AUC（95% CI） |
|---|---|---|---|
| LSTM（nested CV） | 0.888 ± 0.089 | 0.934, 1.000, 0.895, 0.847, 0.764 | —† |
| 隨機森林 | 0.917 ± 0.105 | 0.842, 1.000, 1.000, 0.972, 0.771 | —† |
| 邏輯斯迴歸 | 0.812 ± 0.130 | 0.750, 1.000, 0.684, 0.889, 0.736 | —† |
| NEWS / NEWS2 | 0.857 ± 0.133 | 0.724, 0.993, 0.860, 0.986, 0.722 | 0.855（0.727–0.955） |
| MEWS | 0.807 ± 0.154 | 0.789, 1.000, 0.588, 0.896, 0.764 | 0.815（0.691–0.917） |

† 機器學習模型尚未儲存 out-of-fold 預測，無法計算 pooled AUC 或進行配對統計檢定。

### 表 2. 固定門檻下的表現（0.5 機率門檻 / 臨床切點）

| 模型 | Recall | Precision |
|---|---|---|
| LSTM（門檻 0.5） | 0.683 ± 0.273 | 0.531 ± 0.194 |
| 隨機森林（門檻 0.5） | 0.450 ± 0.411 | 0.800 ± 0.447 |
| 邏輯斯迴歸（門檻 0.5） | 0.617 ± 0.261 | 0.464 ± 0.224 |
| NEWS2 ≥ 5 | 0.526（10/19，合併） | 0.909（10/11） |
| NEWS2 ≥ 7 | 0.211（4/19） | 1.000（4/4） |
| MEWS ≥ 5 | 0.316（6/19） | 1.000（6/6） |

→ 對應 `fig2_臨床門檻recall.png`

### 特徵重要性（LSTM permutation importance，跨折平均 AUC 下降量）
- 分組：循環/呼吸（HR/RR/SBP/DBP/SpO₂）0.342 ≫ 人口學 0.051 > 檢驗值 0.044 > 意識 0.013 > 體溫 0.006
- 單一特徵：HR 0.157 > RR 0.047 > Age 0.034 > SBP 0.032；BT −0.022（無貢獻）
- 解讀：模型主要依賴 HR、RR，這兩項同樣是 NEWS2/MEWS 的核心參數，與臨床直覺一致。

### 資料洩漏檢查：截斷測試（移除每人最後 n 筆後重新評估 LSTM）

| 移除最後 n 筆 | 0 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| AUC 平均 ± SD | 0.888 ± 0.089 | 0.844 ± 0.086 | 0.839 ± 0.127 | 0.900 ± 0.069 | 0.843 ± 0.158 |

解讀：移除最後幾筆後 AUC 沒有崩跌，表示模型並非只靠最末一筆的極端值。但變異大，無法完全排除洩漏。

## 5. 結論（措辭請照此強度，不可加強）

- 在嚴謹的 nested CV 下，LSTM 的 AUC（0.888）與 RF（0.917）及正式臨床評分 NEWS2（0.857）相近，**各模型區間大幅重疊，差異未達可宣稱的程度**。
- 在臨床常用門檻 NEWS2 ≥ 5 下，19 名 IHCA 病人中有 9 名未被觸發，顯示傳統評分在本族群的敏感度有限。機器學習模型可能補足這一點，但需要更大樣本驗證。
- 本研究的貢獻在於以嚴謹的評估框架比較 LSTM 與正式臨床評分，而非宣稱模型效能較佳。

## 6. 研究限制

1. **樣本小**：112 人、19 名陽性，每折僅 3–4 名陽性，估計值變異大。
2. **無時間戳記**：最後 10 筆距離 IHCA 的時間無法確認，prediction horizon 未定義。部分極端值（如 SBP 5 mmHg、HR 7 /min）集中在 IHCA 病人的最後紀錄，可能是急救當下測得，會使所有模型的效能被高估。
3. **臨床評分輸入不完整**：缺給氧與 AVPU 資料，NEWS2 可能被低估。
4. **無單一最終模型**：nested CV 五折選出的超參數不同。
5. **電解質缺失率 > 55%**，填補方式仍待驗證。
6. **尚無統計檢定**：缺 out-of-fold 預測，未做 DeLong 等配對比較。
7. **單一中心公開資料**，未做外部驗證。

## 7. 未來工作

1. 儲存各模型 out-of-fold 預測，計算 pooled AUC 並做 DeLong 檢定
2. Repeated nested CV（多個 seed）以降低估計變異
3. 確認量測時間點，定義 prediction horizon
4. 以序列摘要特徵（mean/min/max/slope）建立混合模型
5. 尋求外部驗證資料

## 8. 建議投影片大綱（約 14 頁）

1. 封面
2. 目錄
3. 研究動機：IHCA 的臨床重要性、傳統評分的限制
4. 資料：資料集概況表（第 2 節）
5. 方法選用邏輯：為何用 nested CV、為何以正式臨床評分作為基準
6. 模型架構與超參數搜尋（第 3.1 節，含各折超參數表）
7. **臨床基準：MEWS / NEWS2 計分方式與假設**（新頁，第 3.2–3.3 節）
8. 結果一：AUC 比較（`fig1_AUC比較.png` + 表 1）
9. **結果二：臨床門檻表現**（新頁，`fig2_臨床門檻recall.png` + 表 2）
10. 特徵重要性
11. 資料洩漏檢查（截斷測試折線圖）
12. 研究限制
13. 結論與未來工作
14. 參考文獻

## 9. 參考文獻（IEEE 編號格式；[1]–[8] 沿用 09-18 版已查證之書目，[9]–[11] 為本次新增）

[1] Andersen et al., *JAMA*, 2019.
[2] Girotra et al., *N. Engl. J. Med.*, 2012.
[3] Sandroni et al., *Intensive Care Med.*, 2007.
[4] Downey et al., *Int. J. Nurs. Stud.*, 2017.
[5] Green et al., *Resuscitation*, 2018.
[6] Rajapaksha et al., *BioMedInformatics*, 2023（資料集；Zenodo record 7603772）.
[7] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, 1997.
[8] T. Akiba et al., "Optuna: A next-generation hyperparameter optimization framework," in *Proc. KDD*, 2019.
[9] C. P. Subbe, M. Kruger, P. Rutherford, and L. Gemmel, "Validation of a modified Early Warning Score in medical admissions," *QJM*, vol. 94, no. 10, pp. 521–526, 2001.
[10] Royal College of Physicians, *National Early Warning Score (NEWS): Standardising the assessment of acute-illness severity in the NHS*. London: RCP, 2012.
[11] Royal College of Physicians, *National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in the NHS*. London: RCP, 2017.

> 註：[1]–[6] 此處只列作者、期刊、年份（已於 09-18 版查證）。完整標題、卷期與 DOI 請直接沿用 09-18 版投影片的參考文獻頁，不要另外生成。
