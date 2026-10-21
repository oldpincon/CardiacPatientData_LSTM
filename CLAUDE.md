# CLAUDE.md

本檔提供 Claude Code 在此 repo 工作時的背景資訊（`experiment` 分支）。

## 專案目的
以 LSTM 預測心臟科病人的院內心跳停止（IHCA），並與 RF / LR 及正式早期預警評分（MEWS、NEWS、NEWS2）比較。資料量很小：112 位病人、約 19 位 IHCA，所以**評估方法的正確性比模型分數重要**。

## 資料
- `CardiacPatientData.csv`：原始資料，每列為一次紀錄，同一病人內列順序視為時間順序。
- `CardiacPatientData_Cleaned.csv`：舊版的全資料常數填補。**有缺失捷徑**（沒抽血的列都被填成同一常數，模型能看出抽血頻率），v2 已不再使用。
- 標籤：`Outcome == 0` 代表 IHCA，程式內一律反轉成 `IHCA = 1 - Outcome`。
- 欄位 `Ceratinine` 的拼字錯誤是原始 CSV 的欄名，不要改。`BT` 單位為 °F。
- 每位病人取最後 `MAX_LEN = 10` 筆，padding 用 `'pre'`。

## 主要程式（v2 修補版，可報告的結果）
- `LSTM_nested_cv_v2.py`：nested CV（外層 5 折、內層 3 折加 Optuna）。缺失值處理是：不合理值 → NaN → 病人內 LOCF → 訓練折中位數。epoch 數取內層最佳 epoch 的中位數，門檻用內層 OOF 的 Youden 指數。輸出 OOF 預測、permutation importance（只在真實時間步之間打亂）、捷徑探針。
- `baselines_nested_v2.py`：RF / LR，匯入 `LSTM_nested_cv_v2.load_patients`，前處理與外層折都相同。
- `run_v2_all.bat`：跑完全部預先指定的分析，需要數小時。
- `v2_修補說明.md`：v2 修了哪些漏洞，以及哪些舊結果不可再報告。
- 工作目錄中（可能尚未 commit）：`early_warning_scores.py`、`evaluate_ews_baseline.py`、`run_ews_baseline.py`、`test_ews.py` 是 MEWS/NEWS/NEWS2 基準與單元測試。

## 舊程式（結果不可報告）
- `LSTM_main.py`、`LSTM_tuning.py`：在全部 112 人上調參，不是 nested。
- `LSTM_nested_cv_full.py`：v1 nested CV（2026-09-17 事後重建的版本），讀 Cleaned CSV，產生下列舊結果檔與 `feature_importance_*.csv`、`truncation_sweep.csv`。
- `results_rf_baseline.csv`、`results_lr_baseline.csv`、`results_lstm_nested.csv`、`results_news_baseline.csv`：用了有捷徑的 Cleaned CSV。

## 執行
Windows，先建立 `.venv`（`setup_env.bat`），依賴見 `requirements.txt`（tensorflow 2.21、numpy、pandas、scikit-learn、optuna）。

```
python LSTM_nested_cv_v2.py --trials 2 --quick           # 只測試程式能不能跑
python LSTM_nested_cv_v2.py --analysis primary --repeats 3 --trials 40
python baselines_nested_v2.py --analysis primary --repeats 3
python test_ews.py                                       # EWS 計分單元測試
```

中文輸出需要 `set PYTHONUTF8=1`。

## 預先指定的分析（三個都要跑、都要報告，不可只挑好看的）
- `primary`：全部 112 人，生命徵象 + Age/Gender + 檢驗值
- `no_labs`：不用檢驗值
- `min10`：只用紀錄 ≥ 10 筆的病人

輸出檔命名：`results_v2_<analysis>_*` 和 `results_baselines_v2_<analysis>_*`。

## 修改時的原則
- 任何需要 fit 的步驟（中位數填補、StandardScaler、調參、門檻）只能用該折的訓練病人。
- 外層切分固定為 `StratifiedKFold(5, shuffle=True, random_state=42 + repeat)`，所有模型與 EWS 必須用相同的折，才能逐折比較。
- 生理合理範圍 `PLAUSIBLE` 是看結果前預先指定的，與 `early_warning_scores.py` 一致，不要事後調整。
- v2 刻意沒改 LSTM 架構與 Optuna 搜尋空間，讓分數變化能歸因到修正本身；要改請另開版本。
- 程式註解與輸出訊息使用繁體中文。
