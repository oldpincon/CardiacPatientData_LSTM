"""
early_warning_scores.py
正式早期預警評分：MEWS (Subbe et al., 2001)、NEWS (RCP, 2012)、NEWS2 (RCP, 2017)
用來取代先前的「簡化版規則式評分」基準。

計分表來源
- NEWS2 / NEWS : Royal College of Physicians. National Early Warning Score (NEWS) 2. London: RCP, 2017.
                 NEWS (2012) 與 NEWS2 在 RR、SBP、HR、BT 閾值相同；差別在 SpO2 Scale 2 與
                 意識（NEWS2 加入 new confusion，ACVPU）。
- MEWS          : Subbe CP, Kruger M, Rutherford P, Gemmel L. QJM 2001;94(10):521-526.

本資料集的必要假設（必須在論文 / 研討會揭露）
1. 資料無「是否使用氧氣」欄位 → 一律視為室內空氣（air, 0 分）。
   若有病人實際使用氧氣，NEWS/NEWS2 會被低估。
2. 資料無高碳酸血症呼吸衰竭（hypercapnic respiratory failure）資訊 → NEWS2 一律使用 SpO2 Scale 1。
3. 資料只有 GCS，沒有 AVPU / ACVPU → 以 GCS 近似對應：
     GCS 15 → A；13–14 → V；9–12 → P；≤8 → U
   （此為近似，非正式量表；可用 GCS_TO_AVPU 調整做敏感度分析）
   NEWS2 的 "new confusion" 無法與既有意識障礙區分，GCS < 15 一律計為非 Alert（3 分）。
   → 在上述假設下，NEWS 與 NEWS2 在本資料集中的分數「完全相同」。
4. 缺失值：同一病人內以前值延續（LOCF）；仍缺失者該項計 0 分（視為正常），
   並由 report_missing() 回報計 0 分的筆數。
5. 本資料集 BT 為 °F 整數（94–100），計分前以 clean_vitals() 轉為 °C。
6. 生理上不可能的數值（範圍見 PLAUSIBLE，於看結果前預先指定）視為缺失再 LOCF。
   GCS 欄位中 76–100 的數值明顯為 SpO2 誤植。主分析清除；敏感度分析保留原值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# GCS → AVPU 近似對應
# ---------------------------------------------------------------------------
def GCS_TO_AVPU(gcs: float) -> str | None:
    if gcs is None or pd.isna(gcs):
        return None
    if gcs >= 15:
        return "A"
    if gcs >= 13:
        return "V"
    if gcs >= 9:
        return "P"
    return "U"


def _na(x) -> bool:
    return x is None or pd.isna(x)


# ---------------------------------------------------------------------------
# NEWS2 (RCP 2017)，SpO2 Scale 1
# ---------------------------------------------------------------------------
def news2_rr(rr):
    if _na(rr): return 0
    if rr <= 8: return 3
    if rr <= 11: return 1
    if rr <= 20: return 0
    if rr <= 24: return 2
    return 3


def news2_spo2_scale1(spo2):
    if _na(spo2): return 0
    if spo2 <= 91: return 3
    if spo2 <= 93: return 2
    if spo2 <= 95: return 1
    return 0


def news2_spo2_scale2(spo2, on_oxygen: bool):
    """保留供未來有 hypercapnic 資訊時使用；本資料集不使用。"""
    if _na(spo2): return 0
    if spo2 <= 83: return 3
    if spo2 <= 85: return 2
    if spo2 <= 87: return 1
    if spo2 <= 92: return 0
    if not on_oxygen: return 0          # ≥93 on air
    if spo2 <= 94: return 1
    if spo2 <= 96: return 2
    return 3


def news2_oxygen(on_oxygen: bool):
    return 2 if on_oxygen else 0


def news2_sbp(sbp):
    if _na(sbp): return 0
    if sbp <= 90: return 3
    if sbp <= 100: return 2
    if sbp <= 110: return 1
    if sbp <= 219: return 0
    return 3


def news2_hr(hr):
    if _na(hr): return 0
    if hr <= 40: return 3
    if hr <= 50: return 1
    if hr <= 90: return 0
    if hr <= 110: return 1
    if hr <= 130: return 2
    return 3


def news2_consciousness(gcs):
    avpu = GCS_TO_AVPU(gcs)
    if avpu is None: return 0
    return 0 if avpu == "A" else 3


def news2_temp(bt):
    if _na(bt): return 0
    if bt <= 35.0: return 3
    if bt <= 36.0: return 1
    if bt <= 38.0: return 0
    if bt <= 39.0: return 1
    return 2


def news2_row(r, on_oxygen: bool = False) -> int:
    return (news2_rr(r["RR"]) + news2_spo2_scale1(r["SpO2"]) + news2_oxygen(on_oxygen)
            + news2_sbp(r["SBP"]) + news2_hr(r["HR"])
            + news2_consciousness(r["GCS"]) + news2_temp(r["BT"]))


def news2_any_red(r) -> bool:
    """任一單項 = 3（RCP 的 'red score' 觸發）"""
    return 3 in (news2_rr(r["RR"]), news2_spo2_scale1(r["SpO2"]), news2_sbp(r["SBP"]),
                 news2_hr(r["HR"]), news2_consciousness(r["GCS"]), news2_temp(r["BT"]))


# NEWS (2012)：本資料集條件下（Scale 1、無氧氣、GCS→AVPU）與 NEWS2 計分相同
def news_row(r, on_oxygen: bool = False) -> int:
    return news2_row(r, on_oxygen)


# ---------------------------------------------------------------------------
# MEWS (Subbe et al., 2001)
# ---------------------------------------------------------------------------
def mews_sbp(sbp):
    if _na(sbp): return 0
    if sbp <= 70: return 3
    if sbp <= 80: return 2
    if sbp <= 100: return 1
    if sbp <= 199: return 0
    return 2


def mews_hr(hr):
    if _na(hr): return 0
    if hr <= 40: return 2
    if hr <= 50: return 1
    if hr <= 100: return 0
    if hr <= 110: return 1
    if hr <= 129: return 2
    return 3


def mews_rr(rr):
    if _na(rr): return 0
    if rr < 9: return 2
    if rr <= 14: return 0
    if rr <= 20: return 1
    if rr <= 29: return 2
    return 3


def mews_temp(bt):
    if _na(bt): return 0
    if bt < 35.0: return 2
    if bt < 38.5: return 0
    return 2


def mews_avpu(gcs):
    return {"A": 0, "V": 1, "P": 2, "U": 3, None: 0}[GCS_TO_AVPU(gcs)]


def mews_row(r) -> int:
    return (mews_sbp(r["SBP"]) + mews_hr(r["HR"]) + mews_rr(r["RR"])
            + mews_temp(r["BT"]) + mews_avpu(r["GCS"]))


# ---------------------------------------------------------------------------
# 病人層級分數
# ---------------------------------------------------------------------------
VITALS = ["SBP", "HR", "RR", "BT", "SpO2", "GCS"]

# 預先指定的生理合理範圍（含端點）；BT 為 °C
PLAUSIBLE = {"SBP": (40, 300), "HR": (20, 250), "RR": (4, 60),
             "SpO2": (50, 100), "BT": (30.0, 45.0), "GCS": (3, 15)}


def clean_vitals(df: pd.DataFrame, temp_unit: str = "F",
                 drop_implausible: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """°F→°C；可選擇把不合理值設為 NaN。回傳 (清理後資料, 各欄被清除筆數)。"""
    out = df.copy()
    out[VITALS] = out[VITALS].astype(float)
    if temp_unit.upper() == "F":
        out["BT"] = (out["BT"] - 32.0) * 5.0 / 9.0
    removed = pd.Series(0, index=VITALS)
    if drop_implausible:
        for c, (lo, hi) in PLAUSIBLE.items():
            m = (out[c] < lo) | (out[c] > hi)
            removed[c] = int(m.sum())
            out.loc[m, c] = np.nan
    return out, removed


def report_missing(df: pd.DataFrame, id_col: str = "patient_id",
                   last_n: int = 10) -> pd.Series:
    """LOCF 後仍缺失、因此被計 0 分的筆數（只計算評估視窗內）。"""
    w = df.groupby(id_col, sort=False).tail(last_n).copy()
    w[VITALS] = w.groupby(id_col, sort=False)[VITALS].ffill()
    return w[VITALS].isna().sum()


def patient_scores(df: pd.DataFrame, id_col: str = "patient_id",
                   last_n: int = 10) -> pd.DataFrame:
    """
    每位病人取最後 last_n 筆紀錄（與 LSTM 輸入視窗一致；列順序視為時間順序）。
    回傳兩種預先指定的彙總方式：
      *_last : 視窗中最後一筆紀錄的分數（主要分析）
      *_max  : 視窗內最高分（敏感度分析）
    """
    w = df.groupby(id_col, sort=False).tail(last_n).copy()
    w[VITALS] = w.groupby(id_col, sort=False)[VITALS].ffill()
    w["NEWS2"] = w.apply(news2_row, axis=1)
    w["MEWS"] = w.apply(mews_row, axis=1)
    w["NEWS2_red"] = w.apply(news2_any_red, axis=1)
    g = w.groupby(id_col, sort=False)
    out = pd.DataFrame({
        "NEWS2_last": g["NEWS2"].last(),
        "NEWS2_max": g["NEWS2"].max(),
        "MEWS_last": g["MEWS"].last(),
        "MEWS_max": g["MEWS"].max(),
        "NEWS2_red_last": g["NEWS2_red"].last(),
    })
    return out
