"""
select_v4.py — 依預先指定的規則挑出 v4 最終版本（見 LSTM_v4_預先指定.md）
規則：F、G、H 三個分析 repeat 1 pooled AUC 平均最高者；差 < 0.001 時依 F → G → H 取較簡單者。
輸出：results_v4_selection.json（印出被選中的版本代號，供執行腳本使用）
"""

import json
import sys

ORDER = ["F", "G", "H"]
ANALYSES = ["primary", "no_labs", "min10"]


def main():
    table = {}
    for v in ORDER:
        aucs = []
        for a in ANALYSES:
            with open(f"results_v4{v}_{a}_summary.json", encoding="utf-8") as fh:
                aucs.append(json.load(fh)["pooled_auc_per_repeat"][0])
        table[v] = {"per_analysis": dict(zip(ANALYSES, aucs)), "mean": sum(aucs) / len(aucs)}
    best = max(t["mean"] for t in table.values())
    chosen = next(v for v in ORDER if table[v]["mean"] > best - 0.001)
    with open("results_v4_selection.json", "w", encoding="utf-8") as fh:
        json.dump({"chosen": chosen, "rule": "三個分析 repeat 1 pooled AUC 平均最高；"
                   "差 < 0.001 依 F→G→H", "table": table}, fh, ensure_ascii=False, indent=2)
    for v, t in table.items():
        print(f"{v}: " + "  ".join(f"{a}={x:.3f}" for a, x in t["per_analysis"].items())
              + f"  平均={t['mean']:.3f}", file=sys.stderr)
    print(chosen)


if __name__ == "__main__":
    main()
