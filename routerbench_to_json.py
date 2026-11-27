#!/usr/bin/env python3
"""
把 RouterBench 的 pkl（routerbench_0shot.pkl / routerbench_5shot.pkl）中的 MMLU 題目
轉成 MMLU 答對率瀏覽器（mmlu_viewer.html）可讀的 JSON。

用法：
    python routerbench_to_json.py routerbench_0shot.pkl -o mmlu_routerbench.json
    python routerbench_to_json.py routerbench_0shot.pkl --limit 500     # 只取前 500 題
    python routerbench_to_json.py routerbench_0shot.pkl --subjects high-school-biology,virology

說明：
- 模型欄位：凡是同時存在「<model>」與「<model>|model_response」兩欄者，視為一個模型。
- 正確答案：RouterBench 沒有存正解欄位，這裡用「答對（score=1）的模型所選字母」的多數決反推；
  如果沒有任何模型答對，gold 會是 null（頁面上會顯示「無法反推」）。
"""
import argparse
import json
import re
from collections import Counter

import pandas as pd

LETTER_RE = re.compile(r"\b([A-D])\b")


def extract_letter(resp):
    if not isinstance(resp, str):
        return None
    s = resp.strip()
    # 常見格式：「B」「B)」「(B)」「Answer: B」
    m = re.match(r"^\(?([A-D])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"[Aa]nswer\s*(?:is)?\s*[:：]?\s*\(?([A-D])\b", s)
    if m:
        return m.group(1)
    m = LETTER_RE.search(s)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("-o", "--out", default="mmlu_routerbench.json")
    ap.add_argument("--limit", type=int, default=0, help="最多輸出幾題（0 = 全部）")
    ap.add_argument("--subjects", default="", help="逗號分隔的科目名稱（eval_name 去掉 mmlu- 前綴）")
    args = ap.parse_args()

    df = pd.read_pickle(args.pkl)
    df = df[df["eval_name"].astype(str).str.lower().str.startswith("mmlu")].copy()
    df["subject"] = df["eval_name"].astype(str).str.replace(r"^mmlu[-_]?", "", regex=True)
    if args.subjects:
        keep = {s.strip() for s in args.subjects.split(",") if s.strip()}
        df = df[df["subject"].isin(keep)]
    if args.limit:
        df = df.head(args.limit)

    models = [c for c in df.columns if isinstance(c, str) and f"{c}|model_response" in df.columns]
    if not models:
        raise SystemExit("找不到模型欄位（需要 <model> 與 <model>|model_response 兩欄）")

    items = []
    for _, row in df.iterrows():
        results, picks = {}, {}
        for m in models:
            v = row.get(m)
            results[m] = None if pd.isna(v) else (1 if float(v) >= 0.5 else 0)
            picks[m] = extract_letter(row.get(f"{m}|model_response"))
        votes = Counter(picks[m] for m in models if results[m] == 1 and picks[m])
        gold = votes.most_common(1)[0][0] if votes else None
        items.append({
            "id": str(row.get("sample_id", len(items))),
            "subject": row["subject"],
            "prompt": str(row.get("prompt", "")),
            "gold": gold,
            "results": results,
            "picks": picks,
        })

    out = {"source": f"RouterBench ({args.pkl.split('/')[-1]})", "models": models, "items": items}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"輸出 {len(items)} 題、{len(models)} 個模型 → {args.out}")


if __name__ == "__main__":
    main()
