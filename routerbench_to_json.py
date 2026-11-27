#!/usr/bin/env python3
"""
把 RouterBench 的 pkl（routerbench_0shot.pkl / routerbench_5shot.pkl）中的 MMLU 題目
轉成 MMLU 答對率瀏覽器（index.html）可讀的 JSON。

用法：
    python routerbench_to_json.py routerbench_0shot.pkl -o data/mmlu.json
    python routerbench_to_json.py routerbench_0shot.pkl --mmlu-dir mmlu_parquet/   # 用原版 MMLU 正解
    python routerbench_to_json.py routerbench_0shot.pkl --limit 500 --verbose-json   # 舊版易讀格式

正確答案：
- 有給 --mmlu-dir（cais/mmlu 的 parquet 檔所在資料夾）時，用題目文字比對原版 MMLU，取官方正解。
- 比對不到的題目，改用「答對（score=1）的模型所選字母」多數決反推；仍無法判斷則為 null。
"""
import argparse
import ast
import glob
import json
import os
import re
import shutil
from collections import Counter

import pandas as pd

OPT_RE = re.compile(r"^\s*\(?([A-D])[\.\)．、:：]\s*(.*)$")


def extract_letter(resp):
    if not isinstance(resp, str):
        return None
    s = resp.strip()
    m = re.match(r"^\(?([A-D])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"[Aa]nswer\s*(?:is)?\s*[:：]?\s*\(?([A-D])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b", s)
    return m.group(1) if m else None


def safe(x):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(x))


def chunk_of(sid):
    return str(int(sid) // 50) if sid.isdigit() else "h" + str(sum(map(ord, sid)) % 64)


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def prompt_text(p):
    """RouterBench 的 prompt 可能是 list（例如 ['...']），統一轉成字串。"""
    if isinstance(p, str):
        s = p.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple)):
                    return "\n".join(str(x) for x in v)
            except (ValueError, SyntaxError):
                pass
        return p
    try:
        return "\n".join(str(x) for x in p)
    except TypeError:
        return "" if p is None else str(p)


def load_mmlu(d):
    """回傳 {normalized question: answer letter}"""
    files = [f for f in glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True) if "auxiliary" not in f]
    table = {}
    for f in files:
        t = pd.read_parquet(f)
        for q, ch, a in zip(t["question"], t["choices"], t["answer"]):
            table[norm(q)] = norm(list(ch)[int(a)])  # 正解的「選項文字」（RouterBench 的選項順序和原版不同）
    print(f"載入原版 MMLU {len(table)} 題（{len(files)} 個 parquet）")
    return table


def parse_options(prompt):
    opts = {}
    for l in prompt_text(prompt).split("\n"):
        m = OPT_RE.match(l)
        if m and m.group(1) not in opts:
            opts[m.group(1)] = norm(m.group(2))
    return opts


def question_keys(prompt):
    """從 prompt 中取出選項之前的題目文字，回傳由長到短的候選（去掉前面可能的指示語）。"""
    lines = prompt_text(prompt).split("\n")
    first = next((i for i, l in enumerate(lines) if OPT_RE.match(l) and OPT_RE.match(l).group(1) == "A"), len(lines))
    head = lines[:first]
    for k in range(len(head)):
        key = norm("\n".join(head[k:]))
        if key:
            yield key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("-o", "--out", default="data/mmlu.json")
    ap.add_argument("--mmlu-dir", default="", help="cais/mmlu parquet 資料夾（用來取官方正解）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose-json", action="store_true", help="輸出舊版（每題 dict）格式")
    args = ap.parse_args()

    df = pd.read_pickle(args.pkl)
    print("欄位範例：", list(df.columns)[:8], "… 共", len(df.columns))
    print("eval_name 範例：", df["eval_name"].astype(str).unique()[:5])
    df = df[df["eval_name"].astype(str).str.lower().str.startswith("mmlu")].copy()
    df["subject"] = df["eval_name"].astype(str).str.replace(r"^mmlu[-_]?", "", regex=True)
    if args.limit:
        df = df.head(args.limit)
    print("MMLU 題數：", len(df))
    if len(df):
        print("prompt 範例：", repr(prompt_text(df["prompt"].iloc[0])[:400]))

    models = [c for c in df.columns if isinstance(c, str) and f"{c}|model_response" in df.columns]
    if not models:
        raise SystemExit("找不到模型欄位（需要 <model> 與 <model>|model_response 兩欄）")
    print("模型：", models)

    mmlu = load_mmlu(args.mmlu_dir) if args.mmlu_dir else {}
    src_count = Counter()
    agree = Counter()
    subjects = sorted(df["subject"].unique())
    sidx = {s: i for i, s in enumerate(subjects)}
    items = []
    raws = {m: {} for m in models}
    for _, row in df.iterrows():
        res, picks = [], []
        for m in models:
            v = row.get(m)
            res.append(None if pd.isna(v) else (1 if float(v) >= 0.5 else 0))
            picks.append(extract_letter(row.get(f"{m}|model_response")))
        gold, gsrc = None, ""
        opts = parse_options(row.get("prompt", ""))
        for key in question_keys(row.get("prompt", "")):
            if key in mmlu:
                hit = [k for k, t in opts.items() if t == mmlu[key]]
                if len(hit) == 1:
                    gold, gsrc = hit[0], "m"
                break
        if not gold:
            votes = Counter(p for p, r in zip(picks, res) if r == 1 and p)
            if votes:
                gold, gsrc = votes.most_common(1)[0][0], "v"
        src_count[gsrc or "none"] += 1
        if gold:
            for p, r in zip(picks, res):
                if r is not None and p:
                    agree[(r == 1) == (p == gold)] += 1
        sid = str(row.get("sample_id", len(items)))
        for m in models:
            t = row.get(f"{m}|model_response")
            if isinstance(t, str) and t.strip():
                t = prompt_text(t)  # 回答同樣可能是 "['...']" 形式
                raws[m][sid] = t if len(t) <= 30000 else t[:30000] + f"\n…（已截斷，原長 {len(t)} 字）"
        items.append((sid, row["subject"], prompt_text(row.get("prompt", "")), gold, gsrc, res, picks))
    print("正解來源統計（m=原版 MMLU, v=模型多數決）：", dict(src_count))
    tot = sum(agree.values())
    if tot:
        print(f"一致性檢查：RouterBench 對錯 與「模型選項 == 正解」相符 {agree[True]}/{tot} = {agree[True]/tot:.2%}")

    if args.verbose_json:
        out = {"source": f"RouterBench ({os.path.basename(args.pkl)})", "models": models,
               "items": [{"id": i, "subject": s, "prompt": p, "gold": g, "goldSrc": gs,
                          "results": dict(zip(models, r)), "picks": dict(zip(models, pk))}
                         for i, s, p, g, gs, r, pk in items]}
    else:
        out = {"format": "compact-v1", "source": f"RouterBench ({os.path.basename(args.pkl)})",
               "models": models, "subjects": subjects,
               "items": [[i, sidx[s], p, g or "", gs,
                          "".join("-" if v is None else str(v) for v in r),
                          "".join(x or "-" for x in pk)] for i, s, p, g, gs, r, pk in items]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"輸出 {len(items)} 題、{len(models)} 個模型 → {args.out}（{os.path.getsize(args.out)/1e6:.1f} MB）")
    if not args.verbose_json:  # 完整回答分片：<out 所在資料夾>/raw/mmlu/<model>/<chunk>.json
        root = os.path.join(os.path.dirname(args.out) or ".", "raw", "mmlu")
        shutil.rmtree(root, ignore_errors=True)
        for m, mp in raws.items():
            parts = {}
            for sid, t in mp.items():
                parts.setdefault(chunk_of(sid), {})[sid] = t
            os.makedirs(os.path.join(root, safe(m)), exist_ok=True)
            for c, d in parts.items():
                with open(os.path.join(root, safe(m), c + ".json"), "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
        print(f"完整回答 → {root}")


if __name__ == "__main__":
    main()
