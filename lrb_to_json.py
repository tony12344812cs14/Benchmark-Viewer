#!/usr/bin/env python3
"""
把 LLMRouterBench 的 bench-release.tar.gz 中「效能-成本（performance-cost）」設定的題目
轉成瀏覽器（index.html）可讀的 JSON，每個子資料集一個檔，並產生 data/index.json 清單。

用法：
    python lrb_to_json.py bench-release.tar.gz -o _site/data

篩選條件照官方 config/baseline_config_performance_cost.yaml：
10 個資料集 × 13 個旗艦模型；同一 dataset/split/model 有多個結果檔時取時間戳最新的一份。
直接串流讀取 tar.gz，不需要先解壓。
"""
import argparse
import json
import os
import re
import shutil
import tarfile
from collections import defaultdict

DATASETS = ["aime", "livemathbench", "gpqa", "hle", "livecodebench", "mmlupro",
            "swe-bench", "simpleqa", "tau2", "arenahard"]
MODELS = ["claude-sonnet-4", "deepseek-v3-0324", "deepseek-v3.1-terminus", "deepseek-r1-0528",
          "gemini-2.5-flash", "gemini-2.5-pro", "gpt-5-chat", "gpt-5", "qwen3-235b-a22b-2507",
          "qwen3-235b-a22b-thinking-2507", "glm-4.6", "kimi-k2-0905", "intern-s1"]
SPLITS = ["test", "hybrid", "v1", "test_3000", "verified"]
NAMES = {"aime": "AIME", "livemathbench": "LiveMathBench", "gpqa": "GPQA", "hle": "HLE（Humanity's Last Exam）",
         "livecodebench": "LiveCodeBench", "mmlupro": "MMLU-Pro", "swe-bench": "SWE-Bench",
         "simpleqa": "SimpleQA", "tau2": "τ²-Bench", "arenahard": "ArenaHard"}
LIM_Q, LIM_P, LIM_G = 6000, 1500, 1500
LIM_RAW = 30000              # 單一完整回答的上限（字）
RAW_BUDGET = 700_000_000     # data/raw 全部完整回答的大小上限（bytes），GitHub Pages 網站上限 1GB
OPTLINE = re.compile(r"(?m)^\s*\(?[A-J][\.\):]\s")
GRADE = {"A": "判定正確", "B": "判定錯誤", "C": "未作答"}


def s(v):
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def trim(t, n):
    t = s(t)
    return t if len(t) <= n else t[:n] + f"\n…（已截斷，原長 {len(t)} 字）"


def safe(x):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(x))


def chunk_of(sid):
    """完整回答分片：數字 id 每 50 題一片，其他 id 依字元碼總和分 64 片（網頁端用同樣算法）。"""
    return str(int(sid) // 50) if sid.isdigit() else "h" + str(sum(map(ord, sid)) % 64)


def write_raw(root, mapping):
    """mapping: {sid: text}，依 chunk 寫成 root/<chunk>.json"""
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    parts = defaultdict(dict)
    for sid, t in mapping.items():
        parts[chunk_of(sid)][sid] = t
    for c, d in parts.items():
        with open(os.path.join(root, c + ".json"), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, separators=(",", ":"))


def dir_size(d):
    return sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(d) for f in fs)


def retrim_dir(d, n):
    for a, _, fs in os.walk(d):
        for fn in fs:
            p = os.path.join(a, fn)
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            ch = False
            for k, v in m.items():
                if len(v) > n + 80:
                    o = re.search(r"原長 (\d+) 字", v[-60:])
                    m[k] = v[:n] + f"\n…（已截斷，原長 {o.group(1) if o else len(v)} 字）"
                    ch = True
            if ch:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(m, f, ensure_ascii=False, separators=(",", ":"))


def ts_of(name, mtime):
    m = re.search(r"(\d{8})_(\d{6})\.json$", name)
    return int(m.group(1) + m.group(2)) if m else int(mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tar")
    ap.add_argument("-o", "--out", default="data")
    args = ap.parse_args()

    best = {}   # (ds, split, model) -> (ts, records dict)
    seen = defaultdict(set)
    n_json = 0
    with tarfile.open(args.tar, "r|gz") as tf:
        for mem in tf:
            if not mem.isfile() or not mem.name.endswith(".json"):
                continue
            f = tf.extractfile(mem)
            try:
                data = json.load(f)
            except Exception:
                continue
            n_json += 1
            if not isinstance(data, dict) or "records" not in data:
                continue
            ds, sp, md = data.get("dataset_name"), data.get("split"), data.get("model_name")
            seen[ds].add((sp, md))
            if data.get("demo") or ds not in DATASETS or md not in MODELS or sp not in SPLITS:
                continue
            ts = ts_of(mem.name, mem.mtime)
            key = (ds, sp, md)
            if key in best and best[key][0] >= ts:
                continue
            recs = {}
            for r in data.get("records", []):
                sc = r.get("score")
                q = s(r.get("origin_query") or r.get("prompt"))
                pr = s(r.get("prompt"))
                if pr and OPTLINE.search(pr) and not OPTLINE.search(q):
                    q = pr  # GPQA / MMLU-Pro：選項只在 prompt 裡
                pred = r.get("prediction")
                if ds == "simpleqa":  # prediction 是評審的等級字母，改顯示模型原始回答
                    g = s(pred).strip()
                    pred = f"[評審：{GRADE.get(g, g)}] " + s(r.get("raw_output"))
                elif ds == "arenahard" and r.get("raw_output"):
                    pred = r.get("raw_output")
                recs[r["index"]] = (trim(q, LIM_Q),
                                    trim(r.get("ground_truth"), LIM_G),
                                    None if sc is None else round(float(sc), 4),
                                    trim(pred, LIM_P),
                                    r.get("cost"))
            best[key] = (ts, recs)
            raws = {str(r["index"]): trim(r.get("raw_output"), LIM_RAW)
                    for r in data.get("records", []) if r.get("raw_output")}
            write_raw(os.path.join(args.out, "raw", f"{safe(ds)}__{sp}", safe(md)), raws)
    print(f"讀了 {n_json} 個 JSON；符合條件的 dataset/split/model 組合：{len(best)}")
    for ds in DATASETS:
        sps = sorted({sp for sp, _ in seen.get(ds, set())})
        print(f"  {ds}: splits 出現 {sps}")

    raw_root = os.path.join(args.out, "raw")
    lrb_dirs = [os.path.join(raw_root, f"{safe(ds)}__{sp}") for ds, sp, _ in best]
    lrb_dirs = sorted(set(d for d in lrb_dirs if os.path.isdir(d)))
    other = dir_size(raw_root) - sum(dir_size(d) for d in lrb_dirs) if os.path.isdir(raw_root) else 0
    lim = LIM_RAW
    total = other + sum(dir_size(d) for d in lrb_dirs)
    print(f"完整回答（截斷上限 {lim} 字）：{total/1e6:.0f} MB（其中 RouterBench {other/1e6:.0f} MB）")
    while total > RAW_BUDGET and lim > 1500:
        lim //= 2
        for d in lrb_dirs:
            retrim_dir(d, lim)
        total = other + sum(dir_size(d) for d in lrb_dirs)
        print(f"  超過上限，改為每則最多 {lim} 字 → {total/1e6:.0f} MB")

    os.makedirs(os.path.join(args.out, "lrb"), exist_ok=True)
    groups = defaultdict(dict)
    for (ds, sp, md), (_, recs) in best.items():
        groups[(ds, sp)][md] = recs

    entries = []
    for (ds, sp) in sorted(groups, key=lambda k: (DATASETS.index(k[0]), k[1])):
        per = groups[(ds, sp)]
        models = [m for m in MODELS if m in per]
        idxs = sorted({i for m in models for i in per[m]}, key=lambda x: (str(type(x)), x))
        items, summ = [], {m: {"n": 0, "score": 0.0, "cost": 0.0, "nc": 0} for m in models}
        for i in idxs:
            q = g = ""
            scores, preds, costs = [], [], []
            for m in models:
                r = per[m].get(i)
                if r is None:
                    scores.append(None); preds.append(None); costs.append(None)
                    continue
                q = q or r[0]; g = g or r[1]
                scores.append(r[2]); preds.append(r[3])
                c = None if r[4] is None else float(r[4])
                costs.append(None if c is None else round(c, 7))
                if r[2] is not None:
                    summ[m]["n"] += 1; summ[m]["score"] += r[2]
                if c is not None:
                    summ[m]["nc"] += 1; summ[m]["cost"] += c
            items.append([str(i), q, g, scores, preds, costs])
        fn = f"lrb/{re.sub(r'[^A-Za-z0-9_.-]', '_', ds)}__{sp}.json"
        with open(os.path.join(args.out, fn), "w", encoding="utf-8") as f:
            json.dump({"format": "bench-v2", "bench": "LLMRouterBench（效能-成本）", "group": NAMES.get(ds, ds),
                       "dataset": ds, "split": sp, "models": models, "items": items},
                      f, ensure_ascii=False, separators=(",", ":"))
        size = os.path.getsize(os.path.join(args.out, fn))
        summary = {m: {"acc": round(v["score"] / v["n"], 4) if v["n"] else None,
                       "avgCost": round(v["cost"] / v["nc"], 7) if v["nc"] else None,
                       "totalCost": round(v["cost"], 4), "n": v["n"]} for m, v in summ.items()}
        label = NAMES.get(ds, ds) + ("" if len([k for k in groups if k[0] == ds]) == 1 else f"（{sp}）")
        entries.append({"key": f"lrb-{ds}-{sp}", "bench": "LLMRouterBench（效能-成本，13 個旗艦模型）",
                        "name": label, "file": f"data/{fn}", "n": len(items), "models": len(models),
                        "hasCost": True, "summary": summary,
                        "raw": f"data/raw/{safe(ds)}__{sp}", "rawLimit": lim})
        print(f"  {label}: {len(items)} 題、{len(models)} 個模型、{size/1e6:.1f} MB")

    idx_path = os.path.join(args.out, "index.json")
    old = []
    if os.path.exists(os.path.join(args.out, "mmlu.json")):
        with open(os.path.join(args.out, "mmlu.json"), encoding="utf-8") as f:
            mm = json.load(f)
        old.append({"key": "mmlu", "bench": "RouterBench（11 個模型）", "name": "MMLU",
                    "file": "data/mmlu.json", "n": len(mm["items"]), "models": len(mm["models"]), "hasCost": False,
                    **({"raw": "data/raw/mmlu", "rawLimit": LIM_RAW} if os.path.isdir(os.path.join(raw_root, "mmlu")) else {})})
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump({"datasets": old + entries}, f, ensure_ascii=False, indent=1)
    print(f"清單 → {idx_path}（{len(old) + len(entries)} 個資料集）")


if __name__ == "__main__":
    main()
