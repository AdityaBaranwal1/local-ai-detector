"""Score a model version on every test set and print one comparable table.

Test sets (never trained on): eval/scores.json texts (the original 743), test/heldout.jsonl (Jabarian 20 %,
ELLIPSE/ICNALE 15 %, textsight 2026), eval/sites/results.json (owner-style texts), and ZeroGPT's verdicts where
we have them (eval/sites/zerogpt_sample.json + results.json + test/zerogpt_heldout.json).

Usage: .venv/bin/python -m eval.suite <model dir or HF id> [--rule p_ai|not_human|hybrid] [--name v1]
Probabilities are cached in eval/runs/<name>.json (re-scoring only new texts).
"""
import argparse
import hashlib
import json
import pathlib
import re
from collections import defaultdict

import numpy as np
import torch

from detector.engines import Classifier
from detector.gpu import GpuLock

E = pathlib.Path(__file__).parent
ROOT = E.parent
RUNS = E / "runs"
RUNS.mkdir(exist_ok=True)


def key(t):
    return hashlib.sha1(re.sub(r"\W+", " ", t.lower()).strip()[:3000].encode()).hexdigest()


def group(src, label):
    s = src
    if s.startswith("human/TOEFL") or s.startswith("learner/") or "nonanglophone" in s:
        return "H non-native"
    if label == 0:
        return "H native"
    if label == 2 and s.startswith("edited/"):
        return "AI-polished human"
    if "humanized" in s or "paraphrase" in s:
        return "AI humanized"
    return "AI raw"


def load_sets():
    rows = []
    for r in json.loads((E / "scores.json").read_text()):
        if len(r["text"].split()) >= 50:
            rows.append({"text": r["text"], "label": r["label"], "src": "eval/" + r["src"], "zerogpt": None})
    for l in (ROOT / "test/heldout.jsonl").read_text().splitlines():
        r = json.loads(l)
        rows.append({"text": r["text"], "label": 0 if r["label"] == 0 else 1, "src": r["src"], "zerogpt": None})
    for r in json.loads((E / "sites/results.json").read_text()):
        rows.append({"text": r["text"], "label": 0 if r["id"].startswith("human") else 1,
                     "src": "sites/" + ("human" if r["id"].startswith("human") else "humanized" if "casual" in r["id"] else "ai"),
                     "zerogpt": r.get("zerogpt")})
    zg = {}
    for f in (E / "sites/zerogpt_sample.json", ROOT / "test/zerogpt_heldout.json"):
        if f.exists():
            for r in json.loads(f.read_text()):
                zg[key(r["text"])] = r["zerogpt"]
    for r in rows:
        if r["zerogpt"] is None:
            r["zerogpt"] = zg.get(key(r["text"]))
    seen, out = set(), []
    for r in rows:
        k = key(r["text"])
        if k not in seen:
            seen.add(k)
            r["k"] = k
            r["g"] = group(r["src"], r["label"]) if not r["src"].startswith("eval/edited") else "AI-polished human"
            out.append(r)
    return out


def score_probs(model, rows, name):
    cache_f = RUNS / f"{name}.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    todo = [r for r in rows if r["k"] not in cache]
    if todo:
        eng = Classifier(name, model, "meanpool", 512)
        lock = GpuLock()
        lock.acquire()
        try:
            eng.load("cuda")
            eng.model.half()
            for i, r in enumerate(todo):
                cache[r["k"]] = eng.score(r["text"])["classes"]
                if i % 500 == 499:
                    cache_f.write_text(json.dumps(cache))
                    print(f"  scored {i + 1}/{len(todo)}", flush=True)
            eng.unload()
            torch.cuda.empty_cache()
        finally:
            lock.release()
        cache_f.write_text(json.dumps(cache))
    return cache


def rule_fn(rule):
    if rule == "p_ai":
        return lambda c: c["ai"]
    if rule == "not_human":
        return lambda c: 1 - c["human"]
    return lambda c: (1 - c["human"]) if c["human"] < 0.01 else c["ai"]   # hybrid = what v0 serves


def report(rows, probs, rule, title):
    f = rule_fn(rule)
    for r in rows:
        r["ours"] = f(probs[r["k"]])
    by = defaultdict(list)
    for r in rows:
        by[r["g"]].append(r)
    print(f"\n== {title} (rule {rule}) · flagged as AI at the 50 % line")
    print(f"{'group':20} {'n':>5} {'ours':>7}   | with ZeroGPT verdict: {'n':>4} {'ours':>6} {'ZeroGPT':>8}")
    for g in ("H native", "H non-native", "AI raw", "AI humanized", "AI-polished human"):
        rs = by.get(g, [])
        if not rs:
            continue
        zz = [r for r in rs if r["zerogpt"] is not None]
        line = f"{g:20} {len(rs):5} {np.mean([r['ours'] >= .5 for r in rs]):7.1%}"
        if zz:
            line += f"   |                       {len(zz):4} {np.mean([r['ours'] >= .5 for r in zz]):6.0%} {np.mean([r['zerogpt'] >= 50 for r in zz]):8.0%}"
        print(line)
    hum = [r for r in rows if r["label"] == 0]
    ai = [r for r in rows if r["label"] == 1]
    from sklearn.metrics import roc_auc_score
    y = [0] * len(hum) + [1] * len(ai)
    print(f"AUROC human vs all AI: {roc_auc_score(y, [r['ours'] for r in hum + ai]):.4f} "
          f"(n={len(rows)}); balanced accuracy @50%: "
          f"{0.5 * (np.mean([r['ours'] < .5 for r in hum]) + np.mean([r['ours'] >= .5 for r in ai])):.3f}")
    zz = [r for r in rows if r["zerogpt"] is not None]
    if zz:
        acc_o = np.mean([(r["ours"] >= .5) == (r["label"] == 1) for r in zz])
        acc_z = np.mean([(r["zerogpt"] >= 50) == (r["label"] == 1) for r in zz])
        print(f"head-to-head on the {len(zz)} texts ZeroGPT has seen: accuracy ours {acc_o:.1%} vs ZeroGPT {acc_z:.1%}")
    by_src = defaultdict(list)
    for r in rows:
        by_src["/".join(r["src"].split("/")[:2])].append(r)
    return {s: (len(v), float(np.mean([(x["ours"] >= .5) == (x["label"] == 1) for x in v]))) for s, v in by_src.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--name", required=True)
    ap.add_argument("--rule", default="all")
    ap.add_argument("--by-source", action="store_true")
    a = ap.parse_args()
    rows = load_sets()
    print(f"{len(rows)} test texts")
    probs = score_probs(a.model, rows, a.name)
    for rule in (["p_ai", "hybrid", "not_human"] if a.rule == "all" else [a.rule]):
        acc = report(rows, probs, rule, a.name)
    if a.by_source:
        print("\naccuracy by source (last rule):")
        for s, (n, v) in sorted(acc.items()):
            print(f"  {s:45} n={n:5} {v:6.1%}")


if __name__ == "__main__":
    main()
