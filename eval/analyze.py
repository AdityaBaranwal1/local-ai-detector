"""Summarise eval/scores.json: AUROC, catch rate at fixed false-positive rates, false positives per human source.

Usage: .venv/bin/python -m eval.analyze [--combo]
"""
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

ROWS = json.loads((pathlib.Path(__file__).parent / "scores.json").read_text())
ENGINES = {  # name -> (key, sign): sign -1 = low score means AI
    "tropa": ("tropa", 1), "desklib": ("desklib", 1), "tmr": ("tmr", 1), "fakespot": ("fakespot", 1),
    "binoculars": ("binoculars", -1), "fastdetect": ("fastdetect", 1),
}


def table(name, score, rows):
    hum = [r for r in rows if r["label"] == 0]
    ai = [r for r in rows if r["label"] == 1]
    y = [0] * len(hum) + [1] * len(ai)
    s = [score(r) for r in hum + ai]
    auc = roc_auc_score(y, s)
    hs = np.sort([score(r) for r in hum])
    out = [f"{name:12} AUROC {auc:.3f}"]
    for fpr in (0.01, 0.05):
        thr = hs[int(np.ceil((1 - fpr) * len(hs))) - 1]
        tpr = np.mean([score(r) > thr for r in ai])
        out.append(f"catch@{int(fpr * 100)}%FP {tpr:5.1%}")
    return "  ".join(out)


def main():
    for r in ROWS:   # rows scored before the switch stored 1 - P(human); the detector uses P(ai) (README)
        if "tropa_classes" in r:
            r["tropa"] = r["tropa_classes"]["ai"]
    allrows = [r for r in ROWS if "fakespot" in r and len(r["text"].split()) >= 50]
    raw = [r for r in allrows if not (r["label"] == 1 and r["src"].startswith("humanized/"))]
    hz = [r for r in allrows if r["label"] == 0 or r["src"].startswith("humanized/")]
    print(f"{len(allrows)} texts: {sum(r['label'] == 0 for r in allrows)} human, "
          f"{sum(r['label'] == 1 and r['src'].startswith('ai/') for r in allrows)} raw AI, "
          f"{sum(r['src'].startswith('humanized/') for r in allrows)} AI rewritten to sound human, "
          f"{sum(r['label'] == 2 for r in allrows)} human essays polished by GPT-4 (reported apart)\n")
    print("Classifiers, every text. Human vs raw AI:")
    for n in ("tropa", "desklib", "tmr", "fakespot"):
        print("  " + table(n, lambda r, n=n: r[n], raw))
    print("Human vs humanized AI:")
    for n in ("tropa", "desklib", "tmr", "fakespot"):
        print("  " + table(n, lambda r, n=n: r[n], hz))

    rows = [r for r in raw if all(k in r for k, _ in ENGINES.values())]
    print(f"\nZero-shot subset ({sum(r['label'] == 0 for r in rows)} human, {sum(r['label'] == 1 for r in rows)} raw AI):")
    for n, (k, sg) in ENGINES.items():
        print("  " + table(n, lambda r, k=k, sg=sg: sg * r[k], rows))

    # learned combination (5-fold, so the numbers are out-of-sample)
    feats = lambda r: [np.log(r["tropa"] + 1e-4) - np.log(1 - r["tropa"] + 1e-4), np.log(r["desklib"] + 1e-4) - np.log(1 - r["desklib"] + 1e-4),
                       np.log(r["tmr"] + 1e-4) - np.log(1 - r["tmr"] + 1e-4), r["binoculars"], r["fastdetect"] / np.sqrt(r["n_tokens"])]
    lab = [r for r in rows if r["label"] in (0, 1)]
    X, y = np.array([feats(r) for r in lab]), np.array([r["label"] for r in lab])
    p = cross_val_predict(LogisticRegression(C=1.0, max_iter=1000), X, y, cv=5, method="predict_proba")[:, 1]
    for r, v in zip(lab, p):
        r["combo"] = v
    print("  " + table("combo(cv)", lambda r: r["combo"], lab))

    rows = allrows
    print("\nFlagged as AI at the 50% line, by source (human rows = false positives, AI rows = catch rate):")
    by = defaultdict(list)
    for r in rows:
        by[r["src"]].append(r)
    cols = ["tropa", "desklib", "tmr", "fakespot", "zerogpt"]
    print(f"{'source':38}" + "".join(f"{c:>9}" for c in cols) + "      n")
    for src in sorted(by, key=lambda s: (s.split("/")[0] != "human", s)):
        rs = by[src]
        cells = []
        for c in cols:
            vals = [r[c] for r in rs if isinstance(r.get(c), (int, float))]
            cells.append(f"{np.mean([v >= 0.5 for v in vals]):9.0%}" if vals else f"{'':>9}")
        print(f"{src[:38]:38}" + "".join(cells) + f"  {len(rs):5}")

    print("\nBy task (raw AI rows, catch rate at the 50% line):")
    bt = defaultdict(list)
    for r in rows:
        if r["label"] == 1 and r["src"].startswith("ai/"):
            bt[r.get("task")].append(r)
    for t, rs in sorted(bt.items()):
        print(f"  {t:14} n={len(rs):2}  tropa {np.mean([r['tropa'] >= .5 for r in rs]):4.0%}  desklib {np.mean([r['desklib'] >= .5 for r in rs]):4.0%}"
              f"  tmr {np.mean([r['tmr'] >= .5 for r in rs]):4.0%}")
    if "--dump" in sys.argv:
        for r in sorted(rows, key=lambda r: -r["tropa"] if r["label"] == 0 else r["tropa"])[:12]:
            print(r["src"], round(r["tropa"], 3), r["text"][:160].replace("\n", " "))


if __name__ == "__main__":
    main()
