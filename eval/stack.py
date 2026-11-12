"""Learned combiner (stacking) over cached model probabilities: fit on one half of the held-out data (hash split),
report on the other half, so reported numbers are never tuned on. Human texts are weighted per domain (every domain
counts equally), so a combiner can't buy average accuracy with one domain's false positives.
  .venv/bin/python -m eval.stack v0 v4 mb1 mb5 [--save models/stacker.json]
"""
import argparse
import json
import math
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

from eval.bias import groups
from eval.suite import RUNS, load_sets

L = lambda p: math.log(max(p, 1e-6) / max(1 - p, 1e-6))


def feats(runs, names, k):
    return sum(([L(1 - runs[n][k]["human"]), L(runs[n][k]["ai"]), L(runs[n][k].get("humanized", 0) + 1e-6)] for n in names), [])


def fit(rows, runs, names, C=0.5):
    dom = [groups(r)["domain"] if r["label"] == 0 else "AI" for r in rows]
    cnt = Counter(dom)
    w = np.array([(1.0 / cnt[d]) * (len(cnt) - 1) if d != "AI" else len(cnt) / cnt["AI"] for d in dom])
    return LogisticRegression(C=C, max_iter=3000).fit(np.array([feats(runs, names, r["k"]) for r in rows]),
                                                      np.array([r["label"] for r in rows]), sample_weight=w)


def report(clf, rows, runs, names, thr):
    p = clf.predict_proba(np.array([feats(runs, names, r["k"]) for r in rows]))[:, 1]
    hum = [(r, s) for r, s in zip(rows, p) if r["label"] == 0]
    ai = [(r, s) for r, s in zip(rows, p) if r["label"] == 1]
    by = {}
    for r, s in hum:
        g = groups(r)
        for dim in ("domain", "length"):
            by.setdefault(f"{dim}={g[dim]}", []).append(s >= thr)
        if r["g"] == "H non-native":
            by.setdefault("non-native", []).append(s >= thr)
    worst = max((np.mean(v), k) for k, v in by.items() if len(v) >= 15)
    hz = [s >= thr for r, s in ai if r["g"] == "AI humanized"]
    zz = [(r, s) for r, s in zip(rows, p) if r["zerogpt"] is not None]
    return {"fpr": float(np.mean([s >= thr for _, s in hum])), "worst": (float(worst[0]), worst[1]),
            "catch": float(np.mean([s >= thr for _, s in ai])), "humanized": float(np.mean(hz)),
            "h2h": float(np.mean([(s >= thr) == (r["label"] == 1) for r, s in zz])),
            "zerogpt": float(np.mean([(r["zerogpt"] >= 50) == (r["label"] == 1) for r, _ in zz])), "n_h2h": len(zz)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--save")
    ap.add_argument("--paths", nargs="*", help="model path / HF id per name, in order (written into --save)")
    ap.add_argument("--thr", type=float, nargs="*", default=[0.5, 0.6, 0.7])
    a = ap.parse_args()
    runs = {n: json.loads((RUNS / f"{n}.json").read_text()) for n in a.names}
    rows = [r for r in load_sets() if all(r["k"] in runs[n] for n in a.names)]
    half = lambda r: int(r["k"][:6], 16) % 2
    tune, rep = [r for r in rows if half(r) == 0], [r for r in rows if half(r) == 1]
    clf = fit(tune, runs, a.names)
    for t in a.thr:
        m = report(clf, rep, runs, a.names, t)
        ok = [m["h2h"] >= m["zerogpt"], m["worst"][0] <= .05, m["fpr"] <= .03, m["humanized"] >= .85]
        print(f"{'+'.join(a.names):24} t={t:.2f} REPORT: FPR {m['fpr']:.1%} | worst {m['worst'][1]} {m['worst'][0]:.1%} | "
              f"catch {m['catch']:.1%} | humanized {m['humanized']:.1%} | h2h {m['h2h']:.1%} vs ZeroGPT {m['zerogpt']:.1%} "
              f"(n={m['n_h2h']}) | bar {sum(ok)}/4")
    if a.save:
        assert a.paths and len(a.paths) == len(a.names), "--save needs --paths, one per model"
        json.dump({"models": a.names, "paths": a.paths, "fit_on": "hash half 0", "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
                   "features": "per model: logit(1-P(human)), logit(P(ai)), logit(P(humanized))"}, open(a.save, "w"), indent=1)
        print("saved", a.save)


if __name__ == "__main__":
    main()
