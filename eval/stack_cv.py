"""2-fold cross-fit of the stacking combiner: each hash half is scored by a combiner fitted only on the other half,
so every reported number is out-of-fold and n doubles.  .venv/bin/python -m eval.stack_cv v0 mb5 [--thr 0.5]"""
import argparse, json
from collections import defaultdict
import numpy as np
from eval.bias import groups
from eval.stack import feats, fit
from eval.suite import RUNS, load_sets


def cv(names, thr=0.5, save_run=None):
    runs = {n: json.loads((RUNS / f"{n}.json").read_text()) for n in names}
    rows = [r for r in load_sets() if all(r["k"] in runs[n] for n in names)]
    half = lambda r: int(r["k"][:6], 16) % 2
    P = {}
    for hv in (0, 1):
        clf = fit([r for r in rows if half(r) != hv], runs, names)
        rs = [r for r in rows if half(r) == hv]
        P.update(zip([r["k"] for r in rs], clf.predict_proba(np.array([feats(runs, names, r["k"]) for r in rs]))[:, 1]))
    if save_run:   # out-of-fold scores as a run file (charts, bias report): p_ai in "ai", the rest in "human"
        (RUNS / f"{save_run}.json").write_text(json.dumps({k: {"human": 1 - float(p), "ai": float(p), "ai_edited": 0.0, "humanized": 0.0}
                                                           for k, p in P.items()}))
    hum = [r for r in rows if r["label"] == 0]
    ai = [r for r in rows if r["label"] == 1]
    by = defaultdict(list)
    for r in hum:
        g = groups(r)
        for d in ("domain", "length"):
            by[f"{d}={g[d]}"].append(P[r["k"]] >= thr)
        if r["g"] == "H non-native":
            by["non-native"].append(P[r["k"]] >= thr)
    worst = max((np.mean(v), k) for k, v in by.items() if len(v) >= 15)
    zz = [r for r in rows if r["zerogpt"] is not None]
    m = {"fpr": np.mean([P[r["k"]] >= thr for r in hum]), "worst": worst,
         "catch": np.mean([P[r["k"]] >= thr for r in ai]),
         "humanized": np.mean([P[r["k"]] >= thr for r in ai if r["g"] == "AI humanized"]),
         "textsight": np.mean([P[r["k"]] >= thr for r in ai if r["src"].startswith("textsight")]),
         "h2h": sum((P[r["k"]] >= thr) == (r["label"] == 1) for r in zz), "zg": sum((r["zerogpt"] >= 50) == (r["label"] == 1) for r in zz), "n": len(zz)}
    ok = [m["h2h"] >= m["zg"], worst[0] <= .05, m["fpr"] <= .03, m["humanized"] >= .85]
    print(f"{'+'.join(names):22} t={thr:.2f} CV: FPR {m['fpr']:.1%} | worst {worst[1]} {worst[0]:.1%} | catch {m['catch']:.1%} | "
          f"humanized {m['humanized']:.1%} | textsight {m['textsight']:.0%} | h2h {m['h2h']}/{m['n']} vs ZeroGPT {m['zg']} | bar {sum(ok)}/4")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="+"); ap.add_argument("--thr", type=float, nargs="*", default=[0.5])
    ap.add_argument("--save-run", help="write the out-of-fold scores to eval/runs/<name>.json")
    a = ap.parse_args()
    for t in a.thr:
        cv(a.names, t, a.save_run)
