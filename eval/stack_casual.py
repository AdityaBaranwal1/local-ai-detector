"""Casual-aware combiner (v7): score the held-back casual-professional dev texts (the v7 package's dev split,
src casual/*) with each member, then cross-fit the combiner on held-out test sets + those texts (casual human texts
weighted as their own domain) and report both, out-of-fold. Why: fitted on the test sets alone, the combiner gave
v7 too little weight -- v7 alone catches 97.5 % of held-back casual AI at 1 % false positives, the old fit 54 %.
  1) score:  .venv/bin/python eval/stack_casual.py score   (needs /tmp/kgl-data7 from train/package.py)
  2) fit:    PYTHONPATH=. .venv/bin/python eval/stack_casual.py fit [--save models/stacker-mb6-mb7.json]
"""
import pathlib as _pl
import sys
REPO = str(_pl.Path(__file__).resolve().parent.parent)
if len(sys.argv) > 1 and sys.argv[1] == "score":
    import json, pathlib, torch
    sys.path.insert(0, REPO)
    from detector.engines import Classifier
    M = {"v0": "wasitaigeneratedcom/tropa-mini-ai-text-detector",
         "mb6": REPO + "/models/rp-modernbert-v6",
         "mb7": REPO + "/models/rp-modernbert-v7", "mb8": REPO + "/models/rp-modernbert-v8"}
    rows = [json.loads(l) for l in open("/tmp/kgl-data7/dev.jsonl")]
    rows = [r for r in rows if r["src"].startswith("casual/")]
    json.dump(rows, open("/tmp/casdev/rows.json", "w"))
    for name, repo in M.items():
        out = pathlib.Path(f"/tmp/casdev/{name}.json")
        if out.exists(): continue
        c = Classifier(name, repo, "meanpool", 512).load("cuda")
        res = {str(i): c.score(r["text"])["classes"] for i, r in enumerate(rows)}
        out.write_text(json.dumps(res)); print(name, "done", flush=True)
        del c; torch.cuda.empty_cache()
else:
    import json, sys, numpy as np
    from collections import Counter, defaultdict
    from sklearn.linear_model import LogisticRegression
    from eval.bias import groups
    from eval.stack import feats
    from eval.suite import RUNS, load_sets
    crow = json.load(open("/tmp/casdev/rows.json"))
    import os as _os
    if _os.environ.get("LEAK_PKG"):             # held-back casual texts that a later package put into training: drop for every engine
        _seen = {json.loads(l)["text"] for f in ("train.jsonl", "extra.jsonl") for l in open(_os.path.join(_os.environ["LEAK_PKG"], f))}
        _keep = [i for i, r in enumerate(crow) if r["text"] not in _seen]
        print(f"casual held-back texts: {len(_keep)} of {len(crow)} still unseen by the newest package", flush=True)
    else:
        _keep = list(range(len(crow)))
    import math
    LEN = "--length" in sys.argv
    def F(runs, names, r):
        x = feats(runs, names, r["k"])
        if not LEN:
            return x
        lw = math.log(max(len(r["text"].split()), 10) / 100)        # 0 at 100 words, -1.6 at 20, +1.6 at 500
        return x + [lw] + [lw * v for v in x]
    def run(names, save=None):
        runs = {n: json.loads((RUNS / f"{n}.json").read_text()) for n in names}
        for n in names:
            d = json.load(open(f"/tmp/casdev/{n}.json"))
            runs[n].update({f"cd{i}": d[str(i)] for i in range(len(crow))})
        test = [r for r in load_sets() if all(r["k"] in runs[n] for n in names)]
        cas = [{"k": f"cd{i}", "label": 1 if r["label"] else 0, "src": r["src"], "text": r["text"], "casual": True,
                "g": ("H native" if r["label"] == 0 else ("AI humanized" if r["label"] == 3 else "AI raw")), "zerogpt": None}
               for i, r in enumerate(crow) if i in set(_keep)]
        for r in test: r["label"] = 1 if r["label"] else 0
        rows = test + cas
        half = lambda r: (int(r["k"][2:]) % 2) if r.get("casual") else int(r["k"][:6], 16) % 2
        dom = lambda r: "AI" if r["label"] else ("casual-pro" if r.get("casual") else groups(r)["domain"])
        def fit(rs):
            d = [dom(r) for r in rs]; c = Counter(d)
            w = np.array([(1 / c[x]) * (len(c) - 1) if x != "AI" else len(c) / c["AI"] for x in d])
            return LogisticRegression(C=0.5, max_iter=3000).fit(np.array([F(runs, names, r) for r in rs]),
                                                               np.array([r["label"] for r in rs]), sample_weight=w)
        P = {}
        for hv in (0, 1):
            clf = fit([r for r in rows if half(r) != hv])
            rs = [r for r in rows if half(r) == hv]
            P.update(zip([r["k"] for r in rs], clf.predict_proba(np.array([F(runs, names, r) for r in rs]))[:, 1]))
        f = lambda r: P[r["k"]] >= .5
        hum = [r for r in test if r["label"] == 0]; ai = [r for r in test if r["label"] == 1]
        by = defaultdict(list)
        for r in hum:
            g = groups(r)
            for d in ("domain", "length"): by[f"{d}={g[d]}"].append(f(r))
            if r["g"] == "H non-native": by["non-native"].append(f(r))
        worst = max((np.mean(v), k) for k, v in by.items() if len(v) >= 15)
        zz = [r for r in test if r["zerogpt"] is not None]
        h2h = sum(f(r) == (r["label"] == 1) for r in zz); zg = sum((r["zerogpt"] >= 50) == (r["label"] == 1) for r in zz)
        ch = [f(r) for r in cas if r["label"] == 0]; ca = [f(r) for r in cas if r["label"] == 1]
        cp = [f(r) for r in cas if r["label"] == 1 and "/priority/" in r["src"]]
        sh = lambda r: len(r["text"].split()) <= 80
        sh_h = [f(r) for r in test + cas if r["label"] == 0 and sh(r)]; sh_a = [f(r) for r in test + cas if r["label"] == 1 and sh(r)]
        print(f"   short (<= 80 words): humans flagged {np.mean(sh_h):.1%} (n={len(sh_h)}), AI caught {np.mean(sh_a):.1%} (n={len(sh_a)})")
        cg = [P[r["k"]] >= .3 for r in cas if r["label"] == 0]
        raw = [f(r) for r in ai if r["g"] == "AI raw"]
        print(f"   test AI caught {np.mean([f(r) for r in ai]):.1%} (raw {np.mean(raw):.1%}, n={len(raw)})")
        print(f"{'+'.join(names):14} TEST: FPR {np.mean([f(r) for r in hum]):.1%} worst {worst[1]} {worst[0]:.1%} "
              f"humanized {np.mean([f(r) for r in ai if r['g'] == 'AI humanized']):.1%} h2h {h2h}/{len(zz)} vs {zg} | "
              f"CASUAL: human flagged {np.mean(ch):.1%} (gate {np.mean(cg):.1%}) AI caught {np.mean(ca):.1%} priority {np.mean(cp):.1%}")
        if save:
            clf = fit([r for r in rows if half(r) == 0])
            paths = {"v0": "wasitaigeneratedcom/tropa-mini-ai-text-detector", "mb6": REPO + "/models/rp-modernbert-v6",
                     "mb7": REPO + "/models/rp-modernbert-v7", "mb8": REPO + "/models/rp-modernbert-v8"}
            json.dump({"models": names, "paths": [paths[n] for n in names], "fit_on": "hash half 0 of the held-out test sets + even rows of the v7 casual dev split",
                       "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
                       "features": "per model: logit(1-P(human)), logit(P(ai)), logit(P(humanized)+1e-6)" + ("; then log(words/100) and log(words/100) x each" if LEN else ""), "length_aware": LEN}, open(save, "w"), indent=1)
            print("saved", save)
    for names in [n.split("+") for n in _os.environ.get("COMBOS", "v0+mb6 mb6+mb7 mb6+mb8 mb7+mb8 v0+mb8").split()]:
        run(names)
    if "--save" in sys.argv: run(_os.environ.get("SAVE_COMBO", "mb6+mb7").split("+"), save=sys.argv[sys.argv.index("--save") + 1])
