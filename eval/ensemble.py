"""Average the cached class probabilities of several scored models into a new run (eval/runs/<out>.json).

  .venv/bin/python -m eval.ensemble k4 mb1 --out ens1 [--weights 0.5 0.5]
Then: python -m eval.suite <any path> --name ens1   (everything is cached, no model is loaded)
      python -m eval.bias v0 k4 mb1 ens1
"""
import argparse
import json

from eval.suite import RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="*", type=float)
    a = ap.parse_args()
    w = a.weights or [1.0 / len(a.runs)] * len(a.runs)
    assert len(w) == len(a.runs) and abs(sum(w) - 1) < 1e-6, "weights must match runs and sum to 1"
    caches = [json.loads((RUNS / f"{r}.json").read_text()) for r in a.runs]
    keys = set.intersection(*(set(c) for c in caches))
    out = {}
    for k in keys:
        classes = caches[0][k].keys()
        out[k] = {c: sum(wi * cache[k][c] for wi, cache in zip(w, caches)) for c in classes}
    (RUNS / f"{a.out}.json").write_text(json.dumps(out))
    print(f"{a.out}: {len(out)} texts, weights {dict(zip(a.runs, w))}")


if __name__ == "__main__":
    main()
