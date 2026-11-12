"""Benchmark engines on the independent reviewer's finished rounds (eval/sol/round-1..3, 400 texts), with ZeroGPT's
saved answers as the reference. Once a round is closed and published it is no longer unseen, so this is a
regression benchmark, not a fresh validation (the next review round builds a new set).

  .venv/bin/python eval/bench_sol.py            # scores members once (cached in eval/runs/sol_<m>.json), then compares
Engines = the saved combiner JSONs in ENGINES. Score = the review's fixed formula:
  10 x (0.40 x human specificity + 0.20 x raw-AI sensitivity + 0.40 x casual/humanized-AI sensitivity)
(AI-polished human excluded, as in the reviews). Flag = score >= 0.5; gate = /verify's 30 % threshold (model part only).
"""
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from detector.engines import Classifier, Stacked  # noqa: E402

SOL = ROOT / "eval/sol"
RUNS = ROOT / "eval/runs"
MEMBERS = {"v0": "wasitaigeneratedcom/tropa-mini-ai-text-detector", "mb6": str(ROOT / "models/rp-modernbert-v6"),
           "mb7": str(ROOT / "models/rp-modernbert-v7"), "mb8": str(ROOT / "models/rp-modernbert-v8")}
ENGINES = {"v0+v6 (round 2's engine)": ("models/stacker-v0-mb6.json", ["v0", "mb6"]),
           "v6+v7 (round 3's engine)": ("models/stacker-mb6-mb7.json", ["mb6", "mb7"]),
           "v6+v7 length-aware (live)": ("models/stacker-mb6-mb7-len.json", ["mb6", "mb7"]),
           "v0+v8 length-aware": ("models/stacker-v0-mb8-len.json", ["v0", "mb8"]),
           "v6+v8 length-aware": ("models/stacker-mb6-mb8-len.json", ["mb6", "mb8"]),
           "v7+v8 length-aware": ("models/stacker-mb7-mb8-len.json", ["mb7", "mb8"])}


def load_rounds():
    rows = []
    for rnd in (1, 2, 3):
        texts = json.loads((SOL / f"round-{rnd}/texts.json").read_text())
        if rnd < 3:
            zz = {r["id"]: r["zerogpt"]["raw"]["data"]["fakePercentage"] for r in json.loads((SOL / f"round-{rnd}/results.json").read_text())["rows"]}
        else:
            zz = {r["id"]: r["score_percent"] for r in json.loads((SOL / "round-3/zerogpt.json").read_text())["rows"]}
        for t in texts:
            rows.append({"round": rnd, "id": t["id"], "text": t["text"], "cat": t["category"], "zg": zz.get(t["id"]),
                         "short": len(t["text"].split()) <= 60, "priority": bool(t.get("priority"))})
    return rows


def score_members(rows):
    for m, repo in MEMBERS.items():
        f = RUNS / f"sol_{m}.json"
        cache = json.loads(f.read_text()) if f.exists() else {}
        todo = [r for r in rows if f"{r['round']}:{r['id']}" not in cache]
        if not todo:
            continue
        c = Classifier(m, repo, "meanpool", 512).load("cuda")
        t0 = time.time()
        for r in todo:
            cache[f"{r['round']}:{r['id']}"] = c.score(r["text"])["classes"]
        f.write_text(json.dumps(cache))
        print(f"scored {len(todo)} texts with {m} in {time.time() - t0:.0f} s", flush=True)
        c.unload()
        import torch
        torch.cuda.empty_cache()


def main():
    rows = load_rounds()
    score_members(rows)
    probs = {m: json.loads((RUNS / f"sol_{m}.json").read_text()) for m in MEMBERS}
    for name, (path, members) in list(ENGINES.items()) + [("ZeroGPT", (None, None))]:
        if path:
            spec = json.loads((ROOT / path).read_text())
            st = Stacked([Classifier(m, "", "meanpool") for m in members], spec["coef"], spec["intercept"], spec.get("length_aware", False))
            s = {f"{r['round']}:{r['id']}": st._stack([probs[m][f"{r['round']}:{r['id']}"] for m in members], len(r["text"].split())) for r in rows}
        else:
            s = {f"{r['round']}:{r['id']}": (r["zg"] or 0) / 100 for r in rows}
        print(f"\n== {name}")
        for rnd in (1, 2, 3, "all"):
            rs = [r for r in rows if rnd == "all" or r["round"] == rnd]
            flag = lambda r: s[f"{r['round']}:{r['id']}"] >= 0.5
            rate = lambda sub: sum(map(flag, sub)) / max(1, len(sub))
            hum = [r for r in rs if r["cat"].startswith("human")]
            raw = [r for r in rs if r["cat"] == "ai_raw"]
            cas = [r for r in rs if r["cat"] == "ai_casual_or_humanized"]
            score = 10 * (0.4 * (1 - rate(hum)) + 0.2 * rate(raw) + 0.4 * rate(cas))
            sh_h = [r for r in hum if r["short"]]
            sh_a = [r for r in raw + cas if r["short"]]
            pr_a = [r for r in raw + cas if r["priority"]]
            print(f"  round {rnd!s:3}  score {score:4.1f}  humans flagged {sum(map(flag, hum))}/{len(hum)}  raw {sum(map(flag, raw))}/{len(raw)}"
                  f"  casual/humanized {sum(map(flag, cas))}/{len(cas)}  | short: humans {sum(map(flag, sh_h))}/{len(sh_h)}, AI {sum(map(flag, sh_a))}/{len(sh_a)}"
                  + (f"  | priority AI {sum(map(flag, pr_a))}/{len(pr_a)}" if pr_a else ""))


if __name__ == "__main__":
    main()
