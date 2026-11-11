"""ZeroGPT verdicts for a stratified sample of test/heldout.jsonl -> test/zerogpt_heldout.json (resumable)."""
import json
import pathlib
import random
import time

from eval.zerogpt_compare import zerogpt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "test/zerogpt_heldout.json"
PLAN = {"learner/ellipse": 25, "learner/icnale": 15, "textsight/human_pmc_ccby_nonanglophone": 20,
        "textsight/human_wikipedia_pre2022": 10, "textsight/human_literature_publicdomain": 10,
        "textsight/ai_generated_2026": 20, "jabarian/human/": 20, "jabarian/ai": 20, "jabarian/humanized": 30}


def main():
    rows = [json.loads(l) for l in (ROOT / "test/heldout.jsonl").read_text().splitlines()]
    random.Random(21).shuffle(rows)
    done = json.loads(OUT.read_text()) if OUT.exists() else []
    have = {d["text"] for d in done}
    for prefix, n in PLAN.items():
        for r in [r for r in rows if r["src"].startswith(prefix)][:n]:
            if r["text"] in have:
                continue
            try:
                z = zerogpt(r["text"][:15000])
            except Exception as e:  # noqa: BLE001
                print("FAIL", r["src"], e, flush=True)
                time.sleep(30)
                continue
            done.append({"src": r["src"], "label": r["label"], "text": r["text"], "zerogpt": z})
            OUT.write_text(json.dumps(done))
            print(f"{r['src'][:40]:40} zerogpt {z}", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
