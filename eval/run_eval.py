"""Score the evaluation set with every engine; writes eval/scores.json (one row per text).

Usage: .venv/bin/python -m eval.run_eval [engine ...]    (from the repo root; holds the GPU lock while it runs)
"""
import json
import pathlib
import sys
import time

import torch

from detector.engines import CLASSIFIERS, ZeroShot
from detector.gpu import GpuLock

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "scores.json"


def load_texts():
    rows = json.loads((DATA / "human_web.json").read_text())
    for d in sorted((DATA / "liang").iterdir()):
        z = json.loads((d / "ZeroGPT.json").read_text())
        label = 2 if "gpt4polished" in d.name else 0     # 2 = human text polished by GPT-4 (reported apart)
        for i, r in enumerate(json.loads((d / "data.json").read_text())):
            rows.append({"src": f"{'edited' if label == 2 else 'human'}/{d.name}", "label": label,
                         "text": r["document"], "zerogpt": z[i].get("score")})
    for f in ("ai_gen.json", "ai_humanized.json"):
        if (DATA / f).exists():
            for r in json.loads((DATA / f).read_text()):
                rows.append({"src": r["src"], "label": 1, "task": r["task"], "text": r["text"]})
    rows = [r for r in rows if len(r["text"].split()) >= 50]      # the detector's stated floor
    for i, r in enumerate(rows):
        r["id"] = i
    return rows


def main(engines):
    rows = load_texts()
    old = {r["text"]: r for r in json.loads(OUT.read_text())} if OUT.exists() else {}
    for r in rows:
        r.update({k: v for k, v in old.get(r["text"], {}).items() if k not in r})
    lock = GpuLock()
    print("waiting for the GPU lock ...", flush=True)
    lock.acquire()
    try:
        for name in engines:
            t0 = time.time()
            if name == "zeroshot":
                eng = ZeroShot().load("cuda")
                # the zero-shot pair is ~2.5 s/text: every AI/edited text + a third of the human ones
                for i, r in enumerate(rows):
                    if "binoculars" not in r and (r["label"] != 0 or i % 3 == 0):
                        s = eng.score(r["text"])
                        r["binoculars"], r["fastdetect"], r["n_tokens"] = s["binoculars"], s["fastdetect"], s["n_tokens"]
            else:
                eng = CLASSIFIERS[name]()
                eng.load("cuda")
                eng.model.half()
                for r in rows:
                    if name not in r:
                        s = eng.score(r["text"])
                        r[name] = s["p_ai"]
                        if name == "tropa":
                            r["tropa_classes"] = s["classes"]
            eng.unload()
            torch.cuda.empty_cache()
            print(f"{name}: {len(rows)} texts in {time.time() - t0:.0f} s", flush=True)
            OUT.write_text(json.dumps(rows))
    finally:
        lock.release()


if __name__ == "__main__":
    main(sys.argv[1:] or ["tropa", "desklib", "tmr", "fakespot", "zeroshot"])
