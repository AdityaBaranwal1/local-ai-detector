"""Back-translation (a common "humanizer" trick): modern AI text -> German or French -> English with open MarianMT
models (Helsinki-NLP opus-mt), run locally -- no API, no tokens. Public benchmarks showed it is our weakest attack
(DetectRL back-translation: 13-33 % caught). Human texts get the same treatment and stay label 0, so translationese
itself can never become an AI tell.
Sources: AI (label 1) from casual / ext / open / transcripts -> label 3 (humanized); human (label 0) from casual and
the short-form set -> label 0. Sentence-level translation, batched, on the GPU under the render workers' lock (run via
`flock ~/All code/shorts-studio/.gpu.lock`), so the live detector falls back to CPU instead of running out of VRAM.
-> train/data/backtrans.jsonl     Usage: .venv/bin/python train/build_backtrans.py [n_ai] [n_human]
"""
import json
import pathlib
import random
import re
import sys
from collections import Counter

import torch
from transformers import MarianMTModel, MarianTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_more import h  # noqa: E402

rng = random.Random(103)
PIVOTS = {"de": ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"), "fr": ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en")}
SPLIT = re.compile(r"(?<=[.!?])\s+")


def load(name):
    return MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name).to("cuda").eval()


@torch.inference_mode()
def translate(tm, sents, bs=32):
    tok, model = tm
    out = []
    for i in range(0, len(sents), bs):
        enc = tok(sents[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        gen = model.generate(**enc, max_new_tokens=300, num_beams=2)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def pick(n_ai, n_hu):
    ai, hu = [], []
    for f in ("casual.jsonl", "ext.jsonl", "open.jsonl", "transcripts.jsonl", "short.jsonl"):
        p = ROOT / "train/data" / f
        if not p.exists():
            continue
        for l in p.read_text().splitlines():
            r = json.loads(l)
            if not (30 <= len(r["text"].split()) <= 300):
                continue
            if r["label"] == 1 and f != "short.jsonl":
                ai.append(r)
            elif r["label"] == 0 and f in ("casual.jsonl", "short.jsonl"):
                hu.append(r)
    return rng.sample(ai, min(n_ai, len(ai))), rng.sample(hu, min(n_hu, len(hu)))


def main():
    n_ai = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    n_hu = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    ai, hu = pick(n_ai, n_hu)
    jobs = [(r, 3 if r["label"] == 1 else 0) for r in ai + hu]
    rng.shuffle(jobs)
    out = []
    for pivot, (fwd, back) in PIVOTS.items():
        part = jobs[: len(jobs) // 2] if pivot == "de" else jobs[len(jobs) // 2:]
        paras = [[SPLIT.split(p.strip()) for p in r["text"].split("\n") if p.strip()] for r, _ in part]
        flat = [s for doc in paras for para in doc for s in para]
        mid = translate(load(fwd), flat)
        torch.cuda.empty_cache()
        back_t = translate(load(back), mid)
        torch.cuda.empty_cache()
        it = iter(back_t)
        for (r, lab), doc in zip(part, paras):
            text = "\n".join(" ".join(next(it) for _ in para) for para in doc).strip()
            if len(text.split()) >= 20 and h(text) != h(r["text"]):
                src = r["src"].split("/")
                out.append({"text": text, "label": lab, "src": f"backtrans/{'human' if lab == 0 else 'ai'}/{pivot}/{'/'.join(src[:3])}"})
        print(f"pivot {pivot}: {len(part)} texts, {len(flat)} sentences", flush=True)
    (ROOT / "train/data/backtrans.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print("backtrans.jsonl:", len(out), dict(Counter(r["label"] for r in out)), flush=True)


if __name__ == "__main__":
    main()
