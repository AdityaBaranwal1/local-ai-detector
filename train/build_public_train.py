"""v9 data from the public benchmarks' TRAINING splits (their test splits, sampled by eval/bench_public.py, are
excluded by hash and stay untouched). Fixes the gaps the public benchmarks found: 2023 Claude-instant / PaLM text and
the back-translation attack.

  * DetectRL train (github.com/NLP2CT/DetectRL): ChatGPT / Claude-instant / PaLM / Llama-2-70B on arXiv, XSum,
    WritingPrompts, Yelp; attacks on AI text (back-translation, DIPPER, polish, typo perturbation, evasion prompts) and
    the same attacks on HUMAN text, so an attack's fingerprint is never an AI tell:
      back-translated / typo-perturbed human -> 0 (human); DIPPER- or LLM-polished human -> 2 (AI-edited);
      plain AI -> 1; AI polished by an LLM -> 1; AI back-translated / DIPPER-paraphrased / perturbed / evasion-prompted -> 3.
  * Ghostbuster (github.com/vivek3141/ghostbuster-data): essays / Reuters / WritingPrompts, human vs ChatGPT vs Claude
    (2023), every file not in the benchmark sample.
  * OUTFOX train (github.com/ryuryukke/OUTFOX): student essays vs ChatGPT / text-davinci-003 / Flan-T5 essays.
Raw files under train/raw/public_bench/ (git-ignored).  -> train/data/pubtrain.jsonl
Usage: .venv/bin/python train/build_public_train.py
"""
import glob
import json
import pathlib
import random
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT))
from build_more import h  # noqa: E402
from eval.bench_public import _pkl  # noqa: E402  (builtins-only unpickler)

RAW = ROOT / "train/raw/public_bench"
D = RAW / "detectrl/Benchmark/Benchmark_Data"
rng = random.Random(101)


def ok(t):
    return isinstance(t, str) and 40 <= len(t.split()) <= 800


def detectrl():
    out = []
    plan = [  # (file, which DetectRL label to take, our label, n, src tag)
        *[(f, "llm", 1, 1200, "plain") for f in sorted(glob.glob(str(D / "Multi_LLM/*_train.json")))],
        *[(f, "human", 0, 300, "human") for f in sorted(glob.glob(str(D / "Multi_LLM/*_train.json")))],
        (D / "Paraphrase_Attacks/paraphrase_back_translation_llm_train.json", "llm", 3, 1500, "back-translation"),
        (D / "Paraphrase_Attacks/paraphrase_dipper_llm_train.json", "llm", 3, 800, "dipper"),
        (D / "Paraphrase_Attacks/paraphrase_polish_llm_train.json", "llm", 1, 800, "polish"),
        (D / "Perturbation_Attacks/perturbation_attacks_llm_train.json", "llm", 3, 800, "perturbation"),
        (D / "Prompt_Attacks/prompt_attacks_llm_train.json", "llm", 3, 800, "evasion-prompt"),
        (D / "Paraphrase_Attacks_Human/paraphrase_back_translation_human_train.json", "human", 0, 900, "human-back-translation"),
        (D / "Paraphrase_Attacks_Human/paraphrase_dipper_human_train.json", "human", 2, 400, "human-dipper"),
        (D / "Paraphrase_Attacks_Human/paraphrase_polish_human_train.json", "human", 2, 400, "human-polish"),
        (D / "Perturbation_Attacks_Human/perturbation_attacks_human_train.json", "human", 0, 700, "human-perturbation"),
    ]
    for f, want, lab, n, tag in plan:
        data = [d for d in json.loads(pathlib.Path(f).read_text()) if str(d.get("label")).lower() == want and ok(d.get("text"))]
        for d in rng.sample(data, min(n, len(data))):
            who = d.get("llm_type") or "unknown"
            out.append({"text": d["text"].strip(), "label": lab,
                        "src": f"pub/{'human' if lab == 0 else 'ai'}/detectrl-{tag}/{who}/{d.get('data_type', '')}"})
    return out


def ghostbuster(bench_h):
    out = []
    for dom in ("essay", "reuter", "wp"):
        for src, lab in (("human", 0), ("gpt", 1), ("claude", 1)):
            files = [f for f in sorted(glob.glob(str(RAW / f"ghostbuster/{dom}/{src}/**/*.txt"), recursive=True)) if "logprobs" not in f]
            picked = 0
            for f in rng.sample(files, len(files)):
                t = pathlib.Path(f).read_text(errors="ignore").strip()
                if ok(t) and h(t) not in bench_h:
                    out.append({"text": t, "label": lab, "src": f"pub/{'human' if lab == 0 else 'ai'}/ghostbuster-{dom}/{src}"})
                    picked += 1
                    if picked >= 600:
                        break
    return out


def outfox():
    out = []
    O = RAW / "outfox/data"
    for f, lab, n, tag in ((O / "common/train/train_humans.pkl", 0, 1000, "human"), (O / "chatgpt/train/train_lms.pkl", 1, 600, "chatgpt"),
                           (O / "text_davinci_003/train/train_lms.pkl", 1, 300, "text-davinci-003"), (O / "flan_t5_xxl/train/train_lms.pkl", 1, 300, "flan-t5-xxl")):
        texts = [t for t in _pkl(f) if ok(t)]
        out += [{"text": t.strip(), "label": lab, "src": f"pub/{'human' if lab == 0 else 'ai'}/outfox/{tag}"} for t in rng.sample(texts, min(n, len(texts)))]
    return out


def main():
    bench_h = {json.loads(l)["k"] for l in (ROOT / "eval/data/public_bench.jsonl").read_text().splitlines()}
    rows = detectrl() + ghostbuster(bench_h) + outfox()
    test_h = set(bench_h)
    for f in (ROOT / "test").glob("*.jsonl"):
        test_h |= {h(json.loads(l)["text"]) for l in f.read_text().splitlines()}
    for f in (ROOT / "eval/sol").rglob("texts.json"):
        test_h |= {h(x["text"]) for x in json.loads(f.read_text()) if isinstance(x, dict) and x.get("text")}
    seen, keep, dropped = set(), [], 0
    for r in rows:
        k = h(r["text"])
        if k in test_h:
            dropped += 1
        elif k not in seen:
            seen.add(k)
            keep.append(r)
    (ROOT / "train/data/pubtrain.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    print(f"pubtrain.jsonl: {len(keep)} ({dropped} dropped: in a test set or the public benchmark sample)", dict(Counter(r["label"] for r in keep)))
    for k, n in sorted(Counter("/".join(r["src"].split("/")[1:3]) for r in keep).items()):
        print(f"  {k:45} {n}")


if __name__ == "__main__":
    main()
