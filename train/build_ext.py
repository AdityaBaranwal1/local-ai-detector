"""Public AI-text datasets for generator families the detector hasn't seen (found 2026-11-12; no LLM calls, no tokens).

  * mild-rgb/aita-human-vs-ai (Apache-2.0) and mild-rgb/eli5-human-vs-ai (CC-BY-4.0): 2026 models -- DeepSeek V4 Pro,
    GLM 5.3, Grok 4.6, Gemini 3.7 Flash, Qwen 3.8 Max, GPT-5.6 Luna Pro, Nemotron 3.5 -- ~415 each per set; first-person
    AITA narratives (casual) and ELI5 explanations, rate-matched to human paragraph/length features (hard positives).
    Their human halves aren't redistributed (ids only), so only the AI side is used.
  * toloka/beemo (MIT): Llama-2 7/13/70B, Mixtral, Mistral-7B, Gemma-1 2B/7B, zephyr, tulu-2 outputs + the human-written
    No Robots reference answer for the same prompt (label 0) -- matched pairs.
  * allenai/WildChat-1M (ODC-BY), shard 0: real ChatGPT replies to real users (gpt-3.5-turbo-0301, gpt-4-0314; 2023),
    English, prose only (no code blocks), 40-400 words.
Markdown is stripped. Deduplicated against every test set and the review rounds.
-> train/data/ext.jsonl.  Usage: .venv/bin/python train/build_ext.py   (inputs under train/raw/ext_src/hf/)
"""
import json
import pathlib
import random
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_expo import clean  # noqa: E402
from build_more import h  # noqa: E402

SRC = ROOT / "train/raw/ext_src/hf"
rng = random.Random(71)


def ok(t, lo=40, hi=450):
    return isinstance(t, str) and lo <= len(t.split()) <= hi and "```" not in t and not re.search(r"(?i)\bas an ai\b", t)


def norm(t):
    t = clean(t or "")
    t = re.sub(r"^\s*\|.*\|\s*$", "", t, flags=re.M)               # tables
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def main():
    import pyarrow.parquet as pq
    rows = []
    for name in ("aita-human-vs-ai", "eli5-human-vs-ai"):
        d = SRC / name
        files = ["dataset.jsonl"] if name.startswith("aita") else ["train.jsonl", "dev.jsonl", "test.jsonl"]
        for f in files:
            if not (d / f).exists():
                continue
            for l in (d / f).read_text().splitlines():
                r = json.loads(l)
                if r.get("label") == "ai" and r.get("text"):
                    t = norm(r["text"])
                    m = (r.get("generator") or r.get("model") or "unknown").split("/")[-1]
                    if ok(t):
                        rows.append({"text": t, "label": 1, "src": f"ext/ai/{name.split('-')[0]}/{m}"})
    b = pq.read_table(SRC / "beemo/data/train-00000-of-00001.parquet", columns=["model", "category", "model_output", "human_output"]).to_pylist()
    for r in b:
        m = r["model"].split("/")[-1]
        t, hu = norm(r["model_output"]), norm(r["human_output"])
        if ok(t) and ok(hu):
            rows.append({"text": t, "label": 1, "src": f"ext/ai/beemo-{r['category']}/{m}"})
            rows.append({"text": hu, "label": 0, "src": f"ext/human/beemo-{r['category']}/no-robots"})
    w = pq.read_table(SRC / "WildChat-1M/data/train-00000-of-00014.parquet", columns=["model", "language", "conversation", "toxic", "redacted"]).to_pylist()
    rng.shuffle(w)
    per = Counter()
    for r in w:
        if r["language"] != "English" or r["toxic"] or r["redacted"]:
            continue
        for turn in r["conversation"]:
            if turn.get("role") != "assistant":
                continue
            raw = turn.get("content") or ""
            if "```" in raw or re.search(r"https?://|\S+@\S+\.\w+", raw):
                continue
            t = norm(raw)
            if ok(t) and per[r["model"]] < 2000:
                rows.append({"text": t, "label": 1, "src": f"ext/ai/wildchat/{r['model']}"})
                per[r["model"]] += 1
            break                                                    # first assistant reply per conversation
    test_h = set()
    for f in (ROOT / "test").glob("*.jsonl"):
        test_h |= {h(json.loads(l)["text"]) for l in f.read_text().splitlines()}
    for f in list((ROOT / "eval/sol").rglob("texts.json")) + list((ROOT / "eval/data").rglob("*.json")):
        try:
            test_h |= {h(x.get("text") or x.get("document")) for x in json.loads(f.read_text()) if isinstance(x, dict) and (x.get("text") or x.get("document"))}
        except Exception:  # noqa: BLE001
            pass
    seen, keep = set(), []
    for r in rows:
        k = h(r["text"])
        if k not in test_h and k not in seen:
            seen.add(k)
            keep.append(r)
    (ROOT / "train/data/ext.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    print("ext.jsonl:", len(keep), dict(Counter(r["label"] for r in keep)))
    for k, n in sorted(Counter("/".join(r["src"].split("/")[2:]) for r in keep).items()):
        print(f"  {k:55} {n}")


if __name__ == "__main__":
    main()
