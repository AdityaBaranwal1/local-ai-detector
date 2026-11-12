"""Matched expository data (v6): the combiner's last weak spot was raw AI text in encyclopedic and biomedical style
(textsight 2026: "Photosynthesis is the biological process...", "Immunohistochemical profiling of the 214 resected
tumours..."), which the models can't yet tell from human Wikipedia and non-anglophone PubMed prose.

Human (label 0, pre-ChatGPT): Wikipedia article openings (20231101 dump; 2-3 paragraphs, 180-340 words) and PubMed
structured abstracts from PubMedQA pqa_artificial (all published before 2019, many non-anglophone authors; properly
cased, unlike ccdv/pubmed, whose lowercase tokenised text would teach "lowercase = human").
AI (label 1): a wide mix of current models, strong ones included, writes the SAME kind of text for the SAME topic:
Wikipedia title -> article opening / encyclopedic overview; PubMedQA study question -> abstract / results / methods
prose with realistic numbers. Markdown is stripped so formatting is never the shortcut. Topics are random real titles,
not the test sets' topics. Deduplicated against every test set and Sol's texts.
-> train/data/expo.jsonl.  Usage: .venv/bin/python train/build_expo.py [human|ai|all]   (resumable)
"""
import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_more import h  # noqa: E402
from build_more2 import rows  # noqa: E402
from gen_train import chat  # noqa: E402

OUT_H = ROOT / "train/raw/expo_human.jsonl"
OUT_A = ROOT / "train/raw/expo_ai.jsonl"
rng = random.Random(31)
MODELS = {"claude-haiku-5-5": 3, "claude-sonnet-5-5": 3, "claude-opus-5-5": 2,
          "gpt-6-sol": 3, "gpt-6-luna": 2, "gpt-5.6-luna": 2, "gpt-oss-120b-medium": 2,
          "gemini-3-flash": 3, "gemini-3.8-flash-high": 2}
N_WIKI, N_MED = 900, 900
_lock = threading.Lock()


def human():
    if OUT_H.exists():
        print("human: already built", flush=True)
        return
    out = []
    for off in rng.sample(range(0, 6_000_000, 100), 40):
        for r in rows("wikimedia/wikipedia", "20231101.en", "train", off):
            paras = [p.strip() for p in r["text"].split("\n") if len(p.split()) >= 30]
            t, n = [], 0
            for p in paras:
                t.append(p)
                n += len(p.split())
                if n >= 180:
                    break
            t = "\n\n".join(t)
            if 180 <= len(t.split()) <= 340 and sum(x["dom"] == "wiki" for x in out) < N_WIKI:
                out.append({"text": t, "label": 0, "src": "expo/human/wikipedia", "dom": "wiki", "topic": r["title"]})
    for off in rng.sample(range(0, 210_000, 100), 25):
        for r in rows("qiaojin/PubMedQA", "pqa_artificial", "train", off):
            t = " ".join(r["context"]["contexts"] + [r["long_answer"]]).strip()
            if 150 <= len(t.split()) <= 380 and sum(x["dom"] == "med" for x in out) < N_MED:
                out.append({"text": t, "label": 0, "src": "expo/human/pubmed", "dom": "med", "topic": r["question"]})
    OUT_H.write_text("".join(json.dumps(r) + "\n" for r in out))
    print("expo human:", {d: sum(x["dom"] == d for x in out) for d in ("wiki", "med")}, flush=True)


WIKI_P = [
    'Write the opening of a Wikipedia article about "{t}" (about {w} words, two or three paragraphs). Plain prose, no headings, no lists, no title.',
    'Write an encyclopedic overview of "{t}" in about {w} words, the way a reference work would. Plain paragraphs only.',
]
MED_P = [
    "Write the abstract of a biomedical research paper for a study asking: \"{t}\" (about {w} words, one paragraph, background, methods, results with realistic numbers, conclusion). Plain text, no section labels, no title.",
    "Write the Results section of a biomedical research paper for a study asking: \"{t}\" (about {w} words of prose with realistic numbers, p-values and cohort sizes). No headings, no tables, no title.",
    "Write a Methods paragraph (about {w} words) for a clinical study asking: \"{t}\". Plain prose, no headings, no title.",
]


def clean(t):
    t = re.sub(r"^\s*#+\s.*$", "", t, flags=re.M)               # headings
    t = re.sub(r"\*\*?([^*]+)\*\*?", r"\1", t)                  # bold / italics
    t = re.sub(r"^\s*[-*•]\s+", "", t, flags=re.M)              # bullets
    t = re.sub(r"^(Here is|Here's|Sure|Certainly)[^\n]*\n", "", t.strip(), flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def ai():
    hum = [json.loads(l) for l in OUT_H.read_text().splitlines()]
    done = {json.loads(l)["key"] for l in OUT_A.read_text().splitlines()} if OUT_A.exists() else set()
    ms = list(MODELS)
    jobs = []
    for i, r in enumerate(hum):
        tpl = rng.choice(WIKI_P if r["dom"] == "wiki" else MED_P)
        w = rng.choice([220, 260, 290, 320])
        jobs.append({"key": f"x{i}", "dom": r["dom"], "model": rng.choices(ms, weights=[MODELS[m] for m in ms])[0],
                     "prompt": tpl.format(t=r["topic"], w=w)})
    jobs = [j for j in jobs if j["key"] not in done]
    print(f"expo ai: {len(done)} done, {len(jobs)} to go", flush=True)

    def run(j):
        try:
            t = clean(chat(j["model"], j["prompt"]))
            if not (120 <= len(t.split()) <= 450):
                return None
            return {"key": j["key"], "text": t, "label": 1, "src": f"expo/ai/{'wikipedia' if j['dom'] == 'wiki' else 'pubmed'}/{j['model']}"}
        except Exception as e:  # noqa: BLE001
            print("FAIL", j["model"], str(e)[:80], flush=True)
            return None
    with cf.ThreadPoolExecutor(int(os.environ.get("EXPO_THREADS", 3))) as ex:
        for i, fut in enumerate(cf.as_completed([ex.submit(run, j) for j in jobs])):   # write as each finishes
            r = fut.result()
            if r:
                with _lock, OUT_A.open("a") as f:
                    f.write(json.dumps(r) + "\n")
            if i % 50 == 0:
                print(f"expo ai {i}/{len(jobs)}", flush=True)


def finish():
    test_h = set()
    for f in (ROOT / "test").glob("*.jsonl"):
        for l in f.read_text().splitlines():
            test_h.add(h(json.loads(l)["text"]))
    for f in list((ROOT / "eval/sol").rglob("texts.json")) + list((ROOT / "eval/data").rglob("*.json")):
        try:
            for r in json.loads(f.read_text()):
                t = isinstance(r, dict) and (r.get("text") or r.get("document"))
                if t:
                    test_h.add(h(t))
        except Exception:  # noqa: BLE001
            pass
    seen, clean_ = set(), []
    hum = [json.loads(l) for l in OUT_H.read_text().splitlines()] if OUT_H.exists() else []
    ai_ = [json.loads(l) for l in OUT_A.read_text().splitlines()] if OUT_A.exists() else []
    # balanced per domain: as many human texts as AI ones (extra2's 2.6 : 1 human tilt taught "formal = human"),
    # preferring the humans whose own topic got an AI counterpart (key x<i> = human row i)
    paired = {int(r["key"][1:]) for r in ai_}
    for dom, tag in (("wiki", "wikipedia"), ("med", "pubmed")):
        n_ai = sum(r["src"].split("/")[2] == tag for r in ai_)
        idx = [i for i, r in enumerate(hum) if r and r["dom"] == dom]
        idx.sort(key=lambda i: i not in paired)
        for i in idx[n_ai:]:
            hum[i] = None
    for r in [x for x in hum if x] + ai_:
        k = h(r["text"])
        if k not in test_h and k not in seen:
            seen.add(k)
            clean_.append({"text": r["text"], "label": r["label"], "src": r["src"]})
    (ROOT / "train/data/expo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in clean_))
    from collections import Counter
    print("expo.jsonl:", len(clean_), dict(Counter((r["label"], r["src"].split("/")[2]) for r in clean_)), flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("human", "all"):
        human()
    if what in ("ai", "all"):
        ai()
    finish()
