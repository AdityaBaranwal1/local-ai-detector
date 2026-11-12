"""Short-form data (40-180 words), both labels, for the weakest group: short emails, form answers, replies, reviews.

Human (label 0, pre-ChatGPT): Enron email (AESLC, offsets not used before), StackExchange best answers (academia,
workplace, interpersonal, writers, ell = English learners, travel, cooking, parenting), Yelp (2015) and Amazon
(pre-2015) reviews, reddit posts (webis/tldr-17, new offsets).
AI (labels 1 / 3): current models through the proxy write the same genres short, in plain / casual / lowercase /
non-native / terse / typo styles; ~1/4 are then rewritten by a "make it human" prompt (label 3, humanized).
-> train/data/short.jsonl, deduplicated against every test set (incl. Sol's rounds) like the other builders.
Usage: .venv/bin/python train/build_short.py [human|ai|all]
"""
import concurrent.futures as cf
import json
import pathlib
import random
import re
import sys
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_more import h  # noqa: E402
from gen_train import chat  # noqa: E402

OUT_H = ROOT / "train/raw/short_human.jsonl"
OUT_A = ROOT / "train/raw/short_ai.jsonl"
rng = random.Random(23)
MODELS = {"claude-haiku-5-5": 4, "gemini-3-flash": 4, "gpt-5.6-luna": 3, "gpt-6-luna": 3, "gpt-oss-120b-medium": 3,
          "claude-sonnet-5-5": 2, "gpt-6-sol": 2, "gemini-3.8-flash-high": 2, "claude-opus-5-5": 1, "claude-fable-5-2": 1}
GENRES = [
    "a short reply email to a coworker who asked whether you can cover their shift on Friday",
    "a two-sentence email asking a recruiter whether the role is still open",
    "an answer for an application form field 'Why are you interested in this role?' for a {role} job",
    "an answer for an application form field 'Describe your relevant experience' for a {role} job",
    "a Slack message telling your team a deploy is delayed and why",
    "a reply on a Q&A site to someone asking how to deal with a coworker who takes credit for their work",
    "a reply on a Q&A site answering how to politely decline a meeting",
    "a short product review of {product}",
    "a restaurant review of a neighborhood {cuisine} place",
    "a forum reply giving advice to someone nervous about their first {thing}",
    "a LinkedIn comment congratulating a former colleague on a new job",
    "a text message to a friend explaining why you'll be late",
    "a short thank-you note to a teacher or mentor",
    "the opening paragraph of a cover letter for a {role} position",
    "a short answer to 'What is your greatest weakness?' in an application",
]
FILL = {"role": ["data analyst", "nurse", "warehouse lead", "junior developer", "teacher", "marketing assistant", "barista"],
        "product": ["a pair of running shoes", "a budget mechanical keyboard", "a rice cooker", "noise-cancelling earbuds", "a phone case"],
        "cuisine": ["Thai", "Ethiopian", "taco", "ramen", "Greek"], "thing": ["job interview", "marathon", "day at a new job", "solo trip"]}
STYLES = [("plain", ""), ("plain", ""), ("casual", " Write it casually, the way a real person quickly types it."),
          ("lowercase", " Write it mostly in lowercase with loose punctuation, like a quick phone message."),
          ("nonnative", " Write it as a non-native English speaker would, with simple words and a small grammar slip or two."),
          ("terse", " Keep it very short and direct."), ("typos", " Type it fast, leaving one or two small typos in.")]
HUMANIZE = ["Rewrite this so it sounds like a real person typed it, not an AI. Keep it about the same length.",
            "Make this sound more human and less polished. Same meaning, same length.",
            "Paraphrase this so AI detectors won't flag it. Keep it short."]
_lock = threading.Lock()


def rows(ds, cfg, split, offset, n=100):
    u = f"https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={offset}&length={n}"
    for i in range(4):
        try:
            return [r["row"] for r in json.load(urllib.request.urlopen(u, timeout=90))["rows"]]
        except Exception as e:  # noqa: BLE001
            print("retry", ds, cfg, offset, e, flush=True)
            time.sleep(5 * (i + 1))
    return []


def ok(t):
    return isinstance(t, str) and 40 <= len(t.split()) <= 180


def human():
    out = []

    def add(t, src, cap):
        t = re.sub(r"\s+\n", "\n", (t or "").strip())
        if ok(t) and sum(1 for x in out if x["src"].startswith(src)) < cap:
            out.append({"text": t, "label": 0, "src": src})
    for off in range(10000, 14000, 200):
        for r in rows("Yale-LILY/aeslc", "default", "train", off):
            add(r["email_body"], "short/human/email-enron", 700)
    for site in ("academia", "workplace", "interpersonal", "writers", "ell", "travel", "cooking", "parenting"):
        for off in rng.sample(range(0, 3000, 100), 3):
            for r in rows("flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl", site, "train", off):
                add(re.sub(r"<[^>]+>", " ", r["upvoted_answer"]), f"short/human/qa-stackexchange-{site}", 110)
    for off in rng.sample(range(0, 640000, 100), 10):
        for r in rows("Yelp/yelp_review_full", "yelp_review_full", "train", off):
            add(r["text"].replace("\\n", "\n"), "short/human/review-yelp", 400)
    for off in rng.sample(range(0, 3_500_000, 100), 10):
        for r in rows("fancyzhx/amazon_polarity", "amazon_polarity", "train", off):
            add(r["content"], "short/human/review-amazon", 400)
    for off in rng.sample(range(500000, 3_000_000, 1000), 15):
        for r in rows("webis/tldr-17", "default", "train", off):
            add(r["content"], "short/human/forum-reddit", 400)
    OUT_H.write_text("".join(json.dumps(r) + "\n" for r in out))
    print("short human:", len(out), flush=True)


def ai(n=1200):
    done = [json.loads(l) for l in OUT_A.read_text().splitlines()] if OUT_A.exists() else []
    ms = list(MODELS)
    jobs = []
    for i in range(n):
        g = rng.choice(GENRES)
        g = re.sub(r"\{(\w+)\}", lambda m: rng.choice(FILL[m.group(1)]), g)
        sname, s = rng.choice(STYLES)
        w = rng.choice([45, 60, 80, 100, 130])
        jobs.append({"key": f"s{i}", "model": rng.choices(ms, weights=[MODELS[m] for m in ms])[0], "style": sname,
                     "genre": g, "prompt": f"Write {g} (about {w} words).{s}"})
    have = {d["key"] for d in done}
    jobs = [j for j in jobs if j["key"] not in have]
    print(f"short ai: {len(done)} done, {len(jobs)} to go", flush=True)

    def run(j):
        try:
            t = chat(j["model"], j["prompt"])
            if not (25 <= len(t.split()) <= 220):
                return None
            r = {"key": j["key"], "text": t, "label": 1, "src": f"short/ai/{j['style']}/{j['model']}", "genre": j["genre"]}
            if rng.random() < 0.25:                      # a quarter go through a humanizer rewrite too
                t2 = chat(rng.choice(ms), rng.choice(HUMANIZE) + "\n\n---\n" + t)
                if 25 <= len(t2.split()) <= 220 and "paste" not in t2[:60].lower():
                    r2 = {"key": j["key"] + "h", "text": t2, "label": 3, "src": f"short/humanized/{j['style']}", "genre": j["genre"]}
                    return [r, r2]
            return [r]
        except Exception as e:  # noqa: BLE001
            print("FAIL", j["model"], str(e)[:80], flush=True)
            return None
    with cf.ThreadPoolExecutor(3) as ex:
        for i, res in enumerate(ex.map(run, jobs)):
            for r in res or []:
                with _lock, OUT_A.open("a") as f:
                    f.write(json.dumps(r) + "\n")
            if i % 100 == 0:
                print(f"short ai {i}/{len(jobs)}", flush=True)


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
    rows_ = [json.loads(l) for f in (OUT_H, OUT_A) if f.exists() for l in f.read_text().splitlines()]
    seen, clean = set(), []
    for r in rows_:
        k = h(r["text"])
        if k not in test_h and k not in seen:
            seen.add(k)
            clean.append({"text": r["text"], "label": r["label"], "src": r["src"]})
    (ROOT / "train/data/short.jsonl").write_text("".join(json.dumps(r) + "\n" for r in clean))
    from collections import Counter
    print("short.jsonl:", len(clean), dict(Counter(r["label"] for r in clean)))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("human", "all"):
        human()
    if what in ("ai", "all"):
        ai()
    finish()
