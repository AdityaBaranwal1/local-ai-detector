"""Matched human/AI data for the domains the models still over-flag: literature, academic, encyclopedic.

Human (label 0): Project Gutenberg excerpts, arXiv abstracts (2021 dump: pre-ChatGPT, many non-native authors),
Wikipedia paragraphs. AI (label 1): a cheap model writes the same kind of text for the SAME title/topic (arXiv title ->
abstract, Wikipedia title -> encyclopedic paragraph, novel -> a passage in its style), so formality or period style is
never a shortcut. -> train/data/extra2.jsonl. Deduplicated against every test set like the other builders.
"""
import concurrent.futures as cf
import json
import pathlib
import random
import re
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_more import h  # noqa: E402
from gen_train import chat  # noqa: E402

rng = random.Random(11)
CHEAP = ["claude-haiku-5-5", "gemini-3-flash", "gpt-5.6-luna", "gpt-6-luna", "gpt-oss-120b-medium"]
N = 450


def rows(ds, cfg, split, offset, n=100):
    u = f"https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={offset}&length={n}"
    for i in range(4):
        try:
            return [r["row"] for r in json.load(urllib.request.urlopen(u, timeout=120))["rows"]]
        except Exception as e:  # noqa: BLE001
            print("retry", ds, offset, e, flush=True)
            time.sleep(5 * (i + 1))
    return []


def excerpt(text, lo=150, hi=380):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 40]
    if not paras:
        return None
    start = rng.randrange(len(paras) // 5, max(len(paras) // 5 + 1, 4 * len(paras) // 5))
    out = []
    for p in paras[start:]:
        out.append(re.sub(r"\s*\n\s*", " ", p))
        if sum(len(x.split()) for x in out) >= lo:
            break
    t = " ".join(out)
    return t if lo <= len(t.split()) <= hi * 2 else None


def main():
    human, jobs = [], []
    # arXiv abstracts (title -> AI abstract for the same title)
    for off in rng.sample(range(0, 1_999_000, 100), 12):
        for r in rows("gfissore/arxiv-abstracts-2021", "default", "train", off):
            a = re.sub(r"\s+", " ", r.get("abstract") or "").strip()
            if 80 <= len(a.split()) <= 350 and len([x for x in human if x["src"].startswith("more2/human/arxiv")]) < N:
                human.append({"text": a, "label": 0, "src": "more2/human/arxiv"})
                jobs.append(("arxiv", r["title"].strip(), f"Write the abstract of a research paper titled \"{r['title'].strip()}\" ({len(a.split())} words). Plain text, no title."))
    # Wikipedia paragraphs (title -> AI encyclopedia paragraph)
    for off in rng.sample(range(0, 6_000_000, 100), 15):
        for r in rows("wikimedia/wikipedia", "20231101.en", "train", off):
            paras = [p for p in r["text"].split("\n") if 100 <= len(p.split()) <= 300]
            if paras and len([x for x in human if x["src"].startswith("more2/human/wikipedia")]) < N:
                human.append({"text": paras[0], "label": 0, "src": "more2/human/wikipedia"})
                jobs.append(("wikipedia", r["title"], f"Write an encyclopedia paragraph (about {len(paras[0].split())} words) about \"{r['title']}\", in the style of Wikipedia. Plain text."))
    # Gutenberg excerpts (book -> AI passage in its style)
    for off in rng.sample(range(0, 48_000, 20), 30):
        for r in rows("sedthh/gutenberg_english", "default", "train", off, n=20):
            ex = excerpt(r.get("TEXT") or "")
            if ex and len([x for x in human if x["src"].startswith("more2/human/gutenberg")]) < N:
                meta = r.get("METADATA") or ""
                title = re.search(r'"title":\s*"([^"]+)"', meta)
                title = title.group(1) if title else "a 19th-century novel"
                human.append({"text": ex, "label": 0, "src": "more2/human/gutenberg"})
                jobs.append(("gutenberg", title, f"Write a passage of about {len(ex.split())} words that could appear in \"{title}\", matching the period's style and vocabulary. Prose only, no title."))
    print("human", {k: sum(1 for x in human if x["src"].endswith(k)) for k in ("arxiv", "wikipedia", "gutenberg")}, flush=True)
    rng.shuffle(jobs)
    jobs = jobs[: N]                                      # AI counterparts: ~1/3 of the human count per domain

    def run(j):
        dom, _, prompt = j
        m = rng.choice(CHEAP)
        try:
            t = chat(m, prompt)
            return {"text": t, "label": 1, "src": f"more2/ai/{dom}/{m}"} if len(t.split()) >= 60 else None
        except Exception as e:  # noqa: BLE001
            print("FAIL", m, str(e)[:80], flush=True)
            return None
    ai = []
    with cf.ThreadPoolExecutor(3) as ex:
        for r in ex.map(run, jobs):
            if r:
                ai.append(r)
    test_h = set()
    for f in (ROOT / "test").glob("*.jsonl"):
        for l in f.read_text().splitlines():
            test_h.add(h(json.loads(l)["text"]))
    out = [r for r in human + ai if h(r["text"]) not in test_h]
    with open(ROOT / "train/data/extra2.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"extra2: {len(out)} rows ({sum(r['label'] == 0 for r in out)} human, {sum(r['label'] == 1 for r in out)} AI)")


if __name__ == "__main__":
    main()
