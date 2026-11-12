"""Domain-balanced extra training data -> train/data/extra.jsonl (labels 0 human / 1 ai).

Why: model k3 (trained on train.jsonl) learned "human looks like our human sources": 70 % of pre-2022 Wikipedia,
40 % of non-native science papers flagged. Fix = every domain has BOTH human and AI text, so domain or formality is
never a shortcut. Sources: MAGE train split, COLING-2025 MGT train split, HC3 (rows not in test/ood.jsonl).
Deduplicated against every test set (test/*.jsonl, eval data, Sol's rounds) by normalised-text hash.
"""
import hashlib
import json
import pathlib
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
rng = random.Random(7)
CAP_MAGE, CAP_COLING, CAP_HC3 = 350, 250, 150     # per (domain, label)


def norm(t):
    return re.sub(r"\W+", " ", t.lower()).strip()


def h(t):
    return hashlib.sha1(norm(t)[:2000].encode()).hexdigest()


def rows(ds, cfg, split, offset, n=100):
    u = f"https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={offset}&length={n}"
    for i in range(4):
        try:
            return [r["row"] for r in json.load(urllib.request.urlopen(u, timeout=90))["rows"]]
        except Exception as e:  # noqa: BLE001
            print("retry", ds, offset, e, flush=True)
            time.sleep(5 * (i + 1))
    return []


def ok(t):
    return isinstance(t, str) and 40 <= len(t.split()) <= 700


def main():
    out, cnt = [], Counter()

    def add(text, label, dom, src, cap):
        if ok(text) and cnt[(dom, label)] < cap:
            cnt[(dom, label)] += 1
            out.append({"text": text.strip(), "label": label, "src": src})

    for i, off in enumerate(rng.sample(range(0, 319071 - 100, 100), 220)):
        for r in rows("yaful/MAGE", "default", "train", off):
            human = "human" in r["src"]
            dom = "mage:" + r["src"].split("_")[0]
            add(r["text"], 0 if human else 1, dom, f"mage/{'human' if human else 'ai'}/{r['src'][:40]}", CAP_MAGE)
        if i % 40 == 0:
            print("mage", i, len(out), flush=True)
    for i, off in enumerate(rng.sample(range(0, 610767 - 100, 100), 300)):
        for r in rows("Jinyan1/COLING_2025_MGT_en", "default", "train", off):
            lab = int(r["label"])
            dom = f"coling:{r.get('source')}:{r.get('sub_source')}"
            add(r.get("text"), lab, dom, f"coling/{'ai' if lab else 'human'}/{r.get('source')}/{r.get('sub_source')}/{(r.get('model') or '')[:20]}", CAP_COLING)
        if i % 50 == 0:
            print("coling", i, len(out), flush=True)
    for cfg in ("finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai"):
        for off in rng.sample(range(0, 900, 100), 4):
            for r in rows("Hello-SimpleAI/HC3", cfg, "train", off):
                for t in (r["human_answers"] or [])[:1]:
                    add(t, 0, "hc3:" + cfg, f"hc3/human/{cfg}", CAP_HC3)
                for t in (r["chatgpt_answers"] or [])[:1]:
                    add(t, 1, "hc3:" + cfg, f"hc3/ai/{cfg}", CAP_HC3)

    # never train on a test text
    test_h = set()
    for f in list((ROOT / "test").glob("*.jsonl")):
        for l in f.read_text().splitlines():
            test_h.add(h(json.loads(l)["text"]))
    for f in list((ROOT / "eval/data").rglob("*.json")) + list((ROOT / "eval/sites").glob("*.json")) + list((ROOT / "eval/sol").rglob("texts.json")) + [ROOT / "test/zerogpt_heldout.json"]:
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        for r in d if isinstance(d, list) else []:
            t = isinstance(r, dict) and (r.get("text") or r.get("document"))
            if t:
                test_h.add(h(t))
    seen, clean = set(), []
    for r in out:
        k = h(r["text"])
        if k in test_h or k in seen:
            continue
        seen.add(k)
        clean.append(r)
    by = defaultdict(Counter)
    for r in clean:
        by[r["src"].split("/")[0]][r["label"]] += 1
    print(f"{len(out)} -> {len(clean)} after removing test overlap/duplicates;", {k: dict(v) for k, v in by.items()})
    doms = Counter()
    for (d, lab), n in cnt.items():
        doms[d.split(":")[0]] += 1
    print("domains x labels:", dict(doms))
    with open(ROOT / "train/data/extra.jsonl", "w") as f:
        for r in clean:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
