"""Out-of-distribution test set: public detection datasets never used for training -> test/ood.jsonl.

MAGE (test split), HC3, ArguGPT (Kaggle alejopaullier/argugpt), COLING-2025 MGT (en dev). Stratified samples,
40-600 words, deduplicated against the training data by normalised-text hash. Label 0 human, 1 AI.
Usage: .venv/bin/python train/build_ood.py
"""
import glob
import hashlib
import json
import pathlib
import random
import re
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "train/raw/ood"
RAW.mkdir(parents=True, exist_ok=True)
rng = random.Random(42)


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
            print("retry", ds, offset, e)
            time.sleep(5 * (i + 1))
    return []


def ok(t):
    return isinstance(t, str) and 40 <= len(t.split()) <= 600


out = []


def add(text, label, src):
    if ok(text):
        out.append({"text": text.strip(), "label": label, "src": src})


def mage():
    # src like "xsum_human", "cmv_machine_continuation_gpt-3.5-trubo", ...; label 1 = human in MAGE
    seen = Counter()
    for off in rng.sample(range(0, 60743 - 100, 100), 40):
        for r in rows("yaful/MAGE", "default", "test", off):
            human = "human" in r["src"]
            dom = r["src"].split("_")[0]
            key = (dom, human)
            if seen[key] >= 25:
                continue
            seen[key] += 1
            add(r["text"], 0 if human else 1, f"mage/{'human' if human else 'ai'}/{r['src'][:40]}")


def hc3():
    for cfg in ("finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai"):
        got = rows("Hello-SimpleAI/HC3", cfg, "train", rng.randrange(0, 800), 100)
        nh = na = 0
        for r in got:
            if r["human_answers"] and nh < 30:
                before = len(out)
                add(r["human_answers"][0], 0, f"hc3/human/{cfg}")
                nh += len(out) - before
            if r["chatgpt_answers"] and na < 30:
                before = len(out)
                add(r["chatgpt_answers"][0], 1, f"hc3/ai/{cfg}")
                na += len(out) - before


def argugpt():
    d = RAW / "argugpt"
    if not d.exists():
        subprocess.run([str(ROOT / ".venv-lightning/bin/kaggle"), "datasets", "download", "-d", "alejopaullier/argugpt",
                        "-p", str(d), "--unzip"], check=True)
    import pandas as pd
    files = glob.glob(str(d / "**/*.csv"), recursive=True)
    print("argugpt files", [pathlib.Path(f).name for f in files])
    for f in files:
        df = pd.read_csv(f)
        cols = {c.lower(): c for c in df.columns}
        tcol = cols.get("text") or cols.get("essay")
        if not tcol:
            continue
        name = pathlib.Path(f).stem.lower()
        if "machine" in name or "gpt" in name and "human" not in name:
            lab = 1
        elif "human" in name:
            lab = 0
        else:
            continue
        sub = df.sample(min(len(df), 150), random_state=1)
        for t in sub[tcol]:
            add(str(t), lab, f"argugpt/{'human' if lab == 0 else 'ai'}/{name}")


def coling():
    seen = Counter()
    for off in rng.sample(range(0, 261758 - 100, 100), 60):
        for r in rows("Jinyan1/COLING_2025_MGT_en", "default", "dev", off):
            text = r.get("text") or r.get("content")
            if text is None:
                continue
            lab = int(r["label"])     # COLING MGT: 1 = machine
            key = (r.get("source"), lab)
            if seen[key] >= 20:
                continue
            seen[key] += 1
            add(text, lab, f"coling/{'ai' if lab else 'human'}/{r.get('source')}/{(r.get('model') or '')[:20]}")


def main():
    for f in (mage, hc3, argugpt, coling):
        n0 = len(out)
        try:
            f()
        except Exception as e:  # noqa: BLE001 - one source failing must not stop the others
            print("SOURCE FAILED", f.__name__, e)
        print(f.__name__, len(out) - n0)
    train_h = set()
    for l in (ROOT / "train/data/train.jsonl").read_text().splitlines():
        train_h.add(h(json.loads(l)["text"]))
    for f in (ROOT / "train/raw").glob("gen_*.jsonl"):
        for l in f.read_text().splitlines():
            train_h.add(h(json.loads(l)["text"]))
    seen, clean = set(), []
    for r in out:
        k = h(r["text"])
        if k in train_h or k in seen:
            continue
        seen.add(k)
        clean.append(r)
    print(f"{len(out)} -> {len(clean)} after removing training overlap/duplicates")
    by = defaultdict(Counter)
    for r in clean:
        by[r["src"].split("/")[0]][r["label"]] += 1
    print({k: dict(v) for k, v in by.items()})
    with open(ROOT / "test/ood.jsonl", "w") as f:
        for r in clean:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
