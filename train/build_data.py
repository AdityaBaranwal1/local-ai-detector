"""Collect public human/AI texts into train/data/{train,val}.jsonl and test/heldout.jsonl.

Labels follow tropa-mini's head: 0 human, 1 ai, 2 ai_edited, 3 humanized.
Every text that appears in any test set (eval/data, eval/sites, the held-out splits here) is removed from training
by a normalised-text hash, so test numbers are never measured on training data.
Usage: .venv/bin/python train/build_data.py   (downloads into train/raw/, ~400 MB)
"""
import hashlib
import io
import json
import pathlib
import random
import re
import time
import urllib.request

import pandas as pd

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
RAW = HERE / "raw"
OUT = HERE / "data"
TEST = ROOT / "test"
for d in (RAW, OUT, TEST):
    d.mkdir(exist_ok=True, parents=True)
HUMAN, AI, EDITED, HUMANIZED = 0, 1, 2, 3


def norm(t):
    return re.sub(r"\W+", " ", t.lower()).strip()


def h(t):
    return hashlib.sha1(norm(t)[:2000].encode()).hexdigest()


def split_of(key, test_frac):
    return "test" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000 < test_frac * 1000 else "train"


def fetch(url, dest):
    dest = RAW / dest
    if not dest.exists():
        for i in range(4):
            try:
                urllib.request.urlretrieve(url, dest)
                break
            except Exception as e:  # noqa: BLE001
                print("retry", url, e)
                time.sleep(5 * (i + 1))
    return dest


def parquet(ds, cfg="default", split="train", name=None):
    url = f"https://huggingface.co/api/datasets/{ds}/parquet/{cfg}/{split}/0.parquet"
    return pd.read_parquet(fetch(url, name or (ds.replace("/", "__") + f"__{cfg}.parquet")))


def rows(ds, cfg, split, offset, n=100):
    u = f"https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={offset}&length={n}"
    for i in range(4):
        try:
            return [r["row"] for r in json.load(urllib.request.urlopen(u, timeout=90))["rows"]]
        except Exception as e:  # noqa: BLE001
            print("retry rows", ds, offset, e)
            time.sleep(5 * (i + 1))
    return []


def ok_len(t, lo=40, hi=1200):
    n = len(str(t).split())
    return lo <= n <= hi


def clip(t, words=600):
    w = str(t).split(" ")
    return " ".join(w[:words]) if len(w) > words else str(t)


train, test = [], []


def add(bucket, text, label, src):
    text = str(text).strip()
    if ok_len(text):
        bucket.append({"text": clip(text), "label": label, "src": src})


def main():
    rng = random.Random(7)

    # --- Jabarian & Imas 2025: human passages + 4 models raw + StealthGPT-humanized, paired; 80/20 by record ---
    B = "https://raw.githubusercontent.com/brianjabarian/DetectionAI/main/all_json"
    models = ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "gemini-2.0-flash", "gpt-4.1-2025-04-14"]
    raw_files = {m: json.loads(fetch(f"{B}/100_ai_generated_text_only_files/generated_output_{m}.json", f"jab_{m}.json").read_text()) for m in models}
    st_files = {m: json.loads(fetch(f"{B}/100_stealth_generated_text_only_files/generated_output_{m}_stealthgpt.json", f"jab_{m}_stealth.json").read_text()) for m in models}
    # files differ in length/order: pair every version by its human source passage
    K = lambda rec: norm(rec["text"])[:200]
    raw_by = {m: {K(r): r["ai_generated"][m] for r in raw_files[m] if (r.get("ai_generated") or {}).get(m)} for m in models}
    st_by = {m: {K(r): r["stealthgpt_rephrased"][m] for r in st_files[m] if (r.get("stealthgpt_rephrased") or {}).get(m)} for m in models}
    humans = {K(r): r for r in raw_files[models[0]]}
    for k, rec in humans.items():
        bucket = test if split_of("jab" + k, 0.2) == "test" else train
        add(bucket, rec["text"], HUMAN, f"jabarian/human/{rec.get('genre')}")
        have_raw = [m for m in models if k in raw_by[m]]
        have_st = [m for m in models if k in st_by[m]]
        if bucket is test:   # test: every model, raw and humanized
            for m in have_raw:
                add(test, raw_by[m][k], AI, f"jabarian/ai/{m}")
            for m in have_st:
                add(test, st_by[m][k], HUMANIZED, f"jabarian/humanized/{m}")
        else:                # train: one model raw + one humanized per record (keeps the classes balanced)
            if have_raw:
                m1 = rng.choice(have_raw)
                add(train, raw_by[m1][k], AI, f"jabarian/ai/{m1}")
            if have_st:
                m2 = rng.choice(have_st)
                add(train, st_by[m2][k], HUMANIZED, f"jabarian/humanized/{m2}")

    # --- learner (non-native) essays: ELLIPSE (US ELL students) + ICNALE (Asian learners); 85/15 ---
    ell = parquet("tcapelle/feedback-prize-english-language-learning-fluency")
    learners = []
    for t in ell["full_text"]:
        b = test if split_of("ell" + norm(t)[:200], 0.15) == "test" else train
        add(b, t, HUMAN, "learner/ellipse")
        if b is train:
            learners.append(t)
    ic = parquet("srky/ICNALE_writing_score")
    for t in ic["OriginalText"].dropna():
        b = test if split_of("icn" + norm(t)[:200], 0.15) == "test" else train
        add(b, t, HUMAN, "learner/icnale")
        if b is train:
            learners.append(t)
    with open(RAW / "learner_train.jsonl", "w") as f:   # input for gen_train.py polish
        for t in learners:
            if ok_len(t, 120, 600):
                f.write(json.dumps({"text": t}) + "\n")

    # --- DAIGT v4: persuade student essays (human) + AI essays from many models ---
    dg = parquet("ramensoft/daigt_v4")
    dg = dg.sample(frac=1, random_state=3)
    hum = dg[dg["label"].astype(str) == "0"].head(1200)
    ai = dg[dg["label"].astype(str) == "1"].head(1200)
    for t in hum["text"]:
        add(train, t, HUMAN, "daigt/human")
    for t, m in zip(ai["text"], ai["model"]):
        add(train, t, AI, f"daigt/ai/{m}")

    # --- RAID: humans + AI across domains, incl. paraphrase attacks (-> humanized) ---
    seen_offsets = rng.sample(range(0, 2_270_000, 100), 70)
    for off in seen_offsets:
        for r in rows("liamdugan/raid", "raid", "train", off):
            g = r.get("generation")
            if not g or r.get("domain") in ("code", "czech", "german"):
                continue
            if r["model"] == "human":
                if r.get("attack") == "none":
                    add(train, g, HUMAN, f"raid/human/{r['domain']}")
            elif r.get("attack") == "paraphrase":
                add(train, g, HUMANIZED, f"raid/paraphrase/{r['model']}")
            elif r.get("attack") in ("none", "synonym", "misspelling", "upper_lower", "insert_paragraphs",
                                     "article_deletion", "alternative_spelling", "number"):
                add(train, g, AI, f"raid/ai/{r['model']}/{r['attack']}")

    # --- everyday human writing outside the eval's offsets: Enron email, reddit posts, stories ---
    for off in range(2000, 10000, 400):
        for r in rows("Yale-LILY/aeslc", "default", "train", off):
            add(train, r["email_body"], HUMAN, "human/email-enron")
    for off in range(100000, 400000, 15000):
        for r in rows("webis/tldr-17", "default", "train", off):
            add(train, r["content"], HUMAN, "human/reddit-post")
    for off in range(0, 6000, 600):
        for r in rows("euclaise/writingprompts", "default", "train", off):
            add(train, re.sub(r" ([.,!?'])", r"\1", r["story"].replace("<newline>", "\n")), HUMAN, "human/story")

    # --- test-only: textsight 2026 benchmark (incl. non-anglophone authors) ---
    for cfg, lab in (("ai_generated_2026", AI), ("human_literature_publicdomain", HUMAN),
                     ("human_pmc_ccby_nonanglophone", HUMAN), ("human_wikipedia_pre2022", HUMAN)):
        df = parquet("textsightai/ai-text-detection-benchmark-2026", cfg)
        for t in df["text"]:
            add(test, t, lab, f"textsight/{cfg}")

    # --- remove anything that is in a test set from training ---
    test_hashes = {h(r["text"]) for r in test}
    for f in (ROOT / "eval/data").rglob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        for r in d if isinstance(d, list) else []:
            t = r.get("text") or r.get("document")
            if t:
                test_hashes.add(h(t))
    for f in (ROOT / "eval/sites").glob("*.json"):
        for r in json.loads(f.read_text()):
            test_hashes.add(h(r["text"]))
    before = len(train)
    seen, clean = set(), []
    for r in train:
        k = h(r["text"])
        if k in test_hashes or k in seen:
            continue
        seen.add(k)
        clean.append(r)
    print(f"train {before} -> {len(clean)} after removing test overlap/duplicates; test {len(test)}")
    rng.shuffle(clean)
    nval = 600
    for name, part in (("val", clean[:nval]), ("train", clean[nval:])):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    with open(TEST / "heldout.jsonl", "w") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print("train labels", Counter(r["label"] for r in clean[nval:]))
    print("train sources", Counter(r["src"].split("/")[0] + "/" + r["src"].split("/")[1] for r in clean).most_common(25))
    print("test sources", Counter("/".join(r["src"].split("/")[:2]) for r in test).most_common(25))


if __name__ == "__main__":
    main()
