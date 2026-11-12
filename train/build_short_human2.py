"""More real, pre-ChatGPT SHORT human writing -- the one place ZeroGPT still beats us on every benchmark is human text
wrongly flagged, mostly short messages. Label 0 only (we have plenty of AI text).
  * GoEmotions raw (google-research-datasets/go_emotions, Apache-2.0): Reddit comments, 2005-2019.
  * SAMSum (knkarthick/samsum mirror, CC BY-NC-ND 4.0): messenger-style chats written by linguists (2019); speaker
    names stripped so "Name:" lines aren't a format tell.
  * EmpatheticDialogues (Estwld/empathetic_dialogues_llm mirror, CC BY-NC 4.0): crowdworkers' first-person situation
    descriptions and their own turns (2018).
Personal / research use only. -> train/data/shorthuman.jsonl (sanitized again at packaging).
Usage: .venv/bin/python train/build_short_human2.py
"""
import json
import pathlib
import random
import re
import sys
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_more import h  # noqa: E402

D = ROOT / "train/raw/ext_src/hf"
rng = random.Random(107)
N = 1500


def ok(t, lo=20, hi=200):
    return isinstance(t, str) and lo <= len(t.split()) <= hi


def main():
    rows = []
    g = pq.read_table(D / "go_emotions/raw/train-00000-of-00001.parquet", columns=["text", "id", "created_utc"]).to_pandas().drop_duplicates("id")
    g = g[g["text"].map(lambda t: ok(t, 20, 150) and "[NAME]" not in t and "[RELIGION]" not in t)]
    for t in g.sample(min(N, len(g)), random_state=1)["text"]:
        rows.append({"text": t.strip(), "label": 0, "src": "shorthuman/human/reddit-goemotions"})
    s = pd.read_csv(D / "samsum/train.csv")
    for d in s.sample(frac=1, random_state=2)["dialogue"]:
        lines = [re.sub(r"^[^:\n]{1,30}:\s*", "", l).strip() for l in str(d).replace("\r", "").split("\n")]
        t = "\n".join(l for l in lines if l and l != "<file_photo>")
        if ok(t, 25, 200):
            rows.append({"text": t, "label": 0, "src": "shorthuman/human/samsum-chat"})
        if sum(r["src"].endswith("samsum-chat") for r in rows) >= N:
            break
    e = pq.read_table(D / "empathetic/data/train-00000-of-00001.parquet").to_pylist()
    rng.shuffle(e)
    for r in e:
        cands = [r["situation"]] + [c.get("content", "") for c in (r.get("conversations") or []) if isinstance(c, dict) and c.get("role") == "user"]
        for t in cands:
            t = (t or "").replace("_comma_", ",").strip()
            if ok(t, 20, 150):
                rows.append({"text": t, "label": 0, "src": "shorthuman/human/empathetic-dialogues"})
                break
        if sum(x["src"].endswith("empathetic-dialogues") for x in rows) >= N:
            break
    test_h = set()
    for f in (ROOT / "test").glob("*.jsonl"):
        test_h |= {h(json.loads(l)["text"]) for l in f.read_text().splitlines()}
    for f in (ROOT / "eval/sol").rglob("texts.json"):
        test_h |= {h(x["text"]) for x in json.loads(f.read_text()) if isinstance(x, dict) and x.get("text")}
    seen, keep = set(), []
    for r in rows:
        k = h(r["text"])
        if k not in test_h and k not in seen:
            seen.add(k)
            keep.append(r)
    (ROOT / "train/data/shorthuman.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    print("shorthuman.jsonl:", len(keep), dict(Counter(r["src"].split("/")[-1] for r in keep)),
          "| median words", sorted(len(r["text"].split()) for r in keep)[len(keep) // 2])


if __name__ == "__main__":
    main()
