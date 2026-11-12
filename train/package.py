"""Package the training data for Kaggle / Runpod: <out>/{train,val,extra,dev}.jsonl.

train = train/data/train.jsonl + generated texts (train/raw/gen_*.jsonl); val = train/data/val.jsonl;
extra/dev = train/data/extra.jsonl split 90/10 per (domain, label) by text hash: dev never trains, it picks the
checkpoint by worst-domain false-positive rate (train/kaggle/train_kaggle.py).
Usage: python3 train/package.py /tmp/kgl-data
"""
import hashlib
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABEL = {"human": 0, "ai": 1, "ai_edited": 2, "humanized": 3}


def main(out):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (ROOT / "train/data/train.jsonl").read_text().splitlines()]
    for f in sorted((ROOT / "train/raw").glob("gen_*.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if "fable" in (r.get("model") or "") or "fable" in (r.get("src") or ""):
                continue
            rows.append({"text": r["text"], "label": LABEL[r["label"]], "src": f"gen/{r['label']}/{r.get('task')}/{r.get('style')}"})
    extra, dev = [], []
    for name in ("extra.jsonl", "extra2.jsonl", "short.jsonl", "expo.jsonl", "open.jsonl", "casual.jsonl", "ext.jsonl", "transcripts.jsonl"):   # domain-balanced, lit/academic/wiki, short-form, expository
        f = ROOT / "train/data" / name
        if not f.exists():
            continue
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if "fable" in r["src"]:              # owner: never use Fable for this (2026-11-11); its texts stay out
                continue
            (dev if int(hashlib.md5(r["text"][:500].encode()).hexdigest(), 16) % 10 == 0 else extra).append(r)
    for name, part in (("train", rows), ("extra", extra), ("dev", dev)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    (out / "val.jsonl").write_text((ROOT / "train/data/val.jsonl").read_text())
    print(f"train {len(rows)} {dict(Counter(r['label'] for r in rows))} | extra {len(extra)} {dict(Counter(r['label'] for r in extra))} "
          f"| dev {len(dev)} {dict(Counter(r['label'] for r in dev))}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/kgl-data")
