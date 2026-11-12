"""The owner's own Claude.ai chat exports as AI training text (owner's OK 2026-11-12): real replies from the Claude
versions used across 2026, no tokens spent. Only ASSISTANT text is used, never the owner's messages.

Privacy: chunks are dropped (not redacted) if they contain an email, phone number, URL, street address, long digit run,
a quoted line, or any identifying string from the git-ignored lists `train/.scrub.local` and `.public-guard.local`.
Code blocks, tables and markdown are removed. Output stays in the git-ignored `train/data/`; it leaves the box only
inside a training package for a Runpod pod that is deleted after the run.
Input: CHAT_EXPORTS (glob, default: the knowledge-base raw exports `*/export/conversations.json`).
-> train/data/transcripts.jsonl   Usage: .venv/bin/python train/build_transcripts.py
"""
import glob
import json
import os
import pathlib
import random
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_expo import clean  # noqa: E402
from build_more import h  # noqa: E402

EXPORTS = os.environ.get("CHAT_EXPORTS", str(pathlib.Path.home() / "docker/obsidian-webdav/data/Main Notes/Knownledge Base/raw/*/export/conversations.json"))
CAP, PER_MSG = 3000, 2
PII = re.compile(r"\S+@\S+\.\w+|https?://|www\.|\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b|\d{6,}|"
                 r"\b\d{1,5} \w+ (st|street|ave|avenue|rd|road|blvd|lane|ln|dr|drive)\b", re.I)


def scrub_terms():
    terms = []
    for f in (ROOT / "train/.scrub.local", ROOT / ".public-guard.local"):
        if f.exists():
            terms += [l.strip().lower() for l in f.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return [t for t in terms if len(t) >= 3]


def chunks(text):
    text = re.sub(r"```.*?```", "\n\n", text, flags=re.S)             # code blocks
    if re.search(r"^\s*>", text, re.M):                                # quotes the user / a document
        text = re.sub(r"^\s*>.*$", "", text, flags=re.M)
    text = re.sub(r"^\s*\|.*\|\s*$", "", text, flags=re.M)              # tables
    text = clean(text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 8]
    cur, out = [], []
    for p in paras:
        cur.append(p)
        if sum(len(x.split()) for x in cur) >= 60:
            out.append("\n\n".join(cur))
            cur = []
    if cur and sum(len(x.split()) for x in cur) >= 40:
        out.append("\n\n".join(cur))
    return [c for c in out if 40 <= len(c.split()) <= 320]


def main():
    terms = scrub_terms()
    rng = random.Random(83)
    rows, dropped = [], Counter()
    for f in glob.glob(EXPORTS):
        for conv in json.loads(pathlib.Path(f).read_text()):
            month = (conv.get("created_at") or "")[:7] or "unknown"
            for m in conv.get("chat_messages", []):
                if m.get("sender") != "assistant":
                    continue
                txt = m.get("text") or " ".join(b.get("text", "") for b in m.get("content") or [] if b.get("type") == "text")
                cs = chunks(txt or "")
                rng.shuffle(cs)
                for c in cs[:PER_MSG]:
                    low = c.lower()
                    if PII.search(c):
                        dropped["pii pattern"] += 1
                    elif any(t in low for t in terms):                  # substring: catches handles like name+digits
                        dropped["identifying name"] += 1
                    elif sum(ch in "{}();=<>$" for ch in c) > len(c) * 0.02:
                        dropped["code-like"] += 1
                    else:
                        rows.append({"text": c, "label": 1, "src": f"transcript/ai/claude-ai/{month}"})
    test_h = set()
    for f in (ROOT / "test").glob("*.jsonl"):
        test_h |= {h(json.loads(l)["text"]) for l in f.read_text().splitlines()}
    seen, keep = set(), []
    rng.shuffle(rows)
    for r in rows:
        k = h(r["text"])
        if k not in test_h and k not in seen and len(keep) < CAP:
            seen.add(k)
            keep.append(r)
    (ROOT / "train/data/transcripts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    print("transcripts.jsonl:", len(keep), "of", len(rows), "candidates; dropped:", dict(dropped),
          "| by month:", dict(sorted(Counter(r["src"].split("/")[-1] for r in keep).items())))


if __name__ == "__main__":
    main()
