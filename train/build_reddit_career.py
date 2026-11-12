"""Human job-search writing from Reddit career subreddits, 2015-2020 (pre-ChatGPT), for the casual-professional set:
cover-letter drafts people post for feedback, outreach / follow-up / thank-you notes, recruiter interactions,
interview answers, and people describing their job search in their own words.

Source: the Arctic Shift archive API (Pushshift lineage; no account), plain per-subreddit day listings sampled over
2015-2020 -- no server-side keyword search (it times out); genre tags are applied locally. One request at a time,
>= 4 s apart, backing off on 422 "slow down" / 429 / 5xx. Rows: 25-250 words, score >= 1, not deleted/removed, not
bots, not edited after 2020, no URLs / emails / phone numbers / code; markdown stripped.
-> train/raw/casual_human_reddit.jsonl  (merged into casual_human.jsonl before train/build_casual.py finish).
Usage: .venv/bin/python -u train/build_reddit_career.py [fetch|extract]   (fetch is resumable; raw pages cached)
"""
import datetime as dt
import json
import pathlib
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_more import h  # noqa: E402

RAW = ROOT / "train/raw/casual_src/reddit"
OUT = ROOT / "train/raw/casual_human_reddit.jsonl"
API = "https://arctic-shift.photon-reddit.com/api/{kind}/search?subreddit={sub}&after={a}&before={b}&limit=100"
SUBS = ["jobs", "careerguidance", "recruitinghell", "cscareerquestions", "resumes", "askHR", "GetEmployed",
        "careeradvice", "interviews", "jobsearchhacks"]
N_REQ = {"posts": 14, "comments": 8}      # day-windows per subreddit per kind
CUTOFF = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc).timestamp()
BOTS = {"AutoModerator", "[deleted]", "RemindMeBot", "sneakpeek_bot"}
GENRE = [("cover-letter", r"\bcover letter\b"),
         ("outreach-or-followup", r"\b(follow(ed|ing)?[- ]up|reached out|reach out|linkedin message|thank[- ]you (note|email)|cold email|networking)\b"),
         ("recruiter", r"\brecruiter"),
         ("interview-answer", r"\b(tell me about yourself|greatest weakness|why do you want|interview question)\b")]


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    rng = random.Random(61)
    days = [dt.date(2015, 1, 1) + dt.timedelta(days=i) for i in range((dt.date(2020, 12, 31) - dt.date(2015, 1, 1)).days)]
    plan = [(k, s, d) for s in SUBS for k, n in N_REQ.items() for d in rng.sample(days, n)]
    rng.shuffle(plan)
    print(f"{len(plan)} requests planned", flush=True)
    wait = 4.0
    for i, (kind, sub, d) in enumerate(plan):
        f = RAW / f"{kind}-{sub}-{d}.json"
        if f.exists():
            continue
        url = API.format(kind=kind, sub=sub, a=d, b=d + dt.timedelta(days=1))
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "local-ai-detector-research"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = json.load(r)["data"] or []
                f.write_text(json.dumps(data))
                wait = max(4.0, wait * 0.8)
                break
            except urllib.error.HTTPError as e:
                ra = float(e.headers.get("retry-after") or 0)
                wait = min(120.0, max(wait * 2, ra, 15))
                print(f"[{i + 1}/{len(plan)}] {kind} {sub} {d}: HTTP {e.code}; backing off {wait:.0f} s", flush=True)
                time.sleep(wait)
            except Exception as e:  # noqa: BLE001
                wait = min(120.0, wait * 2)
                print(f"[{i + 1}/{len(plan)}] {kind} {sub} {d}: {type(e).__name__}; backing off {wait:.0f} s", flush=True)
                time.sleep(wait)
        if i % 20 == 0:
            print(f"[{i + 1}/{len(plan)}] ok", flush=True)
        time.sleep(wait)


def strip_md(t):
    t = re.sub(r"^\s*>.*$", "", t, flags=re.M)                      # quoted text (often a pasted template)
    t = re.sub(r"^\s*\**(edit|update)\b.*$", "", t, flags=re.M | re.I)   # "Edit: thanks everyone" lines
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)                  # links -> text
    t = re.sub(r"[*_~`#]+", "", t)
    t = re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", t, flags=re.M)
    t = t.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<").replace("&#x200B;", "")
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def extract():
    seen = set()
    for f in list((ROOT / "train/data").glob("*.jsonl")) + [ROOT / "train/raw/casual_human.jsonl"]:
        if f.exists():
            seen |= {h(json.loads(l)["text"]) for l in f.read_text().splitlines()}
    rows = []
    for f in sorted(RAW.glob("*.json")):
        kind, sub = f.name.split("-")[:2]
        for x in json.loads(f.read_text()):
            t = x.get("selftext") if kind == "posts" else x.get("body")
            if not t or t in ("[deleted]", "[removed]") or x.get("author") in BOTS or (x.get("score") or 0) < 1:
                continue
            if x.get("created_utc", CUTOFF) >= CUTOFF or (x.get("edited") and float(x["edited"]) >= CUTOFF):
                continue
            if re.search(r"https?://|www\.|\S+@\S+\.\w+|\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}|```|    \S", t):
                continue
            t = strip_md(t)
            n = len(t.split())
            if not (25 <= n <= 250) or sum(c.isascii() for c in t) / len(t) < 0.97:
                continue
            g = next((name for name, rx in GENRE if re.search(rx, t, re.I)), "job-search-story")
            k = h(t)
            if k in seen:
                continue
            seen.add(k)
            rows.append({"text": t, "label": 0, "src": f"casual/human/reddit-{sub}/{g}", "genre": g,
                         "date": dt.datetime.fromtimestamp(x["created_utc"], dt.timezone.utc).strftime("%Y-%m-%d")})
    rng = random.Random(67)
    rng.shuffle(rows)
    rows.sort(key=lambda r: r["genre"] == "job-search-story")         # under-covered genres first
    keep, per_sub = [], Counter()
    for r in rows:
        sub = r["src"].split("/")[2]
        if len(keep) >= 1200 or per_sub[sub] >= 420:
            continue
        keep.append(r)
        per_sub[sub] += 1
    OUT.write_text("".join(json.dumps(r) + "\n" for r in keep))
    print("reddit career:", len(keep), dict(Counter(r["genre"] for r in keep)), dict(per_sub),
          min((r["date"] for r in keep), default="-"), max((r["date"] for r in keep), default="-"), flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if what == "fetch":
        fetch()
    extract()
