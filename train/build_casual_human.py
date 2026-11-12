"""Casual professional HUMAN text (label 0, pre-2021): the human half of the matched set for natural, casual
AI-written job-search messages (networking, referral asks, self-intros, work chat, support replies, career talk).

Every row is human-written and dated per row. Nothing from the owner's own mail. Sources (all public):
  hn-wwtbh        Hacker News "Ask HN: Who wants to be hired?" comments, 2014-2020: job seekers' own intros and
                  outreach. Algolia HN Search API (unauthenticated, no account).
  se-freelancing  Stack Exchange dump 2024-06-30 (archive.org item stackexchange_20240630_json): freelancing, pm,
  se-pm           workplace. Questions and answers whose CreationDate AND LastEditDate are before the cutoff.
  se-workplace    (workplace overlaps build_short.py, which took best-voted answers from the flax mirror; only
                  exact duplicates are removed.)
  twitter-cs      Kaggle thoughtvector/customer-support-on-twitter (2017): AGENT replies only (inbound == False).
                  @mentions removed; a template guard caps near-identical openers at 2 per 8 words.
  clinton-emails  Kaggle kaggle/hillary-clinton-emails: US State Department FOIA releases, sent 2009-2015
                  (work email). Forwards and quoted originals cut; redacted, pasted-news and classification
                  rows dropped.
  ubuntu-irc      HF Tuxifan/UbuntuIRC (#ubuntu channel logs, date taken from the file path). Each run of one
                  speaker's consecutive lines is one row, nick removed, bots dropped.

Filters on every row: 25-250 words; no email or phone number (row dropped, not masked); URLs removed; quoted
reply lines ('>') removed; HTML cleaned; rows with <pre>/<code> or code-like tokens above 3% dropped; English
stop-word share at least 12%. Rows identical (norm()+sha1, as in build_more.h) to any existing train/data or
train/raw jsonl text or any eval/data text are dropped.

Steps (resumable; re-run any time):
  fetch    HN stories and their comments -> train/raw/casual_src/hn/<story_id>.json (skips existing files);
           sampled IRC days (the 3rd and 18th of each month, 2008 to the cutoff) ->
           train/raw/casual_src/ubuntu_irc/YYYY/MM/DD/#ubuntu.txt (skips existing files; 404 days are skipped)
  extract  reads the local data and writes train/raw/casual_human.jsonl (deterministic, seed 17; re-run freely)
  all      fetch, then extract (default)

One-time downloads (not run here; they need the Kaggle CLI and ~/.kaggle credentials, so run them yourself):
  kaggle datasets download -d thoughtvector/customer-support-on-twitter -p train/raw/casual_src/twitter_cs --unzip
  kaggle datasets download -d kaggle/hillary-clinton-emails -p train/raw/casual_src/clinton_emails --unzip
  curl -L -o train/raw/casual_src/stackexchange/<site>.stackexchange.com.7z \
       https://archive.org/download/stackexchange_20240630_json/stackexchange_20240630_json/<site>.stackexchange.com.7z
  (extract mode pulls Posts.json out of each SE archive with the 7z binary when it is missing)

Usage: .venv/bin/python train/build_casual_human.py [fetch|extract|all] [--cutoff 2021-01-01]
Cutoff: default 2021-01-01 (stricter than the 2022-11-30 hard limit). Output: train/raw/casual_human.jsonl.
"""
import argparse
import collections
import csv
import datetime as dt
import glob
import hashlib
import html
import json
import pathlib
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "train" / "raw" / "casual_src"
OUT = ROOT / "train" / "raw" / "casual_human.jsonl"
UA = "casual-human-research/1.0"
SEED = 17
MIN_W, MAX_W = 25, 250
CUTOFF = "2021-01-01"

CAP = {  # per-dataset upper bounds (each well under ~35% of the total)
    "hn-wwtbh": 900, "se-freelancing": 400, "se-pm": 300, "se-workplace": 300,
    "twitter-cs": 800, "clinton-emails": 500, "ubuntu-irc": 400,
}
ORDER = list(CAP)
SE_SITES = {"se-freelancing": "freelancing", "se-pm": "pm", "se-workplace": "workplace"}
IRC_BOTS = {"ubottu", "ubottu2", "ubuntubot", "ubot"}

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?<!\d)(?:\+\d[\d\s.-]{8,}\d|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4})(?!\d)")
URL = re.compile(r"(?:[a-z][a-z0-9+.-]*://|www\.|pic\.twitter\.com/)\S+|"
                 r"\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|io|edu|gov|info|dev|app|me|co\.uk)/\S*", re.I)
# contact fields and obfuscated addresses (HN intros): the whole row is dropped, not masked
CONTACT = re.compile(
    r"(?i)\b(?:e-?mail|contact|linkedin|github|gitlab|r[ée]sum[ée]|cv|phone|tel|mobile|whatsapp|telegram|twitter|website|"
    r"portfolio)[ \t]*:[ \t]*\S|\[(?:at|dot)\]|\s(?:at|@)\s*(?:gmail|hotmail|yahoo|outlook|icloud|protonmail|aol)\b|"
    r"@\s*(?:gmail|hotmail|yahoo|outlook|icloud|protonmail|aol)\b|\b(?:linkedin|github|twitter)\.com/|"
    r"\s@\s*\w+\s*\.\s*(?:com|net|org)\b")
IRC_ADDRESS = re.compile(r"^[\w\[\]{}^`|\\-]{1,32}[:,] ")  # leading "nick: " addressing in chat lines
TAG = re.compile(r"<[^>]+>")
MENTION = re.compile(r"@\w+")
CODE_W = re.compile(r"[{}\[\];=$\\|]")
REDACT = re.compile(r"\(b\)\s*\(\d\)|\bB[1-9]\b|\bb\(\d\)")
NEWS = re.compile(r"\b(Associated Press|Reuters|AFP|Bloomberg|New York Times|Washington Post)\b")
FWD = re.compile(r"^(?:-{2,}\s*Original Message|-{2,}\s*Forwarded|Begin forwarded message|From:\s|Sent:\s|"
                 r"On .{5,80} wrote:)", re.M)
HDR = re.compile(r"^(UNCLASSIFIED|U\.S\. Department of State|Case No\.|Doc No\.|RELEASE IN (FULL|PART)|"
                 r"Sent from my|B6\b)")
WWTBH = re.compile(r"who wants to be hired", re.I)
WWTBH_EXCL = re.compile(r"has anyone|ever been hired", re.I)
STOP = set("a an and are as at be but by for from have i if in is it my not of on or so that the this to was "
           "we with you your they".split())
IRC_LINE = re.compile(r"^\[\d\d:\d\d\] <([^>]+)> ?(.*)$")
IRC_BASE = "https://huggingface.co/datasets/Tuxifan/UbuntuIRC/resolve/main/"
ALGOLIA = "https://hn.algolia.com/api/v1/"

REJ = collections.Counter()


def norm(t):  # same as train/build_more.py norm()
    return re.sub(r"\W+", " ", t.lower()).strip()


def h(t):  # same as train/build_more.py h()
    return hashlib.sha1(norm(t)[:2000].encode()).hexdigest()


def iso_from_ts(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).date().isoformat()


def clean_html(s):
    s = html.unescape(s)
    s = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</pre>|<p>|<li>", "\n", s)
    s = TAG.sub("", s)
    return html.unescape(s)


def normalize(s):
    s = URL.sub("", s)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln and not ln.startswith(">")]
    return "\n".join(lines)


def finish(ds, text, date, genre, out):
    """Clean, check and append one row. Every rejection is counted in REJ."""
    t = normalize(text)
    if EMAIL.search(t) or PHONE.search(t):
        REJ[ds + ":pii"] += 1
        return
    if CONTACT.search(t):
        REJ[ds + ":contact"] += 1
        return
    t = re.sub(r"(?<!\w)@\w+", "", t)  # other users' handles (kept out of the text, not just the row)
    words = t.split()
    if not MIN_W <= len(words) <= MAX_W:
        REJ[ds + ":length"] += 1
        return
    if sum(1 for w in words if CODE_W.search(w) or w.startswith("--")) / len(words) > 0.03:
        REJ[ds + ":code"] += 1
        return
    stop = sum(1 for w in words if w.lower().strip(".,;:!?'\"()*-") in STOP)
    if stop / len(words) < 0.12:
        REJ[ds + ":not-english"] += 1
        return
    out.append({"text": t, "label": 0, "src": f"casual/human/{ds}/{genre}", "genre": genre, "date": date})


# ---------------------------------------------------------------- fetch

def algolia(path, **params):
    url = ALGOLIA + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"  retry {path} ({attempt}): {e}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    raise SystemExit(f"Algolia failed: {path}")


def fetch_hn():
    d = SRC / "hn"
    d.mkdir(parents=True, exist_ok=True)
    cutoff_ts = int(dt.datetime.fromisoformat(CUTOFF).replace(tzinfo=dt.timezone.utc).timestamp())
    stories, page = [], 0
    while True:
        r = algolia("search_by_date", query="who wants to be hired", tags="story", hitsPerPage=1000,
                    page=page, numericFilters=f"created_at_i<{cutoff_ts}")
        stories += r["hits"]
        page += 1
        if page >= r["nbPages"]:
            break
        time.sleep(0.5)
    wanted = [s for s in stories
              if WWTBH.search(s.get("title") or "") and not WWTBH_EXCL.search(s.get("title") or "")]
    print(f"HN: {len(wanted)} 'Who wants to be hired' stories before {CUTOFF} (of {len(stories)} hits)")
    for s in wanted:
        f = d / f"{s['objectID']}.json"
        if f.exists():
            continue
        comments, page = [], 0
        while True:
            r = algolia("search", tags=f"comment,story_{s['objectID']}", hitsPerPage=1000, page=page)
            comments += [{"created_at_i": c["created_at_i"], "comment_text": c.get("comment_text")}
                         for c in r["hits"]]
            page += 1
            if page >= r["nbPages"]:
                break
            time.sleep(0.5)
        f.write_text(json.dumps({"story_id": s["objectID"], "title": s.get("title"),
                                 "created_at_i": s["created_at_i"], "comments": comments}), encoding="utf-8")
        print(f"  story {s['objectID']}: {len(comments)} comments")
        time.sleep(0.5)


def fetch_irc():
    got = skipped = missing = 0
    for y in range(2008, 2026):
        for m in range(1, 13):
            for day in (3, 18):
                rel = f"{y:04d}/{m:02d}/{day:02d}"
                if rel.replace("/", "-") >= CUTOFF:
                    continue
                f = SRC / "ubuntu_irc" / rel / "#ubuntu.txt"
                if f.exists():
                    skipped += 1
                    continue
                req = urllib.request.Request(IRC_BASE + rel + "/%23ubuntu.txt", headers={"User-Agent": UA})
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        missing += 1
                        continue
                    raise
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(data)
                got += 1
                time.sleep(0.4)
    print(f"IRC: downloaded {got}, already present {skipped}, missing on server {missing}")


# ---------------------------------------------------------------- extract

def ensure_se():
    for ds, site in SE_SITES.items():
        out = SRC / "stackexchange" / site / "Posts.json"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["7z", "x", "-y", f"-o{out.parent}",
                        str(SRC / "stackexchange" / f"{site}.stackexchange.com.7z"), "Posts.json"],
                       check=True, stdout=subprocess.DEVNULL)


def hn_rows(out):
    ds = "hn-wwtbh"
    for f in sorted((SRC / "hn").glob("*.json")):
        story = json.loads(f.read_text(encoding="utf-8"))
        for c in story["comments"]:
            raw = c.get("comment_text")
            if not raw:
                REJ[ds + ":deleted"] += 1
                continue
            if "<pre" in html.unescape(raw) or "<code" in html.unescape(raw):
                REJ[ds + ":code"] += 1
                continue
            date = iso_from_ts(c["created_at_i"])
            if date >= CUTOFF:
                REJ[ds + ":after-cutoff"] += 1
                continue
            finish(ds, clean_html(raw), date, "job-seeker-post", out)


def se_rows(ds, site, out):
    posts = json.loads((SRC / "stackexchange" / site / "Posts.json").read_text(encoding="utf-8"))
    for p in posts:
        if p.get("PostTypeId") not in (1, 2):  # questions and answers only
            continue
        created = p["CreationDate"][:10]
        edited = (p.get("LastEditDate") or p["CreationDate"])[:10]
        if created >= CUTOFF or edited >= CUTOFF:
            REJ[ds + ":after-cutoff"] += 1
            continue
        body = p.get("Body") or ""
        if "<pre" in html.unescape(body) or "<code" in html.unescape(body):
            REJ[ds + ":code"] += 1
            continue
        finish(ds, clean_html(body), created, "casual-career-talk", out)


def twitter_rows(out):
    ds = "twitter-cs"
    seen = collections.Counter()
    with (SRC / "twitter_cs" / "twcs" / "twcs.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["inbound"] != "False":  # agent replies only
                continue
            date = dt.datetime.strptime(r["created_at"], "%a %b %d %H:%M:%S %z %Y").date().isoformat()
            if date >= CUTOFF:
                REJ[ds + ":after-cutoff"] += 1
                continue
            text = MENTION.sub("", r["text"]).strip()
            key = " ".join(norm(text).split()[:8])
            if seen[key] >= 2:
                REJ[ds + ":template"] += 1
                continue
            seen[key] += 1
            finish(ds, text, date, "customer-support-reply", out)


def clinton_rows(out):
    ds = "clinton-emails"
    csv.field_size_limit(10 ** 8)
    with (SRC / "clinton_emails" / "Emails.csv").open(newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            body = (r.get("ExtractedBodyText") or "").strip()
            date = (r.get("MetadataDateSent") or "")[:10]
            if not body:
                REJ[ds + ":empty"] += 1
                continue
            if not date or date >= CUTOFF:
                REJ[ds + ":after-cutoff"] += 1
                continue
            m = FWD.search(body)
            if m:  # keep only the author's own text above a forward or quoted original
                body = body[:m.start()]
            if REDACT.search(body):
                REJ[ds + ":redacted"] += 1
                continue
            if NEWS.search(body):
                REJ[ds + ":news-clip"] += 1
                continue
            body = "\n".join(ln for ln in body.splitlines() if not HDR.match(ln))
            finish(ds, body, date, "work-email", out)


def irc_rows(out):
    ds = "ubuntu-irc"
    for f in sorted((SRC / "ubuntu_irc").glob("*/*/*/#ubuntu.txt")):
        date = f"{f.parts[-4]}-{f.parts[-3]}-{f.parts[-2]}"
        if date >= CUTOFF:
            REJ[ds + ":after-cutoff"] += 1
            continue
        runs, cur_nick, cur = [], None, []
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            mm = IRC_LINE.match(line)
            if not mm:
                continue
            nick, msg = mm.group(1), mm.group(2)
            if nick != cur_nick:
                if cur:
                    runs.append((cur_nick, cur))
                cur_nick, cur = nick, []
            cur.append(msg)
        if cur:
            runs.append((cur_nick, cur))
        for nick, msgs in runs:
            if nick.lower() in IRC_BOTS:
                REJ[ds + ":bot"] += 1
                continue
            msgs = [IRC_ADDRESS.sub("", m) for m in msgs]
            finish(ds, "\n".join(msgs), date, "work-chat", out)


def existing_hashes():
    hs = set()
    files = glob.glob(str(ROOT / "train" / "data" / "*.jsonl")) + \
        glob.glob(str(ROOT / "train" / "raw" / "*.jsonl"))
    for f in files:
        if pathlib.Path(f).resolve() == OUT.resolve():
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    t = json.loads(line).get("text")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if isinstance(t, str):
                    hs.add(h(t))
    for f in glob.glob(str(ROOT / "eval" / "data" / "**" / "*.json"), recursive=True) + \
            glob.glob(str(ROOT / "eval" / "data" / "**" / "*.jsonl"), recursive=True):
        try:
            text_blob = open(f, encoding="utf-8").read()
            obj = json.loads(text_blob) if f.endswith(".json") else [json.loads(x) for x in text_blob.splitlines()]
        except (OSError, json.JSONDecodeError):
            continue
        stack = [obj]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                stack.extend(x.values())
            elif isinstance(x, list):
                stack.extend(x)
            elif isinstance(x, str) and len(x) >= 40:
                hs.add(h(x))
    return hs


def extract():
    ensure_se()
    loaders = {
        "hn-wwtbh": hn_rows,
        "twitter-cs": twitter_rows,
        "clinton-emails": clinton_rows,
        "ubuntu-irc": irc_rows,
    }
    for ds, site in SE_SITES.items():
        loaders[ds] = (lambda d, s: lambda out: se_rows(d, s, out))(ds, site)

    print("loading existing text hashes (train/data, train/raw, eval/data) ...")
    seen = existing_hashes()
    print(f"  {len(seen)} existing hashes")

    rng = random.Random(SEED)
    final, stats = [], {}
    for ds in ORDER:
        raw_rows = []
        loaders[ds](raw_rows)
        fresh = []
        for r in raw_rows:
            k = h(r["text"])
            if k in seen:
                REJ[ds + ":duplicate"] += 1
                continue
            seen.add(k)
            fresh.append(r)
        stats[ds] = (len(raw_rows), len(fresh))
        if len(fresh) > CAP[ds]:
            fresh = rng.sample(fresh, CAP[ds])
        final += sorted(fresh, key=lambda r: (r["date"], r["text"]))
        print(f"  {ds}: passed filters {len(raw_rows)}, new after dedupe {len(fresh)}, kept {min(len(fresh), CAP[ds])}")

    rng.shuffle(final)
    tmp = OUT.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in final:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(OUT)

    total = len(final)
    by_ds = collections.Counter(r["src"].split("/")[2] for r in final)
    by_genre = collections.Counter(r["genre"] for r in final)
    print(f"\nWROTE {total} rows -> {OUT}")
    print("by dataset:", {k: f"{v} ({100 * v / total:.0f}%)" for k, v in by_ds.most_common()})
    print("by genre:", dict(by_genre.most_common()))
    print("top rejections:", REJ.most_common(20))


def main():
    global CUTOFF
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="all", choices=["fetch", "extract", "all"])
    ap.add_argument("--cutoff", default="2021-01-01")
    a = ap.parse_args()
    CUTOFF = a.cutoff
    if a.mode in ("fetch", "all"):
        fetch_hn()
        fetch_irc()
    if a.mode in ("extract", "all"):
        extract()


if __name__ == "__main__":
    main()
