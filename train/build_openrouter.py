"""Trickle-generate AI text from open / non-big-3 model families through OpenRouter's FREE models (key in
~/.config/openrouter/api_key; a free-tier key has no credits, so nothing can be charged).

Why: on the held-out sets the combiner catches 98-100 % of Claude / GPT / Gemini text but far less from open-weights
families (Qwen/DeepSeek 29 %, n=17; older open models 58 %). The proxy has one open model (gpt-oss-120b); this adds
NVIDIA Nemotron, Google Gemma, Baidu ERNIE, Rednote dots, Liquid LFM, ... on the SAME prompts and topics as
train/build_expo.py (Wikipedia / PubMed) and train/build_short.py (emails, form answers, reviews, replies).

Rate limits are expected: an upstream 429 skips to the next model; the account's daily free-request cap sleeps until
the reset and carries on. Resumable; one request at a time. -> train/raw/open_ai.jsonl (merged by train/package.py
via build_expo-style finish: train/data/open.jsonl).
Usage: .venv/bin/python -u train/build_openrouter.py [--per-day N]
"""
import argparse
import datetime as dt
import json
import pathlib
import random
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_expo import MED_P, WIKI_P, clean  # noqa: E402
from build_short import FILL, GENRES, STYLES  # noqa: E402

KEY = (pathlib.Path.home() / ".config/openrouter/api_key").read_text().strip()
OUT = ROOT / "train/raw/open_ai.jsonl"
NV_OUT = ROOT / "train/raw/nvidia_ai.jsonl"          # build_nvidia.py's own file (different writer, merged by --finish)
FREE_RX = r":free$"
SKIP = {"thinkingmachines/inkling-2-small:free", "cohere/north-mini-code:free"}      # harness-only / code model
rng = random.Random(41)


def free_models():
    d = json.load(urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60))["data"]
    return [m["id"] for m in d if re.search(FREE_RX, m["id"]) and m["id"] not in SKIP]


def call(model, prompt):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 3000,
                       "reasoning": {"exclude": True}}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", body,
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                  "X-Title": "local-ai-detector"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=240))
    except urllib.error.HTTPError as e:
        d = json.loads(e.read() or b"{}")
    if d.get("error"):
        e = d["error"]
        return None, f"{e.get('code')} {(e.get('metadata') or {}).get('limit_source') or ''} {e.get('message', '')[:120]}"
    return (d["choices"][0]["message"].get("content") or "").strip(), None


def jobs():
    hum = [json.loads(l) for l in (ROOT / "train/raw/expo_human.jsonl").read_text().splitlines()]
    out = []
    for i, r in enumerate(hum):
        tpl = rng.choice(WIKI_P if r["dom"] == "wiki" else MED_P)
        out.append({"key": f"o-x{i}", "kind": "wikipedia" if r["dom"] == "wiki" else "pubmed",
                    "prompt": tpl.format(t=r["topic"], w=rng.choice([220, 260, 290, 320])), "lo": 120, "hi": 450})
    for i in range(1500):
        g = re.sub(r"\{(\w+)\}", lambda m: rng.choice(FILL[m.group(1)]), rng.choice(GENRES))
        sname, s = rng.choice(STYLES)
        out.append({"key": f"o-s{i}", "kind": f"short-{sname}",
                    "prompt": f"Write {g} (about {rng.choice([45, 60, 80, 100, 130])} words).{s}", "lo": 25, "hi": 220})
    rng.shuffle(out)
    return out


def until_reset():
    now = dt.datetime.now(dt.timezone.utc)
    nxt = (now + dt.timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
    return (nxt - now).total_seconds()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-day", type=int, default=1000, help="stop for the day after this many requests")
    ap.add_argument("--finish", action="store_true", help="only write train/data/open.jsonl (deduped vs test sets)")
    a = ap.parse_args()
    if a.finish:
        return finish()
    done = {json.loads(l)["key"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [j for j in jobs() if j["key"] not in done]
    models = free_models()
    print(f"open: {len(done)} done, {len(todo)} to go; free models: {models}", flush=True)
    cool = {}                      # model -> time it may be tried again
    sent_today, day = 0, dt.date.today()
    while todo:
        if dt.date.today() != day:
            sent_today, day, models = 0, dt.date.today(), free_models()
        if sent_today >= a.per_day:
            s = until_reset()
            print(f"per-day cap {a.per_day} reached; sleeping {s / 3600:.1f} h", flush=True)
            time.sleep(s)
            continue
        ready = [m for m in models if cool.get(m, 0) <= time.time()]
        if not ready:
            time.sleep(max(5, min(cool.values()) - time.time()))
            continue
        m, j = rng.choice(ready), todo[0]
        text, err = call(m, j["prompt"])
        sent_today += 1
        if err:
            low = err.lower()
            if "per-day" in low or "per day" in low or "free-models-per" in low or "daily" in low:
                s = until_reset()
                print(f"daily free cap hit ({err[:90]}); sleeping {s / 3600:.1f} h", flush=True)
                time.sleep(s)
            elif err.startswith("429"):
                cool[m] = time.time() + 120
            elif err.startswith(("403", "404", "400")):
                cool[m] = time.time() + 24 * 3600         # not usable this way; retry tomorrow
                print(f"skip {m}: {err[:90]}", flush=True)
            else:
                cool[m] = time.time() + 60
                print(f"{m}: {err[:90]}", flush=True)
            continue
        text = clean(text or "")
        if not (j["lo"] <= len(text.split()) <= j["hi"]):
            cool[m] = time.time() + 10
            continue                                      # keep the job; another model will try it
        with OUT.open("a") as f:
            f.write(json.dumps({"key": j["key"], "text": text, "label": 1, "src": f"open/ai/{j['kind']}/{m.removesuffix(':free')}"}) + "\n")
        todo.pop(0)
        n = len(done) + 1
        done.add(j["key"])
        if n % 10 == 0:
            print(f"{dt.datetime.now():%H:%M} open {n} written; today's requests {sent_today}", flush=True)
        time.sleep(3)                                     # well under the 20/min free limit


def finish():
    import build_expo
    test_h = set()
    for f in (ROOT / "test").glob("*.jsonl"):
        test_h |= {build_expo.h(json.loads(l)["text"]) for l in f.read_text().splitlines()}
    for f in list((ROOT / "eval/sol").rglob("texts.json")) + list((ROOT / "eval/data").rglob("*.json")):
        try:
            test_h |= {build_expo.h(r.get("text") or r.get("document")) for r in json.loads(f.read_text())
                       if isinstance(r, dict) and (r.get("text") or r.get("document"))}
        except Exception:  # noqa: BLE001
            pass
    rows = [json.loads(l) for f in (OUT, NV_OUT) if f.exists() for l in f.read_text().splitlines()]
    meta = re.compile(r"\bhere(?: is|'s) (?:a|an|the|your) (?:template|sample|draft|version)\b", re.I)   # chatbot framing = a false tell
    keep = [{"text": r["text"], "label": 1, "src": r["src"]} for r in rows
            if build_expo.h(r["text"]) not in test_h and not meta.search(r["text"])]
    (ROOT / "train/data/open.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    print("open.jsonl:", len(keep))


if __name__ == "__main__":
    main()
