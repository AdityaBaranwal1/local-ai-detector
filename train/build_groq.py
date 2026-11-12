"""Open-model AI text through Groq's free tier (key in ~/.config/groq/api_key; free tier = no billing, it says
"upgrade to Dev Tier" when a limit is hit). Same prompts and topics as train/build_openrouter.py.

Models: qwen/qwen3.8-27b (the Qwen family is the combiner's worst: 29 % caught), openai/gpt-oss-20b (a second open
size besides the proxy's 120b), allam-2-7b (SDAIA, a family nothing else covers). One worker per model, paced to its
limits (Qwen: 1,000 output tokens/min, 1,000 requests/day) and backing off on 429 by the retry-after header.
Thinking is switched off (reasoning_effort) so the text is the answer, not a trace. Resumable.
-> appends to train/raw/open_ai.jsonl (keys g-<model>-<job>); build_openrouter.py --finish writes train/data/open.jsonl.
Usage: .venv/bin/python -u train/build_groq.py
"""
import json
import pathlib
import random
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_expo import clean  # noqa: E402
from build_openrouter import OUT, jobs  # noqa: E402

KEY = (pathlib.Path.home() / ".config/groq/api_key").read_text().strip()
MODELS = {  # model -> (extra request fields, output tokens per minute budget)
    "qwen/qwen3.8-27b": ({"reasoning_effort": "none"}, 1000),
    "openai/gpt-oss-20b": ({"reasoning_effort": "low"}, 1000),
    "allam-2-7b": ({}, 3000),
}
_lock = threading.Lock()


def call(model, extra, prompt, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, **extra}).encode()
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", body,
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                  "User-Agent": "local-ai-detector"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d, hdr = json.load(r), r.headers
    except urllib.error.HTTPError as e:
        try:
            d = json.loads(e.read() or b"{}")
        except ValueError:
            d = {}
        return None, 0, e.code, float(e.headers.get("retry-after") or 60), (d.get("error") or {}).get("message", "")[:140]
    remaining = hdr.get("x-ratelimit-remaining-requests")
    t = (d["choices"][0]["message"].get("content") or "").strip()
    return t, d.get("usage", {}).get("completion_tokens", 0), 200, 0 if remaining != "0" else 3600, ""


def worker(model, extra, otpm, done, all_jobs):
    tag = model.split("/")[-1]
    todo = [j for j in all_jobs if f"g-{tag}-{j['key']}" not in done]
    random.Random(tag).shuffle(todo)                   # each model works through the topics in its own order
    n = 0
    for j in todo:
        while True:
            text, toks, code, wait, msg = call(model, extra, j["prompt"], 700 if j["hi"] > 300 else 350)
            if code == 200:
                break
            if code == 429:
                time.sleep(max(wait, 5))
                continue
            print(f"{tag}: HTTP {code} {msg}; pausing 10 min", flush=True)
            time.sleep(600)
        text = clean(text or "")
        if j["lo"] <= len(text.split()) <= j["hi"]:
            with _lock, OUT.open("a") as f:
                f.write(json.dumps({"key": f"g-{tag}-{j['key']}", "text": text, "label": 1,
                                    "src": f"open/ai/{j['kind']}/groq-{tag}"}) + "\n")
            n += 1
            if n % 25 == 0:
                print(f"{time.strftime('%H:%M')} {tag}: {n} written", flush=True)
        time.sleep(max(2.0, toks / otpm * 60) + (wait or 0))   # stay under the per-minute output budget


def main():
    done = {json.loads(l)["key"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    all_jobs = jobs()                                  # built once: the shared rng isn't thread-safe
    ts = [threading.Thread(target=worker, args=(m, x, o, done, list(all_jobs)), daemon=True) for m, (x, o) in MODELS.items()]
    for t in ts:
        t.start()
    print(f"groq: {len(ts)} workers ({', '.join(MODELS)})", flush=True)
    for t in ts:
        t.join()


if __name__ == "__main__":
    main()
