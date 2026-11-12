"""Open-model AI text through NVIDIA's API catalog (build.nvidia.com), FREE endpoints only (key in
~/.config/nvidia/api_key, read in code only; nothing is printed or logged). Same prompts and topics as
train/build_openrouter.py.

Why: on the held-out sets the combiner catches 98-100 % of Claude / GPT / Gemini text but far less from open-weights
families (Qwen/DeepSeek 29 %, n=17; older open models 58 %). NVIDIA's catalog adds open models nothing else here
covers. This builder is trickle-paced and resumable, like build_groq.py.

Models: only ones labelled "Free Endpoint" on build.nvidia.com (page or catalog screenshot) that answered a probe with
prose on 2026-11-11:
  minimaxai/minimax-m3.1-flash, nvidia/nemotron-3.5-lightning-30b-a3b, nvidia/nemotron-3.5-bolt-e4b,
  google/diffusiongemma-26b-a4b-it
Not used: reasoning-only (mistral-large-4-1t-a52b, step-5-preview: empty content), timeouts at 90 s (deepseek-v4.1-flash,
glm-5.3-flash, gpt-oss-20b), 404 on this account (kimi-k3.1-pro, gemma-4-31b-it, muse-glimmer-30b, ...), code models
(poolside/laguna-xs-2.1 is labelled Coding), and unconfirmed-free models (no "Free Endpoint" label).
Thinking is switched off (chat_template_kwargs) and any <think> block is stripped; texts with a trace, code fences or
LaTeX/backslash markup are dropped (Bolt wrote `$\text{CCl}_4$` into PubMed abstracts, a shortcut a detector would learn).

Limits: NVIDIA's account limit is ~40 requests/min. All workers share one pacing slot of GAP seconds between request
starts (~37/min). 429 -> wait Retry-After (default 60 s, doubling to 10 min). 401/403/404 -> that model isn't usable
here, its worker stops. Any other error -> 10-minute pause and retry the same job; 6 errors in a row stops the worker.
--per-day (default 1000) caps requests per UTC day across all workers; the cap is in-process, so a restart resets it.
Only the words in each job's [lo, hi] window are kept; rejected jobs are retried on the next run.

Provenance: owner's permission record in data/provenance/nvidia.md (personal, non-production use). Prompts are generic
topics and genres, with no personal data. Outputs stay local (train/raw/ is git-ignored, never published).
-> appends to train/raw/nvidia_ai.jsonl (keys n-<model>-<job>). build_openrouter.py --finish merges it into
train/data/open.jsonl after the test-set dedupe.
Usage: .venv/bin/python -u train/build_nvidia.py [--per-day N]
"""
import argparse
import datetime as dt
import json
import pathlib
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_expo import clean  # noqa: E402
from build_openrouter import NV_OUT as OUT, jobs, until_reset  # noqa: E402

KEY = (pathlib.Path.home() / ".config/nvidia/api_key").read_text().strip()
URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODELS = ["minimaxai/minimax-m3.1-flash", "nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-3.5-bolt-e4b",
          "google/diffusiongemma-26b-a4b-it"]
GAP = 1.6                     # seconds between request starts across ALL workers (~37/min, under the ~40/min limit)
_gate = threading.Lock()      # guards the pacing slot and the daily counter
_lock = threading.Lock()      # guards appends to the output file
_next_slot = [0.0]
_sent = {"day": None, "n": 0}


def reserve(per_day):
    """Block until this worker may send: today's budget has room and the shared pacing slot has come up."""
    while True:
        with _gate:
            today = dt.datetime.now(dt.timezone.utc).date()
            if _sent["day"] != today:
                _sent.update(day=today, n=0)
            if _sent["n"] < per_day:
                start = max(time.time(), _next_slot[0])
                _next_slot[0] = start + GAP
                _sent["n"] += 1
                break
        s = until_reset()
        print(f"per-day cap {per_day} reached; sleeping {s / 3600:.1f} h", flush=True)
        time.sleep(s)
    time.sleep(max(0.0, start - time.time()))


def call(model, prompt, max_tokens):
    """-> (http_code, text, retry_after_seconds, message). Code 0 = network error."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0.8, "stream": False,
                       "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                             "User-Agent": "local-ai-detector"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            content = json.load(r)["choices"][0]["message"].get("content") or ""
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read() or b"{}")
        except ValueError:
            err = {}
        msg = str(err.get("detail") or err.get("title") or "")[:140]
        return e.code, None, float(e.headers.get("retry-after") or 0), msg
    except (OSError, ValueError) as e:
        return 0, None, 0, type(e).__name__
    except (KeyError, IndexError, TypeError):
        return 0, None, 0, "unexpected response shape"
    return 200, content, 0, ""


def worker(model, done, all_jobs, per_day):
    tag = model.split("/")[-1]
    todo = [j for j in all_jobs if f"n-{tag}-{j['key']}" not in done]
    random.Random(tag).shuffle(todo)                   # each model works through the topics in its own order
    wrote = fails = rejects = 0
    delay = 60
    for j in todo:
        key = f"n-{tag}-{j['key']}"
        while True:
            reserve(per_day)
            code, text, retry, msg = call(model, j["prompt"], 700 if j["hi"] > 300 else 350)
            if code == 200:
                fails, delay = 0, 60
                break
            if code in (401, 403, 404):
                print(f"{tag}: HTTP {code} {msg}; not usable on this account, worker stops", flush=True)
                return
            if code == 429:
                time.sleep(retry or delay)
                delay = min(delay * 2, 600)
                continue
            fails += 1
            print(f"{tag}: HTTP {code} {msg}; pausing 10 min", flush=True)
            if fails >= 6:
                print(f"{tag}: 6 errors in a row; worker stops", flush=True)
                return
            time.sleep(600)
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
        text = clean(text)
        if ("<think" in text or "```" in text or "\\" in text or re.search(r"\$[\\{^_]", text)   # traces, fences, LaTeX
                or not (j["lo"] <= len(text.split()) <= j["hi"])):
            rejects += 1
            if rejects >= 30:
                print(f"{tag}: 30 unusable outputs in a row; worker stops", flush=True)
                return
            continue
        rejects = 0
        with _lock, OUT.open("a") as f:
            f.write(json.dumps({"key": key, "text": text, "label": 1,
                                "src": f"open/ai/{j['kind']}/nvidia-{tag}"}) + "\n")
        wrote += 1
        if wrote % 25 == 0:
            print(f"{time.strftime('%H:%M')} {tag}: {wrote} written this run", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-day", type=int, default=1000, help="stop for the UTC day after this many requests (all models)")
    a = ap.parse_args()
    done = {json.loads(l)["key"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    all_jobs = jobs()                                  # built once: the shared rng isn't thread-safe
    ts = [threading.Thread(target=worker, args=(m, done, list(all_jobs), a.per_day), daemon=True) for m in MODELS]
    for t in ts:
        t.start()
    print(f"nvidia: {len(ts)} workers ({', '.join(MODELS)}); {len(done)} already written", flush=True)
    for t in ts:
        t.join()


if __name__ == "__main__":
    main()
