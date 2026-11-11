"""Verify: one PASS/FAIL verdict per text, for agents to check their writing before it goes out under the owner's
name. Modelled on resume-verify (~/docker/resume-parsers): strict, versioned, verdicts stored by hash, never text.

PASS only if every step ran and
  - the model's AI score is below max_ai (default 0.30: text sent as the owner's should read clearly human, so
    the line sits below the 0.50 "reads as AI" verdict), and
  - the tells linter found no strong tell.
Anything that did not run (model failed, text empty/too long) is FAIL with the reason. There is no partial pass.

Stored per verdict (data/verdicts.sqlite, dir 0700): key = sha256(text) + every version; value = the verdict with
sentence/tell offsets and scores, no text. Responses add the quoted sentences back from the text the caller sent.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sqlite3
import threading
import time

from . import tells
from .text import sentence_spans


def pct(x, other=None):
    """'99.4%' when both numbers are near the top (else 100% -> 100% says nothing), else '87%'."""
    near = x >= 0.985 or (other is not None and other >= 0.985)
    return f"{x * 100:.1f}%" if near else f"{round(x * 100)}%"

SERVICE_VERSION = 1
DEFAULT_MAX_AI = 0.30
DATA = pathlib.Path(os.environ.get("VERIFY_DATA", pathlib.Path(__file__).resolve().parent.parent / "data"))
DB = DATA / "verdicts.sqlite"
_db_lock = threading.Lock()

USAGE = """AI-text verify service: one PASS/FAIL verdict per text, for agents to check their writing before it goes out
under the owner's name (emails, application answers, messages). ~/All code/ai-detector/README.md "Verify API".

  POST /verify              JSON {"text": "...", "max_ai": 0.30 (optional), "fresh": false (optional)}
                            -> verdict JSON. Also accepts the raw text as the body (Content-Type: text/plain).
  GET  /verdict/<sha256>    the stored verdict for sha256(text, UTF-8) under the current versions (offsets only), else 404
  GET  /healthz             200 when the model is loadable; "steps" = each check with its version and state
  GET  /verify              this text
  CLI  ai-check [file|-]    the same, from a shell on the box: exit 0 PASS, 1 FAIL, 2 could not check

PASS = the model's AI score < max_ai (default 0.30) AND no strong "tell" (stock LLM openers/closers, overused LLM
vocabulary, leftover placeholders or markdown). Anything else, including an error or no answer, is a no.
"problems" lists what to change: the sentences that drive the score (with the score without each one) and every
tell with a suggested fix. Rewrite those, then verify again.
Privacy: the text is held only for the request. The store keeps hash -> verdict (scores, offsets), never the text.
"""


def _conn():
    DATA.mkdir(parents=True, exist_ok=True)
    os.chmod(DATA, 0o700)
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, sha256 TEXT, verdict TEXT, result TEXT, checked_at REAL)")
    c.execute("CREATE INDEX IF NOT EXISTS v_sha ON verdicts(sha256)")
    return c


def versions(engine_version: str) -> dict:
    return {"service": SERVICE_VERSION, "engine": engine_version, "tells": tells.TELLS_VERSION}


def key_of(sha, max_ai, vers):
    return hashlib.sha256(json.dumps([sha, round(max_ai, 3), vers], sort_keys=True).encode()).hexdigest()


def stored(sha, max_ai, vers):
    with _db_lock, _conn() as c:
        row = c.execute("SELECT result FROM verdicts WHERE key = ?", (key_of(sha, max_ai, vers),)).fetchone()
    return json.loads(row[0]) if row else None


def store(sha, max_ai, vers, result):
    with _db_lock, _conn() as c:
        c.execute("INSERT OR REPLACE INTO verdicts VALUES (?, ?, ?, ?, ?)",
                  (key_of(sha, max_ai, vers), sha, result["verdict"], json.dumps(result), time.time()))


def today_counts():
    start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    try:
        with _db_lock, _conn() as c:
            rows = c.execute("SELECT verdict, COUNT(*) FROM verdicts WHERE checked_at >= ? GROUP BY verdict", (start,)).fetchall()
            total = c.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]
    except sqlite3.Error:
        return {"checks": None, "pass": None, "fail": None, "stored": None}
    d = dict(rows)
    return {"checks": sum(d.values()), "pass": d.get("PASS", 0), "fail": d.get("FAIL", 0), "stored": total}


def build_verdict(text, analysis, hits, max_ai, vers, sha, seconds, device):
    """Pure: analysis (score.analyse output) + tell hits -> verdict dict (offsets, no quotes)."""
    p = analysis["ai_score"]
    problems = []
    if p >= max_ai:
        problems.append({"check": "model", "msg": f"reads {round(p * 100)}% AI (PASS needs under {round(max_ai * 100)}%)"})
    drivers = analysis.get("mode") == "drivers"
    for s in analysis["sentences"]:
        if s["flag"] == "ok":
            continue
        item = {"check": "model", "start": s["start"], "end": s["end"], "flag": s["flag"]}
        if drivers:
            item["without"] = s["without"]
            item["msg"] = f"drives the score: {pct(p, s['without'])} -> {pct(s['without'], p)} without this sentence"
        else:
            item["score"] = s["score"]
            item["msg"] = f"this part reads {round(s['score'] * 100)}% AI"
        if p >= max_ai or s["flag"] == "ai":
            problems.append(item)
    for hit in hits:
        problems.append({"check": "tells", **hit, "msg": f"{hit['why']}: {hit['fix']}"})
    strong = [h for h in hits if h["severity"] == "strong"]
    fail = p >= max_ai or bool(strong)
    verdict = "FAIL" if fail else "PASS"
    reasons = []
    if p >= max_ai:
        n = sum(1 for x in problems if x["check"] == "model" and "start" in x)
        reasons.append(f"reads {round(p * 100)}% AI" + (f"; {n} sentence(s) to rewrite" if n else ""))
    if strong:
        reasons.append(f"{len(strong)} strong AI tell(s): " + ", ".join(sorted({h['rule'] for h in strong})))
    summary = (f"PASS: reads {round(p * 100)}% AI (under {round(max_ai * 100)}%), no strong AI tells"
               + (f"; {len(hits)} soft tell(s) worth a look" if hits else "")) if not fail else "FAIL: " + "; ".join(reasons)
    notes = list(analysis.get("notes") or [])
    return {"verdict": verdict, "summary": summary, "ai_score": p, "max_ai": max_ai, "problems": problems,
            "notes": notes, "words": analysis["words"], "classes": analysis.get("classes"),
            "sha256": sha, "versions": vers, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "seconds": round(seconds, 2), "device": device, "complete": True}


def with_quotes(result, text):
    """Add the quoted sentence / tell text to each problem (the caller's own text; never stored)."""
    out = dict(result)
    out["problems"] = []
    for pr in result["problems"]:
        pr = dict(pr)
        if "start" in pr:
            pr["quote"] = text[pr["start"]:pr["end"]]
        out["problems"].append(pr)
    return out


def sha_of(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tells_ok():
    try:
        tells.find("I hope this email finds you well.")
        return True
    except Exception:  # noqa: BLE001 - a broken rule set must show on the map, not crash /healthz
        return False


def split_ok():
    try:
        return len(sentence_spans("One. Two.")) == 2
    except Exception:  # noqa: BLE001
        return False
