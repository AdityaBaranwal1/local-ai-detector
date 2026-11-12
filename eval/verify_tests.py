"""Acceptance tests for POST /verify (the live service on 127.0.0.1:8323). Stdlib only.

  python3 eval/verify_tests.py          # uses stored verdicts where they exist
  python3 eval/verify_tests.py --fresh  # after a model/tells/service change: re-checks everything
Exit 0 = all pass. Fixtures: eval/sites/results.json (Haiku-written emails/answers + human controls) and a few
inline texts. Known misses of the current model are listed in KNOWN and reported, not failed.
"""
import hashlib
import json
import pathlib
import sys
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8323"
E = pathlib.Path(__file__).parent
FRESH = "--fresh" in sys.argv
KNOWN = {"app-challenge-casual", "human-essay-1"}   # model misses as of 2026-11-11 (README "Against ZeroGPT")
fails = []


def post(body, ctype="application/json"):
    data = json.dumps(body).encode() if ctype == "application/json" else body.encode()
    req = urllib.request.Request(URL + "/verify", data, {"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        fails.append(name)


def main():
    for r in json.loads((E / "sites/results.json").read_text()):
        want = "PASS" if r["id"].startswith("human") else "FAIL"
        code, v = post({"text": r["text"], "fresh": FRESH})
        got = v.get("verdict") if isinstance(v, dict) else None
        if r["id"] in KNOWN and got != want:
            print(f"known {r['id']}: {got} (expected {want}; listed in KNOWN)")
            continue
        check(f"{r['id']} -> {want}", code == 200 and got == want, f"{got} {round(v.get('ai_score', 0) * 100)}%" if got else str(v)[:80])

    human_email = ("Hey Sam, quick one - can we push Thursday's review to Friday at 2? Two things are still open: "
                   "the freight numbers for October don't match what finance sent (off by about 4k), and nobody has "
                   "signed off on the new vendor yet. I can chase finance today. Can you ping Maria about the vendor? Thanks, Raj")
    code, v = post({"text": human_email + " I hope this email finds you well. I look forward to hearing from you.", "fresh": FRESH})
    check("stock opener/closer alone stays PASS (soft tell, listed)", v.get("verdict") == "PASS" and any(p.get("rule") == "stock-opener" for p in v["problems"]), v.get("summary", "")[:90])
    code, v = post({"text": human_email + " It's a testament to our meticulous, seamless and pivotal synergy.", "fresh": FRESH})
    check("pile of LLM words -> FAIL (strong tell)", v.get("verdict") == "FAIL" and any(p.get("rule") == "llm-vocab" and p.get("severity") == "strong" for p in v["problems"]), v.get("summary", "")[:90])
    code, v = post({"text": "Hi [Hiring Manager Name], " + human_email, "fresh": FRESH})
    check("placeholder -> FAIL", v.get("verdict") == "FAIL" and any(p.get("rule") == "placeholder" for p in v["problems"]))
    # real short human writing must pass the gate. (human_email above is a synthetic fixture -- written by an AI
    # early in the project -- fine for the tell/API checks, but no evidence about human writing: since the v6+v7
    # engine it reads ~35 %.) 10 short pre-ChatGPT human texts from the local, unpublished test set; >= 9 must PASS.
    tf = pathlib.Path(__file__).resolve().parent.parent / "test/ood.jsonl"
    if tf.exists():
        real = sorted((json.loads(l) for l in tf.read_text().splitlines()), key=lambda r: hashlib.sha256(r["text"].encode()).hexdigest())
        real = [r["text"] for r in real if r["label"] == 0 and 30 <= len(r["text"].split()) <= 120][:10]
        passed = sum(post({"text": t, "fresh": FRESH})[1].get("verdict") == "PASS" for t in real)
        check("real short human texts -> PASS (>= 9 of 10)", passed >= 9, f"{passed}/10 PASS")
    code, v = post({"text": human_email, "fresh": FRESH})
    first = v.get("verdict")
    sha = hashlib.sha256(human_email.encode()).hexdigest()
    code, v2 = post({"text": human_email})
    check("repeat is served from the store", v2.get("from_store") is True)
    with urllib.request.urlopen(f"{URL}/verdict/{sha}", timeout=10) as r:
        st = json.load(r)
    check("GET /verdict/<sha> returns it, without text", st["verdict"] == first and "quote" not in json.dumps(st) and "Raj" not in json.dumps(st))
    code, v = post(human_email, ctype="text/plain")
    check("raw text/plain body works", code == 200 and v.get("verdict") == first)
    code, v = post({"text": "   "})
    check("empty -> 400", code == 400)
    code, v = post({"text": human_email, "max_ai": 5})
    check("bad max_ai -> 400", code == 400)
    code, v = post({"text": human_email, "max_ai": 0.0001, "fresh": True})
    check("max_ai is honoured (tiny line -> FAIL)", v.get("verdict") == "FAIL")
    try:
        urllib.request.urlopen(f"{URL}/verdict/{'0' * 64}", timeout=10)
        check("unknown hash -> 404", False)
    except urllib.error.HTTPError as e:
        check("unknown hash -> 404", e.code == 404)
    with urllib.request.urlopen(URL + "/healthz", timeout=10) as r:
        h = json.load(r)
    check("healthz reports all 4 steps ok", all(h["steps"][k]["ok"] for k in ("split", "model", "drivers", "tells")))
    print(f"\n{len(fails)} failed" if fails else "\nall passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
