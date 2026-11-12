"""Casual-professional matched data (v7): the independent review's round 2 found the combiner's gap in the owner's own
use case -- naturally casual AI messages (5/8 caught vs ZeroGPT 8/8), e.g. a restrained, concrete LinkedIn networking
message scored 27 % and passed the gate.

Human (label 0, 2008-2020): train/raw/casual_human.jsonl from train/build_casual_human.py -- HN "Who wants to be hired?"
intros, Stack Exchange workplace / freelancing / PM posts, 2017 brand support replies on Twitter, FOIA work email,
Ubuntu IRC support chat. The owner's own email is NOT used (much of it is AI-assisted).
AI (label 1; label 3 = then run through a "make it human" rewrite):
  * matched: for a human text, a model writes ITS OWN text of the same kind for the same situation, about the same
    length (it sees only the first ~15 words as the situation, and is told not to continue or copy them), so the
    pair differs in authorship, not in topic, genre or length;
  * priority: the owner's genres with no public human counterpart (LinkedIn networking / referral asks, recruiter
    replies, follow-ups, short application answers, cover-letter paragraphs, cold outreach), mostly in a natural,
    restrained, specific register (the style that slipped through), plus casual / lowercase / non-native / terse /
    typo styles.
Models: the proxy's Claude Haiku/Sonnet, Gemini Flash, GPT-6/5.6 Luna, GPT-6 Sol, gpt-oss-120b. Never Fable; no Opus.
-> train/data/casual.jsonl (balanced, deduplicated against every test set and Sol's texts).
Usage: .venv/bin/python -u train/build_casual.py [ai|finish]   (resumable)
"""
import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import sys
import threading
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_expo import clean  # noqa: E402
from build_more import h  # noqa: E402
from gen_train import chat  # noqa: E402

IN_H = ROOT / "train/raw/casual_human.jsonl"
OUT_A = ROOT / "train/raw/casual_ai.jsonl"
rng = random.Random(53)
MODELS = {"claude-haiku-5-5": 4, "gemini-3-flash": 4, "gpt-6-luna": 3, "gpt-oss-120b-medium": 3, "gpt-5.6-luna": 2,
          "claude-sonnet-5-5": 2, "gpt-6-sol": 2, "gemini-3.8-flash-high": 2}
assert not any("fable" in m or "opus" in m for m in MODELS)
N_MATCHED, N_PRIORITY, P_HUMANIZE = 2400, 800, 0.2
_lock = threading.Lock()

KIND = {  # human genre -> how to describe it to the model
    "job-seeker-post": "post in a \"Who wants to be hired?\" thread on a tech forum, where a job seeker introduces themselves",
    "casual-career-talk": "post on a workplace / career Q&A site (a question or an answer)",
    "customer-support-reply": "reply from a company's customer-support account on Twitter",
    "work-email": "short work email to a colleague",
    "work-chat": "message in a tech-support chat channel",
}
PRIORITY = [
    "a LinkedIn message to a former colleague asking whether they know anyone on the {team} team at their company, since there's a {role} opening",
    "a LinkedIn message asking someone you met at a meetup for a referral for a {role} role at their company",
    "a short LinkedIn connection note to a {role} at a company you'd like to work for",
    "a reply to a recruiter who reached out about a {role} role, saying you're interested and asking about the team",
    "a reply to a recruiter politely saying you're not looking right now but open to staying in touch",
    "a follow-up email a week after interviewing for a {role} position",
    "a follow-up message after applying for a {role} job and not hearing back",
    "a thank-you email to an interviewer after a {role} interview",
    "a cold email to a hiring manager about a {role} opening",
    "an answer to the application question 'Why do you want to work here?' for a {role} job",
    "an answer to the application question 'Tell us about a time you solved a hard problem' for a {role} job",
    "an answer to the application question 'Anything else you'd like us to know?' for a {role} job",
    "the opening paragraph of a cover letter for a {role} position",
    "a short message asking a former manager whether they'd be a reference",
    "a LinkedIn post announcing you're starting a new job as a {role}",
    "a LinkedIn post saying you were laid off and are looking for {role} roles",
]
FILL = {"role": ["data analyst", "nurse", "warehouse lead", "junior developer", "repair coordinator", "marketing assistant",
                 "project coordinator", "QA tester", "account manager", "lab technician", "UX designer", "IT support specialist"],
        "team": ["service", "data", "operations", "support", "platform", "sales"]}
STYLES = [("natural", " Write it the way a thoughtful person actually would: specific, low-key, no flourishes or stock phrases."),
          ("natural", " Make it sound like a real message from a real person, with one or two concrete details."),
          ("plain", ""),
          ("casual", " Write it casually, the way a real person quickly types it."),
          ("lowercase", " Write it mostly in lowercase with loose punctuation, like a quick phone message."),
          ("nonnative", " Write it as a non-native English speaker would, with simple words and a small grammar slip or two."),
          ("terse", " Keep it short and direct."),
          ("typos", " Type it fast, leaving one or two small typos in.")]
HUMANIZE = ["Rewrite this so it sounds like a real person typed it, not an AI. Keep it about the same length.",
            "Make this sound more human and less polished. Same meaning, same length.",
            "Paraphrase this so AI detectors won't flag it. Keep it about the same length."]
TAIL = " Output only the message itself: no subject line, no greeting placeholder like [Name], no notes."


def model():
    ms = list(MODELS)
    return rng.choices(ms, weights=[MODELS[m] for m in ms])[0]


def jobs():
    hum = [json.loads(l) for l in IN_H.read_text().splitlines()]
    by = {}
    for i, r in enumerate(hum):
        by.setdefault(r["genre"], []).append(i)
    per = N_MATCHED // len(by)
    out = []
    for g, idx in sorted(by.items()):
        for i in rng.sample(idx, min(per, len(idx))):
            r = hum[i]
            n = len(r["text"].split())
            seed = " ".join(r["text"].split()[:15])
            sname, s = rng.choice(STYLES)
            out.append({"key": f"m{i}", "genre": g, "style": sname, "model": model(), "lo": max(20, int(n * .6)), "hi": int(n * 1.6) + 15,
                        "prompt": f"Here is how a {KIND[g]} once started: \"{seed} ...\"\n\nWrite your own complete {KIND[g]} "
                                  f"for the same kind of situation, from scratch and in your own words (don't continue or copy it), "
                                  f"about {n} words.{s}{TAIL}"})
    for i in range(N_PRIORITY):
        g = re.sub(r"\{(\w+)\}", lambda m: rng.choice(FILL[m.group(1)]), rng.choice(PRIORITY))
        sname, s = rng.choice(STYLES[:3] * 2 + STYLES[3:])          # natural/plain weighted up: the style that slipped
        w = rng.choice([40, 60, 80, 110, 150])
        out.append({"key": f"p{i}", "genre": "priority", "style": sname, "model": model(), "lo": 20, "hi": 230,
                    "prompt": f"Write {g} (about {w} words).{s}{TAIL}"})
    return out


def ai():
    done = {json.loads(l)["key"] for l in OUT_A.read_text().splitlines()} if OUT_A.exists() else set()
    todo = [j for j in jobs() if j["key"] not in done]
    hum_rng = random.Random(59)
    print(f"casual ai: {len(done)} done, {len(todo)} to go", flush=True)

    def run(j):
        try:
            t = clean(chat(j["model"], j["prompt"]))
            if not (j["lo"] <= len(t.split()) <= j["hi"]) or re.search(r"\[[A-Z][a-z]+( [A-Z][a-z]+)?\]", t):
                return None
            src = f"casual/ai/{j['genre']}/{j['style']}/{j['model']}"
            rs = [{"key": j["key"], "text": t, "label": 1, "src": src}]
            if hum_rng.random() < P_HUMANIZE:
                t2 = clean(chat(model(), hum_rng.choice(HUMANIZE) + "\n\n---\n" + t))
                if j["lo"] <= len(t2.split()) <= j["hi"] and "paste" not in t2[:60].lower():
                    rs.append({"key": j["key"] + "h", "text": t2, "label": 3, "src": f"casual/humanized/{j['genre']}/{j['style']}"})
            return rs
        except Exception as e:  # noqa: BLE001
            print("FAIL", j["model"], str(e)[:80], flush=True)
            return None
    with cf.ThreadPoolExecutor(int(os.environ.get("CASUAL_THREADS", 6))) as ex:
        for i, fut in enumerate(cf.as_completed([ex.submit(run, j) for j in todo])):
            for r in fut.result() or []:
                with _lock, OUT_A.open("a") as f:
                    f.write(json.dumps(r) + "\n")
            if i % 100 == 0:
                print(f"casual ai {i}/{len(todo)}", flush=True)


def finish():
    test_h = set()
    for f in (ROOT / "test").glob("*.jsonl"):
        test_h |= {h(json.loads(l)["text"]) for l in f.read_text().splitlines()}
    for f in list((ROOT / "eval/sol").rglob("texts.json")) + list((ROOT / "eval/data").rglob("*.json")):
        try:
            test_h |= {h(r.get("text") or r.get("document")) for r in json.loads(f.read_text())
                       if isinstance(r, dict) and (r.get("text") or r.get("document"))}
        except Exception:  # noqa: BLE001
            pass
    rows = [json.loads(l) for l in IN_H.read_text().splitlines()]
    rows += [json.loads(l) for l in OUT_A.read_text().splitlines()] if OUT_A.exists() else []
    seen, keep = set(), []
    for r in rows:
        k = h(r["text"])
        if k not in test_h and k not in seen:
            seen.add(k)
            keep.append({"text": r["text"], "label": r["label"], "src": r["src"]})
    (ROOT / "train/data/casual.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    print("casual.jsonl:", len(keep), dict(Counter(r["label"] for r in keep)), flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "ai"
    if what == "ai":
        ai()
    finish()
