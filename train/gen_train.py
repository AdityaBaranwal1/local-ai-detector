"""Generate AI training texts through CLIProxyAPI (never used for testing).

Stage "ai":       ~700 texts: the owner's genres (emails, application answers, cover letters, LinkedIn...) plus
                  general ones, in several styles (plain, casual, lowercase, non-native, terse), many models.
Stage "humanize": ~350 rewrites of stage-"ai" texts with different "make it sound human" prompts -> label humanized.
Stage "polish":   ~300 learner essays (ELLIPSE/ICNALE train split) polished by a model -> label ai_edited.
Writes train/raw/gen_<stage>.jsonl (resumable). Usage: python train/gen_train.py ai|humanize|polish
"""
import concurrent.futures as cf
import hashlib
import json
import pathlib
import random
import re
import sys
import threading
import urllib.request

HERE = pathlib.Path(__file__).parent
RAW = HERE / "raw"
CFG = pathlib.Path.home() / "docker/cliproxyapi/config.yaml"
KEY = re.search(r'api-keys:\s*\n\s*-\s*"([^"]+)"', CFG.read_text()).group(1)

# weights favour the cheap models so the Claude subscription stays free for the daily job run
MODELS = {"claude-haiku-5-5": 5, "gemini-3-flash": 5, "gemini-3.5-flash-lite": 4, "gpt-5.6-luna": 4, "gpt-6-luna": 4,
          "gpt-oss-120b-medium": 4, "claude-sonnet-5-5": 2, "gpt-6-sol": 2, "gpt-5.6-terra": 2, "claude-opus-5-5": 1,
          "claude-fable-5-2": 1, "claude-sonnet-4-6": 1, "gemini-3.8-flash-high": 2}

ROLES = ["junior data analyst", "software engineer intern", "marketing coordinator", "nurse", "warehouse supervisor",
         "high school teacher", "UX designer", "financial analyst", "customer support lead", "research assistant",
         "mechanical engineer", "barista turned project coordinator", "AI red-teamer", "paralegal", "lab technician",
         "sales development rep", "graphic designer", "logistics planner", "community organizer", "cybersecurity analyst"]
TOPICS = ["remote work", "public transport", "school uniforms", "video games and learning", "climate policy",
          "social media and teens", "four-day work weeks", "studying abroad", "online dating", "AI in classrooms",
          "city bike lanes", "fast fashion", "homework", "self-driving cars", "museum funding", "minimum wage",
          "learning a second language", "cooking at home", "pets in apartments", "mobile phones in school"]
TASKS = [
    ("email-cold", "an email to a hiring manager after applying for a {role} role ({w} words)"),
    ("email-followup", "a thank-you email after an interview for a {role} role ({w} words)"),
    ("email-reference", "an email asking a former manager to be a reference ({w} words)"),
    ("email-networking", "a cold networking email/LinkedIn message to someone who works as a {role} ({w} words)"),
    ("email-work", "a work email to a colleague about rescheduling a meeting and two open issues ({w} words)"),
    ("email-complaint", "an email to a landlord or company about a problem that hasn't been fixed ({w} words)"),
    ("app-why", "an answer for an application form text box: 'Why do you want this {role} job?' ({w} words)"),
    ("app-challenge", "an application text box answer: 'Describe a challenge you overcame' as a {role} ({w} words)"),
    ("app-project", "an application text box answer: 'Tell us about a project you are proud of' as a {role} ({w} words)"),
    ("app-strength", "an application text box answer: 'What is your greatest strength?' as a {role} ({w} words)"),
    ("app-ai", "an application answer: 'Describe a time an AI tool failed or surprised you' ({w} words)"),
    ("cover-letter", "a cover letter for a {role} position ({w} words)"),
    ("linkedin", "a LinkedIn post about something you learned working as a {role} ({w} words)"),
    ("essay", "a short opinion essay about {topic} ({w} words)"),
    ("student-essay", "a school essay about {topic}, as a high school student would hand in ({w} words)"),
    ("reddit", "a Reddit post asking for advice about {topic} ({w} words)"),
    ("review", "an online review of a product, restaurant or app you used ({w} words)"),
    ("story", "a short story that involves {topic} ({w} words)"),
    ("explainer", "a plain explanation of something technical for a general reader ({w} words)"),
    ("blog", "a personal blog post about {topic} ({w} words)"),
]
STYLES = [
    ("plain", ""),
    ("plain", ""),
    ("casual", " Write it the way a real person quickly types it: plain words, specific details, a bit informal."),
    ("lowercase", " Write it like a quick message typed on a phone: mostly lowercase, loose punctuation, short sentences."),
    ("nonnative", " Write it as a non-native English speaker with intermediate English would: simple words, a few small grammar mistakes."),
    ("terse", " Keep it short and direct, no fluff, no greeting clichés."),
    ("human-try", " Make it sound human-written so AI detectors don't flag it."),
]
SYSTEM = "Reply with the requested text only: no preamble, no title, no markdown headings, no placeholders like [Name] — invent names."

HUMANIZE = [
    "Rewrite this so it reads like a real person wrote it: vary sentence length, plain words, contractions, keep the meaning.",
    "Paraphrase this text so AI detectors won't flag it. Change sentence structure and word choice; keep the facts.",
    "Make this sound less like AI: remove polished phrasing and tidy summaries, add a little personality, keep the length.",
    "Rewrite this in a casual, slightly messy human voice, like someone typing quickly. Same content.",
    "Humanize this text: use simpler words, uneven rhythm, one or two small imperfections, no em dashes.",
    "Rewrite this as if a non-native English speaker wrote it themselves, keeping the meaning.",
]
POLISH = [
    "Fix the grammar and spelling of this essay and make it read smoothly. Keep the student's ideas and length.",
    "Polish this essay so it sounds fluent and well written. Keep the same points.",
    "Correct the mistakes in this text. Change as little as possible.",
]

_lock = threading.Lock()


def chat(model, prompt, temperature=1.0):
    body = json.dumps({"model": model, "temperature": temperature, "max_tokens": 6000,
                       "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:8317/v1/chat/completions", body,
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return (json.load(r)["choices"][0]["message"]["content"] or "").strip()


def pick_model(rng):
    ms = list(MODELS)
    return rng.choices(ms, weights=[MODELS[m] for m in ms])[0]


def jobs_ai(rng, n=700):
    out = []
    for i in range(n):
        tid, t = rng.choice(TASKS)
        sname, s = rng.choice(STYLES)
        w = rng.choice([80, 120, 150, 200, 250, 300, 400])
        prompt = t.format(role=rng.choice(ROLES), topic=rng.choice(TOPICS), w=w)
        out.append({"key": f"ai-{i}", "model": pick_model(rng), "task": tid, "style": sname,
                    "prompt": "Write " + prompt + "." + s, "label": "ai"})
    return out


def jobs_humanize(rng, n=350):
    src = [json.loads(l) for l in (RAW / "gen_ai.jsonl").read_text().splitlines()]
    rng.shuffle(src)
    out = []
    for i, s in enumerate(src[:n]):
        h = rng.choice(HUMANIZE)
        out.append({"key": f"hz-{i}", "model": pick_model(rng), "task": s["task"], "style": "humanized",
                    "orig_model": s["model"], "prompt": h + "\n\n---\n" + s["text"], "label": "humanized"})
    return out


def jobs_polish(rng, n=300):
    src = [json.loads(l) for l in (RAW / "learner_train.jsonl").read_text().splitlines()]
    rng.shuffle(src)
    out = []
    for i, s in enumerate(src[:n]):
        out.append({"key": f"pol-{i}", "model": pick_model(rng), "task": "polished-learner", "style": "polish",
                    "prompt": rng.choice(POLISH) + "\n\n---\n" + s["text"], "label": "ai_edited"})
    return out


def main(stage):
    rng = random.Random({"ai": 1, "humanize": 2, "polish": 3}[stage])
    jobs = {"ai": jobs_ai, "humanize": jobs_humanize, "polish": jobs_polish}[stage](rng)
    out = RAW / f"gen_{stage}.jsonl"
    done = {json.loads(l)["key"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [j for j in jobs if j["key"] not in done]
    print(f"{stage}: {len(done)} done, {len(todo)} to go", flush=True)

    def run(j):
        for attempt in range(2):
            try:
                text = chat(j["model"] if attempt == 0 else "gemini-3-flash", j["prompt"])
                if len(text.split()) >= 40 and "paste" not in text[:80].lower():
                    return {**{k: v for k, v in j.items() if k != "prompt"}, "text": text,
                            "model": j["model"] if attempt == 0 else "gemini-3-flash"}
            except Exception as e:  # noqa: BLE001
                print("FAIL", j["key"], j["model"], str(e)[:80], flush=True)
        return None

    with cf.ThreadPoolExecutor(3) as ex:
        for i, r in enumerate(ex.map(run, todo)):
            if r:
                with _lock, out.open("a") as f:
                    f.write(json.dumps(r) + "\n")
            if i % 50 == 0:
                print(f"{stage}: {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
