"""Generate AI-written evaluation texts through the local CLIProxyAPI.

Writes eval/data/ai_gen.json: [{src, label: 1, model, task, text}].
Re-runnable: keeps finished items and only fills in missing ones.
"""
import concurrent.futures as cf
import json
import pathlib
import random
import re
import urllib.request

HERE = pathlib.Path(__file__).parent
OUT = HERE / "data" / "ai_gen.json"
CFG = pathlib.Path.home() / "docker/cliproxyapi/config.yaml"
KEY = re.search(r'api-keys:\s*\n\s*-\s*"([^"]+)"', CFG.read_text()).group(1)

MODELS = [
    "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2",
    "claude-sonnet-4-5-20250929", "gpt-6-sol", "gpt-5.6-terra", "gemini-3-flash",
]

TASKS = {
    "essay": "Write a 300-word essay answering: Do you agree or disagree that it is better to cook at home than to eat at restaurants? Give reasons and examples.",
    "college-essay": "Write a 350-word personal college admissions essay about a small hobby that taught you something important.",
    "story": "Write a 350-word short story for the prompt: 'You wake up and every clock in the world has stopped except yours.'",
    "news": "Write a 300-word news article about a city council approving a new bike-lane network after a heated public meeting.",
    "email": "Write a work email (about 180 words) to a colleague asking to move Thursday's project review and summarising two open issues.",
    "reddit": "Write a Reddit post (about 250 words) asking for advice: your roommate keeps eating your food and you don't know how to bring it up.",
    "cover-letter": "Write a 300-word cover letter for a junior data analyst role at a mid-size logistics company.",
    "linkedin": "Write a LinkedIn post (about 200 words) reflecting on what you learned from your first year as a software engineer.",
    "blog": "Write a 350-word blog post with practical tips for keeping houseplants alive in a dark apartment.",
    "casual-human": "Write a 250-word casual, first-person paragraph about a frustrating trip to the DMV. Make it sound like a real person typed it quickly: plain words, no headings, no lists.",
    "explainer": "Explain in about 300 words how vaccines train the immune system, for a general audience. Plain paragraphs, no headings or lists.",
    "review": "Write a 220-word customer review of a mid-range pair of noise-cancelling headphones you've used for three months.",
}

SYSTEM = "Reply with the requested text only: no preamble, no title, no markdown headings."


def chat(model, prompt, temperature=1.0):
    body = json.dumps({
        "model": model, "temperature": temperature, "max_tokens": 6000,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8317/v1/chat/completions", body,
        {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def main():
    done = json.loads(OUT.read_text()) if OUT.exists() else []
    have = {(d["model"], d["task"]) for d in done}
    # each model gets a deterministic subset of tasks so ~6 tasks x 8 models + a full sweep for two models
    rng = random.Random(7)
    jobs = []
    for m in MODELS:
        tasks = list(TASKS) if m in ("claude-opus-5-5", "gpt-6-sol") else rng.sample(list(TASKS), 7)
        jobs += [(m, t) for t in tasks if (m, t) not in have]
    print(f"{len(done)} done, {len(jobs)} to generate")

    def run(job):
        m, t = job
        try:
            text = chat(m, TASKS[t])
            if len(text.split()) < 60:      # the proxy sometimes returns a cut-off reply
                raise ValueError(f"only {len(text.split())} words")
            return {"src": f"ai/{m}", "label": 1, "model": m, "task": t, "text": text}
        except Exception as e:  # noqa: BLE001 - report and continue; rerun fills gaps
            print("FAIL", m, t, e)
            return None

    with cf.ThreadPoolExecutor(2) as ex:
        for res in ex.map(run, jobs):
            if res:
                done.append(res)
                OUT.write_text(json.dumps(done, indent=0))
                print("ok", res["model"], res["task"], len(res["text"].split()), "words")


if __name__ == "__main__":
    main()
