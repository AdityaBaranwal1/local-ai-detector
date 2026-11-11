"""AI texts rewritten by an LLM to read as human: eval/data/ai_humanized.json (label 1, src humanized/<model>)."""
import concurrent.futures as cf
import json
import random

from gen_ai import OUT as AI_OUT, chat

OUT = AI_OUT.parent / "ai_humanized.json"
PROMPT = ("Rewrite the text below so it reads like a real person wrote it, not an AI: vary sentence length, use plain "
          "everyday words and contractions, drop stock phrases and tidy summaries, keep the meaning and roughly the "
          "same length. Reply with the rewritten text only.\n\n---\n{text}")
REWRITERS = ["claude-sonnet-5-5", "gpt-6-sol"]


def main():
    src = json.loads(AI_OUT.read_text())
    random.Random(3).shuffle(src)
    done = json.loads(OUT.read_text()) if OUT.exists() else []
    have = {(d["orig_model"], d["task"], d["model"]) for d in done}
    jobs = [(s, REWRITERS[i % 2]) for i, s in enumerate(src[:20]) if (s["model"], s["task"], REWRITERS[i % 2]) not in have]

    def run(job):
        s, m = job
        try:
            text = chat(m, PROMPT.format(text=s["text"]))
            if len(text.split()) < 60:
                raise ValueError(f"only {len(text.split())} words")
            return {"src": f"humanized/{m}", "label": 1, "model": m, "orig_model": s["model"], "task": s["task"], "text": text}
        except Exception as e:  # noqa: BLE001
            print("FAIL", m, e)
            return None

    with cf.ThreadPoolExecutor(2) as ex:
        for r in ex.map(run, jobs):
            if r:
                done.append(r)
                OUT.write_text(json.dumps(done, indent=0))
    print(len(done), "humanized texts")


if __name__ == "__main__":
    main()
