"""Run a stratified sample of the eval set through ZeroGPT's web API and compare with ours (scores.json).

Only public or synthetic texts (pre-2023 human corpora, generated AI). Writes eval/sites/zerogpt_sample.json.
Resumable; ~5 s between requests.
"""
import json
import pathlib
import random
import time
import urllib.request

E = pathlib.Path(__file__).parent
OUT = E / "sites" / "zerogpt_sample.json"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
PLAN = {  # source prefix -> how many
    "human/TOEFL_real": 12, "human/CollegeEssay": 12, "human/HewlettStudent": 6, "human/CS224N": 6,
    "human/email-enron": 6, "human/news-2015": 5, "human/reddit-post": 5, "human/story-reddit": 5,
    "ai/": 36, "humanized/": 16, "edited/TOEFL_gpt4polished": 8,
}


def zerogpt(text):
    req = urllib.request.Request("https://api.zerogpt.com/api/detect/detectText", json.dumps({"input_text": text}).encode(),
                                 {"Content-Type": "application/json", "Origin": "https://www.zerogpt.com",
                                  "Referer": "https://www.zerogpt.com/", "User-Agent": UA})
    return json.load(urllib.request.urlopen(req, timeout=90))["data"]["fakePercentage"]


def main():
    rows = [r for r in json.loads((E / "scores.json").read_text()) if "tropa_classes" in r and len(r["text"].split()) >= 50]
    rng = random.Random(11)
    rng.shuffle(rows)
    done = json.loads(OUT.read_text()) if OUT.exists() else []
    have = {d["text"] for d in done}
    for prefix, n in PLAN.items():
        pick = [r for r in rows if r["src"].startswith(prefix)][:n]
        for r in pick:
            if r["text"] in have:
                continue
            try:
                z = zerogpt(r["text"][:15000])
            except Exception as e:  # noqa: BLE001 - keep going, rerun fills gaps
                print("FAIL", r["src"], e, flush=True)
                time.sleep(20)
                continue
            done.append({"src": r["src"], "label": r["label"], "text": r["text"], "ours": r["tropa_classes"]["ai"] * 100, "zerogpt": z})
            OUT.write_text(json.dumps(done, indent=0))
            print(f"{r['src'][:34]:34} ours {done[-1]['ours']:5.1f}  zerogpt {z}", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
