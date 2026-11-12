"""Public detection benchmarks we never trained on, as an independent check:
  * Ghostbuster (Verma et al. 2023): student essays, Reuters news, WritingPrompts stories -- human vs ChatGPT
    (gpt-3.5-turbo) vs Claude (2023). github.com/vivek3141/ghostbuster-data
  * DetectRL (Wu et al., NeurIPS 2024 D&B): ChatGPT, Claude-instant, Google PaLM, Llama-2-70B on arXiv / XSum /
    WritingPrompts / Yelp, plus paraphrase (DIPPER, LLM polish, back-translation) and perturbation attacks.
    github.com/NLP2CT/DetectRL (test splits only)
  * OUTFOX (Koike et al., AAAI 2024): student essays vs ChatGPT essays, incl. ones written to evade a detector.
    github.com/ryuryukke/OUTFOX (test split; pickles loaded with a builtins-only unpickler)
Raw data: train/raw/public_bench/ (git-ignored). Every text whose hash is in our training data is dropped.

  .venv/bin/python eval/bench_public.py build      # sample + dedupe -> eval/data/public_bench.jsonl
  .venv/bin/python eval/bench_public.py score      # members -> eval/runs/pub_<m>.json (cached)
  .venv/bin/python eval/bench_public.py zerogpt    # ZeroGPT on a stratified sample (>= 5 s apart) -> eval/runs/pub_zerogpt.json
  .venv/bin/python eval/bench_public.py report
"""
import glob
import io
import json
import pathlib
import pickle
import random
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))
from build_more import h  # noqa: E402

RAW = ROOT / "train/raw/public_bench"
SET = ROOT / "eval/data/public_bench.jsonl"
RUNS = ROOT / "eval/runs"
MEMBERS = {"v0": "wasitaigeneratedcom/tropa-mini-ai-text-detector", "mb6": str(ROOT / "models/rp-modernbert-v6"),
           "mb7": str(ROOT / "models/rp-modernbert-v7"), "mb8": str(ROOT / "models/rp-modernbert-v8"), "mb9": str(ROOT / "models/rp-modernbert-v9")}
ENGINES = {"v0 alone (tropa-mini as published)": None,
           "v0+v6 (2026-11-11 evening)": ("models/stacker-v0-mb6.json", ["v0", "mb6"]),
           "v6+v7 length-aware (this morning)": ("models/stacker-mb6-mb7-len.json", ["mb6", "mb7"]),
           "v0+v8 length-aware (live)": ("models/stacker-v0-mb8-len.json", ["v0", "mb8"])}
rng = random.Random(97)


class _Safe(pickle.Unpickler):          # untrusted pickles: allow plain containers / strings only, never code
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing {module}.{name}")


def _pkl(path):
    return _Safe(io.BytesIO(pathlib.Path(path).read_bytes())).load()


def ok(t):
    return isinstance(t, str) and 40 <= len(t.split()) <= 800


def build():
    rows = []
    for dom in ("essay", "reuter", "wp"):                       # Ghostbuster: 100 per domain x source
        for src, lab in (("human", 0), ("gpt", 1), ("claude", 1)):
            files = sorted(glob.glob(str(RAW / f"ghostbuster/{dom}/{src}/**/*.txt"), recursive=True))
            files = [f for f in files if "logprobs" not in f]
            for f in rng.sample(files, min(len(files), 160)):
                t = pathlib.Path(f).read_text(errors="ignore").strip()
                if ok(t):
                    rows.append({"bench": "ghostbuster", "sub": f"{dom}/{src}", "label": lab, "text": t})
    D = RAW / "detectrl/Benchmark/Benchmark_Data"
    for f, cap in [(p, 160) for p in sorted(glob.glob(str(D / "Multi_LLM/*_test.json")))] + \
                  [(D / "Paraphrase_Attacks/paraphrase_dipper_llm_test.json", 120),
                   (D / "Paraphrase_Attacks/paraphrase_polish_llm_test.json", 120),
                   (D / "Paraphrase_Attacks/paraphrase_back_translation_llm_test.json", 120),
                   (D / "Perturbation_Attacks/perturbation_attacks_llm_test.json", 120)]:
        data = json.loads(pathlib.Path(f).read_text())
        name = pathlib.Path(f).stem.replace("_test", "")
        for lab_name, lab in (("human", 0), ("llm", 1)):
            part = [d for d in data if str(d.get("label")).lower() == lab_name and ok(d.get("text"))]
            for d in rng.sample(part, min(len(part), cap // 2)):
                rows.append({"bench": "detectrl", "sub": f"{name}/{d.get('llm_type') or lab_name}/{d.get('data_type', '')}",
                             "label": lab, "text": d["text"].strip()})
    O = RAW / "outfox/data"
    for f, sub, lab in ((O / "common/test/test_humans.pkl", "human essays", 0), (O / "chatgpt/test/test_lms.pkl", "chatgpt essays", 1),
                        (O / "chatgpt/test/test_outfox_attacks.pkl", "chatgpt OUTFOX attack", 1)):
        texts = [t for t in _pkl(f) if ok(t)]
        for t in rng.sample(texts, min(len(texts), 200)):
            rows.append({"bench": "outfox", "sub": sub, "label": lab, "text": t.strip()})
    train_h = set()
    for f in list((ROOT / "train/data").glob("*.jsonl")) + list((ROOT / "train/raw").glob("*.jsonl")):
        for l in f.read_text(errors="ignore").splitlines():
            try:
                train_h.add(h(json.loads(l)["text"]))
            except (ValueError, KeyError, TypeError):
                pass
    seen, keep, dropped = set(), [], 0
    for r in rows:
        k = h(r["text"])
        if k in train_h:
            dropped += 1
        elif k not in seen:
            seen.add(k)
            r["k"] = k
            keep.append(r)
    SET.parent.mkdir(parents=True, exist_ok=True)
    SET.write_text("".join(json.dumps(r) + "\n" for r in keep))
    print(f"public bench: {len(keep)} texts ({dropped} dropped as already in our training data)")
    for b in ("ghostbuster", "detectrl", "outfox"):
        print(f"  {b}: {sum(r['bench'] == b and r['label'] == 0 for r in keep)} human, {sum(r['bench'] == b and r['label'] == 1 for r in keep)} AI")


def rows():
    return [json.loads(l) for l in SET.read_text().splitlines()]


def score():
    import torch
    from detector.engines import Classifier
    rs = rows()
    for m, repo in MEMBERS.items():
        f = RUNS / f"pub_{m}.json"
        cache = json.loads(f.read_text()) if f.exists() else {}
        todo = [r for r in rs if r["k"] not in cache]
        if not todo:
            continue
        c = Classifier(m, repo, "meanpool", 512).load("cuda")
        t0 = time.time()
        for r in todo:
            cache[r["k"]] = c.score(r["text"])["classes"]
        f.write_text(json.dumps(cache))
        print(f"scored {len(todo)} with {m} in {time.time() - t0:.0f} s", flush=True)
        c.unload()
        torch.cuda.empty_cache()


def zerogpt(n_per=12, cap=8, gap=3.5):    # gap: seconds between requests (gentle on a free API)
    rs = rows()
    by = defaultdict(list)
    for r in rs:
        by[(r["bench"], r["sub"].split("/")[0] + "/" + r["sub"].split("/")[1] if r["bench"] == "detectrl" else r["sub"], r["label"])].append(r)
    f = RUNS / "pub_zerogpt.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    sample = []
    for g in sorted(by):                     # keep every answer already collected; fill each group up to `cap`
        pick = random.Random(5).sample(by[g], min(len(by[g]), n_per))
        done = [r for r in pick if r["k"] in out]
        sample += done + [r for r in pick if r["k"] not in out][:max(0, cap - len(done))]
    todo = [r for r in sample if r["k"] not in out]
    print(f"ZeroGPT: {len(sample)} sampled, {len(todo)} to go", flush=True)
    ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    for i, r in enumerate(todo):
        for attempt in range(3):
            try:
                req = urllib.request.Request("https://api.zerogpt.com/api/detect/detectText", json.dumps({"input_text": r["text"]}).encode(),
                                             {"Content-Type": "application/json", "Origin": "https://www.zerogpt.com",
                                              "Referer": "https://www.zerogpt.com/", "User-Agent": ua})
                d = json.load(urllib.request.urlopen(req, timeout=60))
                out[r["k"]] = d["data"]["fakePercentage"]
                break
            except Exception as e:  # noqa: BLE001
                print("retry", type(e).__name__, flush=True)
                time.sleep(20 * (attempt + 1))
        f.write_text(json.dumps(out))
        if i % 20 == 0:
            print(f"zerogpt {i}/{len(todo)}", flush=True)
        time.sleep(gap)


def report():
    from detector.engines import Classifier, Stacked
    from sklearn.metrics import roc_auc_score
    from eval.suite import rule_fn
    rs = rows()
    probs = {m: json.loads((RUNS / f"pub_{m}.json").read_text()) for m in MEMBERS}
    zg = json.loads((RUNS / "pub_zerogpt.json").read_text()) if (RUNS / "pub_zerogpt.json").exists() else {}
    scores = {}
    for name, spec in ENGINES.items():
        if spec is None:
            f = rule_fn("hybrid")
            scores[name] = {r["k"]: f(probs["v0"][r["k"]]) for r in rs}
            continue
        path, ms = spec
        j = json.loads((ROOT / path).read_text())
        st = Stacked([Classifier(m, "", "meanpool") for m in ms], j["coef"], j["intercept"], j.get("length_aware", False))
        scores[name] = {r["k"]: st._stack([probs[m][r["k"]] for m in ms], len(r["text"].split())) for r in rs}

    def line(name, sub, s):
        y = [r["label"] for r in sub]
        p = [s[r["k"]] for r in sub]
        hum = [x >= .5 for x, l in zip(p, y) if l == 0]
        ai = [x >= .5 for x, l in zip(p, y) if l == 1]
        auc = roc_auc_score(y, p) if 0 < sum(y) < len(y) else float("nan")
        acc = np.mean([(x >= .5) == (l == 1) for x, l in zip(p, y)])
        return f"    {name:38} AUROC {auc:.3f}  acc {acc:6.1%}  humans flagged {np.mean(hum) if hum else float('nan'):6.1%}  AI caught {np.mean(ai) if ai else float('nan'):6.1%}  (n={len(sub)})"
    for b in ("ghostbuster", "detectrl", "outfox", "ALL"):
        sub = [r for r in rs if b == "ALL" or r["bench"] == b]
        print(f"\n== {b}")
        for name in ENGINES:
            print(line(name, sub, scores[name]))
        zs = [r for r in sub if r["k"] in zg]
        if zs:
            print(f"   on the {len(zs)} texts ZeroGPT also judged:")
            for name in list(ENGINES)[-1:]:
                print(line(name, zs, scores[name]))
            print(line("ZeroGPT", zs, {r["k"]: zg[r["k"]] / 100 for r in zs}))
    print("\n== live engine by subset (AI caught / humans flagged)")
    live = scores[list(ENGINES)[-1]]
    groups = defaultdict(list)
    for r in rs:
        key = r["bench"] + " " + ("/".join(r["sub"].split("/")[:2]) if r["bench"] != "outfox" else r["sub"])
        groups[key].append(r)
    for g in sorted(groups):
        sub = groups[g]
        lab = sub[0]["label"] if len({r["label"] for r in sub}) == 1 else None
        rate = np.mean([live[r["k"]] >= .5 for r in sub])
        print(f"    {g:55} {'AI caught' if lab == 1 else 'humans flagged' if lab == 0 else 'flagged'} {rate:6.1%}  (n={len(sub)})")


if __name__ == "__main__":
    {"build": build, "score": score, "zerogpt": zerogpt, "report": report}[sys.argv[1]]()
