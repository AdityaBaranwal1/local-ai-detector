"""Bias report: false-positive and catch rates per group, side by side for several model versions.

Groups (human texts -> false-positive rate, lower is better; AI texts -> catch rate, higher is better):
  writer   native / non-native (TOEFL, ELLIPSE, ICNALE, non-anglophone researchers)
  domain   encyclopedic, academic, news, fiction, forum, Q&A, email, review, essay, blog/other
  length   50-100 / 100-250 / 250-500 / 500+ words
  register casual (lowercase starts, contractions, texting) / formal
  AI side  generator family, and style (raw / humanized-paraphrased / AI-polished human)
Headline: worst-group FPR (human) and worst-group catch (AI), groups with n >= 20, plus Wilson 95 % intervals.

Usage: .venv/bin/python -m eval.bias v0 k3 [v4 ...] [--rule hybrid|not_human|p_ai] [--min-n 20]
(score a model first with eval.suite so its probabilities are cached in eval/runs/<name>.json)
"""
import argparse
import json
import math
import re
from collections import defaultdict

from eval.suite import RUNS, load_sets, rule_fn

DOMAIN = [  # (regex on src, domain)
    (r"wikipedia|squad|wiki_csai|wikihow|/wiki", "encyclopedic"),
    (r"pmc|CS224N|sci_gen|arxiv|peerread|abstract|medicine|finance", "academic"),
    (r"news|xsum|cnn", "news"),
    (r"literature|novel|story|roct|/wp\b|wp_|writingprompts|hswag", "fiction"),
    (r"reddit|cmv|tldr|eli5|forum|post", "forum"),
    (r"open_qa|hc3|qa", "Q&A"),
    (r"email|enron|sites/", "email"),
    (r"review|yelp|amazon", "review"),
    (r"TOEFL|College|Hewlett|ellipse|icnale|essay|argugpt|persuade", "essay"),
]
FAMILY = [(r"claude|anthropic", "Claude"), (r"gpt-?[456]|gpt-?3\.5|chatgpt|openai|o[13]-|gpt4", "GPT"),
          (r"gemini|palm|bard", "Gemini"), (r"llama", "Llama"), (r"mistral|mixtral", "Mistral"),
          (r"qwen|deepseek|yi-|glm", "Qwen/DeepSeek/other open"),
          (r"opt|gpt2|gpt-j|neo|davinci|curie|babbage|t5|bloom|flan|7b|13b", "older (≤2022)")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def domain_of(src):
    for rx, d in DOMAIN:
        if re.search(rx, src, re.I):
            return d
    return "blog/other"


def family_of(src):
    for rx, f in FAMILY:
        if re.search(rx, src, re.I):
            return f
    return "unknown"


def register_of(text):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if not sents:
        return "formal"
    lower_start = sum(s[0].islower() for s in sents) / len(sents)
    contractions = len(re.findall(r"\b\w+'(t|s|re|ve|ll|d|m)\b", text, re.I)) / max(1, len(text.split()))
    slang = len(re.findall(r"\b(lol|tbh|idk|gonna|wanna|kinda|ok so|yeah|btw|imo)\b", text, re.I))
    return "casual" if lower_start > 0.3 or contractions > 0.02 or slang >= 2 else "formal"


def length_of(text):
    n = len(text.split())
    return "50-100" if n < 100 else "100-250" if n < 250 else "250-500" if n < 500 else "500+"


def style_of(r):
    s = r["src"]
    if "polish" in s or s.startswith("eval/edited"):
        return "AI-polished human"
    if "humaniz" in s or "paraphrase" in s or "stealth" in s:
        return "humanized / paraphrased"
    return "raw"


def groups(r):
    out = {}
    if r["label"] == 0:
        out["writer"] = "non-native" if r["g"] == "H non-native" else "native"
        out["domain"] = domain_of(r["src"])
        out["register"] = register_of(r["text"])
        out["length"] = length_of(r["text"])
    else:
        out["AI family"] = family_of(r["src"])
        out["AI style"] = style_of(r)
        out["AI domain"] = domain_of(r["src"])
        out["AI length"] = length_of(r["text"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--rule", default="hybrid")
    ap.add_argument("--min-n", type=int, default=20)
    a = ap.parse_args()
    rows = load_sets()
    runs = {n: json.loads((RUNS / f"{n}.json").read_text()) for n in a.names}
    rows = [r for r in rows if all(r["k"] in runs[n] for n in a.names)]
    f = rule_fn(a.rule)
    print(f"{len(rows)} texts scored by all of {a.names}; rule {a.rule}; flagged = score >= 0.5\n")
    table = defaultdict(lambda: defaultdict(list))         # dim -> group -> rows
    for r in rows:
        for dim, g in groups(r).items():
            table[dim][g].append(r)
    worst = {n: {"fp": (0, ""), "catch": (1, "")} for n in a.names}
    for dim in ("writer", "domain", "register", "length", "AI family", "AI style", "AI domain", "AI length"):
        human = not dim.startswith("AI")
        print(f"── {dim} ({'false-positive rate: human texts flagged as AI, lower is better' if human else 'catch rate: AI texts flagged, higher is better'})")
        print(f"   {'group':26} {'n':>5} " + " ".join(f"{n:>22}" for n in a.names))
        for g, rs in sorted(table[dim].items(), key=lambda kv: -len(kv[1])):
            cells = []
            for n in a.names:
                k = sum(f(runs[n][r["k"]]) >= .5 for r in rs)
                lo, hi = wilson(k, len(rs))
                cells.append(f"{k / len(rs):6.1%} [{lo:4.0%}-{hi:4.0%}]")
                if len(rs) >= a.min_n:
                    if human and k / len(rs) > worst[n]["fp"][0]:
                        worst[n]["fp"] = (k / len(rs), f"{dim}={g}")
                    if not human and k / len(rs) < worst[n]["catch"][0]:
                        worst[n]["catch"] = (k / len(rs), f"{dim}={g}")
            flag = "" if len(rs) >= a.min_n else "  (small n)"
            print(f"   {g[:26]:26} {len(rs):5} " + " ".join(f"{c:>22}" for c in cells) + flag)
        print()
    print("== headline (groups with n >= %d)" % a.min_n)
    for n in a.names:
        hum = [r for r in rows if r["label"] == 0]
        ai = [r for r in rows if r["label"] == 1]
        fp = sum(f(runs[n][r["k"]]) >= .5 for r in hum) / len(hum)
        ca = sum(f(runs[n][r["k"]]) >= .5 for r in ai) / len(ai)
        print(f"  {n:8} overall FPR {fp:5.1%}  catch {ca:5.1%}  | worst-group FPR {worst[n]['fp'][0]:5.1%} ({worst[n]['fp'][1]})"
              f"  | worst-group catch {worst[n]['catch'][0]:5.1%} ({worst[n]['catch'][1]})")


if __name__ == "__main__":
    main()
