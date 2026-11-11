"""Turn tropa-mini's outputs into what the page shows: a document score, a ZeroGPT-style share of AI-looking text,
and a score per sentence (for highlighting). Every number here was picked on eval/ (README.md "How good is it").

ai_score = P(ai) for the whole text (length-weighted over 500-token chunks). Not the model card's 1 - P(human):
its "humanized" and "ai_edited" classes also fire on polished human writing (real college essays: 49 % flagged
vs 3 % with P(ai)). They are reported, as hints, never added to the score.
"""
from __future__ import annotations

from .text import sentence_spans

AI_LINE = 0.50          # eval: 0-3 % of native-English human texts above it, 98 % of fresh AI text
SENT_AI, SENT_MID = 0.70, 0.40
WINDOW_WORDS = 150      # eval/tune_windows.py: 45 words -> 36 % of AI text flagged, 150 -> 90 %, best splice accuracy
HINT_LINE = 0.50


def _windows(text, spans):
    """For each sentence, a window of neighbouring sentences (the classifier is unreliable on a dozen words)."""
    out = []
    wc = [len(text[s:e].split()) for s, e in spans]
    for i in range(len(spans)):
        lo = hi = i
        n = wc[i]
        while n < WINDOW_WORDS and (lo > 0 or hi < len(spans) - 1):
            if hi < len(spans) - 1:
                hi += 1
                n += wc[hi]
            if n < WINDOW_WORDS and lo > 0:
                lo -= 1
                n += wc[lo]
        out.append(text[spans[lo][0]:spans[hi][1]])
    return out


def verdict(p):
    if p >= 0.90:
        return "ai", "Your text reads as AI-generated."
    if p >= AI_LINE:
        return "ai", "Most likely AI-generated."
    if p >= 0.20:
        return "mixed", "Unclear: it has some of the marks of AI text."
    if p >= 0.05:
        return "human", "Most likely human-written."
    return "human", "Your text reads as human-written."


def analyse(eng, text: str) -> dict:
    doc = eng.score(text)
    p = doc["p_ai"]
    cls = doc["classes"]
    spans = sentence_spans(text)
    if len(spans) > 1:
        uniq = {}
        wins = _windows(text, spans)
        for w in wins:                       # short texts: many sentences share one window
            uniq.setdefault(w, None)
        for w, s in zip(uniq, eng.score_many(list(uniq))):
            uniq[w] = s
        sent_scores = [uniq[w] for w in wins]
    else:
        sent_scores = [p] * len(spans)
    sents, ai_words, total = [], 0, 0
    for (s, e), sp in zip(spans, sent_scores):
        flag = "ai" if sp >= SENT_AI else "mid" if sp >= SENT_MID else "ok"
        w = len(text[s:e].split())
        total += w
        ai_words += w if flag == "ai" else 0
        sents.append({"start": s, "end": e, "score": round(sp, 4), "flag": flag})
    kind, line = verdict(p)
    notes = []
    if total < 50:
        notes.append("Under 50 words: too short for a reliable verdict.")
    if sum(c.isascii() for c in text) < 0.9 * len(text):
        notes.append("The detector is trained on English; other languages give unreliable scores.")
    if p < AI_LINE and cls.get("humanized", 0) >= HINT_LINE:
        notes.append("Hint: it reads like AI text run through a rewriter (\"humanizer\"). This signal also fires on "
                     "some polished human writing, so it is not counted in the score.")
    if p < AI_LINE and cls.get("ai_edited", 0) >= HINT_LINE:
        notes.append("Hint: it reads like human text polished by an AI. Not counted in the score.")
    return {
        "ai_score": round(p, 4), "verdict": kind, "verdict_text": line,
        "ai_share": round(ai_words / total, 4) if total else 0.0,
        "classes": {k: round(v, 4) for k, v in cls.items()},
        "words": total, "sentences": sents, "notes": notes,
        "engines": {"tropa-mini P(ai)": round(p, 4), **{f"class {k}": round(v, 4) for k, v in cls.items()}},
    }
