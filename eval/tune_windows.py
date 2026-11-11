"""Pick the sentence-window size for highlighting: AI texts should be mostly flagged, human texts barely,
and in a human+AI splice the flags should land on the right half. CPU only (runs beside the GPU eval)."""
import json
import pathlib
import random

import numpy as np

from detector import score as S
from detector.engines import CLASSIFIERS
from detector.text import sentence_spans

D = pathlib.Path(__file__).parent / "data"
rng = random.Random(5)
ai = rng.sample(json.loads((D / "ai_gen.json").read_text()), 16)
hu = rng.sample(json.loads((D / "human_web.json").read_text()), 16)
eng = CLASSIFIERS["tropa"]().load("cpu")


def flags(text, ww, combine):
    S.WINDOW_WORDS = ww
    spans = sentence_spans(text)
    sc = eng.score_many(S._windows(text, spans))
    if combine:
        doc = eng.score(text)["p_ai"]
        sc = [np.sqrt(s * doc) for s in sc]          # geometric blend with the whole-text score
    w = [len(text[s:e].split()) for s, e in spans]
    return spans, sc, w


def share(text, ww, combine, line=S.SENT_AI):
    spans, sc, w = flags(text, ww, combine)
    return sum(wi for wi, s in zip(w, sc) if s >= line) / sum(w)


for ww in (45, 90, 150):
    for combine in (False, True):
        a = np.mean([share(r["text"], ww, combine) for r in ai])
        h = np.mean([share(r["text"], ww, combine) for r in hu])
        # splice: human first half, AI second half; accuracy of the word-level flags
        acc = []
        for x, y in zip(hu[:8], ai[:8]):
            t = x["text"].strip() + "\n\n" + y["text"].strip()
            cut = len(x["text"].strip())
            spans, sc, w = flags(t, ww, combine)
            ok = sum(wi for (s, e), sci, wi in zip(spans, sc, w) if (sci >= S.SENT_AI) == (s >= cut))
            acc.append(ok / sum(w))
        print(f"window {ww:3} words blend={combine!s:5}: AI texts flagged {a:5.1%}  human flagged {h:5.1%}  splice accuracy {np.mean(acc):5.1%}", flush=True)
