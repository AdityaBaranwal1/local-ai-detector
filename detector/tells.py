"""AI "tells": phrases and habits that readers and detectors associate with LLM writing.

A second check beside the model: the model says *how AI* a text reads, this says *what to change*. Each hit has
a rule id, a severity, the character span, and a fix an agent can apply.
  strong = FAIL on its own: leftover placeholders, markdown, a pile of distinct LLM-favourite words (no person
           leaves these in a real message)
  soft   = listed, not a FAIL on its own: stock openers/closers ("I look forward to hearing from you" is ordinary
           human email: Sol round 1 found it blocking 4 human texts), em dashes, "not just X but Y", ...
Bump TELLS_VERSION whenever a rule changes: it is part of every verdict key.
"""
from __future__ import annotations

import re

TELLS_VERSION = 3

# (id, severity, regex, why, fix)
_RULES = [
    ("stock-opener", "soft",
     r"\b(I hope (this|my) (email|message|note) finds you( well)?|I hope you('re| are) (doing )?well\b(?! enough)|I am writing to (express|inquire)|"
     r"I wanted to reach out|I trust this (email|message) finds you)",
     "stock LLM email opener", "start with the point, or one specific line about them"),
    ("stock-closer", "soft",
     r"\b(I look forward to (the opportunity|hearing from you|discussing)|Thank you for (your time and consideration|considering my application)|"
     r"Please (do not|don't) hesitate to (reach out|contact)|Feel free to reach out|I would welcome the opportunity)",
     "stock LLM closing line", "end with one concrete next step in your own words, or just your name"),
    ("llm-vocab", "strong",
     r"\b(delve|delving|tapestry|testament to|ever-evolving|ever-changing landscape|in today's (fast-paced|digital|ever)|"
     r"navigate the (complexities|landscape)|realm of|multifaceted|seamless(ly)?|leverag(e|ing) my|synerg(y|ies)|"
     r"pivotal|paramount|meticulous(ly)?|commendable|invaluable (experience|insights?)|foster(ing)? a culture|"
     r"unwavering|underscore[sd]?|resonate[sd]? (deeply|with me)|a deep passion for|I am (deeply )?passionate about)\b",
     "word or phrase LLMs overuse", "use the plain word (look into, use, important, careful) or cut it"),
    ("placeholder", "strong", r"\[[^\]\n]{0,40}\b(name|company|position|role|title|date|address|phone|email|link|url)\b[^\]\n]{0,20}\]|\{\{[^}]+\}\}",
     "template placeholder left in", "fill it in or delete it"),
    ("markdown", "strong", r"(^|\n)\s{0,3}(#{1,6} |\*\*[^*\n]+\*\*|[-*] \*\*)",
     "markdown formatting in plain text", "remove the #, ** and bullet bolding; write plain sentences"),
    ("not-just", "soft", r"\b(it'?s not (just|only|merely) (about )?[^.;:!?]{2,60}[,;—–-]+ (it'?s|but)|not (just|only|merely) [^.;:!?]{2,60}, but (also )?)",
     "'not just X, but Y' construction", "say the one thing you mean"),
    ("hardest-part", "soft", r"\bthe (hardest|real|biggest|most important) (part|lesson|challenge) (was|wasn'?t) (not )?",
     "'the hardest part was not X. It was Y' reveal", "state what happened directly"),
    ("em-dash", "soft", r"\s?—\s?",
     "em dash (several in a short text reads as LLM style)", "use a comma, period or parentheses"),
    ("rule-of-three", "soft", r"\b(\w+(?: \w+)?), (\w+(?: \w+)?),? and (\w+(?: \w+)?)\b(?=[.!])",
     "tidy list of three at a sentence end", "keep the one or two that matter"),
    ("summary-close", "soft", r"(^|\n|\. )(In (summary|conclusion|short)|Overall|Ultimately|All in all),",
     "wrap-up sentence that restates the text", "cut it; end on the last real point"),
    ("eager", "soft", r"\b(I am (truly |incredibly |extremely )?(excited|thrilled|eager) (to|about)|I('m| am) confident (that )?(my|I))",
     "eager-applicant boilerplate", "show it with one specific reason instead"),
]
RULES = [(i, s, re.compile(p, re.I), w, f) for i, s, p, w, f in _RULES]
EM_DASH_SOFT_MAX = 1   # one em dash is normal; two or more in an email or form answer is a tell


def find(text: str) -> list[dict]:
    hits = []
    for rid, sev, rx, why, fix in RULES:
        ms = list(rx.finditer(text))
        if rid == "em-dash" and len(ms) <= EM_DASH_SOFT_MAX:
            continue
        if rid == "rule-of-three" and len(ms) < 2:   # one list of three is normal writing; a habit is the tell
            continue
        if rid == "llm-vocab" and len({m.group(0).lower() for m in ms}) < 2:
            sev = "soft"                              # one "paramount" is a person; two different ones is a pattern
        for m in ms[:6]:
            s, e = m.span()
            while s < e and text[s].isspace():
                s += 1
            hits.append({"rule": rid, "severity": sev, "start": s, "end": e, "why": why, "fix": fix})
    hits.sort(key=lambda h: (h["severity"] != "strong", h["start"]))
    return hits
