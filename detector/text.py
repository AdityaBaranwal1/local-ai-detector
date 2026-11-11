"""Sentence splitting that keeps character spans (for highlighting)."""
from __future__ import annotations

import re

_ABBR = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "inc", "ltd", "co", "no", "fig", "u.s", "a.m", "p.m"}
_END = re.compile(r"[.!?…]+[\"')\]”’]*\s+|\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """[(start, end)] of sentences, whitespace trimmed; paragraph breaks and list items always split."""
    spans, start = [], 0
    for m in _END.finditer(text):
        end = m.end()
        before = text[start:m.start() + 1].rstrip(".!?…\"')]”’ ").split()
        last = before[-1].lower().rstrip(".") if before else ""
        if "\n" not in m.group() and (last in _ABBR or re.fullmatch(r"[a-z]", last)):
            continue  # "Dr. Smith", "J. Doe"
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


def words(text: str) -> int:
    return len(text.split())
