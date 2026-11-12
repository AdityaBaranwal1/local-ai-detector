"""Packaging-time cleanup for the newer data files (open / casual / ext / transcripts), same rules for both labels so
no formatting or contact detail becomes a "this is AI" shortcut (audit 2026-11-12: AI-written job-seeker posts held
invented emails while the human ones had contact lines removed; transcripts kept a few URLs / IPs / paths).

  sanitize(text) -> cleaned text, or None to drop the row.
Drop: anything that looks like personal or contact data (email, phone, URL, street address, IP, home path, token,
SSN/card, date of birth), any of the owner's identifying strings (train/.scrub.local, .public-guard.local), leftover
code, tables, box-drawing / arrow diagrams, LaTeX, JSON/braces, [Placeholders].
Strip (keep the row): list markers, inline backticks, markdown emphasis / headers, separator lines, [1]-style refs.
Audit any file with train/audit_data.py.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DROP = re.compile("|".join([
    r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",                                     # email
    r"(?<!\d)(\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}(?!\d)",       # phone
    r"https?://|www\.\w|\b\w+\.(com|org|net|io|ai|dev|app)/\S*",       # url
    r"\b\d{1,5}\s+\w+(\s\w+)?\s+(st|street|ave|avenue|rd|road|blvd|boulevard|lane|ln|dr|drive|ct|court)\b",
    r"\b\d{1,3}(\.\d{1,3}){3}\b",                                      # ip
    r"/home/\w+|/Users/\w+|\bUsers[\\/]|AppData|\b[A-Z]:[\\/]|~/[\w.-]+",       # home / Windows paths
    r"\b(sk|pk|ghp|gho|xox[bp]|AKIA)[-_][\w-]{10,}|\b[a-f0-9]{32,}\b",  # tokens / hashes
    r"\b\d{3}-\d{2}-\d{4}\b|\b(?:\d{4}[- ]){3}\d{4}\b|\d{7,}",          # ssn / card / long numbers
    r"\b(born on|date of birth|dob)\b",
]), re.I)
FORMAT_DROP = re.compile("|".join([
    r"^\s*(def |class |import |from \w+ import |function |const |var |#include|SELECT |sudo |pip install|npm |git )",
    r"=>|\)\s*\{|\};|\w+\.\w+\([^)]*\)|\w+\(\)",                        # code calls / blocks
    r"[{}]|\|.*\||[│─┌┐└┘├┤┬┴┼═║╔╗╚╝▶►→←⇒]|-->",                        # json, tables, diagrams, arrows
    r"\\(frac|sum|int|alpha|beta|text|mathbb|cdot)|\$[^$\s][^$]{0,40}[a-z^_\\][^$]*\$",   # latex
    r"\[[A-Z][A-Za-z .'/-]{1,40}\]",                                    # [Placeholder]
    r"\b([Ff]lowchart|[Mm]ermaid|sequenceDiagram)\b",
]), re.M)
_TERMS = None


def _terms():
    global _TERMS
    if _TERMS is None:
        _TERMS = []
        for f in (ROOT / "train/.scrub.local", ROOT / ".public-guard.local"):
            if f.exists():
                _TERMS += [l.strip().lower() for l in f.read_text().splitlines() if len(l.strip()) >= 3 and not l.startswith("#")]
    return _TERMS


def sanitize(t):
    if not isinstance(t, str) or DROP.search(t) or any(x in t.lower() for x in _terms()) or FORMAT_DROP.search(t):
        return None
    t = re.sub(r"^\s*([-=_*~]){3,}\s*$", "", t, flags=re.M)              # separator lines
    t = re.sub(r"^\s*#{1,6}\s+", "", t, flags=re.M)                      # headers
    t = re.sub(r"^\s*([-*•]|\d{1,2}[.)])\s+", "", t, flags=re.M)         # list markers
    t = re.sub(r"\*\*([^*]+)\*\*|__([^_]+)__", lambda m: m.group(1) or m.group(2), t)
    t = t.replace("`", "")
    t = re.sub(r"\[\d{1,3}\]", "", t)                                     # [1] refs
    t = re.sub(r"[ \t]{2,}", " ", re.sub(r"\n{3,}", "\n\n", t)).strip()
    return t if len(t.split()) >= 20 else None
