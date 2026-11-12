"""Audit training files for personal data and leftover code / tables / diagrams. Prints counts per check and, for
the non-personal checks only, a few short matched fragments (never whole rows; never PII matches).
  .venv/bin/python train/audit_data.py train/data/transcripts.jsonl [more.jsonl ...]   exit 1 = something found
"""
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
PERSONAL = {
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    "phone": r"(?<!\d)(\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}(?!\d)",
    "url": r"https?://|www\.\w|\b\w+\.(com|org|net|io|ai|dev|app)/\S*",
    "street address": r"\b\d{1,5}\s+\w+(\s\w+)?\s+(st|street|ave|avenue|rd|road|blvd|boulevard|lane|ln|dr|drive|ct|court|way)\b\.?",
    "zip / long number": r"(?<![\d.])\d{5}(-\d{4})?(?![\d.%])|\d{6,}",
    "ssn / card": r"\b\d{3}-\d{2}-\d{4}\b|\b(?:\d{4}[- ]){3}\d{4}\b",
    "ip address": r"\b\d{1,3}(\.\d{1,3}){3}\b",
    "home path": r"/home/\w+|/Users/\w+|C:\\\\Users|~/[\w.-]+",
    "key / token": r"\b(sk|pk|ghp|gho|xox[bp]|AKIA)[-_][\w-]{10,}|\b[A-Za-z0-9+/]{32,}={0,2}\b|\b[a-f0-9]{32,}\b",
    "salutation + name": r"\b(dear|hi|hello|hey)\s+[A-Z][a-z]+\b",
    "dob / age phrase": r"\b(born on|date of birth|dob)\b",
}
FORMAT = {
    "code fence / backticks": r"`",
    "code syntax": r"^\s*(def |class |import |from \w+ import|function |const |let |var |return |if \(|for \(|#include|SELECT |sudo |pip |npm |git |cd |\$ )|=>|\)\s*\{|\};|\w+\(\)|\w+\.\w+\(",
    "cli flag / path": r"(?<!\w)--?[a-z][\w-]{2,}\b|\b\w+/\w+\.\w{2,4}\b",
    "json / braces": r"[{}\[\]]",
    "table pipes": r"\|",
    "box drawing / arrows": r"[│─┌┐└┘├┤┬┴┼═║╔╗╚╝▶►→←↑↓⇒]",
    "diagram words": r"(?i)\b(flowchart|graph (td|lr)|mermaid|-->|sequenceDiagram)\b|-->",
    "markdown header / bold": r"^\s*#{1,6}\s|\*\*|__\w",
    "list bullets": r"^\s*([-*•]|\d+[.)])\s+",
    "latex": r"\\(frac|sum|int|alpha|beta|text|mathbb)|\$[^$]{1,40}\$",
    "emoji": r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
    "separator line": r"^\s*([-=_*~]){3,}\s*$",
}


def audit(path):
    rows = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines()]
    hits, frags = Counter(), defaultdict(list)
    for r in rows:
        t = r["text"]
        for name, rx in {**PERSONAL, **FORMAT}.items():
            ms = list(re.finditer(rx, t, flags=re.M | (0 if name in ("salutation + name",) else re.I)))
            if ms:
                hits[name] += 1
                if name in FORMAT and len(frags[name]) < 4:
                    frags[name].append(ms[0].group(0)[:20])
    print(f"\n{path}: {len(rows)} rows")
    for kind, table in (("PERSONAL", PERSONAL), ("FORMAT", FORMAT)):
        for name in table:
            if hits[name]:
                print(f"  {kind:8} {name:24} {hits[name]:5} rows ({hits[name] / len(rows):.1%})" + (f"   e.g. {frags[name]}" if name in frags else ""))
    return hits


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        h = audit(p)
        bad += sum(h[n] for n in PERSONAL)
    sys.exit(1 if bad else 0)
