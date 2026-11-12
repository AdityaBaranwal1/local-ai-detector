#!/usr/bin/env python3
"""Refuse to publish secrets or private details. Run before every push (installed as .git/hooks/pre-push).

Checks every tracked file (and, with --history, every blob in every commit) for:
  1. the exact values of real credentials found on this machine (Runpod, Kaggle, Lightning, an OpenAI-compatible
     proxy key, the GitHub CLI token, ...) — read from their usual files, never printed;
  2. generic credential patterns (API-key prefixes, private keys, bearer tokens);
  3. private strings listed one per line in `.public-guard.local` (git-ignored: hostnames, home paths, emails);
  4. files that must never be published (third-party test texts) — see FORBIDDEN;
  5. with --history, commit metadata too: author/committer names and emails must not contain a private string or a
     personal email address (use your GitHub no-reply address: <id>+<login>@users.noreply.github.com).
Exit 0 = clean, 1 = found something (prints file + rule, never the secret itself).
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HOME = pathlib.Path.home()
FORBIDDEN = [r"eval/sol/[^/]+/(texts|results|humans|self_ai)\.json$", r"eval/sol/[^/]+/generation/", r"eval/sol/[^/]+/downloads/",
             r"eval/sol/[^/]+/(evaluation\.log|comparison\.html|probe_.*\.json)$", r"eval/sites/results\.json$",
             r"^eval/data/", r"^test/", r"^train/(data|raw)/", r"^models/", r"kernel-metadata\.json$",
             r"train/runpod/(ledger\.jsonl|watchdog\.log)$", r"\.env$", r"(^|/)data/verdicts\.sqlite$"]
PATTERNS = {
    "openai-style key": r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}",
    "runpod key": r"\brpa_[A-Za-z0-9]{20,}",
    "kaggle token": r"\bKGAT_[A-Za-z0-9]{16,}",
    "github token": r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    "hugging face token": r"\bhf_[A-Za-z0-9]{30,}",
    "aws key": r"\bAKIA[0-9A-Z]{16}\b",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "bearer literal": r"Bearer [A-Za-z0-9._-]{30,}",
}


def real_secrets():
    vals = {}

    def read(name, path, fn=lambda s: s.strip()):
        try:
            v = fn(pathlib.Path(path).expanduser().read_text())
            if v and len(v) >= 12:
                vals[name] = v
        except (OSError, ValueError, StopIteration, IndexError):
            pass
    read("runpod api key", "~/.config/runpod/api_key")
    read("kaggle token", "~/.kaggle/access_token")
    read("kaggle key", "~/.kaggle/kaggle.json", lambda s: __import__("json").loads(s)["key"])
    read("proxy api key", "~/docker/cliproxyapi/config.yaml", lambda s: re.search(r'api-keys:\s*\n\s*-\s*"([^"]+)"', s).group(1))
    read("lightning login", "~/.lightning/login.json",
         lambda s: next(v for k, v in __import__("json").loads(s).items() if isinstance(v, str) and len(v) > 20))
    try:
        t = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10).stdout.strip()
        if t:
            vals["github token"] = t
    except (OSError, subprocess.SubprocessError):
        pass
    return vals


def private_strings():
    f = ROOT / ".public-guard.local"
    return [l.strip() for l in f.read_text().splitlines() if l.strip() and not l.startswith("#")] if f.exists() else []


def blobs(history):
    if not history:
        for p in subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True).stdout.decode().split("\0"):
            if p and (ROOT / p).is_file():
                yield p, (ROOT / p).read_bytes()
        return
    out = subprocess.run(["git", "rev-list", "--objects", "--all"], cwd=ROOT, capture_output=True, text=True).stdout
    for line in out.splitlines():
        sha, _, path = line.partition(" ")
        if not path:
            continue
        kind = subprocess.run(["git", "cat-file", "-t", sha], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        if kind == "blob":
            yield path, subprocess.run(["git", "cat-file", "-p", sha], cwd=ROOT, capture_output=True).stdout


def main():
    history = "--history" in sys.argv
    secrets, private = real_secrets(), private_strings()
    problems = set()
    for path, data in blobs(history):
        for rx in FORBIDDEN:
            if re.search(rx, path):
                problems.add(f"{path}: must not be published ({rx})")
        text = data.decode("utf-8", errors="ignore")
        for name, val in secrets.items():
            if val in text:
                problems.add(f"{path}: contains the real {name}")
        for name, rx in PATTERNS.items():
            if re.search(rx, text):
                problems.add(f"{path}: looks like a {name}")
        for s in private:
            if s in text:
                problems.add(f"{path}: contains a private string from .public-guard.local ({s[:3]}…)")
    if history:                                    # commit metadata: authors, committers, message trailers
        meta = subprocess.run(["git", "log", "--all", "--format=%H%x09%an <%ae>%x09%cn <%ce>%x09%B%x00"], cwd=ROOT,
                              capture_output=True, text=True).stdout
        for rec in meta.split("\0"):
            if not rec.strip():
                continue
            sha = rec.strip().split("\t", 1)[0][:8]
            for who in re.findall(r"<([^>]+)>", rec):
                if who.endswith("@users.noreply.github.com") or who.endswith("@anthropic.com"):
                    continue
                if re.search(r"@(gmail|yahoo|outlook|hotmail|icloud|proton(mail)?)\.", who, re.I):
                    problems.add(f"commit {sha}: a personal email address in author/committer/trailer (use the GitHub no-reply address)")
            for sv in private:
                if sv in rec:
                    problems.add(f"commit {sha}: metadata contains a private string from .public-guard.local ({sv[:3]}…)")
    scope = "every commit" if history else "tracked files"
    if problems:
        print(f"check_public: {len(problems)} problem(s) in {scope}:")
        for p in sorted(problems):
            print("  -", p)
        return 1
    print(f"check_public: clean ({scope}; {len(secrets)} real credentials and {len(private)} private strings checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
