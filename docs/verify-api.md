# Verify API — PASS/FAIL for agents


| | |
|---|---|
| `POST /verify` | JSON `{"text": "...", "max_ai": 0.30, "fresh": false}` (only `text` required) or the raw text (`Content-Type: text/plain`) → verdict JSON. `fresh` re-checks even if a verdict is stored |
| `GET /verdict/<sha256>` | the stored verdict for `sha256(text as UTF-8)` under the **current** versions (offsets and scores, no text); `404` if none |
| `GET /healthz` | `200` when the model is on disk and the splitter + tells load, else `503`. `steps` = each check with its version and state (the dashboard map draws them), `today` = verify counts |
| `GET /verify` | usage text |

```bash
ai-check draft.txt                       # PASS: reads 4% AI (under 30%), no strong AI tells        -> exit 0
echo "$TEXT" | ai-check --max-ai 0.2     # stricter line
ai-check --json draft.txt | jq '.problems[] | {check, quote, msg}'
curl -s -H 'Content-Type: application/json' -d '{"text": "..."}' http://127.0.0.1:8323/verify | jq '.verdict, .summary'
```

**PASS only if every check ran and the text is clean:** the model's AI score is under `max_ai` (default **0.30** —
text sent as the owner's should read clearly human, so the line sits below the 0.50 "reads as AI" verdict) **and**
the tells linter found no *strong* tell. A check that did not run is a `FAIL` with `complete: false` (not stored); an
HTTP error or no answer is a no too. **A client's rule: use the text only if `verdict == "PASS"`.**

Response: `verdict`, `summary` (one line, written for an LLM to relay), `problems` (each with `check` = `model` or
`tells`, `quote` = the exact text, `msg` = what to do; model items carry `without` = the score without that sentence
for texts ≤ 225 words, or `score` for its 150-word window; tells carry `rule`, `severity`, `fix`), `ai_score`,
`max_ai`, `notes`, `words`, `classes`, `sha256`, `versions` (`service`, `engine` = model@revision+rule, `tells`),
`checked_at`, `seconds`, `device`, `complete`, `stored`, `from_store`.

The checks (also the map card's `steps`): **split** (sentences, abbreviations kept whole) → **model** (P(ai) as
below) → **drivers** (which sentences make it read as AI) → **tells** (`detector/tells.py`, `TELLS_VERSION` 3):
- *strong* = FAIL on its own: two or more different overused LLM words (delve, tapestry, pivotal, seamless…), leftover
  placeholders (`[Hiring Manager Name]`, `{{first_name}}`), markdown in plain text.
- *soft* = listed only: stock openers/closers ("I hope this email finds you well", "I look forward to hearing from
  you": ordinary in human email — Sol round 1 caught them blocking 4 human texts, so `TELLS_VERSION` 3 made them soft),
  one LLM word, em dashes (2+), "not just X, but Y", "the hardest part was not X", a rule-of-three habit,
  "In conclusion," wrap-ups, eager-applicant boilerplate.
- Measured: a strong tell in 13 of 74 AI texts and 2 of 554 human texts (plus 3 reddit posts with markdown).

**Verdicts are keyed by SHA-256 of the text + `max_ai` + every version** (`SERVICE_VERSION` in `detector/verify.py`,
the model and its revision + scoring rule, `TELLS_VERSION`). Changing any of them re-keys every verdict. Stored in
`data/verdicts.sqlite` (dir 0700, git-ignored): verdict, scores and character offsets — never the text; quotes are
added back from the text the caller sends. Tests: `python3 eval/verify_tests.py [--fresh]` (all pass 2026-11-11;
two known model misses are reported, not failed).
