# AI Detector (local)

**One call tells an agent whether its draft reads as AI:** `ai-check draft.txt` (exit 0 PASS / 1 FAIL / 2 couldn't
check), or `POST http://127.0.0.1:8323/verify {"text": ...}` → `PASS`/`FAIL` + what to rewrite. Agents run it on
anything that goes out under the owner's name (emails, application answers, messages) before calling it final —
the same idea as resume verify (`~/docker/resume-parsers`). See [Verify API](#verify-api).

Behind it: a ZeroGPT-style detector that runs entirely on this box (model from the local Hugging Face cache,
`HF_HUB_OFFLINE=1`; texts never logged, verdicts stored by hash only), plus a paste page for the owner.

- **Agents:** `ai-check` (`bin/ai-check`, symlinked in `~/.local/bin`) or `POST /verify`; `GET /verify` = usage.
- **Page:** https://<host>.<tailnet>.ts.net:8450/ (tailnet only): % AI + highlighted sentences.
- **Service:** user unit `ai-detector.service` (`deploy/ai-detector.service`, installed in `~/.config/systemd/user/`),
  loopback `127.0.0.1:8323`, tailnet `:8450`. `systemctl --user restart ai-detector` after a code change.
- **Homelab:** registry `host_services: ai-detector` (with `steps:`) + `serve: "8450"`; the job pipeline `uses` it;
  `~/CLAUDE.md` §4l.

## Verify API

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

## How it works

- **Engine:** [`wasitaigeneratedcom/tropa-mini-ai-text-detector`](https://huggingface.co/wasitaigeneratedcom/tropa-mini-ai-text-detector)
  (DeBERTa-v3-large fine-tuned from desklib, Apache-2.0), 4 classes: human / ai / ai_edited / humanized.
- **Score = P(ai)**, length-weighted over sentence-aligned 500-token chunks — except when the model is sure the text
  is not human (P(human) < 1 %), then `1 − P(human)`. Plain `1 − P(human)` (the model card's score) flags polished
  human writing (49 % of real college essays); plain P(ai) let half of LLM-"humanized" AI text through because the
  model files it under `humanized`. The rule keeps native-English false positives at ~2 % and catches 72 % of it.
  `humanized` / `ai_edited` are otherwise shown as hints only.
- **Sentences, short texts (≤ 225 words: emails, form answers):** the text is re-scored once without each sentence;
  a sentence is red when removing it drops the score by ≥ 1.0 log-odds (e.g. 61 % → 36 %), amber ≥ 0.5. The tooltip
  shows the score without it — the sentences to rewrite first. (A window would be the whole text, so every sentence
  would get the same score.)
- **Sentences, longer texts:** each sentence is scored inside a ~150-word window of its neighbours (short windows are
  unreliable: 45 words caught 36 % of AI text, 150 words 90 %); ≥ 0.70 = red, ≥ 0.40 = amber. "% of text that reads
  as AI" = words in red sentences.
- **GPU sharing:** the GPU belongs to the render workers (shorts-render, sim-engine), which run every job under
  `flock ~/All code/shorts-studio/.gpu.lock`. The detector takes that same lock, non-blocking, for each check: free →
  fp16 copy in VRAM (~1.1 GB), checks take ~3–5 s; held by a render → CPU (~7 s for 250 words). It keeps the GPU 60 s
  after a check (`GPU_IDLE_S`) and gives it back within ~2 s as soon as anything waits on the lock (`/proc/locks`).
  It never creates the lock file (the workers bind-mount its inode). RAM copy dropped after 30 min idle (`RAM_IDLE_S`).
- **Limits:** English, 50+ words. 6000 words per check.

## How good is it (eval/, 2026-11-11)

743 texts: 554 human (pre-2020 news, Enron email, reddit posts/stories, and Liang et al. 2023's essay sets), 66 fresh
AI texts from 8 current models (Claude Opus/Sonnet/Haiku 5.5, Fable 5.2, Sonnet 4.5, GPT-6 Sol, GPT-5.6 Terra,
Gemini 3 Flash; 12 genres incl. cover letters, LinkedIn, reddit, "casual"), 32 of those rewritten by an LLM "to sound
human", 91 human essays polished by GPT-4. Full output: `eval/RESULTS.txt`.

| flagged as AI at the 50 % line | this detector |
|---|---|
| human: CS224N papers / US college essays / US student essays | 1 / 4 / 0 % |
| human: news / Enron email / reddit posts / reddit stories | 2 / 2 / 0 / 8 % |
| human: TOEFL essays, non-native | **33 %** |
| fresh AI text, 8 models | **98.5 %** (100 % for 7 of 8; Opus 5.5 92 %) · AUROC 0.994 |
| AI text rewritten by an LLM "to sound human" | 72 % |
| human essays polished by GPT-4 | 90 % |

Other engines tried (AUROC, human vs raw AI): desklib 0.890 · fakespot 0.764 · TMR 0.479 · Binoculars (Qwen2.5-1.5B
pair) 0.597 · Fast-DetectGPT 0.557; a learned combination of them all did worse than tropa-mini alone.

### Against ZeroGPT (zerogpt.com today, through its web API; `eval/sites/`)

Stratified sample of 117 of the texts above (`eval/zerogpt_compare.py` → `sites/zerogpt_sample.json`):

| flagged as AI (≥ 50 %) | n | ZeroGPT | this detector |
|---|---|---|---|
| human, native English (essays, email, news, reddit) | 45 | 0 % | 0 % |
| human, non-native (TOEFL) | 12 | 0 % | 33 % |
| raw AI, 8 current models | 36 | 100 % | 100 % |
| AI rewritten by an LLM to sound human | 16 | **100 %** | 75 % |
| human essays polished by GPT-4 | 8 | 100 % | 100 % |

And 12 texts shaped like the real use (`sites/results.json`): 8 written by Claude Haiku 5.5 (3 emails, 3 application
text-box answers, 2 "typed casually"), 4 human (2 Enron emails, 2 pre-2023 college essays): **ZeroGPT 12/12, this
detector 10/12** — it missed an all-lowercase "ok so…" answer (2 %) and scored one real college essay 100 %.

So ZeroGPT (2026) is the stronger detector on humanized and non-native text; this one matches it on plain human and
plain AI text, runs offline, and says which sentences drive the score. Caveat: the human corpora are public, so
ZeroGPT may have trained on them. Only ZeroGPT could be tested automatically: GPTZero and Sapling need an account/API
key, QuillBot and Undetectable sit behind a "verify you are human" check.

## Speed (eval/bench.py → eval/BENCH.txt, 2026-11-11)

| words | GPU | CPU (a render holds the GPU) |
|---|---|---|
| 100 | 0.6 s | 0.9 s |
| 250 | 2.4 s | 4.0 s |
| 500 | 8.4 s | 14.7 s |
| 1000 | 20 s | 38 s |
| 2000 | 44 s | 87 s |

Short texts (≤ 225 words) take one extra pass per sentence for the highlights: ~1–4 s on the GPU (measured before this
mode existed: the table above). Cold start 4.1 s. RAM 3.1 GiB while loaded, VRAM 1.5 GiB while it holds the GPU.
Checks are served one at a time (4 × 250 words at once: 9.7 s). A render that asks for the GPU gets it after
0.1–1.4 s when the detector is idle, or when the check in flight finishes (≤ 44 s for 2000 words).

## Files

```
detector/engines.py   classifiers (+ the zero-shot engines used only in eval)
detector/score.py     thresholds, sentence windows, verdict text, hints
detector/server.py    FastAPI app, GPU-lock handling
detector/gpu.py       the shared GPU lock (non-blocking take, waiters from /proc/locks)
detector/static/index.html   the page
eval/                 gen_ai.py, gen_humanized.py (via CLIProxyAPI), run_eval.py, analyze.py, tune_windows.py,
                      bench.py, zerogpt_compare.py; sites/ = the cross-site test (Haiku texts + results)
```

Re-run the evaluation: `.venv/bin/python eval/gen_ai.py` (needs CLIProxyAPI), then `.venv/bin/python -m eval.run_eval`
(waits for the GPU lock) and `.venv/bin/python -m eval.analyze`. `eval/data/` is git-ignored (third-party texts),
but it is in the nightly backup.

Setup from scratch (versions pinned in `requirements.txt`; torch from the cu126 index — the 1660 Ti is sm_75):
`uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1`,
`uv pip install --python .venv/bin/python -r requirements.txt`, then
`.venv/bin/hf download wasitaigeneratedcom/tropa-mini-ai-text-detector --revision 10fecd0856ed65ddfe92b88416b0a1e1350d8fb8`
(the revision evaluated above). The eval's other models: desklib/ai-text-detector-v1.01, Oxidane/tmr-ai-text-detector,
fakespot-ai/roberta-base-ai-text-detection-v1, Qwen/Qwen2.5-1.5B(-Instruct) — ~8 GB, only needed to re-run eval.
