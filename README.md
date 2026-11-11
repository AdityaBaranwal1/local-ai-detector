# AI Detector (local)

A ZeroGPT-style AI-text detector that runs entirely on this box. Paste text, get a % AI score, the share of the text
that reads as AI, and the sentences that do (highlighted), so a rewrite can be aimed at the right sentences and
re-checked. Nothing leaves the machine: the model loads from the local Hugging Face cache (`HF_HUB_OFFLINE=1`),
texts are never logged or stored.

- **Page:** https://<host>.<tailnet>.ts.net:8450/ (tailnet only)
- **API:** `POST /api/detect {"text": "..."}` → `ai_score`, `verdict`, `ai_share`, `classes`, `sentences[{start,end,score,flag}]`, `notes`;
  `GET /healthz`. Loopback `127.0.0.1:8323`.
- **Service:** user unit `ai-detector.service` (`deploy/ai-detector.service`, installed in `~/.config/systemd/user/`).
  `systemctl --user restart ai-detector` after a code change.
- **Homelab:** registry `host_services: ai-detector` + `serve: "8450"`; `~/CLAUDE.md` §4l.

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
