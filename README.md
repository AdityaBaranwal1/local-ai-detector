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
- **Score = P(ai)**, length-weighted over sentence-aligned 500-token chunks. *Not* the model card's `1 − P(human)`:
  the `humanized` and `ai_edited` classes also fire on polished human writing (real college essays: 49 % flagged vs
  3 % with P(ai)). Those two are shown as hints only.
- **Sentences:** each sentence is scored inside a ~150-word window of its neighbours (short windows are unreliable:
  45 words caught 36 % of AI text, 150 words 90 %); ≥ 0.70 = red, ≥ 0.40 = amber. "% of text that reads as AI" =
  words in red sentences.
- **GPU sharing:** the GPU belongs to the render workers (shorts-render, sim-engine), which run every job under
  `flock ~/All code/shorts-studio/.gpu.lock`. The detector takes that same lock, non-blocking, for each check: free →
  fp16 copy in VRAM (~1.1 GB), checks take ~3–5 s; held by a render → CPU (~7 s for 250 words). It keeps the GPU 60 s
  after a check (`GPU_IDLE_S`) and gives it back within ~2 s as soon as anything waits on the lock (`/proc/locks`).
  It never creates the lock file (the workers bind-mount its inode). RAM copy dropped after 30 min idle (`RAM_IDLE_S`).
- **Limits:** English, 50+ words. 6000 words per check.

## How good is it (eval/, 2026-11-11)

743 texts: 554 human (pre-2020 news, Enron email, reddit posts/stories, and Liang et al. 2023's essay sets, which
come with ZeroGPT's own verdicts), 66 fresh AI texts from 8 current models (Claude Opus/Sonnet/Haiku 5.5, Fable 5.2,
Sonnet 4.5, GPT-6 Sol, GPT-5.6 Terra, Gemini 3 Flash; 12 genres incl. cover letters, LinkedIn, reddit, "casual"),
32 of those rewritten by an LLM "to sound human", 91 human essays polished by GPT-4. Full output: `eval/RESULTS.txt`.

| flagged as AI at the 50 % line | this detector | ZeroGPT (same texts, from Liang et al.) |
|---|---|---|
| CS224N papers (human) | 1 % | 0 % |
| US college essays (human) | 3 % | 3 % |
| US student essays (human) | 0 % | 1 % |
| TOEFL essays, non-native (human) | **25 %** | 53 % |
| news / email / reddit posts / stories (human) | 2 / 2 / 0 / 8 % | – |
| fresh AI text, 8 models | **98 %** (100 % for 7 of 8 models; Opus 5.5 92 %) | not measured* |
| AI text rewritten "to sound human" | 50 % | not measured* |
| human essays polished by GPT-4 | 87 % | 21 % |

\* sending the eval texts to zerogpt.com would publish them; the wasitaigenerated benchmark (Sept 2026) reports
ZeroGPT catching 21 % of raw frontier-model text.

Other engines tried (AUROC, human vs raw AI): tropa-mini 0.990 · desklib 0.890 · fakespot 0.764 · TMR 0.479 ·
Binoculars (Qwen2.5-1.5B pair) 0.597 · Fast-DetectGPT 0.557; a learned combination of all of them (0.974) did worse
than tropa-mini alone, so it runs alone.

Read the numbers this way: a high score on your own writing is likelier if it is formal and non-native, or has been
through an AI polish; rewriting with an LLM gets past it about half the time, rewriting by hand is what moves the
sentence highlights.

## Files

```
detector/engines.py   classifiers (+ the zero-shot engines used only in eval)
detector/score.py     thresholds, sentence windows, verdict text, hints
detector/server.py    FastAPI app, GPU-lock handling
detector/gpu.py       the shared GPU lock (non-blocking take, waiters from /proc/locks)
detector/static/index.html   the page
eval/                 gen_ai.py, gen_humanized.py (via CLIProxyAPI), run_eval.py, analyze.py, tune_windows.py
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
