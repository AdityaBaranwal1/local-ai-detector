# Files and setup


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
