# How it works

- **Live engine: a learned combiner of two models.** tropa-mini (below) and our ModernBERT-large fine-tune
  (`models/rp-modernbert-v6`, trained on Runpod) each score the text; a logistic regression reads three log-odds per
  model (`1 − P(human)`, `P(ai)`, `P(humanized)`) and gives the AI score. Its coefficients are a small JSON file
  (`eval/stack.py --save`) that the service loads via `AI_STACKER`. Fitted on one hash half of the held-out texts;
  the numbers in the README are cross-fitted (`eval/stack_cv.py`). Both models stay in VRAM while the detector holds
  the GPU (fp32, ~3.3 GB): 1.3 s per 250 words. The sentence highlights below work the same way, on the combined score.

The single-model notes below describe tropa-mini on its own (and the service with `AI_MODEL` instead of `AI_STACKER`).

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
  `flock the render workers' lock file (`GPU_LOCK`, default `~/All code/shorts-studio/.gpu.lock`)`. The detector takes that same lock, non-blocking, for each check: free →
  fp16 copy in VRAM (~1.1 GB), checks take ~3–5 s; held by a render → CPU (~7 s for 250 words). It keeps the GPU 60 s
  after a check (`GPU_IDLE_S`) and gives it back within ~2 s as soon as anything waits on the lock (`/proc/locks`).
  It never creates the lock file (the workers bind-mount its inode). RAM copy dropped after 30 min idle (`RAM_IDLE_S`).
- **Limits:** English, 50+ words. 6000 words per check.
