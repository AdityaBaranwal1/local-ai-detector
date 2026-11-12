# The process

How this detector was built, in order — including what didn't work. Numbers are from the held-out tests, never from
training data. Updated each round.

## Round 0 — pick an engine (2026-11-11)

Six open detectors were scored on 743 texts (554 human, 66 fresh AI from 8 current models, 32 LLM-"humanized",
91 GPT-4-polished human essays):

| Engine | AUROC (human vs raw AI) |
|---|---|
| **tropa-mini** (DeBERTa-v3-large, 4 classes) | **0.990** |
| desklib v1.01 | 0.890 |
| fakespot RoBERTa | 0.764 |
| Binoculars on a Qwen2.5-1.5B pair | 0.597 |
| Fast-DetectGPT (same pair) | 0.557 |
| TMR RoBERTa | 0.479 |

A stacked combination did *worse* than tropa-mini alone, so it ran alone.

**Finding 1 — read the classes, not the card.** tropa-mini's suggested score `1 − P(human)` flagged 49 % of real
college essays (its "humanized" and "AI-edited" classes fire on polished human writing). Plain `P(ai)` flagged 3 %,
but let half of LLM-humanized text through. Compromise: `P(ai)`, or `1 − P(human)` when the model is ≥ 99 % sure
the text isn't human.

**Finding 2 — short texts need a different highlight.** For a 120-word email a 150-word window *is* the whole text,
so every sentence got the same score. Re-scoring the text once without each sentence shows which sentences drive
the verdict.

**Finding 3 — fp16 is slower on this GPU.** On a GTX 1660 Ti (no tensor cores) fp16 matmuls ran at 0.6 TFLOPS vs
4.4 in fp32. Switching the service to fp32 made checks ~3.6× faster.

## Round 1 — the first independent review

GPT-6.1 Sol built 120 fresh texts (pre-2022 human letters and learner essays from new sources; AI from several
current models, including casual, lowercase, non-native-imitating and humanized styles) and ran both detectors:
**ours 90.0 % vs ZeroGPT 99.2 %, score 9.0/10, "not at least as good as ZeroGPT".** Its three fixes: calibrate the
class mapping, cover learner-style and understated professional AI, and stop ordinary closers ("I look forward to
hearing from you") from failing human emails. The last one was fixed the same hour.

Also learned: ZeroGPT in 2026 is much better than in 2023 studies (Liang et al. measured 53 % false positives on
TOEFL essays; today it was 0 %), but it only caught 60 % of real StealthGPT-humanized text.

## Round 2 — the first fine-tune and the bias it taught us (k3)

Full fine-tune of tropa-mini on 18k texts (learner essays, Jabarian humanized pairs, emails, reddit, stories, 1,350
newly generated texts in email / application-answer genres) on a free Kaggle T4, ~1 h. Validation looked perfect
(AUROC 0.9998) — and the held-out bias report showed why validation isn't enough:

- humanized AI caught: 48 % → **97 %** (ZeroGPT 77 %), modern Claude/GPT/Gemini: ~75 % → **98 %**;
- but human **academic** writing flagged: 3 % → **48 %**, **encyclopedic** 19 % → 39 %.

The model had learned *formal = AI*, because every formal text in its training data was AI. **Not deployed.**

## Round 3 — domain-balanced data, and a second architecture

- +13,855 texts from MAGE, COLING-2025 MGT and HC3 where **every domain has both human and AI text** (76
  domain×label groups), plus a 1,544-text dev split per domain.
- Checkpoints chosen by **0.5·(1 − worst-domain false-positive rate) + 0.5·catch rate**, so one bad domain can't
  hide behind a good average.
- A second architecture, ModernBERT-large (395M), trained on Runpod (A40 / A6000, $0.59/h cap, scored on the pod).
- Runpod lessons: community GPUs were sold out all evening; requiring a CUDA ≥ 12.8 host driver hid every card;
  the CUDA 12.4 image needed a clean, pinned venv. Each failed attempt cost ~$0.02 thanks to the guards.

Result: better, still not good enough alone. v4 (DeBERTa) caught 98 % of AI text but still flagged 29 % of academic
writing; ModernBERT (mb1, mb5) flagged less but missed more. A plain average of two models didn't fix it either.

## Round 4 — a learned combiner, cross-fitted

Each model outputs four class probabilities (human / AI / AI-edited / humanized). A small logistic regression reads
three log-odds features per model and decides; human texts are weighted per domain so no domain dominates the fit.
To keep the numbers honest it is **cross-fitted**: split the 4,257 held-out texts in two by hash, fit on one half,
score the other, then swap. Every reported number is out-of-fold.

- tropa-mini + mb5: humans flagged 1.6 %, worst group 2.9 %, humanized AI 87.7 % (tropa-mini alone: 43 %) —
  278 vs ZeroGPT's 280 on the shared texts. Deployed (2 models, 1.3 s per 250 words on the 1660 Ti).
- The misses had one theme: **encyclopedic and clinical AI prose** ("Photosynthesis is the biological process…")
  from a 2026 benchmark, only 75–88 % caught. The training data had far more human Wikipedia/PubMed than AI text
  in that style, so the models had learned *encyclopedic = human*.
- A DeBERTa v5 flagged 21 % of native human text alone and added nothing to the combiner: dropped.

## Round 5 — matched pairs (v6)

`train/build_expo.py`: 900 real Wikipedia openings and 900 real PubMed abstracts (all pre-2019), and for **the same
topics** an AI-written opening/abstract from nine current models (markdown stripped so formatting is never the
tell). 3,576 texts, balanced per domain, deduplicated against every test set.

ModernBERT v6 on that data (43 min on an A40, $0.43):

| | alone | combined with tropa-mini (cross-fitted) |
|---|---|---|
| vs ZeroGPT on its 291 texts | 278 vs 280 | **281 vs 280** |
| human texts flagged | 3.6 % | **1.2 %** |
| worst human group | 6.9 % | **2.4 %** |
| humanized AI caught | 89 % | 88 % |
| encyclopedic/clinical AI caught | | **98 %** (was 88 %) |

All four parts of the bar met; deployed 2026-11-11 23:40. 281 vs 280 is parity rather than a clear win (one text),
which is exactly why the independent review builds its own fresh set.

Also running: open-weights AI text (Qwen, gpt-oss-20b, Nemotron, MiniMax, Gemma, …) from free API tiers, because the
combiner catches 98–100 % of Claude/GPT/Gemini text but far less from open-model families. Every model that
generates training text is listed in its row's source; none of the generated texts are published.

_Next: independent review round 2 (fresh, unseen texts + round 1's set again)._
