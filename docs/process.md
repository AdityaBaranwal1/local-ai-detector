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

## Round 3 — domain-balanced data (in progress)

- +13,855 texts from MAGE, COLING-2025 MGT and HC3 where **every domain has both human and AI text** (76
  domain×label groups), plus a 1,544-text dev split per domain.
- Checkpoints chosen by **0.5·(1 − worst-domain false-positive rate) + 0.5·catch rate**, so one bad domain can't
  hide behind a good average.
- In parallel, a second architecture (ModernBERT-large, 395M) on a rented GPU, as a possible ensemble partner.
- Runpod lessons: community GPUs were sold out all evening; requiring a CUDA ≥ 12.8 host driver hid every card;
  the CUDA 12.4 image needed a clean, pinned venv. Each failed attempt cost ~$0.02 thanks to the guards.

_Next: bias report + ZeroGPT head-to-head on v4 and the ensemble, then review round 2._
