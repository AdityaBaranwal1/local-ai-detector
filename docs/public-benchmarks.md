# Public benchmarks

Independent checks on published detection benchmarks we did **not** train on (`eval/bench_public.py`; 2,913 texts,
26 dropped because they were already in our training data). They use 2023–2024 models, so they test breadth —
older models and evasion attacks — rather than the newest models (the independent review rounds cover those).

| benchmark | what | texts |
|---|---|---|
| [Ghostbuster](https://github.com/vivek3141/ghostbuster-data) (Verma et al. 2023) | student essays, Reuters news, WritingPrompts stories: human vs ChatGPT vs Claude (2023) | 1,259 |
| [DetectRL](https://github.com/NLP2CT/DetectRL) (NeurIPS 2024 D&B) | ChatGPT, Claude-instant, PaLM, Llama-2-70B on arXiv / XSum / stories / Yelp, plus paraphrase (DIPPER, LLM polish, back-translation) and perturbation attacks — test splits | 1,057 |
| [OUTFOX](https://github.com/ryuryukke/OUTFOX) (AAAI 2024) | student essays vs ChatGPT essays, incl. essays written to evade a detector — test split | 597 |

ZeroGPT was run on a stratified sample of 510 of the same texts (one request per ~4–5 s).

## Results (2026-11-12, flag = score ≥ 50 %)

```
== ghostbuster
    v0 alone (tropa-mini as published)     AUROC 0.995  acc  98.3%  humans flagged   4.2%  AI caught  99.4%  (n=1259)
    v0+v6 (2026-11-11 evening)             AUROC 0.983  acc  85.4%  humans flagged   1.2%  AI caught  79.0%  (n=1259)
    v6+v7 length-aware (this morning)      AUROC 0.970  acc  78.2%  humans flagged   1.5%  AI caught  68.6%  (n=1259)
    v0+v8 length-aware (live)              AUROC 0.992  acc  91.6%  humans flagged   2.2%  AI caught  88.6%  (n=1259)
   on the 72 texts ZeroGPT also judged:
    v0+v8 length-aware (live)              AUROC 0.998  acc  94.4%  humans flagged   0.0%  AI caught  91.7%  (n=72)
    ZeroGPT                                AUROC 0.998  acc  98.6%  humans flagged   4.2%  AI caught 100.0%  (n=72)

== detectrl
    v0 alone (tropa-mini as published)     AUROC 0.925  acc  84.6%  humans flagged   3.0%  AI caught  73.3%  (n=1057)
    v0+v6 (2026-11-11 evening)             AUROC 0.998  acc  78.7%  humans flagged   0.0%  AI caught  59.4%  (n=1057)
    v6+v7 length-aware (this morning)      AUROC 0.988  acc  76.5%  humans flagged   0.0%  AI caught  55.2%  (n=1057)
    v0+v8 length-aware (live)              AUROC 0.994  acc  81.5%  humans flagged   0.0%  AI caught  64.6%  (n=1057)
   on the 414 texts ZeroGPT also judged:
    v0+v8 length-aware (live)              AUROC 0.992  acc  81.6%  humans flagged   0.0%  AI caught  64.0%  (n=414)
    ZeroGPT                                AUROC 0.972  acc  96.4%  humans flagged   0.0%  AI caught  92.9%  (n=414)

== outfox
    v0 alone (tropa-mini as published)     AUROC 0.977  acc  94.1%  humans flagged   2.5%  AI caught  92.5%  (n=597)
    v0+v6 (2026-11-11 evening)             AUROC 0.999  acc  97.5%  humans flagged   1.5%  AI caught  97.0%  (n=597)
    v6+v7 length-aware (this morning)      AUROC 0.995  acc  94.8%  humans flagged   2.0%  AI caught  93.2%  (n=597)
    v0+v8 length-aware (live)              AUROC 1.000  acc  99.2%  humans flagged   1.0%  AI caught  99.2%  (n=597)
   on the 24 texts ZeroGPT also judged:
    v0+v8 length-aware (live)              AUROC 1.000  acc 100.0%  humans flagged   0.0%  AI caught 100.0%  (n=24)
    ZeroGPT                                AUROC 1.000  acc 100.0%  humans flagged   0.0%  AI caught 100.0%  (n=24)

== ALL
    v0 alone (tropa-mini as published)     AUROC 0.963  acc  92.4%  humans flagged   3.3%  AI caught  89.9%  (n=2913)
    v0+v6 (2026-11-11 evening)             AUROC 0.986  acc  85.4%  humans flagged   0.7%  AI caught  77.0%  (n=2913)
    v6+v7 length-aware (this morning)      AUROC 0.976  acc  81.0%  humans flagged   0.9%  AI caught  70.0%  (n=2913)
    v0+v8 length-aware (live)              AUROC 0.989  acc  89.5%  humans flagged   1.0%  AI caught  83.6%  (n=2913)
   on the 510 texts ZeroGPT also judged:
    v0+v8 length-aware (live)              AUROC 0.988  acc  84.3%  humans flagged   0.0%  AI caught  70.9%  (n=510)
    ZeroGPT                                AUROC 0.978  acc  96.9%  humans flagged   0.4%  AI caught  94.5%  (n=510)

```

## What it shows

- **Strong:** OUTFOX 99.2 % accurate, including **100 %** of the essays built to evade a detector; ChatGPT text caught
  100 % in all three Ghostbuster domains; AUROC ≥ 0.99 everywhere (the ranking is near-perfect).
- **Weak, and ZeroGPT isn't:** DetectRL's **Claude-instant** (40 % caught, ZeroGPT 100 %) and **PaLM** (59 % vs 83 %)
  — 2023 models with no text in our training data — and the **back-translation** attack (translate to another
  language and back: 13–33 % caught, ZeroGPT 67–100 %). Ghostbuster's 2023 Claude stories: 70 %.
- **A lower cutoff is not the fix:** even at a 10 % cutoff DetectRL catch only reaches 79 % (humans still 0 %).

Next (v9): training text from those families (the benchmarks' *training* splits — test splits stay untouched) and
back-translated text made locally with an open translation model (human texts too, so "translationese" can't become
an AI tell).
