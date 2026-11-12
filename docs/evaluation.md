# Evaluation, by version

| Version | Report |
|---|---|
| v0 — tropa-mini as published | [evaluation-v0.md](evaluation-v0.md), independent review round 1: [REPORT.md](../eval/sol/round-1/REPORT.md) |
| k3 — first full fine-tune (narrow data) | [process.md § Round 2](process.md#round-2--the-first-fine-tune-and-the-bias-it-taught-us-k3) |
| v4 DeBERTa, mb1/mb5 ModernBERT | [process.md § Round 3](process.md#round-3--domain-balanced-data-and-a-second-architecture) |
| combiner tropa-mini + mb5 | [process.md § Round 4](process.md#round-4--a-learned-combiner-cross-fitted) |
| **combiner tropa-mini + v6 (live)** | [process.md § Round 5](process.md#round-5--matched-pairs-v6); independent review round 2: running |

Reproduce: `python -m eval.suite <model> --name <run>` (held-out, out-of-distribution, ZeroGPT head-to-head),
`python -m eval.bias <runs...>` (bias report) and `python -m eval.stack_cv v0 mb6 [--save-run cv_v0mb6]` (cross-fitted
combiner; `--save-run` writes its out-of-fold scores for the charts and the bias report). Test texts are rebuilt by `train/build_data.py` and
`train/build_ood.py`; they are not redistributed.
