# Evaluation, by version

| Version | Report |
|---|---|
| v0 — tropa-mini as published | [evaluation-v0.md](evaluation-v0.md), independent review round 1: [REPORT.md](../eval/sol/round-1/REPORT.md) |
| k3 — first full fine-tune (narrow data) | [process.md § Round 2](process.md#round-2--the-first-fine-tune-and-the-bias-it-taught-us-k3) |
| v4 / ModernBERT / ensemble | in progress |

Reproduce: `python -m eval.suite <model> --name <run>` (held-out, out-of-distribution, ZeroGPT head-to-head) and
`python -m eval.bias <runs...>` (bias report). Test texts are rebuilt by `train/build_data.py` and
`train/build_ood.py`; they are not redistributed.
