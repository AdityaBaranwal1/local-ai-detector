# Training

## Data

| Script | What it builds |
|---|---|
| `train/build_data.py` | public human/AI corpora → `train/data/{train,val}.jsonl` and `test/heldout.jsonl` (Jabarian 20 %, ELLIPSE/ICNALE 15 %, the textsight 2026 benchmark — never trained on) |
| `train/gen_train.py ai\|humanize\|polish` | new AI texts through an OpenAI-compatible proxy: 700 in the target genres and styles (plain, casual, lowercase, non-native, terse, "make it sound human"), 350 humanizer rewrites, 300 AI-polished learner essays |
| `train/build_more.py` | domain-balanced MAGE / COLING-2025 / HC3 → `train/data/extra.jsonl` |
| `train/build_ood.py` | the out-of-distribution test set `test/ood.jsonl` |
| `train/package.py` | everything → `{train,val,extra,dev}.jsonl` for a training job (dev = 10 % of extra per domain) |

Every builder removes any text that appears in a test set (normalised-text hash). Labels follow tropa-mini's head:
0 human · 1 ai · 2 ai_edited · 3 humanized. The corpora themselves are not in the repo.

## Where it trains

`train/kaggle/train_kaggle.py` is one self-contained script for both places:

- **Kaggle** (free T4, fp16 + loss scaling): `train/package.py`, upload as a private dataset, `train/kaggle/push.sh`.
  Kaggle only gives GPUs and internet to phone-verified accounts — without that it silently runs on CPU.
- **Runpod** (pay per minute, bf16 on Ampere+): `train/runpod/run_job.py --data <dir> --out models/<name> [--base answerdotai/ModernBERT-large] --cloud SECURE --max-price 0.60 --gpu "NVIDIA RTX A6000"`
  - the API has no price cap, so the runner checks the created pod's `costPerHr` and deletes it within seconds if it's
    over the limit;
  - a detached watchdog deletes the pod after `--max-minutes` whatever happens; the pod is deleted as soon as the
    model is downloaded; every pod is recorded with its price and minutes in a local ledger;
  - the image's preinstalled packages are bypassed with a clean, pinned venv built by `uv` on the pod.

Options: `BASE` (tropa-mini or any backbone — a fresh 4-class head is added), `EPOCHS`, `LR`, `EVAL_EVERY`.
Selection score: 0.5 · (1 − worst-domain false-positive rate) + 0.5 · catch rate on the validation + dev sets.

## Evaluate

```bash
.venv/bin/python -m eval.suite models/<name> --name <run>        # held-out + OOD + ZeroGPT head-to-head
.venv/bin/python -m eval.bias v0 <run>                           # bias report, side by side
.venv/bin/python scripts/make_charts.py --models v0:"start" <run>:"<label>"
```
