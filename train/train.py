"""LoRA fine-tune of tropa-mini on train/data + train/raw/gen_*.jsonl. Holds the render GPU lock while it runs.

Usage: .venv/bin/python -m train.train --out models/tropa-ft-v1 [--epochs 2] [--max-len 320]
Writes the merged model (fp16 safetensors + config + tokenizer) to --out and a log with validation metrics.
"""
import argparse
import json
import math
import pathlib
import random
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from detector.engines import MeanPoolHead
from detector.gpu import GpuLock

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = "wasitaigeneratedcom/tropa-mini-ai-text-detector"
LABEL = {"human": 0, "ai": 1, "ai_edited": 2, "humanized": 3}


def load_rows():
    rows = [json.loads(l) for l in (ROOT / "train/data/train.jsonl").read_text().splitlines()]
    for f in sorted((ROOT / "train/raw").glob("gen_*.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            rows.append({"text": r["text"], "label": LABEL[r["label"]], "src": f"gen/{r['label']}/{r.get('task')}/{r.get('style')}"})
    val = [json.loads(l) for l in (ROOT / "train/data/val.jsonl").read_text().splitlines()]
    return rows, val


def weights(rows):
    """Half human / half AI per epoch; learner essays and the owner's genres (gen/*) count double."""
    w = []
    nh = sum(r["label"] == 0 for r in rows)
    na = len(rows) - nh
    for r in rows:
        x = (0.5 / nh) if r["label"] == 0 else (0.5 / na)
        if r["src"].startswith(("learner/", "gen/")):
            x *= 2.0
        w.append(x)
    return np.array(w) / sum(w)


def crop(text, rng, words=200):
    w = text.split()
    if len(w) <= words:
        return text
    s = rng.randrange(0, len(w) - words)
    return " ".join(w[s:s + words])


@torch.inference_mode()
def evaluate(model, tok, val, max_len, bs=16):
    model.eval()
    probs = []
    for i in range(0, len(val), bs):
        enc = tok([r["text"] for r in val[i:i + bs]], truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
        lg = model(**enc).float()
        probs += torch.softmax(lg, -1).cpu().tolist()
    model.train()
    y = [int(r["label"] != 0) for r in val]
    s = [1 - p[0] for p in probs]
    hum = [x for x, t in zip(s, y) if t == 0]
    ai = [x for x, t in zip(s, y) if t == 1]
    learner = [x for x, r in zip(s, val) if r["src"].startswith("learner/")]
    return {"auroc": roc_auc_score(y, s), "fp@0.5": float(np.mean([x >= .5 for x in hum])),
            "catch@0.5": float(np.mean([x >= .5 for x in ai])), "learner_fp@0.5": float(np.mean([x >= .5 for x in learner])) if learner else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test); 0 = full schedule")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--init", default=BASE, help="start from this model dir (to continue a previous run)")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    rows, val = load_rows()
    say(f"{time.strftime('%F %T')} train {len(rows)} rows, val {len(val)}; args {vars(args)}")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = MeanPoolHead.from_pretrained(args.init)
    lcfg = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05,
                      target_modules=["query_proj", "key_proj", "value_proj", "attention.output.dense", "intermediate.dense", "output.dense"],
                      modules_to_save=["classifier"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    lock = GpuLock()
    say("waiting for the GPU lock ...")
    lock.acquire()
    try:
        model.to("cuda")
        model.base_model.model.model.gradient_checkpointing_enable()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
        w = weights(rows)
        rng = random.Random(0)
        nprng = np.random.default_rng(0)
        steps = int(args.epochs * len(rows) / (args.bs * args.accum))
        warm = max(1, steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
        cw = torch.tensor([1.0, 1.0, 1.0, 1.0], device="cuda")
        if not args.no_save:
            say("before:", evaluate(model, tok, val, args.max_len))
        t0 = time.time()
        model.train()
        best = None
        for step in range(min(steps, args.max_steps) if args.max_steps else steps):
            for _ in range(args.accum):
                idx = nprng.choice(len(rows), size=args.bs, p=w)
                batch = [rows[i] for i in idx]
                enc = tok([crop(r["text"], rng) for r in batch], truncation=True, max_length=args.max_len,
                          padding=True, return_tensors="pt").to("cuda")
                y = torch.tensor([r["label"] for r in batch], device="cuda")
                lg = model(**enc)          # fp32: on this GTX 1660 Ti fp16 matmuls run ~7x slower (0.6 vs 4.4 TFLOPS)
                loss = torch.nn.functional.cross_entropy(lg.float(), y, weight=cw)
                # extra binary term: human vs anything-AI, the decision the detector actually makes
                pb = torch.softmax(lg.float(), -1)[:, 0].clamp(1e-6, 1 - 1e-6)
                loss = loss + torch.nn.functional.binary_cross_entropy(pb, (y == 0).float())
                (loss / args.accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            if step % 50 == 0:
                say(f"step {step}/{steps} loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
            if (step + 1) % 300 == 0 or step == steps - 1:
                m = evaluate(model, tok, val, args.max_len)
                say(f"val @ {step + 1}: {m}")
        if args.no_save:
            say(f"smoke test done, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
            return
        merged = model.merge_and_unload()
        merged.save_pretrained(out, safe_serialization=True)   # fp32: what the GPU runs fastest
        tok.save_pretrained(out)
        say(f"saved {out} in {time.time() - t0:.0f}s")
    finally:
        lock.release()


if __name__ == "__main__":
    main()
