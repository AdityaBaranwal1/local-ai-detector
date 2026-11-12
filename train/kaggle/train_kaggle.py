"""Full fine-tune of tropa-mini on Kaggle (T4, fp16 tensor cores). Self-contained: runs as a Kaggle script kernel.

Input:  /kaggle/input/ai-detector-data/{train,val}.jsonl   (labels 0 human, 1 ai, 2 ai_edited, 3 humanized)
Output: /kaggle/working/model/{config.json, model.safetensors, tokenizer files, train_log.json}
        in tropa-mini's own format (keys model.* + classifier.*), so detector/engines.py MeanPoolHead loads it.
Built and pushed from the homelab: train/kaggle/README.md.
"""
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time

import torch as _t
_old_torch = tuple(int(x) for x in _t.__version__.split("+")[0].split(".")[:2]) < (2, 6)
# transformers >= 4.53 needs torch >= 2.6; ModernBERT needs >= 4.48 (Runpod's CUDA 12.4 image ships torch 2.4)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "safetensors>=0.4", "huggingface_hub", "scikit-learn",
                "sentencepiece", "protobuf", "transformers>=4.48,<4.53" if _old_torch else "transformers>=4.48,<6"], check=False)

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from sklearn.metrics import roc_auc_score
from transformers import AutoConfig, AutoModel, AutoTokenizer

TROPA = "wasitaigeneratedcom/tropa-mini-ai-text-detector"
BASE = os.environ.get("BASE", TROPA)        # or a plain backbone, e.g. answerdotai/ModernBERT-large (new 4-class head)
REV = "10fecd0856ed65ddfe92b88416b0a1e1350d8fb8" if BASE == TROPA else None
import glob
# Kaggle: dataset mounted somewhere under /kaggle/input (path varies by version); Runpod: DATA_DIR / OUT_DIR env
_found = sorted(glob.glob("/kaggle/input/**/train.jsonl", recursive=True))
INP = os.environ.get("DATA_DIR") or (os.path.dirname(_found[0]) if _found else "/kaggle/input/ai-detector-data")
OUT = os.environ.get("OUT_DIR", "/kaggle/working/model")
EPOCHS, BS, MAX_LEN, CROP_WORDS = float(os.environ.get("EPOCHS", 2.0)), 16, 256, 200
LR = float(os.environ.get("LR", 1.5e-5 if BASE == TROPA else 3e-5))   # a fresh backbone needs a higher LR
HEAD_LR, WD, WARM = 1e-4, 0.01, 0.06
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 400))
T0 = time.time()
LOG = []


def say(*a):
    s = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


class MeanPoolHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, getattr(config, "detector_num_labels", 1))

    def forward(self, input_ids, attention_mask):
        h = self.model(input_ids, attention_mask=attention_mask)[0]
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.classifier((h * m).sum(1) / m.sum(1).clamp(min=1e-9))


import re as _re
_DOMAIN = [(r"wikipedia|squad|wiki_csai|wikihow|/wiki", "encyclopedic"), (r"pmc|CS224N|sci_gen|arxiv|peerread|abstract|medicine|finance", "academic"),
           (r"news|xsum|cnn", "news"), (r"literature|novel|story|roct|/wp\b|wp_|writingprompts|hswag", "fiction"),
           (r"reddit|cmv|tldr|eli5|forum|post", "forum"), (r"open_qa|hc3|qa", "Q&A"), (r"email|enron", "email"),
           (r"review|yelp|amazon", "review"), (r"TOEFL|College|Hewlett|ellipse|icnale|essay|persuade|daigt", "essay")]


def domain_of(src):
    for rx, d in _DOMAIN:
        if _re.search(rx, src, _re.I):
            return d
    return "other"


def crop(text, rng):
    w = text.split()
    if len(w) <= CROP_WORDS:
        return text
    s = rng.randrange(0, len(w) - CROP_WORDS)
    return " ".join(w[s:s + CROP_WORDS])


def main():
    say("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    path = snapshot_download(BASE, revision=REV)
    cfg = AutoConfig.from_pretrained(path)
    tok = AutoTokenizer.from_pretrained(path)
    if BASE == TROPA:                                   # detector checkpoint: body + 4-class head
        model = MeanPoolHead(cfg)
        sd = load_file(os.path.join(path, "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        say("loaded base; missing", len(missing), "unexpected", len(unexpected), missing[:3], unexpected[:3])
        assert not [k for k in missing if not k.endswith("position_ids")], missing
    else:                                               # plain backbone: pretrained body, new 4-class head
        cfg.detector_num_labels = 4
        cfg.id2label = {0: "human", 1: "ai", 2: "ai_edited", 3: "humanized"}
        cfg.label2id = {v: k for k, v in cfg.id2label.items()}
        model = MeanPoolHead(cfg)
        model.model = AutoModel.from_pretrained(path, config=cfg)
        say("loaded backbone", BASE, "with a new 4-class head")
    model.cuda()
    model.model.gradient_checkpointing_enable()

    rows = [json.loads(l) for l in open(f"{INP}/train.jsonl")]
    if os.path.exists(f"{INP}/extra.jsonl"):        # domain-balanced MAGE / COLING / HC3 (train/build_more.py)
        extra = [json.loads(l) for l in open(f"{INP}/extra.jsonl")]
        rows += extra
        say(f"+ {len(extra)} domain-balanced extra rows")
    if os.path.exists(f"{INP}/dev.jsonl"):          # held-out slice of the extra data, per domain: model selection
        dev = [json.loads(l) for l in open(f"{INP}/dev.jsonl")]
        say(f"+ {len(dev)} dev rows for selection")
    val = [json.loads(l) for l in open(f"{INP}/val.jsonl")]
    if os.path.exists(f"{INP}/dev.jsonl"):
        val += [json.loads(l) for l in open(f"{INP}/dev.jsonl")]
    nh = sum(r["label"] == 0 for r in rows)
    w = np.array([(0.5 / nh if r["label"] == 0 else 0.5 / (len(rows) - nh)) * (2.0 if r["src"].startswith(("learner/", "gen/")) else 1.0) for r in rows])
    w /= w.sum()
    say(f"train {len(rows)} (human {nh}), val {len(val)}")

    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body_decay = [p for n, p in model.named_parameters() if not n.startswith("classifier") and p.ndim > 1]
    body_nodecay = [p for n, p in model.named_parameters() if not n.startswith("classifier") and p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": body_decay, "lr": LR, "weight_decay": WD},
                             {"params": body_nodecay, "lr": LR, "weight_decay": 0.0},
                             {"params": head, "lr": HEAD_LR, "weight_decay": 0.0}])
    steps = int(EPOCHS * len(rows) / BS)
    warm = int(WARM * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm)))
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    say("autocast dtype", amp)
    rng, nprng = random.Random(0), np.random.default_rng(0)

    @torch.inference_mode()
    def evaluate():
        model.eval()
        probs = []
        for i in range(0, len(val), 32):
            enc = tok([r["text"] for r in val[i:i + 32]], truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=amp):
                lg = model(enc["input_ids"], enc["attention_mask"]).float()
            probs += torch.softmax(lg, -1).cpu().tolist()
        model.train()
        y = np.array([int(r["label"] != 0) for r in val])
        out = {}
        for name, s in (("not_human", np.array([1 - p[0] for p in probs])),
                        ("hybrid", np.array([(1 - p[0]) if p[0] < 0.01 else p[1] for p in probs]))):
            fp = float(np.mean(s[y == 0] >= .5)); tp = float(np.mean(s[y == 1] >= .5))
            out[name] = {"auroc": round(roc_auc_score(y, s), 4), "fp": round(fp, 4), "catch": round(tp, 4), "bal_acc": round(0.5 * (1 - fp + tp), 4)}
        ln = np.array([r["src"].startswith("learner/") for r in val])
        out["learner_fp_not_human"] = round(float(np.mean(np.array([1 - p[0] for p in probs])[ln] >= .5)), 4) if ln.any() else None
        # worst-domain false-positive rate (domains with >= 15 human texts) -> selection score
        s_nh = np.array([1 - p[0] for p in probs])
        dom = np.array([domain_of(r["src"]) for r in val])
        fps = {d: float(np.mean(s_nh[(dom == d) & (y == 0)] >= .5)) for d in set(dom) if ((dom == d) & (y == 0)).sum() >= 15}
        worst_d = max(fps, key=fps.get) if fps else None
        out["domain_fp"] = {d: round(v, 3) for d, v in sorted(fps.items())}
        out["select"] = round(0.5 * (1 - (fps[worst_d] if worst_d else 0)) + 0.5 * out["not_human"]["catch"], 4)
        out["worst_domain"] = worst_d
        return out

    best, best_m = -1.0, None
    m = evaluate()
    say("before:", m)
    model.train()
    for step in range(steps):
        idx = nprng.choice(len(rows), size=BS, p=w)
        batch = [rows[i] for i in idx]
        enc = tok([crop(r["text"], rng) for r in batch], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        y = torch.tensor([r["label"] for r in batch], device="cuda")
        with torch.autocast("cuda", dtype=amp):
            lg = model(enc["input_ids"], enc["attention_mask"])
        lg = lg.float()
        loss = nn.functional.cross_entropy(lg, y)
        pb = torch.softmax(lg, -1)[:, 0].clamp(1e-6, 1 - 1e-6)                 # human vs anything-AI: the real decision
        loss = loss + nn.functional.binary_cross_entropy(pb, (y == 0).float())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 50 == 0:
            say(f"step {step}/{steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}")
        if (step + 1) % EVAL_EVERY == 0 or step == steps - 1:
            m = evaluate()
            score = m["select"]        # 0.5 (1 - worst-domain FPR) + 0.5 catch: an average can't hide one domain
            say(f"val @ {step + 1}: {m}")
            if score > best:
                best, best_m = score, m
                os.makedirs(OUT, exist_ok=True)
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, f"{OUT}/model.safetensors")
                say(f"  saved (best select {best}, worst domain {m['worst_domain']})")
    if BASE == TROPA:
        for f in os.listdir(path):
            if f.endswith((".json", ".model", ".txt")) and f != "model.safetensors":
                shutil.copy(os.path.join(path, f), OUT)
    else:
        cfg.architectures = ["DesklibAIDetectionModelV2"]   # same mean-pool + linear head as tropa (detector/engines.py)
        cfg.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    json.dump({"best": best_m, "steps": steps, "log": LOG}, open(f"{OUT}/train_log.json", "w"), indent=1)
    say("done; best", best_m)


if __name__ == "__main__":
    main()
