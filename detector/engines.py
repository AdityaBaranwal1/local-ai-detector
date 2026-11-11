"""Local AI-text detection engines.

Two kinds:
  * classifiers fine-tuned to tell human from AI text (tropa-mini, desklib, tmr, fakespot)
  * zero-shot statistics from small language models (Binoculars, Fast-DetectGPT):
    AI text is unsurprising to a language model; human text is not.

Every engine returns P(AI) in [0, 1] (zero-shot ones return a raw statistic that
`calibrate.json` maps to a probability).
"""
from __future__ import annotations

import gc
import math
import os
import threading

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel

torch.set_num_threads(min(8, os.cpu_count() or 4))


class MeanPoolHead(PreTrainedModel):
    """DeBERTa + mean pooling + linear head (desklib's architecture; tropa-mini fine-tunes it to 4 classes)."""

    config_class = AutoConfig
    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, getattr(config, "detector_num_labels", 1))

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.model(input_ids, attention_mask=attention_mask)[0]
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.classifier((h * m).sum(1) / m.sum(1).clamp(min=1e-9))


def _chunks(tok, text: str, max_tokens: int):
    """Split a long text into <= max_tokens pieces on sentence boundaries; yields (piece, n_tokens)."""
    from .text import sentences

    cur, cur_n = [], 0
    for s in sentences(text):
        n = len(tok(s, add_special_tokens=False)["input_ids"])
        if cur and cur_n + n > max_tokens:
            yield " ".join(cur), cur_n
            cur, cur_n = [], 0
        cur.append(s)
        cur_n += n
    if cur:
        yield " ".join(cur), cur_n


class Classifier:
    kind = "classifier"

    def __init__(self, name, repo, arch, max_len=512, ai_label="ai"):
        self.name, self.repo, self.arch, self.max_len = name, repo, arch, max_len
        self.ai_label = ai_label
        self.model = self.tok = None
        self.device = "cpu"

    def load(self, device):
        if self.model is None:
            self.tok = AutoTokenizer.from_pretrained(self.repo)
            cls = MeanPoolHead if self.arch == "meanpool" else AutoModelForSequenceClassification
            self.model = cls.from_pretrained(self.repo).eval()
        if device != self.device:
            self.model.to(device)
            self.device = device
        return self

    @torch.inference_mode()
    def _probs(self, texts):
        enc = self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
        out = self.model(**enc)
        logits = (out if torch.is_tensor(out) else out.logits).float()
        if logits.shape[-1] == 1:
            p = torch.sigmoid(logits[:, 0])
            return [{"ai": float(x)} for x in p]
        sm = torch.softmax(logits, -1)
        labels = [self.model.config.id2label[i] for i in range(sm.shape[-1])]
        return [dict(zip([str(l).lower() for l in labels], map(float, row))) for row in sm]

    def p_ai(self, probs):
        # tropa-mini: P(ai) alone. Its model card's 1 - P(human) also counts "humanized" and "ai_edited", which
        # fire on polished human writing (eval: 49 % vs 3 % of real college essays flagged at the 50 % line)
        return probs.get(self.ai_label, 0.0)

    def score(self, text: str) -> dict:
        """Whole text: length-weighted mean over sentence-aligned chunks."""
        pieces = list(_chunks(self.tok, text, self.max_len - 16)) or [(text, 1)]
        probs = []
        for i in range(0, len(pieces), 8):
            probs += self._probs([p for p, _ in pieces[i:i + 8]])
        w = [n for _, n in pieces]
        tot = sum(w)
        merged = {k: sum(pr[k] * wi for pr, wi in zip(probs, w)) / tot for k in probs[0]}
        return {"p_ai": self.p_ai(merged), "classes": merged}

    def score_many(self, texts: list[str], batch=8) -> list[float]:
        """Short texts (sentence windows): one forward per batch, no chunking."""
        out = []
        for i in range(0, len(texts), batch):
            out += [self.p_ai(p) for p in self._probs(texts[i:i + batch])]
        return out

    def unload(self):
        self.model = None
        self.device = "cpu"


class ZeroShot:
    """Binoculars (Hans et al. 2024) and Fast-DetectGPT (Bao et al. 2024) from one observer/performer pair.

    binoculars = log-perplexity(performer) / cross-entropy(observer || performer); low = AI.
    fastdetect = analytic Fast-DetectGPT curvature with the observer as both sampler and scorer; high = AI.
    """

    kind = "zeroshot"

    def __init__(self, observer="Qwen/Qwen2.5-1.5B", performer="Qwen/Qwen2.5-1.5B-Instruct", max_len=512):
        self.observer_repo, self.performer_repo, self.max_len = observer, performer, max_len
        self.obs = self.perf = self.tok = None
        self.device = "cpu"

    def load(self, device):
        if self.obs is not None and device == self.device:
            return self
        self.unload()
        self.tok = AutoTokenizer.from_pretrained(self.observer_repo)
        if device == "cuda":
            # two 1.5B models in fp16 don't fit 6 GB together: observer fp16, performer int8
            from transformers import BitsAndBytesConfig

            self.obs = AutoModelForCausalLM.from_pretrained(self.observer_repo, dtype=torch.float16).to("cuda").eval()
            self.perf = AutoModelForCausalLM.from_pretrained(
                self.performer_repo, dtype=torch.float16, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map={"": 0}).eval()
        else:
            self.obs = AutoModelForCausalLM.from_pretrained(self.observer_repo, dtype=torch.bfloat16).eval()
            self.perf = AutoModelForCausalLM.from_pretrained(self.performer_repo, dtype=torch.bfloat16).eval()
        self.device = device
        return self

    @torch.inference_mode()
    def token_stats(self, text: str):
        """Per-token statistics for the whole text (first max_len tokens), with character offsets."""
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_len, return_offsets_mapping=True)
        ids = enc["input_ids"].to(self.device)
        offsets = enc["offset_mapping"][0, 1:].tolist()
        # hidden states once, then the 152k-wide vocabulary projection 128 positions at a time (VRAM)
        ho = self.obs.model(ids).last_hidden_state[0, :-1]
        hp = self.perf.model(ids).last_hidden_state[0, :-1]
        tgt = ids[0, 1:]
        cols = {k: [] for k in ("nll", "xent", "ll", "mu", "var")}
        for i in range(0, len(tgt), 128):
            t = tgt[i:i + 128, None]
            lso = torch.log_softmax(self.obs.lm_head(ho[i:i + 128]).float(), -1)
            lsp = torch.log_softmax(self.perf.lm_head(hp[i:i + 128].to(self.perf.lm_head.weight.dtype)).float(), -1)
            po = lso.exp()
            mu = (po * lso).sum(-1)
            cols["nll"].append(-lsp.gather(-1, t)[:, 0])           # performer surprise at the real token
            cols["xent"].append(-(po * lsp).sum(-1))               # expected performer surprise under observer
            cols["ll"].append(lso.gather(-1, t)[:, 0])
            cols["mu"].append(mu)
            cols["var"].append((po * lso.square()).sum(-1) - mu.square())
            del lso, lsp, po
        out = {k: torch.cat(v).cpu() for k, v in cols.items()}
        out["offsets"] = offsets
        return out

    @staticmethod
    def reduce(st, sl=slice(None)):
        nll, xent = st["nll"][sl], st["xent"][sl]
        if len(nll) == 0:
            return None
        bino = float(nll.mean() / xent.mean())
        fast = float((st["ll"][sl] - st["mu"][sl]).sum() / st["var"][sl].sum().clamp(min=1e-6).sqrt())
        return {"binoculars": bino, "fastdetect": fast, "n_tokens": len(nll)}

    def score(self, text: str) -> dict:
        st = self.token_stats(text)
        return {**self.reduce(st), "stats": st}

    def unload(self):
        self.obs = self.perf = None
        self.device = "cpu"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_lock = threading.Lock()

CLASSIFIERS = {
    "tropa": lambda: Classifier("tropa-mini", "wasitaigeneratedcom/tropa-mini-ai-text-detector", "meanpool", 512),
    "desklib": lambda: Classifier("desklib", "desklib/ai-text-detector-v1.01", "meanpool", 512),
    "tmr": lambda: Classifier("tmr", "Oxidane/tmr-ai-text-detector", "auto", 512),
    "fakespot": lambda: Classifier("fakespot", "fakespot-ai/roberta-base-ai-text-detection-v1", "auto", 512),
}
