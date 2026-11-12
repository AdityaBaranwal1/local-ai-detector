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
import json
import math
import os
import pathlib
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
            self.model = cls.from_pretrained(self.repo, dtype=torch.float32).eval()   # fp16 checkpoints too
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

    SURE_NOT_HUMAN = 0.01

    def p_ai(self, probs):
        # tropa-mini: P(ai), except when the model is sure the text is not human (P(human) < 1 %): then 1 - P(human),
        # so text it files under "humanized" still counts. Plain 1 - P(human) (the model card's score) flags polished
        # human writing (49 % of real college essays vs 3-4 %); plain P(ai) let half of LLM-"humanized" AI text
        # through (eval 2026-11-11: 50 % -> 72 % caught, native-English human false positives 1.7 % -> 1.9 %).
        if "human" in probs and probs["human"] < self.SURE_NOT_HUMAN:
            return 1.0 - probs["human"]
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

    def gpu_copy(self):
        """A copy of this classifier on the GPU (fp32: fp16 is ~7x slower on a GTX 16xx); the CPU master stays."""
        import copy as _copy
        g = _copy.copy(self)
        cls = MeanPoolHead if self.arch == "meanpool" else AutoModelForSequenceClassification
        try:      # straight from disk onto the GPU: no second CPU copy (an ensemble would hit the unit's MemoryMax)
            g.model = cls.from_pretrained(self.repo, dtype=torch.float32, device_map="cuda").eval()
        except (ValueError, ImportError):
            g.model = _copy.deepcopy(self.model).to("cuda").eval()
        g.device = "cuda"
        return g


class Ensemble:
    """Several classifiers whose class probabilities are averaged (equal weights unless given).
    Same interface as Classifier (score / score_many / p_ai / device), so verdicts and sentence drivers just work."""

    def __init__(self, members, weights=None):
        self.members = members
        self.weights = weights or [1.0 / len(members)] * len(members)
        self.device = members[0].device if members else "cpu"
        self.name = "+".join(m.name for m in members)

    def load(self, device):
        for m in self.members:
            m.load(device)
        self.device = device
        return self

    def _merge(self, dicts):
        keys = dicts[0].keys()
        return {k: sum(w * d[k] for w, d in zip(self.weights, dicts)) for k in keys}

    def p_ai(self, probs):
        return self.members[0].p_ai(probs)

    def score(self, text):
        parts = [m.score(text) for m in self.members]
        classes = self._merge([p["classes"] for p in parts])
        return {"p_ai": self.p_ai(classes), "classes": classes}

    def score_many(self, texts, batch=8):
        per = [[] for _ in texts]
        for m in self.members:
            for i in range(0, len(texts), batch):
                for j, pr in enumerate(m._probs(texts[i:i + batch])):
                    per[i + j].append(pr)
        return [self.p_ai(self._merge(ps)) for ps in per]

    def unload(self):
        for m in self.members:
            m.unload()

    def gpu_copy(self):
        e = Ensemble([m.gpu_copy() for m in self.members], self.weights)
        e.device = "cuda"
        return e


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class Stacked(Ensemble):
    """A learned combiner (eval/stack.py): logistic regression over each member's logit(1-P(human)), logit(P(ai)),
    logit(P(humanized)). Fitted on one hash half of the held-out sets, reported on the other. The class
    probabilities shown are the members' average; p_ai is the combiner's. Same interface as Classifier.
    length_aware (eval/stack_casual.py --length): the features also get lw = log(words/100) and lw x each feature, so
    the combiner weighs the members differently for a 40-word message than for a 400-word essay."""

    def __init__(self, members, coef, intercept, length_aware=False):
        super().__init__(members)
        self.coef, self.intercept, self.length_aware = list(coef), float(intercept), bool(length_aware)
        k = 3 * len(members)
        assert len(self.coef) == (2 * k + 1 if self.length_aware else k), "stacker / member count mismatch"

    @classmethod
    def from_file(cls, path, load_member):
        spec = json.loads(pathlib.Path(path).read_text())
        return cls([load_member(m) for m in spec["paths"]], spec["coef"], spec["intercept"], spec.get("length_aware", False))

    def _stack(self, per_member, n_words=100):
        x = []
        for p in per_member:
            x += [_logit(1 - p.get("human", 0.0)), _logit(p.get("ai", 0.0)), _logit(p.get("humanized", 0.0) + 1e-6)]
        if self.length_aware:                     # must match eval/stack_casual.py F()
            lw = math.log(max(n_words, 10) / 100)
            x = x + [lw] + [lw * v for v in x]
        z = self.intercept + sum(c * v for c, v in zip(self.coef, x))
        return 1 / (1 + math.exp(-max(min(z, 50), -50)))

    def score(self, text):
        parts = [m.score(text)["classes"] for m in self.members]
        return {"p_ai": self._stack(parts, len(text.split())), "classes": self._merge(parts)}

    def score_many(self, texts, batch=8):
        per = [[] for _ in texts]
        for m in self.members:
            for i in range(0, len(texts), batch):
                for j, pr in enumerate(m._probs(texts[i:i + batch])):
                    per[i + j].append(pr)
        return [self._stack(ps, len(t.split())) for ps, t in zip(per, texts)]

    def gpu_copy(self):
        e = Stacked([m.gpu_copy() for m in self.members], self.coef, self.intercept, self.length_aware)
        e.device = "cuda"
        return e


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
