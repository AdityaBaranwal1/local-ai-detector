"""AI Detector web service.

  POST /verify  {text}  -> PASS/FAIL verdict for agents (detector/verify.py; usage: GET /verify)
  POST /api/detect      -> score + per-sentence highlights for the page; GET / = the page

Run: .venv/bin/uvicorn detector.server:app --host 127.0.0.1 --port 8323
Nothing is logged or stored: texts live only for the duration of a request.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import pathlib
import threading
import time

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import tells
from . import verify as V
from .engines import Classifier
from .gpu import GpuLock
from .score import analyse

log = logging.getLogger("ai-detector")
STATIC = pathlib.Path(__file__).parent / "static"
GPU_IDLE_S = float(os.environ.get("GPU_IDLE_S", 60))     # keep the GPU (and its lock) this long after a check
RAM_IDLE_S = float(os.environ.get("RAM_IDLE_S", 1800))   # drop the model from RAM after this long unused
MAX_WORDS = 6000
# The model: an HF id (tropa-mini at the evaluated revision) or a fine-tuned dir under models/ (train/train.py).
MODEL = os.environ.get("AI_MODEL", "wasitaigeneratedcom/tropa-mini-ai-text-detector")
RULE = "hybrid"     # engines.Classifier.p_ai: P(ai), or 1 - P(human) when P(human) < 1 %


def model_version():
    p = pathlib.Path(MODEL)
    if p.is_dir():
        cfg = (p / "config.json").read_bytes() if (p / "config.json").exists() else b""
        w = p / "model.safetensors"
        stamp = f"{w.stat().st_size}-{int(w.stat().st_mtime)}" if w.exists() else "missing"
        return f"{p.name}@{stamp}+{RULE}"
    ref = pathlib.Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL.replace("/", "--")) / "refs/main"
    rev = ref.read_text().strip()[:8] if ref.exists() else "missing"
    return f"{MODEL.split('/')[-1]}@{rev}+{RULE}"


def model_present():
    p = pathlib.Path(MODEL)
    if p.is_dir():
        return (p / "model.safetensors").exists()
    return "missing" not in model_version()


class Engine:
    """The tropa-mini classifier: an fp32 master copy in RAM, an fp16 copy in VRAM only while we hold the GPU lock."""

    def __init__(self):
        self.mu = threading.Lock()
        self.lock = GpuLock()
        self.cpu = None
        self.gpu = None
        self.last = 0.0
        self.busy = False
        threading.Thread(target=self._reaper, daemon=True).start()

    def _cpu(self):
        if self.cpu is None:
            self.cpu = Classifier("detector", MODEL, "meanpool", 512).load("cpu")
        return self.cpu

    def _gpu(self):
        if not torch.cuda.is_available() or not self.lock.try_acquire():
            return None
        if self.gpu is None:
            try:
                g = copy.copy(self._cpu())
                g.model = copy.deepcopy(g.model).half().to("cuda").eval()
                g.device = "cuda"
                self.gpu = g
            except torch.OutOfMemoryError:
                self._drop_gpu()
                return None
        return self.gpu

    def _drop_gpu(self):
        self.gpu = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.lock.release()

    def run(self, text):
        with self.mu:
            self.busy = True
            try:
                eng = self._gpu() or self._cpu()
                try:
                    return analyse(eng, text), eng.device
                except torch.OutOfMemoryError:
                    self._drop_gpu()
                    return analyse(self._cpu(), text), "cpu"
            finally:
                self.busy = False
                self.last = time.time()

    def _reaper(self):
        while True:
            time.sleep(2)
            with self.mu:
                idle = time.time() - self.last
                # a render is waiting for the GPU, or we've been idle: give the GPU back
                if self.gpu is not None and (self.lock.waiters() > 0 or idle > GPU_IDLE_S):
                    self._drop_gpu()
                if self.cpu is not None and idle > RAM_IDLE_S:
                    self.cpu = None

    def state(self):
        return {"in_ram": self.cpu is not None, "on_gpu": self.gpu is not None, "gpu_lock_held": self.lock.held}


ENGINE = Engine()
app = FastAPI(title="AI Detector", docs_url=None, redoc_url=None)


class Req(BaseModel):
    text: str = Field(..., max_length=60000)


@app.post("/api/detect")
def detect(req: Req):
    text = req.text.replace("\r\n", "\n")
    n = len(text.split())
    if n == 0:
        raise HTTPException(400, "empty text")
    if n > MAX_WORDS:
        raise HTTPException(413, f"{n} words; the limit is {MAX_WORDS}: check it in parts")
    t0 = time.time()
    res, device = ENGINE.run(text)
    res.update(device=device, ms=int((time.time() - t0) * 1000))
    return res


@app.get("/healthz")
def healthz():
    ok_model = model_present()
    ok_split, ok_tells = V.split_ok(), V.tells_ok()
    st = ENGINE.state()
    steps = {
        "split": {"ok": ok_split, "version": "regex v1"},
        "model": {"ok": ok_model, "version": model_version(),
                  "where": "GPU" if st["on_gpu"] else "CPU (loaded)" if st["in_ram"] else "on disk, loads on use"},
        "drivers": {"ok": ok_model, "version": "leave-one-out <= 225 words, 150-word windows above"},
        "tells": {"ok": ok_tells, "version": f"v{tells.TELLS_VERSION}, {len(tells.RULES)} rules"},
    }
    ok = ok_model and ok_split and ok_tells
    body = {"ok": ok, "engine": model_version(), "steps": steps, "today": V.today_counts(), **st}
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/verify", response_class=PlainTextResponse)
def verify_usage():
    return V.USAGE


@app.post("/verify")
async def verify(request: Request, fresh: bool = False):
    ctype = request.headers.get("content-type", "")
    raw = await request.body()
    max_ai = V.DEFAULT_MAX_AI
    try:
        if "json" in ctype:
            body = json.loads(raw or b"{}")
            text = body.get("text", "")
            max_ai = float(body.get("max_ai", max_ai))
            fresh = fresh or bool(body.get("fresh"))
        else:
            text = raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as e:
        raise HTTPException(400, f"send JSON {{'text': ...}} or UTF-8 text: {e}")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "empty text")
    if not 0.0 < max_ai <= 1.0:
        raise HTTPException(400, "max_ai must be in (0, 1]")
    text = text.replace("\r\n", "\n")
    n = len(text.split())
    if n > MAX_WORDS:
        raise HTTPException(413, f"{n} words; the limit is {MAX_WORDS}: verify it in parts")
    sha = V.sha_of(text)
    vers = V.versions(model_version())
    if not fresh:
        hit = V.stored(sha, max_ai, vers)
        if hit:
            return {**V.with_quotes(hit, text), "from_store": True, "stored": True}
    t0 = time.time()
    try:
        analysis, device = ENGINE.run(text)
        hits = tells.find(text)
    except Exception as e:  # noqa: BLE001 - strict: a check that did not run is a FAIL, never a pass
        log.exception("verify failed")
        return {"verdict": "FAIL", "summary": f"FAIL: could not check ({type(e).__name__})", "problems": [],
                "complete": False, "stored": False, "sha256": sha, "versions": vers}
    result = V.build_verdict(text, analysis, hits, max_ai, vers, sha, time.time() - t0, device)
    V.store(sha, max_ai, vers, result)
    print(f"verify {sha[:12]} {result['verdict']} {round(result['ai_score'] * 100)}% {result['seconds']}s", flush=True)
    return {**V.with_quotes(result, text), "from_store": False, "stored": True}


@app.get("/verdict/{sha}")
def verdict(sha: str, max_ai: float = V.DEFAULT_MAX_AI):
    hit = V.stored(sha.lower(), max_ai, V.versions(model_version()))
    if not hit:
        raise HTTPException(404, "no verdict for that hash under the current versions")
    return {**hit, "from_store": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})
