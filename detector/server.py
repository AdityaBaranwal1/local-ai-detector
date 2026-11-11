"""AI Detector web service: POST /api/detect {text} -> score + per-sentence highlights; GET / = the page.

Run: .venv/bin/uvicorn detector.server:app --host 127.0.0.1 --port 8323
Nothing is logged or stored: texts live only for the duration of a request.
"""
from __future__ import annotations

import copy
import logging
import os
import pathlib
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .engines import CLASSIFIERS
from .gpu import GpuLock
from .score import analyse

log = logging.getLogger("ai-detector")
STATIC = pathlib.Path(__file__).parent / "static"
GPU_IDLE_S = float(os.environ.get("GPU_IDLE_S", 60))     # keep the GPU (and its lock) this long after a check
RAM_IDLE_S = float(os.environ.get("RAM_IDLE_S", 1800))   # drop the model from RAM after this long unused
MAX_WORDS = 6000


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
            self.cpu = CLASSIFIERS["tropa"]().load("cpu")
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
    return {"ok": True, "engine": "tropa-mini (DeBERTa-v3-large)", **ENGINE.state()}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})
