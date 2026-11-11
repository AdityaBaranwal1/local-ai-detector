"""Speed + resource benchmark of the live service (127.0.0.1:8323).

GPU runs: the service takes the free GPU lock itself. CPU runs: this script holds the GPU lock (like a render
would), so the service falls back to the CPU. Writes eval/BENCH.txt.
Usage: .venv/bin/python -m eval.bench   (needs the service up and no render running)
"""
import concurrent.futures as cf
import fcntl
import json
import os
import pathlib
import statistics as st
import subprocess
import threading
import time
import urllib.request

from detector.gpu import LOCK

URL = "http://127.0.0.1:8323"
D = pathlib.Path(__file__).parent / "data"
OUT = pathlib.Path(__file__).parent / "BENCH.txt"
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def post(text):
    req = urllib.request.Request(URL + "/api/detect", json.dumps({"text": text}).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return time.perf_counter() - t0, r


def health():
    return json.load(urllib.request.urlopen(URL + "/healthz"))


def service_mem():
    out = subprocess.run(["systemctl", "--user", "show", "ai-detector", "-p", "MemoryCurrent", "-p", "MainPID"],
                         capture_output=True, text=True).stdout
    kv = dict(l.split("=", 1) for l in out.split())
    return int(kv["MemoryCurrent"]) / 2**30, int(kv["MainPID"])


def vram(pid):
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return sum(int(m) for p, m in (l.split(", ") for l in out.strip().splitlines() if l) if int(p) == pid)


def text_of(words):
    pool = " \n\n".join(r["text"] for r in json.loads((D / "ai_gen.json").read_text()))
    w = pool.split(" ")
    return " ".join(w[:words])


class HoldGpu:
    """Hold the render lock like a render job would (blocking take)."""

    def __enter__(self):
        self.fd = os.open(LOCK, os.O_RDONLY)
        t0 = time.perf_counter()
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        self.waited = time.perf_counter() - t0
        return self

    def __exit__(self, *a):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def main():
    sizes = [100, 250, 500, 1000, 2000]
    texts = {n: text_of(n) for n in sizes}
    say(f"AI detector benchmark {time.strftime('%Y-%m-%d %H:%M')} · GTX 1660 Ti 6 GB / Ryzen 7 4800H (8 threads) · "
        f"tropa-mini DeBERTa-v3-large · page = whole-text score + one 150-word window per sentence")
    h = health()
    say(f"\nstate before: in RAM {h['in_ram']}, on GPU {h['on_gpu']}")

    # cold start: model loaded from the SSD page cache, then copied to the GPU
    dt, r = post(texts[250])
    say(f"cold start (first check, 250 words): {dt:.1f} s on {r['device']}")

    say("\nlatency per check (median of runs), by text length:")
    say(f"  {'words':>6} {'sentences':>9} {'GPU s':>7} {'CPU s':>7}  {'GPU words/s':>11}")
    gpu, cpu = {}, {}
    for n in sizes:
        runs = [post(texts[n]) for _ in range(5 if n <= 1000 else 3)]
        assert all(r["device"] == "cuda" for _, r in runs), "expected GPU runs"
        gpu[n] = (st.median(d for d, _ in runs), len(runs[0][1]["sentences"]))
    mem_gpu, pid = service_mem()
    vram_mb = vram(pid)
    with HoldGpu() as hg:
        waited_idle = hg.waited
        for n in sizes:
            runs = [post(texts[n]) for _ in range(3 if n <= 500 else 2)]
            assert all(r["device"] == "cpu" for _, r in runs), "expected CPU runs"
            cpu[n] = st.median(d for d, _ in runs)
    for n in sizes:
        say(f"  {n:6} {gpu[n][1]:9} {gpu[n][0]:7.1f} {cpu[n]:7.1f}  {n / gpu[n][0]:11.0f}")

    say(f"\nmemory: service RAM {mem_gpu:.2f} GiB (MemoryCurrent, incl. the CPU master copy); VRAM {vram_mb} MiB while it holds the GPU")

    # concurrency: requests are served one at a time (one model, one lock)
    post(texts[250])  # back on the GPU
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(4) as ex:
        lat = list(ex.map(lambda _: post(texts[250])[0], range(4)))
    say(f"4 checks of 250 words sent at once: all done in {time.perf_counter() - t0:.1f} s "
        f"(each waited {min(lat):.1f}-{max(lat):.1f} s; served one at a time)")

    # GPU hand-back: a render arriving while the detector idles on the GPU / while it is mid-check
    hb = []
    for _ in range(3):
        post(texts[100])
        with HoldGpu() as hg:
            hb.append(hg.waited)
    say(f"render arriving while the detector idles on the GPU: got the lock after {min(hb):.1f}-{max(hb):.1f} s (3 tries); "
        f"first CPU block above waited {waited_idle:.1f} s")
    post(texts[100])
    res = {}

    def long_check():
        res["d"] = post(texts[2000])[0]
    th = threading.Thread(target=long_check)
    th.start()
    time.sleep(0.5)
    with HoldGpu() as hg:
        mid = hg.waited
    th.join()
    say(f"render arriving mid-check (2000 words): got the lock after {mid:.1f} s (the check in flight finishes first)")
    h = health()
    say(f"state after: on GPU {h['on_gpu']} (gives the GPU back 60 s after the last check; RAM copy dropped after 30 min)")
    OUT.write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
