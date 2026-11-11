"""Share the GPU with the render workers.

shorts-render and sim-engine run every GPU job under `flock ~/All code/shorts-studio/.gpu.lock`.
The detector takes the same lock (non-blocking) whenever it has models in VRAM, so:
  * a render that is running -> the detector uses the CPU instead;
  * the detector holding the GPU -> a render waits; `waiters()` lets the detector notice and give it back.
"""
from __future__ import annotations

import fcntl
import os
import pathlib

LOCK = pathlib.Path(os.environ.get("GPU_LOCK", pathlib.Path.home() / "All code/shorts-studio/.gpu.lock"))


class GpuLock:
    def __init__(self, path=LOCK):
        self.path = pathlib.Path(path)
        self.fd = None

    @property
    def held(self):
        return self.fd is not None

    def try_acquire(self) -> bool:
        if self.fd is not None:
            return True
        if not self.path.exists():
            return False  # never create it: the workers bind-mount this exact inode
        fd = os.open(self.path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self.fd = fd
        return True

    def acquire(self):
        """Blocking (offline evaluation only)."""
        fd = os.open(self.path, os.O_RDONLY)
        fcntl.flock(fd, fcntl.LOCK_EX)
        self.fd = fd

    def release(self):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None

    def waiters(self) -> int:
        """Processes blocked waiting for this lock (lines with '->' in /proc/locks)."""
        try:
            st = self.path.stat()
            want = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
            return sum(1 for ln in open("/proc/locks") if "->" in ln and want in ln.split())
        except OSError:
            return 0
