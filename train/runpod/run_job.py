"""Run train/kaggle/train_kaggle.py on one Runpod RTX 4090, with hard cost guards, and fetch the model back.

Guards (owner, 2026-11-11: "only 34 cents"):
  * request: RTX 4090, COMMUNITY cloud, on-demand, no persistent volume;
  * the create response's costPerHr is checked at once: above MAX_PRICE the pod is deleted within seconds and we stop;
  * a detached watchdog deletes the pod after MAX_MINUTES no matter what (crash, hang, lost connection);
  * the pod is deleted as soon as the model is downloaded.
The API key is read from ~/.config/runpod/api_key and never printed.

Usage: .venv/bin/python train/runpod/run_job.py --data <dir with train/val/extra.jsonl> --out models/<name> [--epochs 2]
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
KEY = (pathlib.Path.home() / ".config/runpod/api_key").read_text().strip()
SSH_KEY = pathlib.Path.home() / ".ssh/runpod_ed25519"
API = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"   # CUDA 12.4: runs on every host driver
MAX_PRICE = 0.34
LEDGER = ROOT / "train/runpod/ledger.jsonl"


def api(method, path, body=None):
    req = urllib.request.Request(API + path, json.dumps(body).encode() if body is not None else None,
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                  "User-Agent": "ai-detector-trainer/1.0 (+homelab)"}, method=method)   # default urllib UA = Cloudflare 1010
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:500]


def delete(pid, why):
    for _ in range(5):
        code, _ = api("DELETE", f"/pods/{pid}")
        if code in (200, 204, 404):
            print(f"pod {pid} deleted ({why})", flush=True)
            return True
        time.sleep(3)
    print(f"!! could not delete pod {pid}: delete it at https://console.runpod.io/pods", flush=True)
    return False


def ledger(**kw):
    with LEDGER.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%F %T"), **kw}) + "\n")


def ssh_args(ip, port):
    return ["-i", str(SSH_KEY), "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", default="2")
    ap.add_argument("--max-minutes", type=int, default=90)
    ap.add_argument("--gpu", action="append", help="gpuTypeIds in order of preference (default RTX 4090)")
    ap.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    ap.add_argument("--max-price", type=float, default=MAX_PRICE, help="hard $/h cap, checked on the created pod")
    ap.add_argument("--base", default="", help="backbone for train_kaggle.py (BASE env); default tropa-mini")
    ap.add_argument("--eval-name", default="", help="also score the test sets on the pod -> eval/runs/<name>.json")
    a = ap.parse_args()
    if not SSH_KEY.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "runpod-ai-detector", "-f", str(SSH_KEY)], check=True)
    pub = SSH_KEY.with_suffix(".pub").read_text().strip()

    body = {"name": "ai-detector-train", "imageName": IMAGE, "cloudType": a.cloud, "computeType": "GPU",
            "gpuTypeIds": a.gpu or ["NVIDIA GeForce RTX 4090"], "gpuTypePriority": "custom", "gpuCount": 1, "interruptible": False,
            "containerDiskInGb": 40, "volumeInGb": 0, "ports": ["22/tcp"], "supportPublicIp": True,
            "env": {"PUBLIC_KEY": pub}, "minRAMPerGPU": 16}
    code, pod = api("POST", "/pods", body)
    if code not in (200, 201) or not isinstance(pod, dict) or "id" not in pod:
        print("create failed:", code, pod)
        return 2
    pid = pod["id"]
    price = float(pod.get("costPerHr") or pod.get("adjustedCostPerHr") or 99)
    print(f"pod {pid} created: ${price}/h on {((pod.get('machine') or {}).get('gpuTypeId') or (pod.get('gpu') or {}).get('displayName'))} (limit ${a.max_price})", flush=True)
    ledger(event="create", pod=pid, price=price)
    if price > a.max_price + 1e-9:
        delete(pid, f"price ${price} > ${a.max_price}")
        ledger(event="refused", pod=pid, price=price)
        return 3
    # watchdog: survives this script; deletes the pod after max-minutes whatever happens
    subprocess.Popen(["bash", "-c", f'sleep {a.max_minutes * 60}; curl -s -X DELETE -H "Authorization: Bearer $(cat ~/.config/runpod/api_key)" '
                      f'{API}/pods/{pid} >/dev/null; echo "watchdog deleted {pid} at $(date)" >> "{ROOT}/train/runpod/watchdog.log"'],
                     start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        ip = port = None
        for _ in range(90):                                         # up to ~15 min for the image to start
            code, p = api("GET", f"/pods/{pid}")
            if isinstance(p, dict):
                ip = p.get("publicIp")
                port = (p.get("portMappings") or {}).get("22")
                if ip and port and p.get("desiredStatus") == "RUNNING":
                    break
            time.sleep(10)
        if not (ip and port):
            print("pod never became reachable", flush=True)
            return 4
        print(f"pod up at {ip}:{port} after {time.time() - t0:.0f}s", flush=True)
        S = ssh_args(ip, port)
        for _ in range(30):
            if subprocess.run(["ssh", *S, f"root@{ip}", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"]).returncode == 0:
                break
            time.sleep(10)
        subprocess.run(["ssh", *S, f"root@{ip}", "mkdir -p /root/job/data"], check=True)
        files = [str(pathlib.Path(a.data) / f) for f in ("train.jsonl", "val.jsonl", "extra.jsonl", "dev.jsonl") if (pathlib.Path(a.data) / f).exists()]
        subprocess.run(["scp", *[x.replace("-p", "-P") if x == "-p" else x for x in S], *files, f"root@{ip}:/root/job/data/"], check=True)
        subprocess.run(["scp", *[x.replace("-p", "-P") if x == "-p" else x for x in S], str(ROOT / "train/kaggle/train_kaggle.py"), f"root@{ip}:/root/job/"], check=True)
        base = f"BASE={a.base} " if a.base else ""
        # a clean, pinned env (the image's preinstalled packages clash with current transformers/numpy)
        boot = ("pip install -q uv >/dev/null 2>&1 && uv venv -q /root/venv --python 3.11 && "
                "uv pip install -q --python /root/venv/bin/python --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 && "
                "uv pip install -q --python /root/venv/bin/python 'transformers>=4.48,<6' safetensors huggingface_hub "
                "scikit-learn sentencepiece protobuf numpy")
        r = subprocess.run(["ssh", *S, f"root@{ip}", boot + " && /root/venv/bin/python -c 'import torch,transformers;print(torch.__version__, transformers.__version__, torch.cuda.is_available())'"],
                           capture_output=True, text=True)
        print("env:", (r.stdout.strip() or r.stderr.strip()[-300:]), flush=True)
        if r.returncode != 0 or not r.stdout.strip().endswith("True"):   # CUDA must work, or we'd bill a CPU run
            print("CUDA not available on this host: giving the pod back", flush=True)
            return 6
        run = (f"cd /root/job && {base}DATA_DIR=/root/job/data OUT_DIR=/root/job/model EPOCHS={a.epochs} "
               f"nohup /root/venv/bin/python train_kaggle.py > train.log 2>&1 & echo started")
        subprocess.run(["ssh", *S, f"root@{ip}", run], check=True)
        last = 0
        while True:                                                   # follow the log until done or dead
            time.sleep(30)
            r = subprocess.run(["ssh", *S, f"root@{ip}", f"tail -n +{last + 1} /root/job/train.log"], capture_output=True, text=True)
            if r.returncode != 0:
                continue
            new = r.stdout.splitlines()
            last += len(new)
            for l in new:
                if any(k in l for k in ("step ", "val @", "saved", "done;", "Traceback", "Error", "autocast", "extra", "train ")):
                    print(l, flush=True)
            tail = subprocess.run(["ssh", *S, f"root@{ip}", "tail -n 60 /root/job/train.log; pgrep -f train_kaggle.py >/dev/null && echo __ALIVE__"],
                                  capture_output=True, text=True).stdout
            if "done;" in tail:
                break
            if "__ALIVE__" not in tail:
                print("trainer exited without 'done'", flush=True)
                break
        if a.eval_name:                                              # score the test sets here (GPU) instead of on the laptop
            bundle = "/tmp/eval_bundle.tgz"
            files = ["detector", "eval/__init__.py", "eval/suite.py", "eval/scores.json", "eval/sites", "test"]
            subprocess.run(["tar", "czf", bundle, "-C", str(ROOT), "--exclude=__pycache__", "--exclude=*.sqlite", *files], check=True)
            subprocess.run(["scp", *[x.replace("-p", "-P") if x == "-p" else x for x in S], bundle, f"root@{ip}:/root/job/"], check=True)
            r = subprocess.run(["ssh", *S, f"root@{ip}", "cd /root/job && mkdir -p ev && tar xzf eval_bundle.tgz -C ev && cd ev && "
                                f"/root/venv/bin/python -m eval.suite /root/job/model --name {a.eval_name} --rule hybrid 2>&1 | "
                                "grep -E 'texts|group|^H |^AI|AUROC|head-to-head'"], capture_output=True, text=True)
            print(r.stdout[-2500:] or r.stderr[-800:], flush=True)
            subprocess.run(["scp", *[x.replace("-p", "-P") if x == "-p" else x for x in S],
                            f"root@{ip}:/root/job/ev/eval/runs/{a.eval_name}.json", str(ROOT / "eval/runs/")], check=False)
            subprocess.run(["ssh", *S, f"root@{ip}", "rm -rf /root/job/ev /root/job/eval_bundle.tgz"], check=False)
            pathlib.Path(bundle).unlink(missing_ok=True)
        out = ROOT / a.out
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", "-r", *[x.replace("-p", "-P") if x == "-p" else x for x in S], f"root@{ip}:/root/job/model/.", str(out)], check=False)
        subprocess.run(["scp", *[x.replace("-p", "-P") if x == "-p" else x for x in S], f"root@{ip}:/root/job/train.log", str(out)], check=False)
        print("downloaded to", out, flush=True)
        return 0 if (out / "model.safetensors").exists() else 5
    finally:
        mins = (time.time() - t0) / 60
        delete(pid, "job finished")
        ledger(event="delete", pod=pid, price=price, minutes=round(mins, 1), est_cost=round(price * mins / 60, 3))
        print(f"~{mins:.0f} min at ${price}/h = ~${price * mins / 60:.2f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
