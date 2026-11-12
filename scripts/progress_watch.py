"""Write data/progress.json every 30 s for the /progress page: one entry per Runpod training job + the plan.

Reads the runner logs (/tmp/rp_<name>.log: pod id, price, IP:port), tails each pod's train.log over SSH (step N/M,
latest validation), and adds cost so far and an ETA. The plan (stages and their state) is data/plan.json, edited by
hand / by the agent as work moves on. Never reads or writes any text being checked.
Usage: .venv/bin/python scripts/progress_watch.py   (runs until killed)
"""
import ast
import json
import os
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data/progress.json"
KEY = pathlib.Path.home() / ".ssh/runpod_ed25519"
LEDGER = ROOT / "train/runpod/ledger.jsonl"


def ssh(ip, port, cmd):
    r = subprocess.run(["ssh", "-i", str(KEY), "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                        "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=8", f"root@{ip}", cmd], capture_output=True, text=True, timeout=25)
    return r.stdout if r.returncode == 0 else None


def job(log):
    name = log.stem.replace("rp_", "")
    txt = log.read_text(errors="ignore")
    created = re.findall(r"pod (\w+) created: \$([\d.]+)/h on ([^(]+?) \(limit", txt)
    up = re.findall(r"pod up at ([\d.]+):(\d+)", txt)
    if not created:
        return None
    pid, price, gpu = created[-1]
    j = {"name": name, "pod": pid, "price": float(price), "gpu": gpu.replace("NVIDIA ", "").strip(), "state": "starting"}
    t0 = os.path.getmtime(log) if not txt else None
    # creation time = first ledger "create" event for this pod
    if LEDGER.exists():
        for l in LEDGER.read_text().splitlines():
            e = json.loads(l)
            if e.get("pod") == pid and e.get("event") == "create":
                j["created"] = time.mktime(time.strptime(e["ts"], "%Y-%m-%d %H:%M:%S"))
            if e.get("pod") == pid and e.get("event") == "delete":
                j["deleted"] = True
                j["minutes"] = e.get("minutes")
    if "downloaded to" in txt or j.get("deleted"):
        j["state"] = "done" if "downloaded to" in txt else "stopped"
    if "head-to-head" in txt:
        m = re.search(r"head-to-head.*accuracy ours ([\d.]+)% vs ZeroGPT ([\d.]+)%", txt)
        if m:
            j["h2h"] = [float(m.group(1)), float(m.group(2))]
    if up and j["state"] not in ("done", "stopped"):
        ip, port = up[-1]
        out = ssh(ip, port, "grep -E 'step [0-9]+/|val @|saved' /root/job/train.log | tail -40; "
                            "test -d /root/job/ev && echo __SCORING__; true")
        if out is not None:
            steps = re.findall(r"\[\s*(\d+)s\] step (\d+)/(\d+)", out)
            if steps:
                el, s, n = map(int, steps[-1])
                j.update(step=s, total=n, elapsed_s=el, state="training")
                if s > 0:
                    j["eta_s"] = int(el / s * (n - s) * 1.25)        # +25 % for validation passes
            vals = re.findall(r"val @ (\d+): (\{.*\})", out)
            if vals:
                try:
                    d = ast.literal_eval(re.sub(r"np\.(float64|str_)\(([^)]*)\)", r"\2", vals[-1][1]))
                    j["val"] = {"at": int(vals[-1][0]), "auroc": d["not_human"]["auroc"], "fp": d["not_human"]["fp"],
                                "catch": d["not_human"]["catch"], "worst_domain": d.get("worst_domain"),
                                "worst_fp": max((d.get("domain_fp") or {"-": 0}).values())}
                except (ValueError, SyntaxError, KeyError):
                    pass
            if "__SCORING__" in out:
                j["state"] = "scoring tests"
    if j.get("created"):
        mins = j.get("minutes") or (time.time() - j["created"]) / 60
        j["cost"] = round(j["price"] * mins / 60, 2)
    return j


def main():
    while True:
        jobs = []
        for log in sorted(pathlib.Path("/tmp").glob("rp_*.log"), key=os.path.getmtime, reverse=True)[:6]:
            try:
                j = job(log)
                if j:
                    jobs.append(j)
            except Exception as e:  # noqa: BLE001 - one bad log must not stop the page
                jobs.append({"name": log.stem, "state": f"watch error: {type(e).__name__}"})
        spent = 0.0
        if LEDGER.exists():
            for l in LEDGER.read_text().splitlines():
                spent += json.loads(l).get("est_cost", 0) or 0
        live = sum(j.get("cost", 0) for j in jobs if j.get("state") not in ("done", "stopped"))
        plan = json.loads((ROOT / "data/plan.json").read_text()) if (ROOT / "data/plan.json").exists() else {}
        OUT.parent.mkdir(exist_ok=True)
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps({"updated": time.time(), "jobs": jobs, "spent": round(spent + live, 2), "budget": 10.0, **plan}))
        tmp.replace(OUT)
        time.sleep(30)


if __name__ == "__main__":
    main()
