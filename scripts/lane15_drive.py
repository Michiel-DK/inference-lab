"""Lane 15 driver: start the Runpod worker fresh, time it stage by stage, append one JSON row per trial.

    HF_HUB_OFFLINE=1 LAB_WILCO_DIR=<pin clone> ~/.pyenv/versions/lewagon/bin/python3 scripts/lane15_drive.py --phase gate
    HF_HUB_OFFLINE=1 LAB_WILCO_DIR=<pin clone> ~/.pyenv/versions/lewagon/bin/python3 scripts/lane15_drive.py --phase trial

Design: predictions/cold-start-worker.md ("Pre-run notes"). Per worker: cat the pack's weight files (warm cache, timed), spawn
the worker in its own session, poll GET / every 20 ms until it answers (server accepting), POST /runsync twice, then SIGSTOP the
process group for 60 s, SIGCONT, POST /runsync a third time (arm C), kill the group. Order A, B, A, B, ... Rows are appended to
docs/probes/cold-start-worker/rows.jsonl; `--phase trial` refuses if trial rows already exist (committed evidence, never
regenerated). The swap stop fires if swap used grows more than 2048 MB above the block's first reading.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab import cases as C  # noqa: E402
from lab.run import resolve_model  # noqa: E402

VENV_PY = Path.home() / ".venvs" / "lab-inf-15" / "bin" / "python"
WORKER = ROOT / "scripts" / "lane15_worker.py"
ROWS = ROOT / "docs" / "probes" / "cold-start-worker" / "rows.jsonl"
LOGS = Path(os.environ.get("TMPDIR", "/tmp")) / "lane15-logs"
MODELS = {"A": "mlx-community/gemma-4-e4b-it-4bit", "B": "mlx-community/gemma-4-e4b-it-8bit"}
CASE_ID = "overdue-invoice"   # a TRAIN case; the second id of lab/cases.py: SMOKE
EXAM = "email-triage"
MAX_TOKENS = 64
POLL_S = 0.020
PAUSE_S = 60
SWAP_STOP_MB = 2048
TRIALS = 10


def swap_used_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", out).group(1))


def messages():
    """email-triage TRAIN case `overdue-invoice`, assembled as wilco's run_plain_case does (no library at the pin)."""
    problems = C.verify_pin()
    if problems:
        raise SystemExit(f"refused: wilco pin: {problems}")
    doc = json.loads(C.cases_path(EXAM).read_text())
    case = next(c for c in doc["cases"] if c["id"] == CASE_ID)
    if case.get("split") != "train":
        raise SystemExit(f"refused: {CASE_ID} is not a train case")
    system = (C.WILCO_DIR / "agents" / EXAM / "prompt.md").read_text()
    return [{"role": "system", "content": system}, {"role": "user", "content": case["input"]}]


def weight_files(path):
    return sorted(Path(path).glob("*.safetensors"))


def warm(path):
    files = weight_files(path)
    t0 = time.perf_counter()
    subprocess.run(["cat", *map(str, files)], stdout=subprocess.DEVNULL, check=True)
    return time.perf_counter() - t0, sum(f.stat().st_size for f in files), [f.name for f in files]


def http(method, url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def request(port, msgs):
    t_send = time.time()
    status, raw = http("POST", f"http://localhost:{port}/runsync", {"input": {"messages": msgs, "max_tokens": MAX_TOKENS}})
    t_recv = time.time()
    out = json.loads(raw)
    if out.get("status") != "COMPLETED":
        raise RuntimeError(f"runsync not completed: {out}")
    o = out["output"]
    text = o.pop("text")
    o["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    o["text_head"] = text[:120]
    return {"t_send": t_send, "t_recv": t_recv, "http_status": status, **o}


def kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGCONT)
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    try:   # anything left in the group (none expected: no heartbeat without RUNPOD_WEBHOOK_PING)
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def one_worker(cond, idx, phase, port, msgs, swap0):
    path, snapshot = resolve_model(MODELS[cond])
    cat_s, wbytes, wnames = warm(path)
    row = {"lane": "cold-start-worker", "phase": phase, "condition": cond, "trial": idx, "model": MODELS[cond],
           "snapshot": snapshot, "weight_files": wnames, "weight_bytes": wbytes, "cat_s": cat_s, "port": port,
           "poll_s": POLL_S, "pause_s": PAUSE_S, "max_tokens": MAX_TOKENS, "case": f"{EXAM}/{CASE_ID}",
           "swap_used_mb_block_start": swap0, "swap_used_mb_before": swap_used_mb()}
    env = {k: v for k, v in os.environ.items() if k != "RUNPOD_API_KEY"}
    env.update({"HF_HUB_OFFLINE": "1", "LANE15_MODEL": path, "PYTHONUNBUFFERED": "1"})
    LOGS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / f"{phase}-{cond}-{idx}.log", "w")
    t_spawn = time.time()
    env["LANE15_SPAWN_T"] = repr(t_spawn)
    proc = subprocess.Popen([str(VENV_PY), str(WORKER), "--rp_serve_api", "--rp_api_port", str(port)],
                            cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    row["t_spawn"] = t_spawn
    row["worker_pid"] = proc.pid
    try:
        polls = 0
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"worker exited {proc.returncode} before ready; log {log.name}")
            if time.time() - t_spawn > 300:
                raise RuntimeError("worker not ready within 300 s")
            polls += 1
            try:
                http("GET", f"http://localhost:{port}/", timeout=2)
                row["t_ready"] = time.time()
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(POLL_S)
        row["polls"] = polls
        row["req1"] = request(port, msgs)
        row["req2"] = request(port, msgs)
        row["t_stop"] = time.time()
        os.killpg(proc.pid, signal.SIGSTOP)
        row["swap_used_mb_paused"] = None
        time.sleep(PAUSE_S)
        row["swap_used_mb_paused"] = swap_used_mb()
        row["t_cont"] = time.time()
        os.killpg(proc.pid, signal.SIGCONT)
        row["req3"] = request(port, msgs)
        row["ok"] = True
    except Exception as exc:
        row["ok"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        kill_group(proc)
        log.close()
    row["swap_used_mb_after"] = swap_used_mb()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gate", "trial"], required=True)
    a = ap.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("refused: set HF_HUB_OFFLINE=1 (cached models only)")
    loaded = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.strip().splitlines()[1:]
    if loaded:
        raise SystemExit(f"refused: ollama has a model loaded: {loaded}")
    existing = [json.loads(line) for line in ROWS.read_text().splitlines()] if ROWS.exists() else []
    if a.phase == "trial" and any(r["phase"] == "trial" for r in existing):
        raise SystemExit("refused: trial rows already exist in rows.jsonl (never regenerated)")
    msgs = messages()
    plan = [("A", 0), ("B", 0)] if a.phase == "gate" else [(c, i) for i in range(TRIALS) for c in ("A", "B")]
    swap0 = swap_used_mb()
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    for n, (cond, idx) in enumerate(plan):
        row = one_worker(cond, idx, a.phase, 18100 + len(existing) + n, msgs, swap0)
        with open(ROWS, "a") as f:
            f.write(json.dumps(row) + "\n")
        r1 = row.get("req1", {})
        print(f"{a.phase} {cond}{idx} ok={row['ok']} cat={row['cat_s']:.2f}s "
              f"ready={row.get('t_ready', float('nan')) - row['t_spawn']:.2f}s "
              f"first_tok={r1.get('stamps', {}).get('first_token', [float('nan')])[0] - row['t_spawn']:.2f}s "
              f"peak={r1.get('peak_memory_gb', float('nan')):.2f}GB swap {row['swap_used_mb_before']:.0f}"
              f"->{row['swap_used_mb_after']:.0f}MB {row.get('error', '')}", flush=True)
        if not row["ok"]:
            raise SystemExit(2)
        if max(row["swap_used_mb_after"], row["swap_used_mb_paused"] or 0) - swap0 > SWAP_STOP_MB:
            raise SystemExit(f"swap stop: grew {row['swap_used_mb_after'] - swap0:.0f} MB above the block start")


if __name__ == "__main__":
    main()
