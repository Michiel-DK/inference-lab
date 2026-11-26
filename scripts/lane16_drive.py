"""Lane 16 drive: capture the prompts (no model), then every job of `lane16_batch.py plan` in its own fresh process.

Stops (operator rules in the brief): swap may not grow more than 2048 MB over its value at the start of the drive; the
running job's process group is killed and the drive exits 3. Gates run first and are read before the arms: a gate that
fails stops the drive (exit 2) so the fallback can be chosen by hand. A job whose raw file already has a footer is skipped,
so an interrupted evening resumes. Log lines go to docs/probes/batch-invariance/drive.log.
"""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "probes" / "batch-invariance"
RAW = OUT / "raw"
PY = sys.executable
SWAP_LIMIT_MB = 2048


def swap_used_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", s).group(1))


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "drive.log", "a") as fh:
        fh.write(line + "\n")


def done(path):
    if not path.exists():
        return False
    lines = path.read_text().splitlines()
    return bool(lines) and "_footer" in lines[-1]


def run_job(job, base_swap):
    out = RAW / f"{job}.jsonl"
    if done(out):
        log(f"{job}: kept (footer present)")
        return 0
    t0 = time.time()
    p = subprocess.Popen([PY, str(ROOT / "scripts" / "lane16_batch.py"), "gen", "--job", job, "--out", str(out)],
                         start_new_session=True)
    peak = [swap_used_mb()]
    breach = [False]

    def watch():
        while p.poll() is None:
            s = swap_used_mb()
            peak[0] = max(peak[0], s)
            if s - base_swap > SWAP_LIMIT_MB:
                breach[0] = True
                os.killpg(p.pid, signal.SIGKILL)
                return
            time.sleep(2)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    rc = p.wait()
    th.join()
    log(f"{job}: rc {rc}  {time.time() - t0:.1f}s  swap peak {peak[0]:.1f} MB (start of drive {base_swap:.1f}, "
        f"delta {peak[0] - base_swap:+.1f})")
    if breach[0]:
        log(f"STOP: swap grew past {SWAP_LIMIT_MB} MB over the drive's start")
        return 3
    return rc


# The swap budget is the LANE's, not the drive's: measured once, at the first drive start of the lane (drive-8bit.log,
# 2026-11-26T23:18:59, 2208.5 MB used). A re-started drive must not reset its own meter.
LANE_START_SWAP_MB = 2208.5


def main(which):
    RAW.mkdir(parents=True, exist_ok=True)
    base = LANE_START_SWAP_MB
    log(f"drive start ({which}): swap used {swap_used_mb():.1f} MB now; lane start {base:.1f} MB; stop at lane start + "
        f"{SWAP_LIMIT_MB} MB")
    if swap_used_mb() - base > SWAP_LIMIT_MB:
        log("STOP: swap already past the lane's budget")
        return 3
    if not (OUT / "prompts.jsonl").exists():
        subprocess.run([PY, str(ROOT / "scripts" / "lane16_batch.py"), "capture"], check=True)
    jobs = subprocess.run([PY, str(ROOT / "scripts" / "lane16_batch.py"), "plan"], capture_output=True, text=True,
                          check=True).stdout.split()
    if which == "gates":
        jobs = [j for j in jobs if j.startswith("gate-")]
    elif which == "arms":
        jobs = [j for j in jobs if not j.startswith("gate-")]
    for j in jobs:
        rc = run_job(j, base)
        if rc != 0:
            log(f"drive stopped at {j} (rc {rc})")
            return rc
    log("drive done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "all"))
