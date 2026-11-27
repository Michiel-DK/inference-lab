"""Lane 21 (macos15-resit): ONE fresh load of the champion over all 83 cases, through lab.run.run_loads.

  HF_HUB_OFFLINE=1 LAB_WILCO_DIR=<pin clone> python scripts/lane21_drive.py

`lab run --exams` refuses --loads < 3 (wilco's majority rule). The lane brief asks for load 0 only, so this applies the
CLI's own gates (committed prediction, verify_pin, never overwrite) and calls the same run_loads with n_loads=1,
context=None, spec=None: the `lab sit` child runs the default path lane 10 ran. A watchdog stops the run (the `lab sit`
child included) if swap grows by more than 2 GB over its value at start."""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import cases as C, results  # noqa: E402
from lab.cli import _prediction_committed  # noqa: E402

MODEL, NAME, PRED = "mlx-community/gemma-4-e4b-it-8bit", "macos15-resit", "macos15-resit"
SWAP_STOP_MB = 2048.0
SCRATCH = ".lab-scratch"


def swap_used_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", out).group(1))


def watchdog(base, stop):
    peak = base
    while not stop.is_set():
        used = swap_used_mb()
        peak = max(peak, used)
        if used - base > SWAP_STOP_MB:
            print(f"SWAP STOP: used {used:.0f} MB, base {base:.0f} MB (+{used - base:.0f})", file=sys.stderr, flush=True)
            subprocess.run(["pkill", "-f", f"lab sit {MODEL}"])
            os.kill(os.getpid(), signal.SIGTERM)
            return
        stop.wait(15)
    print(f"swap: base {base:.0f} MB, peak {peak:.0f} MB (+{peak - base:.0f})", file=sys.stderr, flush=True)


def main():
    ok, why, commit = _prediction_committed(PRED)
    if not ok:
        sys.exit(f"refused: {why}")
    problems = C.verify_pin()
    if problems:
        sys.exit("refused: wilco pin does not hold:\n  " + "\n  ".join(problems))
    out = Path("results") / f"{NAME}.json"
    if out.exists():
        sys.exit(f"refused: {out} exists; results are never overwritten")
    plan = [[e, [x["id"] for x in json.loads(C.cases_path(e).read_text())["cases"]]] for e in sorted(C.EXAMS)]
    base = swap_used_mb()
    stop = threading.Event()
    t = threading.Thread(target=watchdog, args=(base, stop), daemon=True)
    t.start()
    from lab.run import run_loads
    try:
        run_loads(MODEL, NAME, plan, 1, PRED, commit, out, SCRATCH, True)
    finally:
        stop.set(); t.join(timeout=5)
    ok, msg = results.check(str(out))
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
