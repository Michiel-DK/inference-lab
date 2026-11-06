"""`lab run`: one model sits fixed wilco cases in-process under mlx-lm; wilco grades; lab/metering.py meters.

Grading is wilco's own code at the pin, imported read-only (no bytecode written into its tree): `sandbox/runner.py`
`load_agent` + `run_exam`, called once per case with wilco's `adapter_for` swapped for the metered mlx adapter. Prompt
assembly, JSON extraction, the re-ask, and the labels / fields / properties scoring are therefore exactly wilco's. wilco
at the pin has no `grade` subcommand (its CLI: new, validate, list, run, diff, pick, adopt, check, live); `run_exam` is
the function its `run` command scores with. None of the three exams has a rubric.md, so no judge is ever loaded.
"""
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import cases as C
from . import results


def _git(*args, cwd=None):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd).stdout.strip()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def import_wilco_runner(wilco_dir=None):
    sys.dont_write_bytecode = True                       # read-only: no __pycache__ in the pinned checkout
    sandbox = str(Path(wilco_dir or C.WILCO_DIR) / "sandbox")
    if sandbox not in sys.path:
        sys.path.insert(0, sandbox)
    import runner  # noqa: E402
    import router  # noqa: E402
    return runner, router


def resolve_model(ref):
    """(path, snapshot id) of a model already on disk. Never downloads (HF_HUB_OFFLINE)."""
    p = Path(ref).expanduser()
    if p.exists():
        return str(p), p.name
    from huggingface_hub import snapshot_download
    path = snapshot_download(ref, local_files_only=True)
    return path, Path(path).name


def run(model_ref, name, exam, case_ids, prediction_commit, out_path):
    runner, router = import_wilco_runner()
    import mlx.core as mx
    import mlx_lm
    from .metering import MeteredMLX

    path, snapshot = resolve_model(model_ref)
    t_load = time.perf_counter()
    adapter = MeteredMLX(path, empty_answer_exc=router.TerminationError)
    load_s = time.perf_counter() - t_load

    agent = runner.load_agent(exam)
    exam_doc = json.loads(C.cases_path(exam).read_text())
    by_id = {c["id"]: c for c in exam_doc["cases"]}
    missing = [i for i in case_ids if i not in by_id]
    if missing:
        raise SystemExit(f"refused: case ids not in {exam}: {missing}")

    rows = []
    real_adapter_for = runner.adapter_for
    runner.adapter_for = lambda *a, **k: adapter
    try:
        for cid in case_ids:
            case = by_id[cid]
            n_before = len(adapter.calls)
            start = _now()
            t0 = time.perf_counter()
            res = runner.run_exam(agent, {**exam_doc, "cases": [case]}, provider="mlx", model=model_ref)
            wall_s = time.perf_counter() - t0
            end = _now()
            graded = res["cases"][0]
            rows.append({"exam": exam, "case": cid, "split": graded["split"], "passed": bool(graded["passed"]),
                         "failed_checks": graded.get("failed_checks", []),
                         "start_utc": start, "end_utc": end, "wall_s": wall_s,
                         "calls": adapter.calls[n_before:]})
            print(f"{cid}: {'PASS' if graded['passed'] else 'FAIL'}  {wall_s:.1f}s  "
                  f"{len(adapter.calls) - n_before} call(s)", file=sys.stderr)
    finally:
        runner.adapter_for = real_adapter_for

    agg = results.derive(rows)
    doc = {
        "schema": results.SCHEMA, "run": name, "posted": False,
        "prediction_commit": prediction_commit, "lab_commit": _git("rev-parse", "HEAD"),
        "model": {"ref": model_ref, "snapshot": snapshot, "load_s": load_s,
                  "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__},
        "wilco": {"dir": str(C.WILCO_DIR), "pin": C.WILCO_PIN, "cases_sha256": C.EXAMS[exam]["sha256"],
                  "grader": "sandbox/runner.py run_exam (per case), adapter_for -> lab.metering.MeteredMLX",
                  "agent": {k: agent.get(k) for k in ("temperature", "max_tokens")}},
        "machine": {"model": _sysctl("hw.model"), "macos": platform.mac_ver()[0], "python": platform.python_version()},
        "exam": exam, "case_ids": case_ids,
        "metric": "pass_rate", "headline": agg["pass_rate"], "aggregates": agg,
        "energy": None, "energy_note": "no powermetrics capture in this run (needs the operator's sudo); Wh not metered",
        "rows": rows, "rows_sha256": results.digest(rows),
    }
    Path(out_path).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def _sysctl(key):
    return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()


def add_energy(src, capture, idle_start, idle_end, out_path):
    """A NEW results file: src plus an energy block integrated from an operator capture. src is never rewritten."""
    from .energy import energy_block
    d = results.load(src)
    d["energy"] = energy_block(Path(capture).read_bytes(), d["rows"], idle_start, idle_end)
    d.pop("energy_note", None)
    d["aggregates"] = results.derive(d["rows"], d["energy"])
    d["headline"] = d["aggregates"][d["metric"]]
    d["energy_source"] = {"from": str(src), "capture": str(capture)}
    Path(out_path).write_text(json.dumps(d, indent=1) + "\n")
    return d
