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
                         # wilco's own error string when the case hit its `error` bucket (e.g. no JSON in the output), so a
                         # wiring fault and a bad answer can be told apart from the committed file alone (lane 01 lesson)
                         **({"wilco_error": str(graded["got"]["error"])[:300]}
                            if isinstance(graded.get("got"), dict) and "error" in graded["got"] else {}),
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


# ---- lane 02 on: several exams x N fresh loads, one process per load, rows written as they land

def _battery():
    return subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.strip().replace("\t", " ")


def sit_load(model_ref, plan, load_idx, partial_path, context=None):
    """One FRESH load (this process) sits every (exam, case) in plan; one JSON line per row, flushed as it lands.

    The last line is a footer {"_footer": ...}: a partial file without it is an interrupted load and is never assembled.
    """
    runner, router = import_wilco_runner()
    from .metering import MeteredMLX
    path, _ = resolve_model(model_ref)
    t_load = time.perf_counter()
    adapter = MeteredMLX(path, empty_answer_exc=router.TerminationError, context=context)
    load_s = time.perf_counter() - t_load
    batt0 = _battery()
    real_adapter_for = runner.adapter_for
    runner.adapter_for = lambda *a, **k: adapter
    n = 0
    try:
        with open(partial_path, "w") as fh:
            for exam, ids in plan:
                agent = runner.load_agent(exam)
                exam_doc = json.loads(C.cases_path(exam).read_text())
                by_id = {c["id"]: c for c in exam_doc["cases"]}
                for cid in ids:
                    n_before = len(adapter.calls)
                    adapter.current_case, adapter.current_exam = by_id[cid], exam   # read only on the lane-05 path
                    start = _now()
                    t0 = time.perf_counter()
                    res = runner.run_exam(agent, {**exam_doc, "cases": [by_id[cid]]}, provider="mlx", model=model_ref)
                    wall_s = time.perf_counter() - t0
                    graded = res["cases"][0]
                    row = {"exam": exam, "case": cid, "split": graded["split"], "load": load_idx,
                           "passed": bool(graded["passed"]), "failed_checks": graded.get("failed_checks", []),
                           **({"wilco_error": str(graded["got"]["error"])[:300]}
                              if isinstance(graded.get("got"), dict) and "error" in graded["got"] else {}),
                           "start_utc": start, "end_utc": _now(), "wall_s": wall_s,
                           "calls": adapter.calls[n_before:]}
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    n += 1
                    print(f"load {load_idx} {exam}/{cid}: {'PASS' if row['passed'] else 'FAIL'}  {wall_s:.1f}s  "
                          f"{len(row['calls'])} call(s)", file=sys.stderr, flush=True)
            fh.write(json.dumps({"_footer": {"load": load_idx, "load_s": load_s, "n_rows": n,
                                             "battery_start": batt0, "battery_end": _battery()}}) + "\n")
    finally:
        runner.adapter_for = real_adapter_for


def _read_partial(path):
    lines = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if not lines or "_footer" not in lines[-1]:
        return None, None
    return lines[:-1], lines[-1]["_footer"]


def run_loads(model_ref, name, plan, n_loads, prediction, prediction_commit, out_path, scratch, posted, context=None):
    """Orchestrate N fresh loads (one subprocess each; a load whose partial file is complete is reused, so an interrupted
    evening resumes at the next load), then write ONE results file. Never overwrites out_path (the CLI checks)."""
    import mlx.core as mx
    import mlx_lm
    scratch = Path(scratch); scratch.mkdir(parents=True, exist_ok=True)
    plan_arg = json.dumps(plan)
    rows, loads_meta = [], []
    for i in range(n_loads):
        part = scratch / f"{name}.load{i}.jsonl"
        got = _read_partial(part) if part.exists() else (None, None)
        if got[0] is None:
            cmd = [sys.executable, "-m", "lab", "sit", model_ref, "--plan", plan_arg, "--load", str(i), "--out", str(part)]
            if context is not None:
                cmd += ["--context", json.dumps(context)]
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                raise SystemExit(f"load {i} exited {rc}; partial kept at {part}")
            got = _read_partial(part)
            if got[0] is None:
                raise SystemExit(f"load {i}: no footer in {part}")
        rows += got[0]
        loads_meta.append(got[1])
    agg = results.derive(rows)
    _, snapshot = resolve_model(model_ref)
    exams = [e for e, _ in plan]
    doc = {
        "schema": results.SCHEMA, "run": name, "posted": posted,
        "prediction": f"predictions/{prediction}.md", "prediction_commit": prediction_commit,
        "lab_commit": _git("rev-parse", "HEAD"),
        "model": {"ref": model_ref, "snapshot": snapshot, "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
                  "loads": [{"load": m["load"], "load_s": m["load_s"], "n_rows": m["n_rows"]} for m in loads_meta],
                  "load_protocol": "each load is its own python process (fresh mlx-lm load, weights re-read from disk)"},
        "wilco": {"dir": str(C.WILCO_DIR), "pin": C.WILCO_PIN,
                  "cases_sha256": {e: C.EXAMS[e]["sha256"] for e in exams},
                  "grader": "sandbox/runner.py run_exam (per case), adapter_for -> lab.metering.MeteredMLX",
                  "snapshot_rule": "sandbox/runner.py aggregate_loads: majority verdict over loads, stable = all agree",
                  "agents": {e: {k: runner_agent.get(k) for k in ("temperature", "max_tokens")}
                             for e, runner_agent in ((e, import_wilco_runner()[0].load_agent(e)) for e in exams)}},
        "machine": {"model": _sysctl("hw.model"), "macos": platform.mac_ver()[0], "python": platform.python_version(),
                    "power": [{"load": m["load"], "battery_start": m["battery_start"], "battery_end": m["battery_end"]}
                              for m in loads_meta]},
        "exams": exams, "plan": plan,
        "metric": "heldout_majority_pass_rate", "headline": agg["heldout_majority_pass_rate"], "aggregates": agg,
        "energy": None, "energy_note": "no powermetrics capture in this run (needs the operator's sudo); Wh not metered",
        "rows": rows, "rows_sha256": results.digest(rows),
    }
    if context is not None:
        doc["context"] = _context_stamp(context, model_ref, rows)
    Path(out_path).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def _context_stamp(context, model_ref, rows):
    """Lane 05's stamp: what was requested AND what the runtime applied (cache classes read off the cache objects)."""
    import hashlib
    from . import context as X
    path, _ = resolve_model(model_ref)
    tmpl = Path(path) / "chat_template.jinja"
    tcfg = json.loads((Path(path) / "tokenizer_config.json").read_text())
    tsrc = tmpl.read_text() if tmpl.exists() else (tcfg.get("chat_template") or "")
    cfg = json.loads((Path(path) / "config.json").read_text())
    calls = [c for r in rows for c in r["calls"]]
    applied = sorted({json.dumps(c["context"]["cache_applied"], sort_keys=True) for c in calls if "context" in c})
    padded = [r for r in rows if r["calls"] and ((r["calls"][0].get("context") or {}).get("pad") or {}).get("pad_chars", 0) > 0]
    return {"requested": context,
            "applied_cache_classes": [json.loads(a) for a in applied],
            "prefill_step_size": 2048,
            "prefill_step_size_source": "mlx_lm.generate.generate_step default (stream_generate is passed none)",
            "padding_rule": {"header": X.HEADER, "ack": X.ACK, "sep": X.SEP, "slack_tokens": X.SLACK,
                             "pool": "sitting exam's train inputs, then the other exams' train inputs (sorted), minus the case and any train input containing its text"},
            "rows_padded": len(padded), "rows_total": len(rows),
            "quantization": cfg.get("quantization") or cfg.get("quantization_config"),
            "chat_template_sha256": hashlib.sha256(tsrc.encode()).hexdigest()}
