import argparse
import subprocess
import sys
from pathlib import Path

from . import cases as C
from . import results


def _prediction_committed(name):
    """(ok, why, commit) — the run's prediction must exist, have both sections, be tracked, and be clean."""
    p = Path("predictions") / f"{name}.md"
    if not p.exists():
        return False, f"no predictions/{name}.md", None
    text = p.read_text()
    if "## Prediction" not in text or "## Mechanism" not in text:
        return False, f"predictions/{name}.md needs '## Prediction' and '## Mechanism'", None
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], capture_output=True).returncode != 0:
        return False, f"predictions/{name}.md is not tracked by git", None
    if subprocess.run(["git", "status", "--porcelain", str(p)], capture_output=True, text=True).stdout.strip():
        return False, f"predictions/{name}.md has uncommitted edits", None
    commit = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(p)], capture_output=True, text=True).stdout.strip()
    return True, "", commit


def main(argv=None):
    p = argparse.ArgumentParser(prog="lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="re-derive every posted aggregate from a committed results file; exit 1 on drift")
    c.add_argument("paths", nargs="+")
    d = sub.add_parser("diff", help="difference of two results files' aggregates (b - a)")
    d.add_argument("a"); d.add_argument("b")
    r = sub.add_parser("run", help="one model sits fixed wilco cases; refuses without a committed prediction")
    r.add_argument("model", help="mlx-lm model already on disk: HF id in the local cache, or a path")
    r.add_argument("--name", required=True)
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help=f"the fixed 2-case smoke: {C.SMOKE['exam']} {C.SMOKE['cases']}")
    g.add_argument("--exam", choices=sorted(C.EXAMS))
    g.add_argument("--exams", help="comma-separated exams or 'all': every case of each, sat on --loads fresh loads")
    r.add_argument("--cases", help="comma-separated case ids (with --exam); default: every case of the exam")
    r.add_argument("--prediction", help="prediction file stem when one prediction covers several runs (default: --name)")
    r.add_argument("--loads", type=int, default=3, help="fresh model loads with --exams (odd, >= 3: wilco's snapshot rule)")
    r.add_argument("--post", action="store_true", help="write posted: true (the lane posts this run)")
    r.add_argument("--scratch", default=".lab-scratch", help="per-load partial rows (gitignored); a complete load is reused")
    s = sub.add_parser("sit", help=argparse.SUPPRESS)        # internal: one fresh load, run by `lab run --exams`
    s.add_argument("model"); s.add_argument("--plan", required=True)
    s.add_argument("--load", type=int, required=True); s.add_argument("--out", required=True)
    e = sub.add_parser("energy", help="new results file = SRC + Wh integrated from an operator powermetrics capture")
    e.add_argument("src"); e.add_argument("--capture", required=True)
    e.add_argument("--idle-start", required=True); e.add_argument("--idle-end", required=True)
    e.add_argument("--out", required=True)
    a = p.parse_args(argv)

    if a.cmd == "check":
        rc = 0
        for path in a.paths:
            ok, msg = results.check(path)
            print(msg)
            rc = rc or (0 if ok else 1)
        return rc
    if a.cmd == "diff":
        ra, rb = results.load(a.a), results.load(a.b)
        for k, va in ra["aggregates"].items():
            vb = rb["aggregates"].get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                print(f"{k:22s} {vb - va:+.6f}")
            else:
                print(f"{k:22s} {va!r} -> {vb!r}")
        return 0
    if a.cmd == "energy":
        from .run import add_energy
        if Path(a.out).exists():
            print(f"refused: {a.out} exists; results are never overwritten", file=sys.stderr)
            return 2
        add_energy(a.src, a.capture, a.idle_start, a.idle_end, a.out)
        ok, msg = results.check(a.out)
        print(msg)
        return 0 if ok else 1
    if a.cmd == "sit":
        import json
        from .run import sit_load
        problems = C.verify_pin()
        if problems:
            print("refused: wilco pin does not hold:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 2
        sit_load(a.model, json.loads(a.plan), a.load, a.out)
        return 0
    if a.cmd == "run":
        ok, why, commit = _prediction_committed(a.prediction or a.name)
        if not ok:
            print(f"refused: {why}", file=sys.stderr)
            return 2
        problems = C.verify_pin()
        if problems:
            print("refused: wilco pin does not hold:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 2
        out = Path("results") / f"{a.name}.json"
        if out.exists():
            print(f"refused: {out} exists; results are never overwritten", file=sys.stderr)
            return 2
        if a.exams:
            import json
            exams = sorted(C.EXAMS) if a.exams == "all" else a.exams.split(",")
            unknown = [e for e in exams if e not in C.EXAMS]
            if unknown or a.loads < 3 or a.loads % 2 == 0:
                print(f"refused: unknown exams {unknown} or loads {a.loads} not odd >= 3", file=sys.stderr)
                return 2
            plan = [[e, [x["id"] for x in json.loads(C.cases_path(e).read_text())["cases"]]] for e in exams]
            from .run import run_loads
            run_loads(a.model, a.name, plan, a.loads, a.prediction or a.name, commit, out, a.scratch, a.post)
            ok, msg = results.check(str(out))
            print(msg)
            return 0 if ok else 1
        if a.smoke:
            exam, ids = C.SMOKE["exam"], list(C.SMOKE["cases"])
        else:
            import json
            exam = a.exam
            all_ids = [x["id"] for x in json.loads(C.cases_path(exam).read_text())["cases"]]
            ids = a.cases.split(",") if a.cases else all_ids
        from .run import run
        run(a.model, a.name, exam, ids, commit, out)
        ok, msg = results.check(str(out))
        print(msg)
        return 0 if ok else 1
    return 1
