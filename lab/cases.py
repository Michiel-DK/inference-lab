"""The fixed case set: three wilco exams, pinned to one wilco commit and to the sha256 of each cases.json.

Nothing here reads a model. `verify_pin()` is what `lab run` calls before any case is graded: a moved wilco HEAD, a dirty
exam file or a changed case file is a refusal, never a warning. docs/CASES.md is the prose version of this table.
"""
import hashlib
import os
import subprocess
from pathlib import Path

WILCO_DIR = Path(os.environ.get("LAB_WILCO_DIR", Path.home() / "code" / "Michiel-DK" / "agent-exam-suite"))
WILCO_PIN = "62fcd449d5cb3ef7359a0a0262f0f403c929062d"

# sizes are what wilco's own cases.json states (counted from its "split" field at the pin), not chosen here
EXAMS = {
    "email-triage": {"mode": "labels", "train": 8, "heldout": 20,
                     "sha256": "955dd3000007048b6472b802a5b8825144bcafa63e584f3096df5a8939314bfd"},
    "call-fields": {"mode": "fields", "train": 13, "heldout": 19,
                    "sha256": "16ea5b0295a4ff2cf4911cbfa55a451014d99f6759875d1eb5098b717d4951af"},
    "recap": {"mode": "properties", "train": 7, "heldout": 16,
              "sha256": "315a6132da7f78c8f9e8a140ff711656f24b8ce12d893e8bc248ecd931aaf120"},
}

# The lane-01 smoke: two TRAIN cases of the shortest-input exam. Fixed here so a smoke cannot pick its cases after the fact.
SMOKE = {"exam": "email-triage", "cases": ["client-quote-request", "overdue-invoice"]}


def cases_path(exam, wilco_dir=None):
    return Path(wilco_dir or WILCO_DIR) / "evals" / exam / "cases.json"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pin(wilco_dir=None):
    """Problems with the wilco checkout, as a list of strings; empty means the pin holds."""
    d = Path(wilco_dir or WILCO_DIR)
    if not (d / ".git").exists():
        return [f"no wilco checkout at {d} (set LAB_WILCO_DIR)"]
    problems = []
    head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != WILCO_PIN:
        problems.append(f"wilco HEAD is {head or '?'}, pinned {WILCO_PIN}")
    for exam, meta in EXAMS.items():
        paths = [f"evals/{exam}", f"agents/{exam}"]
        dirty = subprocess.run(["git", "-C", str(d), "status", "--porcelain", "--", *paths],
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            problems.append(f"wilco {exam} has uncommitted edits: {dirty.splitlines()[0]}")
        p = cases_path(exam, d)
        if not p.exists():
            problems.append(f"missing {p}")
        elif sha256_file(p) != meta["sha256"]:
            problems.append(f"{p} sha256 {sha256_file(p)} != pinned {meta['sha256']}")
    return problems
