"""Lane 02's row-level reads: every SHOWCASE number that is not a `lab check` aggregate. Stdlib, read-only.

  python3 scripts/lane02_rows.py [WILCO_PIN_CLONE]

Reads the committed results files (rows are covered by rows_sha256, so `lab check` vouches for them) and, if given, wilco's
snapshots at the pin for the rank-B champion figures.
"""
import collections
import json
import sys
from pathlib import Path

RUNS = ["dense-gemma-4-e2b-it-4bit", "dense-gemma-4-e4b-it-4bit", "dense-llama-3.1-8b-instruct-4bit"]

for run in RUNS:
    d = json.load(open(f"results/{run}.json"))
    rows, calls = d["rows"], [c for r in d["rows"] for c in r["calls"]]
    rch = sum(c.get("reasoning_chars", 0) for c in calls)
    share = rch / (rch + sum(c.get("content_chars", 0) for c in calls))
    print(f"{run}: thinking share of generated chars {share:.3f}; gen tokens per sitting "
          f"{d['aggregates']['gen_tokens_total'] / d['aggregates']['n_cases']:.1f}")
    for exam in d["exams"]:
        er = [r for r in rows if r["exam"] == exam]
        fc = collections.Counter(f["check"] for r in er if r["load"] == 0 and r["split"] == "heldout"
                                 for f in r["failed_checks"])
        fails = sorted({r["case"] for r in er if r["split"] == "heldout" and not r["passed"]})
        capped = sorted({(r["case"], r["split"]) for r in er if any(c["finish_reason"] == "length" for c in r["calls"])})
        print(f"  {exam}: held-out failed checks (load 0) {dict(fc)}; length-capped {capped}")
        print(f"    held-out fails: {fails}")

if len(sys.argv) > 1:
    for exam in ["email-triage", "call-fields", "recap"]:
        s = json.load(open(Path(sys.argv[1]) / "evals" / exam / "snapshot.json"))
        h = [c for c in s["cases"] if c["split"] == "heldout"]
        print(f"wilco champion {exam}: {s['provider']} {s['model']} {s['date'][:10]}, held-out "
              f"{sum(c['passed'] for c in h)}/{len(h)}, unstable {sum(not c.get('stable', True) for c in s['cases'])}")
