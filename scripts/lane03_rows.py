"""Lane 03's row-level reads: every SHOWCASE number that is not a `lab check` aggregate. Stdlib, read-only.

  python3 scripts/lane03_rows.py RUN [RUN ...]       # results/<RUN>.json, the MoE runs and their dense comparators

Rows are covered by rows_sha256, so `lab check` vouches for what this reads. Per run: thinking share of generated chars,
generated tokens per sitting, length-capped calls, and per exam the held-out fails and failed checks (load 0). Per call:
the empty-answer count (no final channel / unterminated thought -> wilco's termination bucket).
"""
import collections
import json
import sys

SMOKE = ("email-triage", {"client-quote-request", "overdue-invoice"})   # lab/cases.py SMOKE

for run in sys.argv[1:]:
    d = json.load(open(f"results/{run}.json"))
    rows, calls = d["rows"], [c for r in d["rows"] for c in r["calls"]]
    rch = sum(c.get("reasoning_chars", 0) for c in calls)
    tot = rch + sum(c.get("content_chars", 0) for c in calls)
    a = d["aggregates"]
    print(f"{run}: thinking share of generated chars {rch / tot if tot else 0:.3f}; gen tokens per sitting "
          f"{a['gen_tokens_total'] / a['n_cases']:.1f}; calls {len(calls)}; length-capped calls "
          f"{sum(c['finish_reason'] == 'length' for c in calls)}; empty-answer calls "
          f"{sum(c.get('content_chars', 1) == 0 for c in calls)}")
    sc = [c for r in rows if r["exam"] == SMOKE[0] and r["case"] in SMOKE[1] for c in r["calls"]]
    if sc:   # token-weighted decode speed on the lane-03 gate smoke cases, every load: the like-for-like gpt-oss comparison
        tps_smoke = sum(c["gen_tokens"] for c in sc) / sum(c["gen_tokens"] / c["gen_tps"] for c in sc)
        print(f"  gen tok/s on the gate smoke cases {tps_smoke:.1f} over {len(sc)} calls")
    for exam in d.get("exams") or sorted({r["exam"] for r in rows}):
        er = [r for r in rows if r["exam"] == exam]
        fc = collections.Counter(f["check"] for r in er if r.get("load", 0) == 0 and r["split"] == "heldout"
                                 for f in r["failed_checks"])
        fails = sorted({r["case"] for r in er if r["split"] == "heldout" and not r["passed"]})
        capped = sorted({(r["case"], r["split"]) for r in er if any(c["finish_reason"] == "length" for c in r["calls"])})
        tps = sum(c["gen_tokens"] for r in er for c in r["calls"]) / len(er)
        print(f"  {exam}: gen tokens per sitting {tps:.1f}; held-out failed checks (load 0) {dict(fc)}; "
              f"length-capped {capped}")
        print(f"    held-out fails: {fails}")

# The gate's 2-case projection (supports the gate; the pre-registered formula also needed the call-fields case, which
# did not finish): gpt-oss smoke wall / e4b load-0 wall on the same two cases x e4b's whole 3 x 83 wall.
g = json.load(open("results/moe-gate-gpt-oss-20b-smoke.json"))
e = json.load(open("results/dense-gemma-4-e4b-it-4bit.json"))
gw = sum(r["wall_s"] for r in g["rows"])
ew = sum(r["wall_s"] for r in e["rows"] if r["exam"] == SMOKE[0] and r["case"] in SMOKE[1] and r["load"] == 0)
print(f"gate 2-case projection: gpt-oss {gw:.1f} s / e4b load-0 {ew:.1f} s x e4b 3x83 {e['aggregates']['wall_s_total']:.0f} s"
      f" = {gw / ew * e['aggregates']['wall_s_total'] / 60:.0f} min (threshold 300)")
