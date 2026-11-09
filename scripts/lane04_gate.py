"""Lane 04's gate (predictions/quantisation.md addendum §4). Stdlib, read-only.

  python3 scripts/lane04_gate.py SMOKE_RUN CALL_RUN BASELINE_RUN [--minutes-left N]

SMOKE_RUN: the `--smoke` file (two email-triage train cases); CALL_RUN: call-fields `support-long-train`; BASELINE_RUN:
the lane-02 4-bit file of the same model. Projected 3 x 83 wall = (gate wall / the same three cases' load-0 wall in the
baseline) x the baseline's wall_s_total. Prints peak memory against mlx's 11.45 GB working set and, with
--minutes-left, the verdict. Exit 0 iff the gate passes (memory, and time when --minutes-left is given).
"""
import json
import sys

WORKING_SET_GB = 11.453251584   # mx.device_info()['max_recommended_working_set_size'] on this M1 Pro, 9 Nov 2026
args = sys.argv[1:]
left = None
if "--minutes-left" in args:
    i = args.index("--minutes-left"); left = float(args[i + 1]); del args[i:i + 2]
smoke, call, base = (json.load(open(f"results/{r}.json")) for r in args)
gate_rows = smoke["rows"] + call["rows"]
keys = {(r["exam"], r["case"]) for r in gate_rows}
gate_wall = sum(r["wall_s"] for r in gate_rows)
base_wall = sum(r["wall_s"] for r in base["rows"] if r["load"] == 0 and (r["exam"], r["case"]) in keys)
proj_min = gate_wall / base_wall * base["aggregates"]["wall_s_total"] / 60
peak = max(smoke["aggregates"]["peak_memory_gb_max"], call["aggregates"]["peak_memory_gb_max"])
calls = [c for r in gate_rows for c in r["calls"]]
tps = sum(c["gen_tokens"] for c in calls) / sum(c["gen_tokens"] / c["gen_tps"] for c in calls)
ok_mem = peak < WORKING_SET_GB
ok_time = left is None or proj_min <= left
print(f"gate cases {sorted(k[1] for k in keys)}: wall {gate_wall:.1f} s vs 4-bit {base_wall:.1f} s "
      f"(x{gate_wall / base_wall:.2f}); projected 3x83 {proj_min:.0f} min"
      + (f" vs {left:.0f} left" if left is not None else "")
      + f"; peak {peak:.2f} GB vs {WORKING_SET_GB:.2f}; gen tok/s {tps:.1f}; "
      f"passed {sum(r['passed'] for r in gate_rows)}/{len(gate_rows)}")
print("GATE", "PASS" if ok_mem and ok_time else "FAIL", "(memory)" if not ok_mem else "", "(time)" if not ok_time else "")
sys.exit(0 if ok_mem and ok_time else 1)
