"""Lane calibrated-4bit: per-load speed and cross-load identity for arms A and B, and the per-case gate smokes of
A and C. Stdlib, read-only, committed results/*.json only (rows are covered by rows_sha256, so `lab check` vouches).

  python3 scripts/calib4_contention.py
"""
import collections
import json


def load(run):
    return json.load(open(f"results/{run}.json"))


for run in ("calib4-gemma-4-e4b-it-qat-mixed", "calib4-gemma-4-e4b-it-qat-4bit-requant"):
    d = load(run)
    print(run)
    for i in sorted({r["load"] for r in d["rows"]}):
        rows = [r for r in d["rows"] if r["load"] == i]
        calls = [c for r in rows for c in r["calls"]]
        tps = sum(c["gen_tokens"] for c in calls) / sum(c["gen_tokens"] / c["gen_tps"] for c in calls)
        print(f"  load {i}: gen {tps:.1f} tok/s; wall {sum(r['wall_s'] for r in rows):.0f} s; "
              f"{rows[0]['start_utc'][11:19]}-{rows[-1]['end_utc'][11:19]} UTC")
    by = collections.defaultdict(dict)
    for r in d["rows"]:
        by[(r["exam"], r["case"])][r["load"]] = (r["passed"], tuple(c["gen_tokens"] for c in r["calls"]))
    same_v = sum(len({v[0] for v in x.values()}) == 1 for x in by.values())
    same_t = sum(len({v[1] for v in x.values()}) == 1 for x in by.values())
    print(f"  cases {len(by)}: verdict identical across loads {same_v}; per-call gen_tokens identical {same_t}")

for run in ("calib4-gate-qat-mixed-smoke", "calib4-gate-qat-mixed-call",
            "calib4-gate-splice-mlp8-smoke", "calib4-gate-splice-mlp8-call"):
    for r in load(run)["rows"]:
        print(f"{run} {r['case']}: {r['start_utc'][11:19]}-{r['end_utc'][11:19]} UTC, wall {r['wall_s']:.1f} s, "
              f"gen tok/s {[round(c['gen_tps'], 2) for c in r['calls']]}, gen tokens {[c['gen_tokens'] for c in r['calls']]}")
