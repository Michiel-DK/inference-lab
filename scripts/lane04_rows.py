"""Lane 04's row-level reads: flips between precisions of one model, and the cost deltas. Stdlib, read-only.

  python3 scripts/lane04_rows.py LOW_RUN HIGH_RUN [HIGH_RUN ...]   # results/<RUN>.json; LOW is the 4-bit lane-02 file

Rows are covered by rows_sha256, so `lab check` vouches for what this reads. A flip is a case id whose majority verdict
differs between two precisions of the same model (predictions/quantisation.md addendum §5). Each flip is tagged
stable-in-both (all loads agree at each precision) or unstable. The shipping-rule trigger: a held-out case that is a
stable fail at LOW and a stable pass at HIGH.
"""
import collections
import json
import sys


def verdicts(run):
    d = json.load(open(f"results/{run}.json"))
    by = collections.defaultdict(list)
    split = {}
    for r in d["rows"]:
        by[(r["exam"], r["case"])].append(bool(r["passed"]))
        split[(r["exam"], r["case"])] = r["split"]
    out = {}
    for k, v in by.items():
        out[k] = (sum(v) * 2 > len(v), len(set(v)) == 1)   # (majority pass, stable)
    return d, out, split


low_run, highs = sys.argv[1], sys.argv[2:]
dl, vl, split = verdicts(low_run)
al = dl["aggregates"]
for hr in highs:
    dh, vh, _ = verdicts(hr)
    ah = dh["aggregates"]
    print(f"{low_run} -> {hr}")
    for exam in sorted({k[0] for k in vl}):
        ups = sorted(k[1] for k in vl if k[0] == exam and not vl[k][0] and vh[k][0])
        downs = sorted(k[1] for k in vl if k[0] == exam and vl[k][0] and not vh[k][0])
        tag = lambda c: f"{c}[{split[(exam, c)]}{'' if vl[(exam, c)][1] and vh[(exam, c)][1] else ',UNSTABLE'}]"
        trig = [c for c in ups if split[(exam, c)] == "heldout" and vl[(exam, c)][1] and vh[(exam, c)][1]]
        print(f"  {exam}: held-out {al[exam + '.heldout_majority_pass']} -> {ah[exam + '.heldout_majority_pass']}"
              f"/{ah[exam + '.heldout_n']}; up {len(ups)} {[tag(c) for c in ups]}; down {len(downs)} "
              f"{[tag(c) for c in downs]}; rule trigger (held-out, stable fail -> stable pass) {trig}")
        print(f"    min/1000 cases {al[exam + '.min_per_1000_cases']:.1f} -> {ah[exam + '.min_per_1000_cases']:.1f} "
              f"(x{ah[exam + '.min_per_1000_cases'] / al[exam + '.min_per_1000_cases']:.2f}); gen tokens per sitting "
              f"{sum(c['gen_tokens'] for r in dl['rows'] if r['exam'] == exam for c in r['calls']) / al[exam + '.n_cases'] / al['n_loads']:.1f}"
              f" -> {sum(c['gen_tokens'] for r in dh['rows'] if r['exam'] == exam for c in r['calls']) / ah[exam + '.n_cases'] / ah['n_loads']:.1f}")
    n_flip = sum(vl[k][0] != vh[k][0] for k in vl)
    print(f"  distinct case ids flipped {n_flip} of {len(vl)}; held-out total {al['heldout_majority_pass']} -> "
          f"{ah['heldout_majority_pass']}/55; unstable {al['n_unstable']} -> {ah['n_unstable']}")
    print(f"  gen tok/s {al['gen_tps']:.1f} -> {ah['gen_tps']:.1f} (x{ah['gen_tps'] / al['gen_tps']:.2f}); peak GB "
          f"{al['peak_memory_gb_max']:.2f} -> {ah['peak_memory_gb_max']:.2f} ({ah['peak_memory_gb_max'] - al['peak_memory_gb_max']:+.2f}); "
          f"wall 3x83 {al['wall_s_total'] / 60:.1f} -> {ah['wall_s_total'] / 60:.1f} min (x{ah['wall_s_total'] / al['wall_s_total']:.2f}); "
          f"gen tokens per sitting {al['gen_tokens_total'] / al['n_cases']:.1f} -> {ah['gen_tokens_total'] / ah['n_cases']:.1f}; "
          f"Wh/case {al['wh_per_case']} -> {ah['wh_per_case']}")
