"""Lane 21 (macos15-resit): the re-sit on the new OS against load 0 of lane 10, per case and per call (stdlib only).

  python3 scripts/lane21_compare.py            # writes docs/probes/macos15-resit/diff.txt and prints it
  python3 scripts/lane21_compare.py --check    # re-derives every posted count; exit 1 on a mismatch

Key per row: (exam, case). Baseline rows: load 0 of results/champion-e4b-8bit.json (macOS 14.3.1). New rows: the single
load of results/macos15-resit.json. --check also requires each posted value, as formatted in POSTED_TEXT, to appear in
the LESSONS.md row for macos15-resit."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE, NEW = "champion-e4b-8bit", "macos15-resit"
OUT = ROOT / "docs/probes/macos15-resit/diff.txt"

# Posted in LESSONS.md (filled after the run from this script's own output; --check re-derives each one).
POSTED = {
    "cases": 83,
    "verdicts_identical": 83,
    "verdicts_moved": 0,
    "gen_tokens_identical_cases": 83,
    "outputs_identical_cases": 83,
    "heldout_pass_base": 41,
    "heldout_pass_new": 41,
    "gen_tps_new": 29.2,
    "gen_tps_base_load0": 29.3,
    "peak_gb_new": 9.18,
    "peak_gb_base_load0": 9.18,
    "macos_base": "14.3.1",
    "macos_new": "27.2.1",
    "snapshot_same": True,
    "swap_growth_mb": 933,
}


def load(run, load_idx=0):
    d = json.loads((ROOT / "results" / f"{run}.json").read_text())
    rows = {(r["exam"], r["case"]): r for r in d["rows"] if r["load"] == load_idx}
    return d, rows


def tps(rows):
    calls = [c for r in rows.values() for c in r["calls"]]
    return sum(c["gen_tokens"] for c in calls) / sum(c["gen_tokens"] / c["gen_tps"] for c in calls if c["gen_tps"])


def swap_growth_mb():
    """The watchdog's line in docs/probes/macos15-resit/run.log (the full load's stderr, committed verbatim): peak minus
    base over the full load, in sysctl's M units. A safety reading; the results schema keeps no swap."""
    import re
    m = re.search(r"base (\d+) MB, peak (\d+) MB", (OUT.parent / "run.log").read_text())
    return int(m.group(2)) - int(m.group(1))


def derive():
    db, b = load(BASE)
    dn, n = load(NEW)
    assert set(b) == set(n), (sorted(set(b) ^ set(n)))
    keys = sorted(b)
    moved = [k for k in keys if b[k]["passed"] != n[k]["passed"]]
    tok_diff = [k for k in keys if [c["gen_tokens"] for c in b[k]["calls"]] != [c["gen_tokens"] for c in n[k]["calls"]]]
    shape = lambda r: (r["failed_checks"], r.get("wilco_error"),
                       [(c["gen_tokens"], c["content_chars"], c["reasoning_chars"], c["finish_reason"]) for c in r["calls"]])
    out_diff = [k for k in keys if shape(b[k]) != shape(n[k])]
    held = lambda rows: sum(r["passed"] for r in rows.values() if r["split"] != "train")
    got = {
        "cases": len(keys),
        "verdicts_identical": len(keys) - len(moved),
        "verdicts_moved": len(moved),
        "gen_tokens_identical_cases": len(keys) - len(tok_diff),
        "outputs_identical_cases": len(keys) - len(out_diff),
        "heldout_pass_base": held(b),
        "heldout_pass_new": held(n),
        "gen_tps_new": round(tps(n), 1),
        "gen_tps_base_load0": round(tps(b), 1),
        "peak_gb_new": round(max(c["peak_memory_gb"] for r in n.values() for c in r["calls"]), 2),
        "peak_gb_base_load0": round(max(c["peak_memory_gb"] for r in b.values() for c in r["calls"]), 2),
        "macos_base": db["machine"]["macos"],
        "macos_new": dn["machine"]["macos"],
        "snapshot_same": db["model"]["snapshot"] == dn["model"]["snapshot"],
        "swap_growth_mb": swap_growth_mb(),
    }
    lines = [f"{BASE} load 0 (macOS {got['macos_base']}) vs {NEW} (macOS {got['macos_new']})",
             f"snapshot {db['model']['snapshot']} vs {dn['model']['snapshot']} (same: {got['snapshot_same']})",
             f"mlx {db['model']['mlx']} / {dn['model']['mlx']}; mlx-lm {db['model']['mlx_lm']} / {dn['model']['mlx_lm']}", ""]
    lines += [f"{k}: {v}" for k, v in got.items()]
    lines += ["", "verdicts moved (exam/case: base -> new):"]
    lines += [f"  {e}/{c}: {b[(e, c)]['passed']} -> {n[(e, c)]['passed']}" for e, c in moved] or ["  none"]
    lines += ["", "failed_checks, wilco error or per-call (gen_tokens, content_chars, reasoning_chars, finish_reason) differ:"]
    lines += [f"  {e}/{c}" for e, c in out_diff] or ["  none"]
    lines += ["", "per-call gen_tokens differ (exam/case: base -> new):"]
    lines += [f"  {e}/{c}: {[x['gen_tokens'] for x in b[(e, c)]['calls']]} -> {[x['gen_tokens'] for x in n[(e, c)]['calls']]}"
              for e, c in tok_diff] or ["  none"]
    return got, "\n".join(lines) + "\n"


def main():
    got, text = derive()
    if "--check" not in sys.argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text)
        print(text, end="")
        return 0
    bad = [f"{k}: posted {POSTED[k]!r}, derived {got[k]!r}" for k in POSTED if POSTED[k] != got[k]]
    if OUT.read_text() != text:
        bad.append(f"{OUT.relative_to(ROOT)} differs from a fresh derivation")
    row = next((l for l in (ROOT / "LESSONS.md").read_text().splitlines() if l.startswith(f"| {NEW} ")), None)
    if row is None:
        bad.append("no LESSONS.md row starts with '| macos15-resit '")
    else:
        for k, s in POSTED_TEXT.items():
            if s.format(**got) not in row:
                bad.append(f"LESSONS row lacks {k}: {s.format(**got)!r}")
    for b_ in bad:
        print("MISMATCH", b_)
    print(f"lane21_compare --check: {len(POSTED)} posted values, {len(POSTED_TEXT)} LESSONS strings, "
          f"{'ok' if not bad else f'{len(bad)} mismatch(es)'}")
    return 1 if bad else 0


# How each posted value is written in the LESSONS row (filled after the run).
POSTED_TEXT = {
    "verdicts": "Verdicts {verdicts_identical} of {cases} identical ({verdicts_moved} moved)",
    "heldout": "held-out {heldout_pass_new}/55 on both",
    "tokens": "per-call `gen_tokens` identical on {gen_tokens_identical_cases} of {cases} cases",
    "outputs": "finish reasons identical on {outputs_identical_cases} of {cases}",
    "tps": "gen {gen_tps_new} tok/s (lane 10 load 0: {gen_tps_base_load0})",
    "peak": "peak {peak_gb_new} GB ({peak_gb_base_load0})",
    "os": "macOS {macos_base} to {macos_new}",
    "swap": "Swap +{swap_growth_mb} MB over the run",
}

if __name__ == "__main__":
    raise SystemExit(main())
