"""Results files: the derivation every posted number comes from, and `check`, which re-derives all of them. Stdlib only.

A results file stores RAW rows (per case: verdict from wilco, per-call meter readings, UTC window) and the aggregates it
posts. `check` recomputes every aggregate from the rows and fails on any drift, a tampered row (rows_sha256), or an energy
block whose arithmetic does not close. The capture itself is not committed (hundreds of MB); its integration is
re-derivable by `lab energy` from the capture, and its arithmetic from the block here.
"""
import hashlib
import json

SCHEMA = "inference-lab/results/v1"
TOL = 1e-9


def load(path):
    with open(path) as fh:
        return json.load(fh)


def digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _gen_seconds(call):
    return call["gen_tokens"] / call["gen_tps"] if call["gen_tps"] else 0.0


def derive(rows, energy=None):
    """Every posted aggregate, from the rows (and the energy block, if any). Deterministic."""
    if not rows:
        raise ValueError("no rows")
    calls = [c for r in rows for c in r["calls"]]
    gen_tokens = sum(c["gen_tokens"] for c in calls)
    gen_s = sum(_gen_seconds(c) for c in calls)
    prompt_tokens = sum(c["prompt_tokens"] for c in calls)
    prompt_s = sum(c["prompt_tokens"] / c["prompt_tps"] for c in calls if c["prompt_tps"])
    out = {
        "n_cases": len(rows),
        "n_calls": len(calls),
        "pass_rate": sum(bool(r["passed"]) for r in rows) / len(rows),
        "gen_tokens_total": gen_tokens,
        "gen_tps": gen_tokens / gen_s if gen_s else None,              # token-weighted over all calls
        "prompt_tps": prompt_tokens / prompt_s if prompt_s else None,
        "peak_memory_gb_max": max((c["peak_memory_gb"] for c in calls), default=None),
        "wall_s_total": sum(r["wall_s"] for r in rows),
        "wall_s_per_case": sum(r["wall_s"] for r in rows) / len(rows),
        "wh_per_case": None,                                            # None = not metered; never 0
    }
    if energy is not None:
        out["wh_per_case"] = energy["net_j"] / 3600.0 / len(rows)
    return out


def _energy_problems(e, rows):
    probs = []
    closed = e["gross_j"] - e["idle_mw"] / 1000.0 * e["window_s"]
    if abs(closed - e["net_j"]) > 1e-6:
        probs.append(f"energy net_j {e['net_j']} != gross - idle x window = {closed}")
    if e["n_cases"] != len(rows):
        probs.append(f"energy n_cases {e['n_cases']} != {len(rows)} rows")
    if e["coverage"] < 0.9:
        probs.append(f"energy coverage {e['coverage']} < 0.9")
    return probs


def _differs(a, b):
    if a is None or b is None:
        return a is not b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) > TOL * max(1.0, abs(float(b)))
    return a != b


def check(path):
    """(ok, message). Re-derives every aggregate and the headline; exit-1 material on any drift."""
    d = load(path)
    rows = d["rows"]
    if d.get("rows_sha256") != digest(rows):
        return False, f"{path}: rows_sha256 does not match the rows"
    energy = d.get("energy")
    if energy is not None:
        probs = _energy_problems(energy, rows)
        if probs:
            return False, f"{path}: " + "; ".join(probs)
    got = derive(rows, energy)
    posted = d["aggregates"]
    drift = [f"{k}: posted {posted.get(k)!r}, rows derive {got[k]!r}" for k in got if _differs(posted.get(k), got[k])]
    drift += [f"{k}: posted but not derivable" for k in posted if k not in got]
    if _differs(d["headline"], got[d["metric"]]):
        drift.append(f"headline: posted {d['headline']!r}, rows derive {got[d['metric']]!r}")
    if drift:
        return False, f"{path}: " + "; ".join(drift)
    status = "POSTED" if d.get("posted") else "not posted (wiring/smoke)"
    wh = "not metered" if got["wh_per_case"] is None else f"{got['wh_per_case']:.4f} Wh/case"
    gtps = "n/a" if got["gen_tps"] is None else f"{got['gen_tps']:.2f}"
    return True, (f"{path}: matches — {d['metric']} = {got[d['metric']]:.6f} over {got['n_cases']} cases "
                  f"({got['n_calls']} calls); gen {gtps} tok/s; peak {got['peak_memory_gb_max']:.3f} GB; "
                  f"wall {got['wall_s_total']:.2f} s; {wh}; {status}")
