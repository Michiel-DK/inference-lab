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


def _peak(calls):
    """Max peak memory over calls that measured one (a hosted call has None: no local memory to read)."""
    vals = [c["peak_memory_gb"] for c in calls if c["peak_memory_gb"] is not None]
    return max(vals, default=None)


def call_cost(call, prices):
    """$ of one call: prompt tokens x input rate + completion tokens (thinking included) x output rate. None if unmetered."""
    if call.get("prompt_tokens") is None or call.get("gen_tokens") is None:
        return None
    return call["prompt_tokens"] * prices["input_usd_per_mtok"] / 1e6 + call["gen_tokens"] * prices["output_usd_per_mtok"] / 1e6


def _cost(calls, prices):
    costs = [call_cost(c, prices) for c in calls]
    return sum(x for x in costs if x is not None), sum(x is None for x in costs)


def derive(rows, energy=None, prices=None):
    """Every posted aggregate, from the rows (and the energy block, if any; and the price block, if any). Deterministic.

    `prices` (lane 09 on, hosted files only): {input_usd_per_mtok, output_usd_per_mtok}. With it, cost is derived from the
    rows' token counts; without it no cost key exists, so files written before lane 09 derive exactly what they posted.
    """
    if not rows:
        raise ValueError("no rows")
    calls = [c for r in rows for c in r["calls"]]
    gen_tokens = sum(c["gen_tokens"] or 0 for c in calls)
    gen_s = sum(_gen_seconds(c) for c in calls)
    prompt_tokens = sum(c["prompt_tokens"] or 0 for c in calls)
    prompt_s = sum(c["prompt_tokens"] / c["prompt_tps"] for c in calls if c["prompt_tps"])
    out = {
        "n_cases": len(rows),
        "n_calls": len(calls),
        "pass_rate": sum(bool(r["passed"]) for r in rows) / len(rows),
        "gen_tokens_total": gen_tokens,
        "gen_tps": gen_tokens / gen_s if gen_s else None,              # token-weighted over all calls
        "prompt_tps": prompt_tokens / prompt_s if prompt_s else None,
        "peak_memory_gb_max": _peak(calls),
        "wall_s_total": sum(r["wall_s"] for r in rows),
        "wall_s_per_case": sum(r["wall_s"] for r in rows) / len(rows),
        "wh_per_case": None,                                            # None = not metered; never 0
    }
    if energy is not None:
        out["wh_per_case"] = energy["net_j"] / 3600.0 / len(rows)
    if prices is not None:
        total, unmetered = _cost(calls, prices)
        out.update({"prompt_tokens_total": prompt_tokens, "cost_usd_total": total, "cost_usd_per_case": total / len(rows),
                    "unmetered_calls": unmetered})
    if "load" in rows[0]:
        out.update(_derive_loads(rows, prices))
    return out


def _derive_loads(rows, prices=None):
    """Multi-exam, multi-load files (lane 02 on): wilco's snapshot rule, per exam, plus per-exam cost.

    A row is one sitting of one case on one fresh model load. Per (exam, case) over the N loads, as wilco's
    `sandbox/runner.py: aggregate_loads` at the pin: `passed` = MAJORITY verdict (N odd), `stable` = every load agreed;
    the score is majority passes over ALL cases of the split (the denominator never moves). Beside it, the stricter count
    the lane brief names ("a case counts when all three agree"): stable passes only. Cost per exam is over every sitting:
    wall_s_per_case = mean wall of one sitting; min_per_1000_cases = that x 1000 / 60. Wh per 1,000 cases is None
    unless an energy block exists (per exam it is never apportioned here, so it stays None).
    """
    loads = sorted({r["load"] for r in rows})
    out = {"n_loads": len(loads)}
    tot = {"heldout_n": 0, "heldout_majority_pass": 0, "heldout_stable_pass": 0, "n_unstable": 0, "n_unique_cases": 0}
    for exam in sorted({r["exam"] for r in rows}):
        er = [r for r in rows if r["exam"] == exam]
        by_case = {}
        for r in er:
            by_case.setdefault(r["case"], []).append(r)
        bad = {c: len(v) for c, v in by_case.items() if sorted(x["load"] for x in v) != loads}
        if bad:
            raise ValueError(f"{exam}: cases not sat exactly once on every load {loads}: {bad}")
        n_unstable = 0
        split = {"train": [0, 0, 0], "heldout": [0, 0, 0]}          # n, majority passes, stable passes
        for cid, v in by_case.items():
            n_pass = sum(bool(x["passed"]) for x in v)
            majority = n_pass > len(v) - n_pass
            stable = n_pass in (0, len(v))
            n_unstable += 0 if stable else 1
            s = split[v[0]["split"]]
            s[0] += 1; s[1] += majority; s[2] += stable and majority
        calls = [c for r in er for c in r["calls"]]
        gt = sum(c["gen_tokens"] or 0 for c in calls)
        gs = sum(_gen_seconds(c) for c in calls)
        wall = sum(r["wall_s"] for r in er) / len(er)
        out.update({
            f"{exam}.n_cases": len(by_case),
            f"{exam}.train_n": split["train"][0], f"{exam}.train_majority_pass": split["train"][1],
            f"{exam}.heldout_n": split["heldout"][0], f"{exam}.heldout_majority_pass": split["heldout"][1],
            f"{exam}.heldout_stable_pass": split["heldout"][2],
            f"{exam}.all_majority_pass": split["train"][1] + split["heldout"][1],
            f"{exam}.n_unstable": n_unstable,
            f"{exam}.gen_tps": gt / gs if gs else None,
            f"{exam}.peak_memory_gb_max": _peak(calls),
            f"{exam}.wall_s_per_case": wall,
            f"{exam}.min_per_1000_cases": wall * 1000 / 60,
            f"{exam}.finish_length_calls": sum(c["finish_reason"] == "length" for c in calls),
            f"{exam}.wh_per_1000_cases": None,
        })
        if prices is not None:
            out[f"{exam}.cost_usd_per_case"] = _cost(calls, prices)[0] / len(er)
            out[f"{exam}.length_no_answer_cases"] = sum(
                1 for r in er if any(c["finish_reason"] == "length" for c in r["calls"]) and "wilco_error" in r)
        tot["heldout_n"] += split["heldout"][0]
        tot["heldout_majority_pass"] += split["heldout"][1]
        tot["heldout_stable_pass"] += split["heldout"][2]
        tot["n_unstable"] += n_unstable
        tot["n_unique_cases"] += len(by_case)
    out.update(tot)
    if prices is not None:          # hosted: one total per sit, so the sit-to-sit spread is re-derivable
        per = [sum(bool(r["passed"]) for r in rows if r["load"] == i and r["split"] == "heldout") for i in loads]
        out["heldout_pass_per_load"] = per
        out["heldout_pass_spread"] = max(per) - min(per)
    out["heldout_majority_pass_rate"] = tot["heldout_majority_pass"] / tot["heldout_n"] if tot["heldout_n"] else None
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
    got = derive(rows, energy, d.get("prices"))
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
    peak = "n/a (hosted)" if got["peak_memory_gb_max"] is None else f"{got['peak_memory_gb_max']:.3f} GB"
    cost = "" if "cost_usd_total" not in got else (
        f"; ${got['cost_usd_total']:.4f} total = ${got['cost_usd_per_case']:.5f}/case "
        f"({got['prompt_tokens_total']} in + {got['gen_tokens_total']} out tokens; unmetered calls {got['unmetered_calls']})")
    return True, (f"{path}: matches — {d['metric']} = {got[d['metric']]:.6f} over {got['n_cases']} cases "
                  f"({got['n_calls']} calls); gen {gtps} tok/s; peak {peak}; "
                  f"wall {got['wall_s_total']:.2f} s; {wh}{cost}; {status}") + _per_exam_lines(got)


def _per_exam_lines(got):
    if "n_loads" not in got:
        return ""
    exams = sorted({k.split(".")[0] for k in got if "." in k})
    lines = [f"\n  {got['n_loads']} loads; held-out majority {got['heldout_majority_pass']}/{got['heldout_n']}, "
             f"stable-pass {got['heldout_stable_pass']}/{got['heldout_n']}; unstable {got['n_unstable']}/{got['n_unique_cases']}"
             + (f"; held-out per load {got['heldout_pass_per_load']} (spread {got['heldout_pass_spread']})"
                if "heldout_pass_per_load" in got else "")]
    for e in exams:
        g = lambda k: got[f"{e}.{k}"]
        tps = "n/a" if g("gen_tps") is None else f"{g('gen_tps'):.1f}"
        pk = "n/a" if g("peak_memory_gb_max") is None else f"{g('peak_memory_gb_max'):.2f} GB"
        usd = "" if f"{e}.cost_usd_per_case" not in got else (
            f" · ${g('cost_usd_per_case'):.5f}/case · length-capped no-answer cases {g('length_no_answer_cases')}")
        lines.append(f"\n  {e:13s} held-out {g('heldout_majority_pass')}/{g('heldout_n')} (stable-pass "
                     f"{g('heldout_stable_pass')}) · all {g('all_majority_pass')}/{g('n_cases')} · unstable {g('n_unstable')} · "
                     f"gen {tps} tok/s · peak {pk} · {g('wall_s_per_case'):.2f} s/case = "
                     f"{g('min_per_1000_cases'):.1f} min/1000 · length-capped calls {g('finish_length_calls')}{usd}")
    return "".join(lines)
