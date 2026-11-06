"""The lab gate. Stdlib + pytest only: no GPU, no mlx, no model. Run from the repo root: python3 -m pytest -q"""
import json
import plistlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab import cases as C
from lab import energy, results
from lab.cli import main

T0 = datetime(2026, 11, 6, 12, 0, 0, tzinfo=timezone.utc)


def _row(case, passed, start_s, end_s, calls):
    return {"exam": "email-triage", "case": case, "split": "train", "passed": passed, "failed_checks": [],
            "start_utc": (T0 + timedelta(seconds=start_s)).isoformat(), "end_utc": (T0 + timedelta(seconds=end_s)).isoformat(),
            "wall_s": end_s - start_s, "calls": calls}


def _call(gen_tokens, gen_tps, prompt_tokens=100, prompt_tps=500.0, peak=1.5):
    return {"prompt_tokens": prompt_tokens, "prompt_tps": prompt_tps, "gen_tokens": gen_tokens, "gen_tps": gen_tps,
            "peak_memory_gb": peak, "finish_reason": "stop", "wall_s": 1.0}


def _doc(rows, energy_block=None):
    agg = results.derive(rows, energy_block)
    return {"schema": results.SCHEMA, "run": "t", "posted": False, "metric": "pass_rate", "headline": agg["pass_rate"],
            "aggregates": agg, "energy": energy_block, "rows": rows, "rows_sha256": results.digest(rows)}


ROWS = [_row("a", True, 0, 10, [_call(40, 20.0)]), _row("b", False, 10, 30, [_call(10, 10.0, peak=2.0), _call(30, 30.0)])]


def _write(tmp_path, doc):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(doc))
    return str(p)


def test_derive_is_the_arithmetic():
    agg = results.derive(ROWS)
    assert agg["pass_rate"] == 0.5
    assert agg["n_calls"] == 3 and agg["gen_tokens_total"] == 80
    assert agg["gen_tps"] == pytest.approx(80 / (2.0 + 1.0 + 1.0))       # token-weighted, not a mean of rates
    assert agg["peak_memory_gb_max"] == 2.0
    assert agg["wall_s_total"] == 30 and agg["wall_s_per_case"] == 15
    assert agg["wh_per_case"] is None                                     # unmetered is None, never 0


def test_check_passes_clean_file(tmp_path):
    ok, msg = results.check(_write(tmp_path, _doc(ROWS)))
    assert ok, msg
    assert "not metered" in msg and "not posted" in msg


@pytest.mark.parametrize("key,val", [("gen_tps", 99.0), ("peak_memory_gb_max", 1.0), ("pass_rate", 1.0),
                                     ("wh_per_case", 0.0)])
def test_check_catches_a_tampered_aggregate(tmp_path, key, val):
    d = _doc(ROWS)
    d["aggregates"][key] = val
    ok, msg = results.check(_write(tmp_path, d))
    assert not ok and key in msg
    assert main(["check", _write(tmp_path, d)]) == 1


def test_check_catches_tampered_headline_and_rows(tmp_path):
    d = _doc(ROWS); d["headline"] = 1.0
    assert not results.check(_write(tmp_path, d))[0]
    d = _doc(ROWS); d["rows"][0]["calls"][0]["gen_tps"] = 200.0
    ok, msg = results.check(_write(tmp_path, d))
    assert not ok and "rows_sha256" in msg


# ---- energy: the sandbox method on a synthetic NUL-separated plist capture

def _capture(watts, start, n, interval_s=1.0, combined=True):
    out = []
    for i in range(n):
        end = (start + timedelta(seconds=(i + 1) * interval_s)).replace(tzinfo=None)   # plist dates are naive UTC
        proc = ({"combined_power": watts * 1000.0} if combined
                else {"cpu_power": watts * 600.0, "gpu_power": watts * 400.0, "ane_power": 0.0})
        out.append(plistlib.dumps({"timestamp": end, "elapsed_ns": int(interval_s * 1e9), "processor": proc}))
    return b"\x00".join(out)


def test_energy_integrates_and_subtracts_idle():
    idle = _capture(1.0, T0 - timedelta(seconds=100), 50)            # 1 W idle window, an hour of nothing
    run = _capture(10.0, T0, 30)                                     # 10 W over the 30 s both rows span
    e = energy.energy_block(idle + b"\x00" + run, ROWS,
                            (T0 - timedelta(seconds=100)).isoformat(), (T0 - timedelta(seconds=50)).isoformat())
    assert e["gross_j"] == pytest.approx(300.0)
    assert e["idle_mw"] == pytest.approx(1000.0)
    assert e["net_j"] == pytest.approx(270.0)
    assert e["coverage"] == pytest.approx(1.0)
    assert e["per_row"]["a"]["net_j"] == pytest.approx(90.0)
    d = _doc(ROWS, e)
    assert d["aggregates"]["wh_per_case"] == pytest.approx(270.0 / 3600 / 2)


def test_energy_sums_parts_when_no_combined():
    run = _capture(10.0, T0, 30, combined=False)
    e = energy.energy_block(run, ROWS, T0.isoformat(), (T0 + timedelta(seconds=30)).isoformat())
    assert e["gross_j"] == pytest.approx(300.0)


def test_energy_refuses_low_coverage_and_unknown_format():
    with pytest.raises(ValueError, match="coverage"):
        energy.energy_block(_capture(10.0, T0, 20), ROWS, T0.isoformat(), (T0 + timedelta(seconds=30)).isoformat())
    bad = plistlib.dumps({"timestamp": T0.replace(tzinfo=None), "elapsed_ns": 10**9, "processor": {}})
    with pytest.raises(ValueError, match="none of"):
        energy.energy_block(bad, ROWS, T0.isoformat(), (T0 + timedelta(seconds=1)).isoformat())


def test_check_catches_energy_arithmetic(tmp_path):
    e = energy.energy_block(_capture(10.0, T0, 30), ROWS, T0.isoformat(), (T0 + timedelta(seconds=30)).isoformat())
    d = _doc(ROWS, e)
    assert results.check(_write(tmp_path, d))[0]
    d["energy"]["net_j"] += 50.0
    ok, msg = results.check(_write(tmp_path, d))
    assert not ok and "net_j" in msg


# ---- gates

def test_run_refuses_without_prediction():
    assert main(["run", "some/model", "--name", "no-such-prediction", "--smoke"]) == 2


def test_case_set_is_the_three_exams():
    assert sorted(C.EXAMS) == ["call-fields", "email-triage", "recap"]
    assert C.SMOKE["exam"] in C.EXAMS and len(C.SMOKE["cases"]) == 2


@pytest.mark.skipif(not (C.WILCO_DIR / ".git").exists(), reason="no wilco checkout on this machine")
def test_wilco_pin_holds_and_sizes_match():
    assert C.verify_pin() == []
    for exam, meta in C.EXAMS.items():
        cs = json.loads(C.cases_path(exam).read_text())["cases"]
        assert sum(c["split"] == "train" for c in cs) == meta["train"]
        assert sum(c["split"] == "heldout" for c in cs) == meta["heldout"]
    smoke = {c["id"]: c for c in json.loads(C.cases_path(C.SMOKE["exam"]).read_text())["cases"]}
    assert all(smoke[i]["split"] == "train" for i in C.SMOKE["cases"])   # the smoke never touches held-out


def test_committed_results_check():
    for p in sorted(Path("results").glob("*.json")):
        ok, msg = results.check(str(p))
        assert ok, msg
