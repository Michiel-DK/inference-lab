"""Lane 06 helpers (lab/spec.py): fault rows, acceptance, the control-vs-draft comparison. No GPU, no model."""
from lab import results
from lab import spec as S


def _call(raw, gen=10, drafted=6, fault=None):
    c = {"prompt_tokens": 100, "prompt_tps": 500.0, "gen_tokens": gen, "gen_tps": 40.0, "peak_memory_gb": 5.0,
         "finish_reason": "stop", "wall_s": 1.0, "content_chars": len(raw), "reasoning_chars": 0,
         "spec": {"draft": True, "num_draft_tokens": 3, "from_draft_tokens": drafted, "raw_chars": len(raw),
                  "raw_sha256": S.sha(raw), "raw": raw}}
    if fault:
        c.update({"prompt_tps": None, "gen_tokens": None, "gen_tps": None, "finish_reason": "fault"})
        c["spec"] = {"draft": True, "num_draft_tokens": 3, "fault": fault, "tokens_before_fault": 0,
                     "raw_chars": 0, "raw_sha256": S.sha("")}
    return c


def _row(case, calls, passed=True, exam="email-triage", split="heldout"):
    return {"exam": exam, "case": case, "split": split, "load": 0, "passed": passed, "failed_checks": [],
            "start_utc": "t0", "end_utc": "t1", "wall_s": 1.0, "calls": calls}


def test_compare_marks_identical_differs_and_fault():
    ctl = [_row("a", [_call("same")]), _row("b", [_call("abcX")]), _row("c", [_call("x")])]
    spc = [_row("a", [_call("same")]), _row("b", [_call("abcY")], passed=False),
           _row("c", [_call("", fault="ValueError: broadcast")], passed=False)]
    got = {r["case"]: r for r in S.compare_rows(ctl, spc)}
    assert got["a"]["status"] == "identical"
    assert got["b"]["status"] == "differs" and got["b"]["first_divergence_char"] == 3
    assert got["b"]["verdict_control"] is True and got["b"]["verdict_spec"] is False
    assert got["c"]["status"] == "fault" and "broadcast" in got["c"]["fault"]


def test_acceptance_skips_faulted_calls():
    calls = [_call("x", gen=10, drafted=6), _call("", fault="E"), _call("y", gen=5, drafted=1)]
    assert S.acceptance(calls) == (7, 15)


def test_fault_call_records_derive_and_check_cleanly():
    rows = [_row("a", [_call("ok")]), _row("b", [_call("", fault="ValueError: x")], passed=False)]
    agg = results.derive(rows)               # None token fields on a fault call must not break the derivation
    assert agg["n_calls"] == 2 and agg["gen_tokens_total"] == 10
    assert agg["email-triage.finish_length_calls"] == 0
    assert S.is_fault_row(rows[1]) and not S.is_fault_row(rows[0])


def test_first_divergence_prefix_case():
    assert S.first_divergence("abc", "abc") is None
    assert S.first_divergence("abc", "abcd") == 3
