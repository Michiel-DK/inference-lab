"""Lane 09 gate: the hosted meter and its cost arithmetic. No network, no key, no GPU (a fake router stands in)."""
import json
import types

import pytest

from lab import hosted as H
from lab import results

PRICES = {"input_usd_per_mtok": 2.0, "output_usd_per_mtok": 10.0}


def _hcall(p, g, finish="stop"):
    return {"prompt_tokens": p, "prompt_tps": None, "gen_tokens": g, "gen_tps": None, "peak_memory_gb": None,
            "finish_reason": finish, "wall_s": 1.0, "content_chars": 10, "reasoning_chars": 0}


def _hrow(exam, case, split, load, passed, calls, err=False):
    r = {"exam": exam, "case": case, "split": split, "load": load, "passed": passed, "failed_checks": [],
         "start_utc": "x", "end_utc": "y", "wall_s": 2.0, "calls": calls}
    if err:
        r["wilco_error"] = "model emitted no answer"
    return r


ROWS = [_hrow("email-triage", "a", "heldout", 0, True, [_hcall(1000, 500)]),
        _hrow("email-triage", "a", "heldout", 1, False, [_hcall(1000, 2048, "length")], err=True),
        _hrow("email-triage", "a", "heldout", 2, True, [_hcall(1000, 300)]),
        _hrow("email-triage", "b", "heldout", 0, True, [_hcall(2000, 100), _hcall(2100, 100)]),
        _hrow("email-triage", "b", "heldout", 1, True, [_hcall(2000, 100)]),
        _hrow("email-triage", "b", "heldout", 2, True, [_hcall(2000, None)])]


def test_cost_is_tokens_times_prices_and_unmetered_is_counted():
    got = results.derive(ROWS, None, PRICES)
    metered = [(1000, 500), (1000, 2048), (1000, 300), (2000, 100), (2100, 100), (2000, 100)]
    want = sum(p * 2.0 / 1e6 + g * 10.0 / 1e6 for p, g in metered)
    assert got["cost_usd_total"] == pytest.approx(want)
    assert got["cost_usd_per_case"] == pytest.approx(want / 6)
    assert got["unmetered_calls"] == 1
    assert got["peak_memory_gb_max"] is None and got["gen_tps"] is None
    assert got["email-triage.length_no_answer_cases"] == 1
    assert got["heldout_majority_pass"] == 2 and got["n_unstable"] == 1
    assert got["heldout_pass_per_load"] == [2, 1, 2] and got["heldout_pass_spread"] == 1


def test_no_prices_means_no_cost_keys():
    assert not any("cost" in k for k in results.derive(ROWS))


def test_one_sit_heldout_only():
    one = [r for r in ROWS if r["load"] == 0]
    got = results.derive(one, None, PRICES)
    assert got["n_loads"] == 1 and got["n_unstable"] == 0 and got["heldout_majority_pass"] == 2
    assert got["email-triage.train_n"] == 0 and got["heldout_pass_spread"] == 0


def _doc(rows):
    agg = results.derive(rows, None, PRICES)
    return {"schema": results.SCHEMA, "run": "h", "posted": True, "metric": "heldout_majority_pass_rate",
            "headline": agg["heldout_majority_pass_rate"], "aggregates": agg, "energy": None,
            "prices": dict(PRICES), "rows": rows, "rows_sha256": results.digest(rows)}


def test_check_rederives_cost_and_catches_a_price_edit(tmp_path):
    p = tmp_path / "h.json"
    p.write_text(json.dumps(_doc(ROWS)))
    ok, msg = results.check(str(p))
    assert ok, msg
    assert "$" in msg and "spread 1" in msg
    d = _doc(ROWS)
    d["prices"]["output_usd_per_mtok"] = 5.0                       # cheaper on paper, same tokens: must fail
    p.write_text(json.dumps(d))
    assert not results.check(str(p))[0]
    d = _doc(ROWS)
    d["aggregates"]["cost_usd_total"] *= 0.5
    p.write_text(json.dumps(d))
    assert not results.check(str(p))[0]


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class _TerminationError(Exception):
    pass


def _fake_router(responses):
    req = types.SimpleNamespace()
    req.post = lambda *a, **k: responses.pop(0)

    class Inner:
        base_url = "https://example.invalid/v1"

        def generate(self, messages, model, temperature=0.0, max_tokens=512):
            resp = req.post("u", data="{}")
            body = resp.json()
            text = body["choices"][0]["message"]["content"]
            if not text:
                raise _TerminationError("no answer")
            return text, {}

    return types.SimpleNamespace(requests=req, get_adapter=lambda name: Inner())


def test_meter_records_usage_even_when_wilco_raises(tmp_path):
    body = lambda content, fin: {"model": "claude-haiku-5-5", "choices": [{"message": {"content": content},
                                                                           "finish_reason": fin}],
                                 "usage": {"prompt_tokens": 100, "completion_tokens": 2048, "total_tokens": 2148,
                                           "completion_tokens_details": {"reasoning_tokens": 2000}}}
    router = _fake_router([_Resp(200, body("", "length")), _Resp(200, body('{"label": "ignore"}', "stop"))])
    m = H.MeteredHosted(router, "claude-haiku-5-5", tmp_path, "t")
    with pytest.raises(_TerminationError):
        m.generate([], "claude-haiku-5-5")
    assert m.generate([], "claude-haiku-5-5")[0] == '{"label": "ignore"}'
    assert [c["finish_reason"] for c in m.calls] == ["length", "stop"]
    assert m.calls[0]["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2000
    assert m.calls[0]["peak_memory_gb"] is None and m.calls[0]["content_chars"] == 0
    assert H.ledger_spent(tmp_path) == pytest.approx(2 * (100 * 0.10 + 2048 * 0.50) / 1e6)


def test_meter_keeps_failed_http_out_of_cost(tmp_path):
    router = _fake_router([_Resp(429, {"error": {"type": "rate_limit_error"}})])
    m = H.MeteredHosted(router, "claude-haiku-5-5", tmp_path, "t")
    with pytest.raises(Exception):
        m.generate([], "claude-haiku-5-5")
    assert m.calls == [] and m.http_failures[0]["http_status"] == 429
    assert H.ledger_spent(tmp_path) == 0.0


def test_worst_case_bounds_two_full_calls():
    w = H.worst_case_usd({"input": "x" * 3000}, 4096, H.PRICES["claude-opus-5-5"])
    assert w > 2 * 4096 * 20 / 1e6


def test_hosted_refuses_without_prediction():
    from lab.cli import main
    assert main(["hosted", "claude-haiku-5-5", "--name", "nope", "--smoke", "--prediction", "no-such-file"]) == 2
