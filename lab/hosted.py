"""`lab hosted`: a hosted model sits fixed wilco cases through wilco's OWN `anthropic` provider; wilco grades; every
billed response's usage is kept per call (lane 09).

The adapter is wilco's: `router.get_adapter("anthropic")`. That is the OpenAI-compatible `/v1/chat/completions`
endpoint with wilco's `drop_body_keys: ["temperature"]`, the path wilco's 28 Oct frontier probe used. The lab does not
build or alter one request byte. While one `generate()` runs, `MeteredHosted` wraps `requests.post` and reads each HTTP
response's `usage`, `finish_reason` and `model`. It reads nothing else: no headers, and no text beyond its length.

Cost is not stored as a number to trust. The results file stores the token counts and the price block, and
`results.derive` multiplies them, so `lab check` re-derives every $ figure. The spend ceiling is enforced here, before
each case, from a ledger of every billed response in the lane (smokes included).
"""
import json
import os
import sys
import time
from pathlib import Path

from . import cases as C
from . import results
from .run import _git, _now, _sysctl, import_wilco_runner

# $ per million tokens, list price, standard tier, no caching. Thinking is billed as output.
PRICES = {
    "claude-haiku-5-5": {"input_usd_per_mtok": 0.10, "output_usd_per_mtok": 0.50},
    "claude-sonnet-5-5": {"input_usd_per_mtok": 2.00, "output_usd_per_mtok": 10.00},
    "claude-opus-5-5": {"input_usd_per_mtok": 4.00, "output_usd_per_mtok": 20.00},
}
PRICE_SOURCE = ("claude-api reference model table, cached 2026-10-06 (rank C); the same rates wilco's "
                "docs/probes/frontier-direct-2026-10-28 used")
CEILING_USD = 2.00
LEDGER = "hosted-ledger.jsonl"


class CeilingReached(SystemExit):
    pass


def _numbers(d):
    """The numeric leaves of a usage dict (nested *_details included), nothing else."""
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[k] = v
        elif isinstance(v, dict):
            sub = _numbers(v)
            if sub:
                out[k] = sub
    return out


def ledger_spent(scratch):
    p = Path(scratch) / LEDGER
    if not p.exists():
        return 0.0
    return sum(json.loads(x)["cost_usd"] for x in p.read_text().splitlines() if x.strip())


class MeteredHosted:
    """A wilco Adapter that delegates to wilco's own ChatCompletionsAdapter and records each billed response."""

    def __init__(self, router, model, scratch, run_name):
        self.router = router
        self.inner = router.get_adapter("anthropic")          # key read by wilco from ANTHROPIC_API_KEY; never printed
        self.endpoint = self.inner.base_url
        self.model = model
        self.prices = PRICES[model]
        self.ledger = Path(scratch) / LEDGER
        self.run_name = run_name
        self.calls, self.http_failures = [], []

    def generate(self, messages, model, temperature=0.0, max_tokens=512):
        req = self.router.requests
        real_post = req.post
        seen = []

        def post(*a, **k):
            t0 = time.perf_counter()
            try:
                resp = real_post(*a, **k)
            except Exception as exc:                          # network: no response, nothing billed
                self.http_failures.append({"http_status": None, "error": type(exc).__name__,
                                           "wall_s": time.perf_counter() - t0})
                raise
            wall = time.perf_counter() - t0
            try:
                body = resp.json()
            except ValueError:
                body = None
            if resp.status_code != 200 or not isinstance(body, dict):
                self.http_failures.append({"http_status": resp.status_code, "wall_s": wall,
                                           "error_type": ((body or {}).get("error") or {}).get("type")
                                           if isinstance(body, dict) else None})
                return resp
            usage = body.get("usage") or {}
            choice = (body.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            call = {"prompt_tokens": usage.get("prompt_tokens"), "prompt_tps": None,
                    "gen_tokens": usage.get("completion_tokens"), "gen_tps": None, "peak_memory_gb": None,
                    "finish_reason": choice.get("finish_reason"), "wall_s": wall,
                    "content_chars": len(msg.get("content") or ""), "reasoning_chars": len(reasoning),
                    "usage": _numbers(usage), "response_model": body.get("model")}
            seen.append(call)
            cost = results.call_cost(call, self.prices)
            with open(self.ledger, "a") as fh:
                fh.write(json.dumps({"utc": _now(), "run": self.run_name, "model": self.model,
                                     "prompt_tokens": call["prompt_tokens"], "gen_tokens": call["gen_tokens"],
                                     "cost_usd": cost or 0.0, "unmetered": cost is None}) + "\n")
            return resp

        req.post = post
        try:
            return self.inner.generate(messages, model, temperature=temperature, max_tokens=max_tokens)
        finally:
            req.post = real_post
            self.calls.extend(seen)


def worst_case_usd(case, max_tokens, prices):
    """Upper bound of one case: two billed calls (wilco re-asks once), each the case text at ~3 chars/token + 2k tokens
    of system prompt and template, and a full max_tokens of output."""
    prompt = len(json.dumps(case)) / 3 + 2000
    return 2 * (prompt * prices["input_usd_per_mtok"] + max_tokens * prices["output_usd_per_mtok"]) / 1e6


def _read_sit(path):
    """(rows, footer) of a sit's partial file; footer None while the sit is unfinished (it resumes, case by case)."""
    if not Path(path).exists():
        return [], None
    lines = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if lines and "_footer" in lines[-1]:
        return lines[:-1], lines[-1]["_footer"]
    return lines, None


def sit(model, name, plan, sit_idx, part, scratch, ceiling):
    """One sit: every (exam, case) in plan, once. Rows are appended and flushed as they land. A rerun skips cases already
    in the partial file, so an interruption never pays twice for a case."""
    runner, router = import_wilco_runner()
    adapter = MeteredHosted(router, model, scratch, name)
    done_rows, footer = _read_sit(part)
    if footer is not None:
        return done_rows, footer
    done = {(r["exam"], r["case"]) for r in done_rows}
    real_adapter_for = runner.adapter_for
    runner.adapter_for = lambda *a, **k: adapter
    t_sit = time.perf_counter()
    try:
        with open(part, "a") as fh:
            for exam, ids in plan:
                agent = runner.load_agent(exam)
                exam_doc = json.loads(C.cases_path(exam).read_text())
                by_id = {c["id"]: c for c in exam_doc["cases"]}
                for cid in ids:
                    if (exam, cid) in done:
                        continue
                    spent = ledger_spent(scratch)
                    worst = worst_case_usd(by_id[cid], agent.get("max_tokens", 2048), adapter.prices)
                    if spent + worst > ceiling:
                        raise CeilingReached(f"ceiling: spent ${spent:.4f} + worst case ${worst:.4f} for {exam}/{cid} "
                                             f"> ${ceiling:.2f}; sit {sit_idx} of {name} stopped, not sat (ceiling)")
                    n_calls, n_fail = len(adapter.calls), len(adapter.http_failures)
                    start = _now()
                    t0 = time.perf_counter()
                    res = runner.run_exam(agent, {**exam_doc, "cases": [by_id[cid]]}, provider="anthropic", model=model)
                    wall_s = time.perf_counter() - t0
                    graded = res["cases"][0]
                    row = {"exam": exam, "case": cid, "split": graded["split"], "load": sit_idx,
                           "passed": bool(graded["passed"]), "failed_checks": graded.get("failed_checks", []),
                           **({"wilco_error": str(graded["got"]["error"])[:300]}
                              if isinstance(graded.get("got"), dict) and "error" in graded["got"] else {}),
                           "start_utc": start, "end_utc": _now(), "wall_s": wall_s,
                           "calls": adapter.calls[n_calls:], "http_failures": adapter.http_failures[n_fail:]}
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    done_rows.append(row)
                    usd = sum(results.call_cost(c, adapter.prices) or 0 for c in row["calls"])
                    print(f"sit {sit_idx} {exam}/{cid}: {'PASS' if row['passed'] else 'FAIL'}  {wall_s:.1f}s  "
                          f"{len(row['calls'])} call(s)  ${usd:.5f}  (lane spent ${ledger_spent(scratch):.4f})",
                          file=sys.stderr, flush=True)
            footer = {"load": sit_idx, "n_rows": len(done_rows), "sit_wall_s_this_process": time.perf_counter() - t_sit,
                      "endpoint": adapter.endpoint}
            fh.write(json.dumps({"_footer": footer}) + "\n")
    finally:
        runner.adapter_for = real_adapter_for
    return done_rows, footer


def run_hosted(model, name, plan, n_sits, prediction, prediction_commit, out_path, scratch, posted, ceiling):
    scratch = Path(scratch); scratch.mkdir(parents=True, exist_ok=True)
    rows, footers = [], []
    for i in range(n_sits):
        r, f = sit(model, name, plan, i, scratch / f"{name}.sit{i}.jsonl", scratch, ceiling)
        rows += r
        footers.append(f)
    runner, router = import_wilco_runner()
    prices = PRICES[model]
    agg = results.derive(rows, None, prices)
    exams = [e for e, _ in plan]
    metric = "heldout_majority_pass_rate" if agg["heldout_n"] else "pass_rate"     # a train-only gate smoke has no held-out
    doc = {
        "schema": results.SCHEMA, "run": name, "posted": posted,
        "prediction": f"predictions/{prediction}.md", "prediction_commit": prediction_commit,
        "lab_commit": _git("rev-parse", "HEAD"),
        "model": {"ref": model, "provider": "anthropic (wilco sandbox/router.py PROVIDERS, get_adapter)",
                  "endpoint": footers[0]["endpoint"] + "/chat/completions",
                  "response_models": sorted({c["response_model"] for r in rows for c in r["calls"]
                                             if c.get("response_model")}),
                  "effort": "API default (this endpoint sends none): haiku/opus medium, sonnet high (rank C)",
                  "temperature": "not sent (wilco drop_body_keys); the endpoint samples, so sits are samples, not loads",
                  "loads": [{"load": f["load"], "n_rows": f["n_rows"]} for f in footers],
                  "load_protocol": "'load' = one sit: every case once, sequentially, one process; hosted, so no "
                                   "weights are loaded and n_unstable counts sampling disagreement between sits"},
        "prices": {**prices, "source": PRICE_SOURCE, "tier": "standard, no prompt caching, no batch"},
        "ceiling_usd": ceiling,
        "wilco": {"dir": str(C.WILCO_DIR), "pin": C.WILCO_PIN,
                  "cases_sha256": {e: C.EXAMS[e]["sha256"] for e in exams},
                  "grader": "sandbox/runner.py run_exam (per case, provider='anthropic'), adapter_for -> "
                            "lab.hosted.MeteredHosted around router.get_adapter('anthropic')",
                  "snapshot_rule": "sandbox/runner.py aggregate_loads: majority verdict over sits, stable = all agree",
                  "tier": "agents say tier: local; lifted for this lane by the operator's go (synthetic public cases)",
                  "agents": {e: {k: runner.load_agent(e).get(k) for k in ("temperature", "max_tokens")} for e in exams}},
        "machine": {"model": _sysctl("hw.model"), "python": sys.version.split()[0]},
        "exams": exams, "plan": plan,
        "metric": metric, "headline": agg[metric], "aggregates": agg,
        "energy": None, "energy_note": "hosted: no local energy to meter; cost is $ from metered tokens x list price",
        "rows": rows, "rows_sha256": results.digest(rows),
    }
    Path(out_path).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def key_present():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
