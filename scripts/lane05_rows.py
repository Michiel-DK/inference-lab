"""Lane 05 post: every figure the SHOWCASE quotes, re-derived from committed results files.

Runs `lab check` on every file first and exits 1 on drift. Then, per arm: held-out majority per exam and total, p50/p95
wall per held-out sitting (all exams, and per exam), prompt tokens, rows padded, prompt tok/s, gen tok/s, peak memory,
length-capped calls; flips by id against the unpadded baseline (lane 02 / lane 03 files, majority over 3 loads), KV
flips against the 8k default arm, and load-to-load disagreement for the 8k repeat. Missing arm files print as NOT SAT.
Usage: python3 scripts/lane05_rows.py
"""
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMS = ["call-fields", "email-triage", "recap"]
MODELS = {
    "e4b": ("results/dense-gemma-4-e4b-it-4bit.json", "gemma-4-e4b-it-4bit"),
    "lfm": ("results/moe-lfm2.5-8b-a1b-mlx-4bit.json", "lfm2.5-8b-a1b-mlx-4bit"),
}
SUFFIXES = ["2k", "4k", "8k", "8k-kv8", "8k-kv4", "8k-repeat"]


def load(p):
    return json.loads((ROOT / p).read_text())


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q                       # linear interpolation (numpy's default)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def verdicts(d):
    """{(exam, case): majority passed} over held-out rows, any number of loads."""
    by = {}
    for r in d["rows"]:
        if r["split"] == "heldout":
            by.setdefault((r["exam"], r["case"]), []).append(bool(r["passed"]))
    return {k: sum(v) > len(v) - sum(v) for k, v in by.items()}


def summary(d):
    h = [r for r in d["rows"] if r["split"] == "heldout"]
    v = verdicts(d)
    calls = [c for r in h for c in r["calls"]]
    first = [r["calls"][0] for r in h if r["calls"]]
    out = {"pass": {e: sum(p for (x, _), p in v.items() if x == e) for e in EXAMS}, "n_rows": len(h)}
    out["total"] = sum(out["pass"].values())
    w = [r["wall_s"] for r in h]
    out["p50"], out["p95"] = pct(w, .5), pct(w, .95)
    out["per_exam_wall"] = {e: (pct([r["wall_s"] for r in h if r["exam"] == e], .5),
                                pct([r["wall_s"] for r in h if r["exam"] == e], .95)) for e in EXAMS}
    out["prompt_tok_p50"] = pct([c["prompt_tokens"] for c in first], .5)
    out["prompt_tok_max"] = max(c["prompt_tokens"] for c in first)
    out["prompt_tps_p50"] = pct([c["prompt_tps"] for c in first], .5)
    gt = sum(c["gen_tokens"] for c in calls); gs = sum(c["gen_tokens"] / c["gen_tps"] for c in calls if c["gen_tps"])
    out["gen_tps"] = gt / gs
    out["gen_tokens_per_sitting"] = gt / len(h)
    out["peak_gb"] = max(c["peak_memory_gb"] for c in calls)
    nl = len({r.get("load", 0) for r in h})                 # count columns are PER LOAD (baseline files have 3)
    f = lambda x: x / nl if x % nl else x // nl
    out["length_calls"] = f(sum(c["finish_reason"] == "length" for c in calls))
    out["calls_per_sitting"] = len(calls) / len(h)
    pads = [(c.get("context") or {}).get("pad") for c in first]
    out["padded"] = {e: sum(1 for r, p in zip(h, pads) if p and p["pad_chars"] > 0 and r["exam"] == e) for e in EXAMS}
    out["errors"] = f(sum("wilco_error" in r for r in h))
    out["no_answer"] = {e: f(sum(1 for r in h if r["exam"] == e and any(c["bucket"] == "termination" for c in r["failed_checks"])))
                        for e in EXAMS}
    return out


def flips(a, b):
    va, vb = verdicts(a), verdicts(b)
    up = sorted(k for k in va if not va[k] and vb.get(k))
    down = sorted(k for k in va if va[k] and not vb.get(k))
    return up, down


def main():
    files = [m[0] for m in MODELS.values()]
    arms = {}
    for key, (_, stem) in MODELS.items():
        for s in SUFFIXES:
            p = f"results/ctx-{stem}-{s}.json"
            if (ROOT / p).exists():
                arms[(key, s)] = p
                files.append(p)
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "results").glob("ctx-gate-*.json"))]
    rc = subprocess.run([sys.executable, "-m", "lab", "check", *files], cwd=ROOT, capture_output=True, text=True)
    print(f"lab check on {len(files)} files: exit {rc.returncode}")
    if rc.returncode:
        print(rc.stdout, rc.stderr); return 1

    repro = {k: [0, 0] for k in MODELS}
    for key, (base_p, stem) in MODELS.items():
        base = load(base_p)
        print(f"\n## {key} ({stem})  baseline {base_p} (3 loads, majority; walls over all 3 loads' held-out sittings; count columns per load)")
        print("| arm | email /20 | call /19 | recap /16 | total /55 | padded e·c·r | prompt tok p50 / max | p50 s | p95 s "
              "| p50/p95 s email · call · recap | prompt tok/s p50 | gen tok/s | gen tok / sitting | peak GB | length calls | wilco errors |")
        print("|" + "---|" * 16)
        rows = [("0 (baseline)", base)] + [(s, load(arms[(key, s)])) for s in SUFFIXES if (key, s) in arms]
        for s, d in rows:
            m = summary(d)
            pe = " · ".join(f"{m['per_exam_wall'][e][0]:.1f}/{m['per_exam_wall'][e][1]:.1f}" for e in ["email-triage", "call-fields", "recap"])
            pad = " · ".join(str(m["padded"][e]) for e in ["email-triage", "call-fields", "recap"])
            print(f"| {s} | {m['pass']['email-triage']} | {m['pass']['call-fields']} | {m['pass']['recap']} | **{m['total']}** "
                  f"| {pad} | {m['prompt_tok_p50']:.0f} / {m['prompt_tok_max']} | {m['p50']:.1f} | {m['p95']:.1f} | {pe} "
                  f"| {m['prompt_tps_p50']:.0f} | {m['gen_tps']:.1f} | {m['gen_tokens_per_sitting']:.0f} | {m['peak_gb']:.2f} "
                  f"| {m['length_calls']} | {m['errors']} | {' · '.join(str(m['no_answer'][e]) for e in ['email-triage', 'call-fields', 'recap'])} |")
        for s in SUFFIXES:
            if (key, s) not in arms:
                print(f"NOT SAT: {key} {s}")
        print("\nflips by id (held-out; up = FAIL→PASS):")
        for s, d in rows[1:]:
            ref_name, ref = ("8k default", load(arms[(key, "8k")])) if s.startswith("8k-") and (key, "8k") in arms else ("baseline", base)
            up, down = flips(ref, d)
            print(f"  {s} vs {ref_name}: +{len(up)} / -{len(down)}  up {[c for _, c in up]}  down {[c for _, c in down]}")
        # zero-pad equivalence: rows the rule left unpadded must reproduce the baseline (same code path in effect)
        for s, d in rows[1:]:
            if "kv" in s:
                continue
            b0 = {(r["exam"], r["case"]): r for r in base["rows"] if r["load"] == 0}
            same = diff = 0
            for r in d["rows"]:
                p = (r["calls"][0].get("context") or {}).get("pad") if r["calls"] else None
                if p and p["pad_chars"] == 0:
                    br = b0[(r["exam"], r["case"])]
                    eq = (br["passed"] == r["passed"] and [c["gen_tokens"] for c in br["calls"]] == [c["gen_tokens"] for c in r["calls"]])
                    same += eq; diff += not eq
            print(f"  unpadded rows in {s}: {same} reproduce baseline load 0 (verdict + gen tokens per call), {diff} do not")
            repro[key] = [repro[key][0] + same, repro[key][1] + same + diff]
        derived(key, base, {s: d for s, d in rows[1:]}, repro[key])
    return 0


def derived(key, base, arms, repro):
    """Every delta, ratio and count the SHOWCASE quotes that the table does not print as-is."""
    print(f"\nDerived figures ({key}):")
    print(f"  unpadded rows reproducing baseline load 0, summed over padded default arms: {repro[0]} of {repro[1]}")
    b = summary(base)
    for s, d in arms.items():
        m = summary(d)
        h = [r for r in d["rows"] if r["split"] == "heldout"]
        understood = sum(1 for r in h if "Understood" in r.get("wilco_error", ""))
        len_na = sum(1 for r in h if any(c["bucket"] == "termination" for c in r["failed_checks"])
                     and any(c["finish_reason"] == "length" for c in r["calls"]))
        na = sum(m["no_answer"].values())
        pe = " · ".join(f"{m['per_exam_wall'][e][0] - b['per_exam_wall'][e][0]:+.1f}" for e in ["email-triage", "call-fields", "recap"])
        print(f"  {s}: p50 {m['p50'] - b['p50']:+.1f} s vs baseline (email · call · recap {pe}); "
              f"gen tok/s x{m['gen_tps'] / b['gen_tps']:.3f}; peak {m['peak_gb'] - b['peak_gb']:+.2f} GB; "
              f"no-answer sittings {na}, of which with a length-capped call {len_na}; "
              f"rows whose wilco error is an 'Understood' reply {understood}")
        # prefill account: extra first-call prompt tokens / that arm's prompt tok/s, vs the measured p50 shift
        extra = st.median([r["calls"][0]["prompt_tokens"] for r in h]) - b["prompt_tok_p50"]
        print(f"      prefill account: median prompt +{extra:.0f} tok / {m['prompt_tps_p50']:.0f} tok/s = {extra / m['prompt_tps_p50']:+.1f} s")
    if "8k" in arms:
        e8 = summary(arms["8k"])
        for s in ("8k-kv8", "8k-kv4"):
            if s in arms:
                m = summary(arms[s])
                print(f"  {s} vs 8k: prompt tok/s x{m['prompt_tps_p50'] / e8['prompt_tps_p50']:.3f}; gen tok/s "
                      f"x{m['gen_tps'] / e8['gen_tps']:.3f}; p50 {m['p50'] - e8['p50']:+.1f} s; peak "
                      f"{m['peak_gb'] - e8['peak_gb']:+.2f} GB ({(m['peak_gb'] / e8['peak_gb'] - 1) * 100:+.0f}%)")


if __name__ == "__main__":
    sys.exit(main())
