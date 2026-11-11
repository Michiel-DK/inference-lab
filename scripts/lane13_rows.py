"""Lane 13 post: every figure the SHOWCASE and LESSONS quote, re-derived from committed results files (read-only).

Runs `lab check` on every lane-13 file and lane 05's two 8k files first and exits 1 on drift. Then, per arm: held-out
majority per exam and total; the requested KV kwargs beside the applied cache classes (the stamp); format/schema-failure
sittings (failed_checks with bucket `format` or check `error`, rule fixed in the prediction's addendum) and termination
sittings, per exam; reasoning_chars > 0 sittings, gen tokens per sitting, length-capped calls; prompt and gen tok/s, p50 /
p95 wall, peak GB; arm 2 - arm 1 per exam with the ids that moved; repeat vs arm 2; the kill reading (i)-(iii); lane 05's
two 8k rows for scale. Missing arm files print as NOT SAT.
Usage: python3 scripts/lane13_rows.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lane05_rows import EXAMS, flips, load, pct, verdicts  # noqa: E402  (same derivations as lane 05)

ARMS = [("1 control", "results/ctx-qwen3-4b-4bit-8k.json"),
        ("2 maxkv4096", "results/ctx-qwen3-4b-4bit-8k-maxkv4096.json"),
        ("3 maxkv4096 repeat", "results/ctx-qwen3-4b-4bit-8k-maxkv4096-repeat.json")]
SMOKES = ["results/ctx-gate-qwen3-4b-4bit-8k-smoke.json", "results/ctx-gate-qwen3-4b-4bit-8k-maxkv4096-smoke.json"]
LANE05 = [("lane 05 e4b 8k", "results/ctx-gemma-4-e4b-it-4bit-8k.json"),
          ("lane 05 LFM 8k", "results/ctx-lfm2.5-8b-a1b-mlx-4bit-8k.json")]
ORDER = ["email-triage", "call-fields", "recap"]


def is_format(r):
    return any(c.get("bucket") == "format" or c.get("check") == "error" for c in r["failed_checks"])


def is_term(r):
    return any(c.get("bucket") == "termination" for c in r["failed_checks"])


def summary(d):
    h = [r for r in d["rows"] if r["split"] == "heldout"]
    v = verdicts(d)
    calls = [c for r in h for c in r["calls"]]
    first = [r["calls"][0] for r in h if r["calls"]]
    m = {"pass": {e: sum(p for (x, _), p in v.items() if x == e) for e in EXAMS}, "n": len(h)}
    m["total"] = sum(m["pass"].values())
    w = [r["wall_s"] for r in h]
    m["p50"], m["p95"] = pct(w, .5), pct(w, .95)
    m["prompt_tok"] = (min(c["prompt_tokens"] for c in first), max(c["prompt_tokens"] for c in first))
    m["prompt_tps"] = pct([c["prompt_tps"] for c in first], .5)
    gt = sum(c["gen_tokens"] for c in calls)
    gs = sum(c["gen_tokens"] / c["gen_tps"] for c in calls if c["gen_tps"])
    m["gen_tps"] = gt / gs if gs else None
    m["gen_per_sitting"] = gt / len(h)
    m["peak"] = max(c["peak_memory_gb"] for c in calls)
    m["length_calls"] = {e: sum(c["finish_reason"] == "length" for r in h if r["exam"] == e for c in r["calls"]) for e in EXAMS}
    m["reasoning"] = sum(1 for r in h if any(c.get("reasoning_chars", 0) > 0 for c in r["calls"]))
    m["format"] = {e: sum(1 for r in h if r["exam"] == e and is_format(r)) for e in EXAMS}
    m["term"] = {e: sum(1 for r in h if r["exam"] == e and is_term(r)) for e in EXAMS}
    m["calls"] = len(calls)
    m["padded"] = sum(1 for c in first if ((c.get("context") or {}).get("pad") or {}).get("pad_chars", 0) > 0)
    ctx = d.get("context") or {}
    m["requested"] = ctx.get("requested")
    m["applied"] = ctx.get("applied_cache_classes")
    return m


def main():
    have = [(n, p) for n, p in ARMS if (ROOT / p).exists()]
    files = [p for _, p in have] + [p for p in SMOKES if (ROOT / p).exists()] + [p for _, p in LANE05]
    rc = subprocess.run([sys.executable, "-m", "lab", "check", *files], cwd=ROOT, capture_output=True, text=True)
    print(f"lab check on {len(files)} files: exit {rc.returncode}")
    if rc.returncode:
        print(rc.stdout, rc.stderr)
        return 1
    for n, p in ARMS:
        if (n, p) not in have:
            print(f"NOT SAT: arm {n} ({p})")

    print("\n## Smokes (train, not posted): requested -> applied")
    for p in SMOKES:
        if (ROOT / p).exists():
            d = load(p)
            r = d["rows"]
            print(f"  {p}: requested {d['context']['requested']} -> applied {d['context']['applied_cache_classes']}; "
                  f"wall {sum(x['wall_s'] for x in r):.1f} s over {len(r)} cases; "
                  f"peak {max(c['peak_memory_gb'] for x in r for c in x['calls']):.2f} GB; "
                  f"passed {sum(x['passed'] for x in r)}/{len(r)}")
        else:
            print(f"  NOT SAT: {p}")

    S = {n: summary(load(p)) for n, p in have}
    L = {n: summary(load(p)) for n, p in LANE05}
    print("\n## Held-out arms (one load each; count columns per arm)")
    print("| arm | email /20 | call /19 | recap /16 | **total /55** | requested | applied cache classes | format-fail e·c·r "
          "| termination e·c·r | reasoning>0 /55 | gen tok / sitting | length calls e·c·r | prompt tok min–max "
          "| prompt tok/s p50 | gen tok/s | p50 s | p95 s | peak GB |")
    print("|" + "---|" * 18)
    for n, m in list(S.items()) + list(L.items()):
        f3 = lambda k: " · ".join(str(m[k][e]) for e in ORDER)
        print(f"| {n} | {m['pass']['email-triage']} | {m['pass']['call-fields']} | {m['pass']['recap']} | **{m['total']}** "
              f"| {m['requested']} | {m['applied']} | {f3('format')} | {f3('term')} | {m['reasoning']} "
              f"| {m['gen_per_sitting']:.0f} | {f3('length_calls')} | {m['prompt_tok'][0]}–{m['prompt_tok'][1]} "
              f"| {m['prompt_tps']:.0f} | {m['gen_tps']:.1f} | {m['p50']:.1f} | {m['p95']:.1f} | {m['peak']:.2f} |")

    print("\n## Derived figures")
    a1, a2, a3 = (S.get(n) for n, _ in ARMS)
    for n, m in S.items():
        print(f"  {n}: format-fail sittings email + call = {m['format']['email-triage'] + m['format']['call-fields']} of 39; "
              f"termination sittings {sum(m['term'].values())}; calls {m['calls']} over {m['n']} sittings; padded {m['padded']}")
    for n, p in have:
        h = [r for r in load(p)["rows"] if r["split"] == "heldout"]
        nojson = sum(1 for r in h if r.get("wilco_error", "").startswith("no JSON object found"))
        stray = sum(1 for r in h if r.get("wilco_error", "").split(": ", 1)[-1].lstrip("'\"").startswith("</think>"))
        reask = sum(1 for r in h if len(r["calls"]) > 1)
        pe = " · ".join(f"{e} {pct([r['wall_s'] for r in h if r['exam'] == e], .5):.1f}" for e in ORDER)
        print(f"  {n}: wilco 'no JSON object found' sittings {nojson}; of them the stored answer opens with a stray "
              f"'</think>' {stray}; sittings re-asked (2 calls) {reask}; p50 wall per exam {pe}")
    if a1 and a2:
        d1, d2 = load(ARMS[0][1]), load(ARMS[1][1])
        up, down = flips(d1, d2)
        print(f"  arm 2 - arm 1: total {a2['total'] - a1['total']:+d}; per exam "
              + " · ".join(f"{e} {a2['pass'][e] - a1['pass'][e]:+d}" for e in ORDER))
        print(f"    moved +{len(up)} / -{len(down)}; up {[f'{e}/{c}' for e, c in up]}; down {[f'{e}/{c}' for e, c in down]}")
        print(f"    peak {a2['peak'] - a1['peak']:+.2f} GB; gen tok/s x{a2['gen_tps'] / a1['gen_tps']:.3f}; "
              f"prompt tok/s x{a2['prompt_tps'] / a1['prompt_tps']:.3f}; p50 {a2['p50'] - a1['p50']:+.1f} s")
    if a2 and a3:
        up, down = flips(load(ARMS[1][1]), load(ARMS[2][1]))
        print(f"  repeat vs arm 2: moved {len(up) + len(down)} of 55 (+{len(up)} / -{len(down)}); "
              f"ids {[f'{e}/{c}' for e, c in up + down]}")
        ra, rb = load(ARMS[1][1])["rows"], load(ARMS[2][1])["rows"]
        same = sum(1 for x, y in zip(ra, rb) if (x["exam"], x["case"]) == (y["exam"], y["case"])
                   and [c["gen_tokens"] for c in x["calls"]] == [c["gen_tokens"] for c in y["calls"]]
                   and x.get("wilco_error") == y.get("wilco_error"))
        print(f"    rows identical in per-call gen_tokens and wilco's error string: {same} of {len(ra)}")

    print("\n## Kill reading")
    if a2:
        k1 = not any("RotatingKVCache" in json.dumps(x) for x in (a2["applied"] or []))
        print(f"  (i) arm 2 applied cache lists no RotatingKVCache: {'FIRED' if k1 else 'did not fire'} ({a2['applied']})")
    else:
        print("  (i) arm 2 not sat: unread")
    print("  (ii) runtime faults: a fault ends the load (lab adapter -> RuntimeError), so a sat arm file has 0; "
          + "; ".join(f"{n}: {m['n']} rows" for n, m in S.items()))
    if a2 and a3:
        up, down = flips(load(ARMS[1][1]), load(ARMS[2][1]))
        print(f"  (iii) repeat moved {len(up) + len(down)} of 55 vs arm 2: {'FIRED (> 3)' if len(up) + len(down) > 3 else 'did not fire'}")
    else:
        print("  (iii) repeat not sat: unread")
    return 0


if __name__ == "__main__":
    sys.exit(main())
