"""Lane 09 table: the 55 held-out cases, local 4-bit / 8-bit rows beside the hosted rows. Stdlib only.

Every cell comes from a committed results file. The file is first re-derived by `lab.results.check`, and the script
exits 1 if any file drifts, so the table cannot quote a number `check` would not reproduce.

    python3 scripts/lane09_rows.py            # markdown table + Haiku sit detail + per-case hosted-vs-local ids
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab import results  # noqa: E402

EXAMS = ("email-triage", "call-fields", "recap")
CEILING = 2.00
LOCAL = [
    ("gemma-4-e2b-it", "4-bit", "results/dense-gemma-4-e2b-it-4bit.json"),
    ("gemma-4-e2b-it", "8-bit", "results/quant-gemma-4-e2b-it-8bit.json"),
    ("gemma-4-e4b-it", "4-bit", "results/dense-gemma-4-e4b-it-4bit.json"),
    ("gemma-4-e4b-it", "8-bit", "results/quant-gemma-4-e4b-it-8bit.json"),
    ("Llama-3.1-8B-Instruct", "4-bit", "results/dense-llama-3.1-8b-instruct-4bit.json"),
    ("Llama-3.1-8B-Instruct", "8-bit", "results/quant-llama-3.1-8b-instruct-8bit.json"),
]
HOSTED = [
    ("claude-haiku-5-5", "results/hosted-claude-haiku-5-5.json"),
    ("claude-sonnet-5-5", "results/hosted-claude-sonnet-5-5.json"),
    ("claude-opus-5-5", "results/hosted-claude-opus-5-5.json"),
]


def _checked(path):
    ok, msg = results.check(path)
    if not ok:
        sys.exit(f"drift: {msg}")
    d = results.load(path)
    return d, results.derive(d["rows"], d.get("energy"), d.get("prices"))


def _heldout_rows(d):
    return [r for r in d["rows"] if r["split"] == "heldout"]


def _majority(d):
    """{(exam, case): majority pass} over held-out cases, wilco's rule (N odd)."""
    by = {}
    for r in _heldout_rows(d):
        by.setdefault((r["exam"], r["case"]), []).append(bool(r["passed"]))
    return {k: sum(v) > len(v) - sum(v) for k, v in by.items()}


def main():
    lines = ["| arm | sits | email-triage /20 | call-fields /19 | recap /16 | **total /55** | unstable held-out ids "
             "| wall s / held-out case | $ / held-out case | $ / case: email · call · recap |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    maj, local_walls = {}, []
    for model, prec, path in LOCAL:
        d, g = _checked(path)
        h = _heldout_rows(d)
        wall = sum(r["wall_s"] for r in h) / len(h)
        local_walls.append(wall)
        unst = sum(1 for v in _by_case(h).values() if 0 < sum(v) < len(v))
        cells = [str(g[f"{e}.heldout_majority_pass"]) for e in EXAMS]
        lines.append(f"| {model} {prec} (local) | {g['n_loads']} loads | " + " | ".join(cells) +
                     f" | **{g['heldout_majority_pass']}** | {unst} | {wall:.2f} | void (Wh not metered) | void |")
        maj[f"{model} {prec}"] = _majority(d)
    hosted_docs = {}
    for model, path in HOSTED:
        if not Path(path).exists():
            lines.append(f"| {model} (hosted) | not sat | | | | | | | | |")
            continue
        d, g = _checked(path)
        h = _heldout_rows(d)
        wall = sum(r["wall_s"] for r in h) / len(h)
        cells = [str(g[f"{e}.heldout_majority_pass"]) for e in EXAMS]
        unst = "unmeasured (1 sit)" if g["n_loads"] == 1 else str(g["n_unstable"])
        per = " · ".join(f"{g[f'{e}.cost_usd_per_case']:.5f}" for e in EXAMS)
        lines.append(f"| {model} (hosted) | {g['n_loads']} | " + " | ".join(cells) +
                     f" | **{g['heldout_majority_pass']}** | {unst} | {wall:.2f} | {g['cost_usd_per_case']:.5f} | {per} |")
        maj[model] = _majority(d)
        hosted_docs[model] = (d, g)
    print("\n".join(lines))

    print("\nHosted spend (every billed response in each posted file):")
    tot = 0.0
    for model, (d, g) in hosted_docs.items():
        tot += g["cost_usd_total"]
        print(f"- {model}: ${g['cost_usd_total']:.4f} over {g['n_cases']} sittings, {g['n_calls']} calls, "
              f"{g['prompt_tokens_total']} in + {g['gen_tokens_total']} out tokens, unmetered calls {g['unmetered_calls']}, "
              f"length-capped no-answer sittings "
              f"{sum(g[f'{e}.length_no_answer_cases'] for e in EXAMS)}")
    print(f"- total posted hosted spend: ${tot:.4f}")

    if "claude-haiku-5-5" in hosted_docs:
        d, g = hosted_docs["claude-haiku-5-5"]
        print(f"\nHaiku per sit, held-out /55: {g['heldout_pass_per_load']} (spread {g['heldout_pass_spread']}); "
              f"majority {g['heldout_majority_pass']}; stable-pass {g['heldout_stable_pass']}")
        for e in EXAMS:
            per = [sum(bool(r["passed"]) for r in _heldout_rows(d) if r["exam"] == e and r["load"] == i)
                   for i in range(g["n_loads"])]
            print(f"  {e}: per sit {per}")
        flips = sorted(k for k, v in _by_case(_heldout_rows(d)).items() if 0 < sum(v) < len(v))
        print(f"  ids that differ between sits ({len(flips)}): " + ", ".join(f"{e}/{c}" for e, c in flips))

    ref = "gemma-4-e4b-it 8-bit"
    for model in hosted_docs:
        a, b = maj[model], maj[ref]
        up = sorted(k for k in a if a[k] and not b[k])
        down = sorted(k for k in a if b[k] and not a[k])
        print(f"\n{model} vs {ref} (majority, held-out): hosted-only passes {len(up)}, local-only passes {len(down)}")
        print("  hosted-only: " + (", ".join(f"{e}/{c}" for e, c in up) or "none"))
        print("  local-only:  " + (", ".join(f"{e}/{c}" for e, c in down) or "none"))
    if hosted_docs:
        derived(hosted_docs, maj, local_walls)
        probe(maj)


def derived(hosted_docs, maj, local_walls):
    """Every other figure the post quotes, from the same checked files (lane 09 rule: no number without its command)."""
    print("\nDerived figures:")
    g = {m: v[1] for m, v in hosted_docs.items()}
    d = {m: v[0] for m, v in hosted_docs.items()}
    for m in g:
        p = d[m]["prices"]
        inp = g[m]["prompt_tokens_total"] * p["input_usd_per_mtok"] / 1e6
        print(f"- {m}: input share of cost {100 * inp / g[m]['cost_usd_total']:.1f}%; call/email $ per case "
              f"{g[m]['call-fields.cost_usd_per_case'] / g[m]['email-triage.cost_usd_per_case']:.1f}x; "
              f"output tokens per sitting {g[m]['gen_tokens_total'] / g[m]['n_cases']:.0f}; "
              f"$ per 1,000 held-out cases {1000 * g[m]['cost_usd_per_case']:.2f}")
    if {"claude-opus-5-5", "claude-sonnet-5-5"} <= set(g):
        print(f"- Opus / Sonnet $ per case: {g['claude-opus-5-5']['cost_usd_per_case'] / g['claude-sonnet-5-5']['cost_usd_per_case']:.2f}x")
    hw = [sum(r["wall_s"] for r in _heldout_rows(d[m])) / len(_heldout_rows(d[m])) for m in d]
    print(f"- wall per held-out case: hosted {min(hw):.2f}-{max(hw):.2f} s, local {min(local_walls):.2f}-{max(local_walls):.2f} s; "
          f"local/hosted {min(local_walls) / max(hw):.1f}-{max(local_walls) / min(hw):.1f}x")
    smoke = "results/hosted-gate-claude-haiku-5-5-smoke.json"
    sd, sg = _checked(smoke)
    spend = sg["cost_usd_total"] + sum(x["cost_usd_total"] for x in g.values())
    print(f"- lane spend (smoke ${sg['cost_usd_total']:.4f} + posted files): ${spend:.4f} of ${CEILING:.2f}")
    if {"claude-haiku-5-5", "claude-sonnet-5-5"} <= set(g):
        before = sg["cost_usd_total"] + g["claude-haiku-5-5"]["cost_usd_total"] + g["claude-sonnet-5-5"]["cost_usd_total"]
        proj = g["claude-sonnet-5-5"]["cost_usd_total"] * 2.5
        print(f"- Opus gate: spent ${before:.4f} + Sonnet ${g['claude-sonnet-5-5']['cost_usd_total']:.4f} x 2.5 = "
              f"${before + proj:.4f} vs ${CEILING:.2f} -> {'sit' if before + proj <= CEILING else 'not sat (ceiling)'}")
    ref = maj["gemma-4-e4b-it 8-bit"]
    hs = [maj[m] for m in g]
    allh = sorted(k for k in ref if all(h[k] for h in hs) and not ref[k])
    alll = sorted(k for k in ref if ref[k] and not any(h[k] for h in hs))
    print(f"- every hosted model passes, e4b 8-bit fails ({len(allh)}): " + ", ".join(f"{e}/{c}" for e, c in allh))
    print(f"- e4b 8-bit passes, every hosted model fails ({len(alll)}): " + ", ".join(f"{e}/{c}" for e, c in alll))


def probe(maj):
    """Case-level diff against wilco's 28 Oct frontier probe (same models, exams, provider; one pass each), read from the
    pinned wilco clone. Skipped with a note if the clone or the probe files are absent."""
    root = Path(results_dir_wilco()) / "docs" / "probes" / "frontier-direct-2026-10-28" / "results"
    print("\nAgainst wilco's 28 Oct probe (held-out, by case id):")
    if not root.exists():
        print(f"- skipped: {root} not found (set LAB_WILCO_DIR to the pin clone)")
        return
    import json
    for m in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"):
        if m not in maj:
            continue
        parts = []
        for e in EXAMS:
            f = root / f"{e}__{m}.json"
            old = {c["id"]: bool(c["passed"]) for c in json.loads(f.read_text())["cases"] if c["split"] == "heldout"}
            now = {c: v for (ex, c), v in maj[m].items() if ex == e}
            same_ids = set(old) == set(now)
            moved = sorted(c for c in now if c in old and old[c] != now[c])
            parts.append(f"{e} {sum(old.values())} -> {sum(now.values())} (ids match: {same_ids}; "
                         f"{len(moved)} moved{': ' + ', '.join(moved) if moved else ''})")
        print(f"- {m}: " + "; ".join(parts))


def results_dir_wilco():
    from lab import cases as C
    return C.WILCO_DIR


def _by_case(rows):
    by = {}
    for r in rows:
        by.setdefault((r["exam"], r["case"]), []).append(bool(r["passed"]))
    return by


if __name__ == "__main__":
    main()
