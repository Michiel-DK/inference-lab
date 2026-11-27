"""Lane 15 post: every number the LESSONS row, SHOWCASE and the article quote, re-derived from the committed rows (read-only).

    python3 scripts/lane15_rows.py            # prints the tables, the prediction lines, both kills, the decider; checks POSTED
Exit 1 when a number in POSTED disagrees with the rows at its posted precision, or when the rows are not 10 ok trials per
condition. Input: docs/probes/cold-start-worker/rows.jsonl (phase "trial" only; the two "gate" rows are reported apart).

Stages (predictions/cold-start-worker.md, pre-run notes), in seconds:
  s1 spawn (driver time.time()) -> imports done (worker); s1a of it: spawn -> `import runpod` done
  s2 imports done -> mlx_lm.load returned (worker perf_counter)
  s3 load returned -> first successful GET / (driver, 20 ms poll)
  s4 request 1 sent (driver) -> its first token (handler)
  s5 first token -> last (64th) token (handler perf_counter)
  gaps: ready -> request sent; last token -> response received (driver)
  cold start (Runpod's definition) = s1+s2+s3; fresh start to first token = s1+s2+s3+s4
  C = SIGCONT (driver) -> request 3's first token, after a 60 s SIGSTOP of the whole process group
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "docs" / "probes" / "cold-start-worker" / "rows.jsonl"

# The numbers posted in LESSONS.md / SHOWCASE.md / the article, at the precision posted. Filled after the run.
POSTED = {
    'A.saved_by_resume': (3.6, 1),
    'B.saved_by_resume': (5.4, 1),
    'A.load_gb_per_s': (2.8, 1),
    'B.load_gb_per_s': (2.8, 1),
    'bytes_B_over_A': (1.73, 2),
    'A.gen_tps1.median': (45.7, 1),
    'B.gen_tps1.median': (29.7, 1),
    'A.gaps.max_ms': (1.0, 0),
    'B.gaps.max_ms': (2.0, 0),
    'A.swap_growth_mb.max': (-16.0, 0),
    'B.swap_growth_mb.max': (-1052.0, 0),
    'gate.A.ftft': (8.1, 2),
    'gate.A.s3': (2.75, 2),
    'gate.A.s4': (2.46, 2),
    'gate.A.prefill_tps1': (95.0, 0),
    'A.s1.median': (1.3, 2),
    'A.s1a.median': (0.06, 2),
    'A.s1b.median': (1.23, 2),
    'A.s2.median': (1.87, 2),
    'A.s3.median': (0.37, 2),
    'A.cold.median': (3.53, 2),
    'A.s4.median': (0.81, 2),
    'A.ftft.median': (4.34, 2),
    'A.s5.median': (1.42, 2),
    'A.e2e.median': (5.76, 2),
    'A.ttft2.median': (0.64, 2),
    'A.ttft3.median': (0.72, 2),
    'A.C.median': (0.72, 2),
    'A.cat_s.median': (0.82, 2),
    'B.s1.median': (1.29, 2),
    'B.s1a.median': (0.06, 2),
    'B.s1b.median': (1.23, 2),
    'B.s2.median': (3.15, 2),
    'B.s3.median': (0.36, 2),
    'B.cold.median': (4.8, 2),
    'B.s4.median': (1.38, 2),
    'B.ftft.median': (6.18, 2),
    'B.s5.median': (2.18, 2),
    'B.e2e.median': (8.36, 2),
    'B.ttft2.median': (0.66, 2),
    'B.ttft3.median': (0.79, 2),
    'B.C.median': (0.79, 2),
    'B.cat_s.median': (1.32, 2),
    'A.ftft.min': (4.31, 2),
    'A.ftft.max': (4.49, 2),
    'A.cold.min': (3.5, 2),
    'A.cold.max': (3.7, 2),
    'B.ftft.min': (6.07, 2),
    'B.ftft.max': (6.31, 2),
    'B.cold.min': (4.73, 2),
    'B.cold.max': (4.92, 2),
    'A.peak_gb.max': (4.59, 2),
    'B.peak_gb.max': (8.29, 2),
    'A.ftft.spread': (1.04, 2),
    'B.ftft.spread': (1.04, 2),
    'A.ttft1_over_ttft2': (1.25, 2),
    'B.ttft1_over_ttft2': (2.1, 2),
    'B_over_A.s2': (1.68, 2),
    'B_minus_A.ftft': (1.84, 2),
    'A.ttft3_minus_ttft2': (0.07, 2),
    'B.ttft3_minus_ttft2': (0.13, 2),
    'A.kill1.gap_pct': (0.01, 2),
    'B.kill1.gap_pct': (0.02, 2),
    'A_ftft_over_CA': (6.1, 1),
    'A.ftft_over_C': (6.1, 1),
    'B.ftft_over_C': (7.9, 1),
    'A.s1.share_pct': (30, 0),
    'A.s2.share_pct': (43, 0),
    'A.s3.share_pct': (8, 0),
    'A.s4.share_pct': (19, 0),
    'B.s1.share_pct': (21, 0),
    'B.s2.share_pct': (51, 0),
    'B.s3.share_pct': (6, 0),
    'B.s4.share_pct': (22, 0),
    'A.prefill_tps1.median': (378, 0),
    'A.prefill_tps2.median': (464, 0),
    'B.prefill_tps1.median': (187, 0),
    'B.prefill_tps2.median': (451, 0),
    'A.n': (10, 0),
    'B.n': (10, 0),
    'A.texts_distinct': (1, 0),
    'B.texts_distinct': (1, 0),
}


def stages(r):
    m, ts = r["req1"]["module_stamps"], r["t_spawn"]
    q1, q2, q3 = r["req1"], r["req2"], r["req3"]
    s = {
        "s1a": m["import_runpod_done"][0] - ts,
        "s1": m["imports_done"][0] - ts,
        "s2": m["weights_loaded"][1] - m["imports_done"][1],
        "s3": r["t_ready"] - m["weights_loaded"][0],
        "gap_ready_send": q1["t_send"] - r["t_ready"],
        "s4": q1["stamps"]["first_token"][0] - q1["t_send"],
        "s5": q1["stamps"]["last_token"][1] - q1["stamps"]["first_token"][1],
        "gap_tail": q1["t_recv"] - q1["stamps"]["last_token"][0],
        "e2e": q1["t_recv"] - ts,
        "ttft2": q2["stamps"]["first_token"][0] - q2["t_send"],
        "ttft3": q3["stamps"]["first_token"][0] - q3["t_send"],
        "C": q3["stamps"]["first_token"][0] - r["t_cont"],
        "peak_gb": max(q["peak_memory_gb"] for q in (q1, q2, q3)),
        "cat_s": r["cat_s"],
        "gen_tokens": q1["gen_tokens"],
        "prefill_tps1": q1["prompt_tps"], "prefill_tps2": q2["prompt_tps"],
        "gen_tps1": q1["gen_tps"],
    }
    s["s1b"] = s["s1"] - s["s1a"]
    s["cold"] = s["s1"] + s["s2"] + s["s3"]
    s["ftft"] = s["cold"] + s["s4"]
    s["sum5"] = s["ftft"] + s["s5"]
    return s


def med(xs):
    return st.median(xs)


def derive(rows):
    out, per = {}, {}
    for c in ("A", "B"):
        rs = [r for r in rows if r["condition"] == c]
        S = [stages(r) for r in rs]
        per[c] = S
        col = lambda k: [s[k] for s in S]  # noqa: E731
        for k in ("s1a", "s1b", "s1", "s2", "s3", "gap_ready_send", "s4", "s5", "gap_tail", "e2e", "cold", "ftft",
                  "ttft2", "ttft3", "C", "cat_s", "prefill_tps1", "prefill_tps2", "gen_tps1"):
            out[f"{c}.{k}.median"] = med(col(k))
            out[f"{c}.{k}.min"] = min(col(k))
            out[f"{c}.{k}.max"] = max(col(k))
        out[f"{c}.n"] = len(S)
        out[f"{c}.peak_gb.max"] = max(col("peak_gb"))
        out[f"{c}.ftft.spread"] = max(col("ftft")) / min(col("ftft"))
        out[f"{c}.cold.spread"] = max(col("cold")) / min(col("cold"))
        tot = sum(out[f"{c}.{k}.median"] for k in ("s1", "s2", "s3", "s4"))
        for k in ("s1", "s2", "s3", "s4"):
            out[f"{c}.{k}.share_pct"] = 100 * out[f"{c}.{k}.median"] / tot
        # kill 1 on the median trial (the trial whose driver end-to-end is the median; 10 trials: the lower middle)
        srt = sorted(S, key=lambda s: s["e2e"])
        mid = srt[(len(srt) - 1) // 2]
        out[f"{c}.kill1.gap_pct"] = 100 * abs(mid["sum5"] - mid["e2e"]) / mid["e2e"]
        out[f"{c}.ttft1_over_ttft2"] = out[f"{c}.s4.median"] / out[f"{c}.ttft2.median"]
        out[f"{c}.ftft_over_C"] = out[f"{c}.ftft.median"] / out[f"{c}.C.median"]
        out[f"{c}.ttft3_minus_ttft2"] = out[f"{c}.ttft3.median"] - out[f"{c}.ttft2.median"]
        out[f"{c}.texts_distinct"] = len({q["text_sha256"] for r in rs for q in (r["req1"], r["req2"], r["req3"])})
        out[f"{c}.gen_tokens.min"] = min(col("gen_tokens"))
        out[f"{c}.swap_growth_mb.max"] = max(max(r["swap_used_mb_after"], r["swap_used_mb_paused"])
                                             - r["swap_used_mb_block_start"] for r in rs)
        out[f"{c}.saved_by_resume"] = out[f"{c}.ftft.median"] - out[f"{c}.C.median"]
        out[f"{c}.gaps.max_ms"] = 1000 * max(max(s["gap_ready_send"], s["gap_tail"]) for s in S)
        wb = rs[0]["weight_bytes"]
        out[f"{c}.load_gb_per_s"] = wb / 1e9 / out[f"{c}.s2.median"]
    out["B_over_A.s2"] = out["B.s2.median"] / out["A.s2.median"]
    out["B_minus_A.ftft"] = out["B.ftft.median"] - out["A.ftft.median"]
    out["A_ftft_over_CA"] = out["A.ftft.median"] / out["A.C.median"]
    out["weight_bytes.A"] = rows[0]["weight_bytes"] if rows[0]["condition"] == "A" else rows[1]["weight_bytes"]
    out["weight_bytes.B"] = next(r["weight_bytes"] for r in rows if r["condition"] == "B")
    out["bytes_B_over_A"] = out["weight_bytes.B"] / out["weight_bytes.A"]
    return out, per


def gate_stats(allrows):
    g = {}
    for r in allrows:
        if r["phase"] == "gate":
            s = stages(r)
            g[f"gate.{r['condition']}.ftft"] = s["ftft"]
            g[f"gate.{r['condition']}.prefill_tps1"] = s["prefill_tps1"]
            g[f"gate.{r['condition']}.s3"] = s["s3"]
            g[f"gate.{r['condition']}.s4"] = s["s4"]
    return g


# The headline values as the posts print them: each must appear, at this precision, in the lane's LESSONS row AND its
# SHOWCASE section, so an edit to the text that leaves the rows behind fails this check (refuter of 27 Nov).
TEXT_KEYS = [("A.ftft.median", 2), ("B.ftft.median", 2), ("A.cold.median", 2), ("B.cold.median", 2), ("A.C.median", 2),
             ("B.C.median", 2), ("A_ftft_over_CA", 1), ("B.ftft_over_C", 1), ("B_over_A.s2", 2), ("bytes_B_over_A", 2),
             ("A.peak_gb.max", 2), ("B.peak_gb.max", 2), ("A.ttft1_over_ttft2", 2), ("B.ttft1_over_ttft2", 2),
             ("B_minus_A.ftft", 2), ("A.s2.median", 2), ("B.s2.median", 2), ("A.s4.median", 2), ("B.s4.median", 2),
             ("A.ttft3_minus_ttft2", 2), ("B.ttft3_minus_ttft2", 2), ("A.ftft.spread", 2), ("gate.A.ftft", 2)]


def post_texts():
    les = next(line for line in (ROOT / "LESSONS.md").read_text().splitlines() if line.startswith("| cold-start-worker"))
    show = (ROOT / "SHOWCASE.md").read_text()
    i = show.index("## Lane 15 cold-start-worker")
    j = show.find("\n## ", i + 1)
    return {"LESSONS.md row": les, "SHOWCASE.md lane 15": show[i:j if j > 0 else None]}


NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
# Numbers allowed in the LESSONS row's result and verdict cells that are not row-derived, each with its reason.
CONSTANTS = {
    "0": "exit code of this script", "3": "stage label (1-3)", "64": "max new tokens (design)",
    "1.0": "prediction threshold (C <= 1.0 s)", "5": "prediction / decider threshold (>= 5x)", "15": "decider threshold (15 s)",
    "324": "Runpod's quoted median", "91": "Runpod's quoted median", "2511.05502": "cited paper id", "31": "cited paper's MLX total",
}


def text_drift(o):
    """Two checks on the posted text. (1) Every number in the LESSONS row's result and verdict cells is, as a whole token, a
    POSTED value at its posted precision, the POSTED count, or a named CONSTANT. (2) Each TEXT_KEYS value appears as a whole
    token in the LESSONS row and in the SHOWCASE lane-15 section."""
    drift = []
    allowed = {f"{o[k]:.{nd}f}" for k, (_, nd) in POSTED.items()} | {str(len(POSTED))} | set(CONSTANTS)
    texts = post_texts()
    cells = [c.strip() for c in texts["LESSONS.md row"].split(" | ")]
    for name, cell in (("result", cells[2]), ("verdict", cells[3])):
        for tok in NUM.findall(cell):
            if tok not in allowed:
                drift.append(f"LESSONS.md row, {name} cell: {tok} is not a row-derived posted value or a named constant")
    for where, text in texts.items():
        toks = set(NUM.findall(text))
        for key, nd in TEXT_KEYS:
            want = f"{o[key]:.{nd}f}"
            if want not in toks:
                drift.append(f"{where}: {key} = {want} (from the rows) is not in the text")
    return drift


def verdict(ok):
    return "held" if ok else "REFUTED"


def main():
    allrows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    rows = [r for r in allrows if r["phase"] == "trial"]
    bad = [f"{r['condition']}{r['trial']}" for r in rows if not r.get("ok")]
    counts = {c: sum(1 for r in rows if r["condition"] == c and r.get("ok")) for c in ("A", "B")}
    if bad or counts != {"A": 10, "B": 10}:
        print(f"rows not complete: ok counts {counts}, failed {bad}")
        return 1
    o, per = derive(rows)
    o.update(gate_stats(allrows))

    print(f"rows: {ROWS.relative_to(ROOT)}; trial rows {len(rows)} (A {counts['A']}, B {counts['B']}); "
          f"gate rows {sum(1 for r in allrows if r['phase'] == 'gate')} (not counted)")
    for r in (r for r in allrows if r["phase"] == "gate"):   # the first worker ever started in the new venv, n = 1 each
        g = stages(r)
        print(f"gate {r['condition']}{r['trial']} (not counted): imports {g['s1']:.2f}  weights {g['s2']:.2f}  "
              f"server {g['s3']:.2f}  first request {g['s4']:.2f}  fresh start to first token {g['ftft']:.2f} s")
    print(f"weight bytes A {o['weight_bytes.A']:,}  B {o['weight_bytes.B']:,}  ratio {o['weight_bytes.B'] / o['weight_bytes.A']:.2f}")
    print("\nstage table, seconds, median [min, max] over 10 fresh workers")
    names = [("s1", "1 imports (spawn -> runpod + mlx_lm imported)"), ("s1a", "  of which spawn -> runpod imported"),
             ("s1b", "  of which mlx_lm import"), ("s2", "2 weights (mlx_lm.load, eager eval)"),
             ("s3", "3 server accepting (load -> first GET / answers)"), ("cold", "= cold start, Runpod's definition (1-3)"),
             ("gap_ready_send", "  gap: ready -> request sent"), ("s4", "4 first request -> first token"),
             ("ftft", "= fresh start to first token (1-4)"), ("s5", "5 first -> 64th token"),
             ("gap_tail", "  gap: last token -> response at driver"), ("e2e", "driver: spawn -> response 1 received"),
             ("ttft2", "request 2: sent -> first token (same worker)"),
             ("ttft3", "request 3 (after 60 s SIGSTOP): sent -> first token"),
             ("C", "C: SIGCONT -> request 3's first token"), ("cat_s", "warm-up cat of the weight files")]
    print(f"{'':52s} {'A 4-bit':>22s} {'B 8-bit':>22s}")
    for k, label in names:
        cells = [f"{o[f'{c}.{k}.median']:7.3f} [{o[f'{c}.{k}.min']:.3f}, {o[f'{c}.{k}.max']:.3f}]" for c in ("A", "B")]
        print(f"{label:52s} {cells[0]:>22s} {cells[1]:>22s}")
    print("\nshare of fresh start to first token (median stage / sum of the four medians), %")
    for k in ("s1", "s2", "s3", "s4"):
        print(f"  {k}: A {o[f'A.{k}.share_pct']:.0f}  B {o[f'B.{k}.share_pct']:.0f}")
    for c in ("A", "B"):
        print(f"{c}: peak {o[f'{c}.peak_gb.max']:.2f} GB; swap growth max {o[f'{c}.swap_growth_mb.max']:.0f} MB; "
              f"prefill tok/s req1 {o[f'{c}.prefill_tps1.median']:.0f} vs req2 {o[f'{c}.prefill_tps2.median']:.0f}; "
              f"gen tok/s {o[f'{c}.gen_tps1.median']:.1f}; gen tokens min {o[f'{c}.gen_tokens.min']}; "
              f"distinct texts over 30 requests {o[f'{c}.texts_distinct']}")

    for c in ("A", "B"):
        print(f"{c}: weights read at {o[f'{c}.load_gb_per_s']:.1f} GB/s (bytes / stage 2); resume saves "
              f"{o[f'{c}.saved_by_resume']:.1f} s per run (fresh start to first token - C); largest measured gap "
              f"{o[f'{c}.gaps.max_ms']:.1f} ms")
    print(f"bytes B / A {o['bytes_B_over_A']:.2f}; gate A0: fresh start to first token {o['gate.A.ftft']:.2f} s, server "
          f"{o['gate.A.s3']:.2f} s, first request {o['gate.A.s4']:.2f} s, prefill {o['gate.A.prefill_tps1']:.0f} tok/s")
    print("\nprediction lines")
    lines = [
        ("A fresh start to first token median in [5, 12] s", o["A.ftft.median"], 5 <= o["A.ftft.median"] <= 12),
        ("A imports in [1.5, 5] s", o["A.s1.median"], 1.5 <= o["A.s1.median"] <= 5),
        ("A weights in [2.5, 4] s", o["A.s2.median"], 2.5 <= o["A.s2.median"] <= 4),
        ("A first request -> first token in [0.5, 3] s", o["A.s4.median"], 0.5 <= o["A.s4.median"] <= 3),
        ("B weights / A weights in [1.3, 1.6]", o["B_over_A.s2"], 1.3 <= o["B_over_A.s2"] <= 1.6),
        ("B weights in [4, 5.5] s", o["B.s2.median"], 4 <= o["B.s2.median"] <= 5.5),
        ("B ftft - A ftft in [1, 4] s", o["B_minus_A.ftft"], 1 <= o["B_minus_A.ftft"] <= 4),
        ("C-A first token within 1.0 s (median)", o["A.C.median"], o["A.C.median"] <= 1.0),
        ("A ftft / C-A >= 5", o["A_ftft_over_CA"], o["A_ftft_over_CA"] >= 5),
        ("A ttft req1 / req2 >= 1.5 (unsure)", o["A.ttft1_over_ttft2"], o["A.ttft1_over_ttft2"] >= 1.5),
        ("B ttft req1 / req2 >= 1.5 (unsure)", o["B.ttft1_over_ttft2"], o["B.ttft1_over_ttft2"] >= 1.5),
        ("A spread (slowest / fastest ftft) <= 1.5", o["A.ftft.spread"], o["A.ftft.spread"] <= 1.5),
        ("B spread (slowest / fastest ftft) <= 1.5", o["B.ftft.spread"], o["B.ftft.spread"] <= 1.5),
    ]
    for label, v, ok in lines:
        print(f"  {verdict(ok):8s} {label}: {v:.3f}")
    print(f"  (also) C-B median {o['B.C.median']:.3f} s; B ftft / C-B {o['B.ftft_over_C']:.2f}; "
          f"request 3 - request 2 time to first token: A {o['A.ttft3_minus_ttft2']:+.3f} s, B {o['B.ttft3_minus_ttft2']:+.3f} s")

    print("\nkills")
    k1 = [o[f"{c}.kill1.gap_pct"] for c in ("A", "B")]
    print(f"  kill 1 (median trial: |sum of stages 1-5 - driver end-to-end| > 10%): A {k1[0]:.2f}%  B {k1[1]:.2f}%  "
          f"-> {'FIRED' if max(k1) > 10 else 'not fired'}")
    k2 = o["A.ftft.spread"] > 2 and o["B.ftft.spread"] > 2
    print(f"  kill 2 (slowest > 2x fastest in both A and B): A {o['A.ftft.spread']:.2f}  B {o['B.ftft.spread']:.2f}  "
          f"-> {'FIRED' if k2 else 'not fired'}")
    print("\ndecider (operator): flips when B fresh start to first token median > 15 s, or B ftft / C-B < 5")
    flip = o["B.ftft.median"] > 15 or o["B.ftft_over_C"] < 5
    print(f"  B ftft median {o['B.ftft.median']:.2f} s; B ftft / C-B {o['B.ftft_over_C']:.2f} -> "
          f"{'FLIPS' if flip else 'does not flip'}")

    drift = text_drift(o)
    for key, (val, nd) in POSTED.items():
        got = round(o[key], nd)
        if got != val:
            drift.append(f"{key}: posted {val}, rows give {got}")
    print(f"\nPOSTED check: {len(POSTED)} numbers against the rows, every number in the LESSONS row's result and verdict "
          f"cells traced to one of them or a named constant, {len(TEXT_KEYS)} also found in the SHOWCASE section; " + ("all match" if not drift else "DRIFT:\n  " + "\n  ".join(drift)))
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
