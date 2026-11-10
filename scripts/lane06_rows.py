"""Lane 06 rows: every figure the post quotes, from the committed results files. Runs `lab check` on each file first and
exits 1 on drift. Optional: the e4b tokenizer (local snapshot, no network) to place each divergence in tokens.

    python3 scripts/lane06_rows.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import results  # noqa: E402
from lab import spec as S  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"
BASE = R / "dense-gemma-4-e4b-it-4bit.json"
CTL = R / "spec-gemma-4-e4b-it-4bit-control.json"
ARMS = {"k3": R / "spec-gemma-4-e4b-it-4bit-e2b-k3.json", "k3-repeat": R / "spec-gemma-4-e4b-it-4bit-e2b-k3-repeat.json"}
SMOKES = sorted(R.glob("spec-gate-*.json"))
WINDOW = 512
EXAMS = ("call-fields", "email-triage", "recap")


def tps(calls):
    calls = [c for c in calls if c.get("gen_tokens") and c.get("gen_tps")]
    t = sum(c["gen_tokens"] for c in calls)
    s = sum(c["gen_tokens"] / c["gen_tps"] for c in calls)
    return t / s if s else None


def window_class(row):
    """Where the target's sliding-window cache stands, from the CONTROL row (what the plain path would have sat)."""
    if any(c["prompt_tokens"] >= WINDOW for c in row["calls"]):
        return "starts>512"
    if any(c["prompt_tokens"] + (c["gen_tokens"] or 0) >= WINDOW for c in row["calls"]):
        return "crosses-in-gen"
    return "under-512"


def fmt(x, n=2):
    return "n/a" if x is None else f"{x:.{n}f}"


def main():
    files = [p for p in [BASE, CTL, *ARMS.values(), *SMOKES] if p.exists()]
    bad = 0
    for p in files:
        ok, msg = results.check(str(p))
        print(msg.splitlines()[0])
        bad += not ok
    if bad:
        print(f"{bad} file(s) drift; nothing below is posted", file=sys.stderr)
        return 1

    if not CTL.exists():
        print('\n(no control arm file yet)')
        smokes()
        return 0
    base, ctl = results.load(BASE), results.load(CTL)
    key = lambda r: (r["exam"], r["case"])
    b0 = {key(r): r for r in base["rows"] if r["load"] == 0}
    bmaj = {}
    for r in base["rows"]:
        bmaj.setdefault(key(r), []).append(r["passed"])
    bmaj = {k: sum(v) > len(v) - sum(v) for k, v in bmaj.items()}
    crow = {key(r): r for r in ctl["rows"]}

    print("\n## Control vs lane 02 load 0 (verdict + per-call gen_tokens, content_chars, reasoning_chars)")
    sig = lambda r: (r["passed"], [(c["gen_tokens"], c["content_chars"], c["reasoning_chars"]) for c in r["calls"]])
    same = [k for k in crow if sig(crow[k]) == sig(b0[k])]
    vsame = [k for k in crow if crow[k]["passed"] == b0[k]["passed"]]
    print(f"reproduced exactly: {len(same)} of {len(crow)}; verdict equal: {len(vsame)} of {len(crow)}")
    for k in sorted(set(crow) - set(same)):
        print(f"  differs: {k[0]}/{k[1]} lane02 {sig(b0[k])} control {sig(crow[k])}")
    print("control gen tok/s per exam vs lane 02 (all loads):",
          {e: (fmt(tps([c for r in ctl["rows"] if r["exam"] == e for c in r["calls"]]), 1),
               fmt(base["aggregates"][f"{e}.gen_tps"], 1)) for e in EXAMS})

    cls = {k: window_class(r) for k, r in crow.items()}
    print("window classes (from the control rows):", {c: sum(v == c for v in cls.values())
                                                       for c in ("under-512", "crosses-in-gen", "starts>512")})

    tok = None
    try:
        from transformers import AutoTokenizer
        from lab.run import resolve_model
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        tok = AutoTokenizer.from_pretrained(resolve_model(ctl["model"]["ref"])[0])
    except Exception as exc:  # the token position is a convenience; every other figure stands without it
        print(f"(no tokenizer: {type(exc).__name__}; divergence positions in chars only)")

    for arm, path in ARMS.items():
        if not path.exists():
            print(f"\n## {arm}: no results file")
            continue
        d = results.load(path)
        cmp = S.compare_rows(ctl["rows"], d["rows"])
        srow = {key(r): r for r in d["rows"]}
        print(f"\n## {arm} vs control ({path.name})")
        sp = d.get("speculative", {})
        print("stamp:", json.dumps({k: sp.get(k) for k in ("requested", "tokenizer_identical")}),
              "| draft", (sp.get("draft") or {}).get("snapshot"))
        for c in ("under-512", "crosses-in-gen", "starts>512"):
            ks = [x for x in cmp if cls[(x["exam"], x["case"])] == c]
            n = {s: sum(x["status"] == s for x in ks) for s in ("identical", "differs", "fault")}
            print(f"  {c:15s} n={len(ks):2d}  identical {n['identical']:2d} · differs {n['differs']:2d} · fault {n['fault']:2d}")
        for e in EXAMS:
            ks = [x for x in cmp if x["exam"] == e]
            print(f"  {e:13s} identical {sum(x['status'] == 'identical' for x in ks)} · differs "
                  f"{sum(x['status'] == 'differs' for x in ks)} · fault {sum(x['status'] == 'fault' for x in ks)} of {len(ks)}")
        faults = [x for x in cmp if x["status"] == "fault"]
        kinds = {}
        for x in faults:
            kinds[x["fault"].split(":")[0]] = kinds.get(x["fault"].split(":")[0], 0) + 1
        print("  fault kinds:", kinds)
        for x in faults[:3]:
            print(f"    e.g. {x['exam']}/{x['case']}: {x['fault'][:160]}")
        print("  differing / faulted ids:")
        for x in cmp:
            if x["status"] == "identical":
                continue
            k = (x["exam"], x["case"])
            where = ""
            if x["status"] == "differs" and tok is not None:
                # first call whose bytes differ; its divergence as an absolute cache position
                for cc, sc in zip(crow[k]["calls"], srow[k]["calls"]):
                    if cc["spec"]["raw_sha256"] != sc["spec"]["raw_sha256"]:
                        i = S.first_divergence(cc["spec"]["raw"], sc["spec"].get("raw", ""))
                        pos = cc["prompt_tokens"] + len(tok.encode(cc["spec"]["raw"][:i], add_special_tokens=False))
                        where = f" diverges at cache position ~{pos} ({'past' if pos >= WINDOW else 'under'} 512)"
                        break
            v = "" if x["status"] == "fault" else (f" verdict {x['verdict_control']}->{x['verdict_spec']}")
            print(f"    {x['split']:7s} {x['exam']}/{x['case']}: {x['status']} [{cls[k]}]{v}{where}")

        flips_c = [k for k in srow if srow[k]["passed"] != crow[k]["passed"]]
        flips_b = [k for k in srow if srow[k]["passed"] != bmaj[k]]
        up = lambda ks, ref: sum(srow[k]["passed"] and not ref(k) for k in ks)
        print(f"  verdict flips vs control: {len(flips_c)} (+{up(flips_c, lambda k: crow[k]['passed'])}/"
              f"-{len(flips_c) - up(flips_c, lambda k: crow[k]['passed'])}); "
              f"vs lane 02 majority: {len(flips_b)} (+{up(flips_b, lambda k: bmaj[k])}/-{len(flips_b) - up(flips_b, lambda k: bmaj[k])})")
        past = lambda k: max(c["prompt_tokens"] + (c["gen_tokens"] or 0) for c in srow[k]["calls"]) >= WINDOW
        for lab_, ks in (("arm sitting passed 512", [k for k in srow if past(k)]),
                         ("arm sitting stayed under 512", [k for k in srow if not past(k)])):
            st = {(x["exam"], x["case"]): x["status"] for x in cmp}
            print(f"  {lab_}: n={len(ks)}; bytes differ {sum(st[k] == 'differs' for k in ks)}; verdict flips vs control "
                  f"{sum(k in flips_c for k in ks)}")
        em = [k for k in srow if k[0] == "email-triage" and past(k)]
        mx_ = [max(c["prompt_tokens"] + (c["gen_tokens"] or 0) for c in srow[k]["calls"]) for k in em]
        print(f"  (post-hoc split, classed on the arm's own length) email sittings past 512: {len(em)}, reaching "
              f"{min(mx_, default=None)}-{max(mx_, default=None)}; their flips {sum(k in flips_c for k in em)}")
        if tok is not None and arm == "k3":
            print("  email flips: the arm's text around the point its first call's cache reaches 512")
            for k in em:
                if k not in flips_c:
                    continue
                c0 = srow[k]["calls"][0]
                raw = c0["spec"]["raw"]
                need = WINDOW - c0["prompt_tokens"]
                ids = tok.encode(raw, add_special_tokens=False)
                if need >= len(ids):
                    print(f"    {k[1]}: first call ends under 512 (a later call passes it)")
                    continue
                cut = len(tok.decode(ids[:need]))
                print(f"    {k[1]}: ...{raw[max(0, cut - 150):cut]!r} || {raw[cut:cut + 150]!r}...")
        cpast = [k for k in crow if any(c["prompt_tokens"] + (c["gen_tokens"] or 0) >= WINDOW for c in crow[k]["calls"])]
        print(f"  control sittings past 512: {len(cpast)}, passing {sum(crow[k]['passed'] for k in cpast)}"
              f" (same ids in lane 02 majority: {sum(bmaj[k] for k in cpast)})")
        nf_flips = [k for k in flips_c if not S.is_fault_row(srow[k])]
        print(f"  of the flips vs control, on non-fault cases: {len(nf_flips)}")
        hs = lambda rows: {e: sum(r["passed"] for r in rows if r["exam"] == e and r["split"] == "heldout") for e in EXAMS}
        h_s, h_c = hs(d["rows"]), hs(ctl["rows"])
        h_b = {e: base["aggregates"][f"{e}.heldout_majority_pass"] for e in EXAMS}
        print(f"  held-out passes (call · email · recap = total): arm {h_s} = {sum(h_s.values())}; control {h_c} = "
              f"{sum(h_c.values())}; lane 02 majority {h_b} = {sum(h_b.values())}")
        nfault_ho = sum(S.is_fault_row(r) for r in d["rows"] if r["split"] == "heldout")
        print(f"  held-out cases that faulted (not model FAILs): {nfault_ho} of {sum(r['split'] == 'heldout' for r in d['rows'])}")

        print("  speed, token-weighted gen tok/s (arm / control = ratio), over cases that did not fault:")
        for e in (*EXAMS, "all"):
            ks = [k for k in srow if (e == "all" or k[0] == e) and not S.is_fault_row(srow[k])]
            a_ = tps([c for k in ks for c in srow[k]["calls"]])
            c_ = tps([c for k in ks for c in crow[k]["calls"]])
            acc = S.acceptance([c for k in ks for c in srow[k]["calls"]])
            r_ = None if not (a_ and c_) else a_ / c_
            lb = base["aggregates"].get(f"{e}.gen_tps") if e != "all" else base["aggregates"]["gen_tps"]
            print(f"    {e:13s} n={len(ks):2d} arm {fmt(a_, 1)} / control {fmt(c_, 1)} = x{fmt(r_, 3)}"
                  f" (lane 02 {fmt(lb, 1)}; arm/lane02 x{fmt(a_ / lb if a_ and lb else None, 3)});"
                  f" acceptance {acc[0]}/{acc[1]} = {fmt(acc[0] / acc[1] if acc[1] else None, 3)}")
        ks = [x for x in cmp if x["status"] == "identical"]
        a_ = tps([c for x in ks for c in srow[(x["exam"], x["case"])]["calls"]])
        c_ = tps([c for x in ks for c in crow[(x["exam"], x["case"])]["calls"]])
        acc = S.acceptance([c for x in ks for c in srow[(x["exam"], x["case"])]["calls"]])
        print(f"    byte-identical cases only: n={len(ks)} arm {fmt(a_, 1)} / control {fmt(c_, 1)} = "
              f"x{fmt(a_ / c_ if a_ and c_ else None, 3)}; acceptance {acc[0]}/{acc[1]}")
        cnt = lambda rows: (sum(len(r["calls"]) for r in rows), sum("wilco_error" in r for r in rows),
                            sum(c["gen_tokens"] or 0 for r in rows for c in r["calls"]))
        print("  calls · cases in wilco's error bucket (no parseable answer) · gen tokens: arm %s, control %s"
              % (cnt(d["rows"]), cnt(ctl["rows"])))
        print("  identical ids:", [f"{x['exam']}/{x['case']} ({x['split']})" for x in cmp if x["status"] == "identical"])
        print("  flips by exam and split:", {f"{e}/{sp}": sum(1 for k in flips_c if k[0] == e and srow[k]["split"] == sp)
                                             for e in EXAMS for sp in ("train", "heldout")})
        wall = lambda rows: sum(r["wall_s"] for r in rows) / len(rows)
        print(f"  wall per case: arm {wall(d['rows']):.2f} s, control {wall(ctl['rows']):.2f} s; "
              f"peak GB arm {d['aggregates']['peak_memory_gb_max']:.2f}, control {ctl['aggregates']['peak_memory_gb_max']:.2f}; "
              f"Wh/case arm {d['aggregates']['wh_per_case']}, control {ctl['aggregates']['wh_per_case']}")

    if all(p.exists() for p in ARMS.values()):
        a, b = (results.load(p) for p in ARMS.values())
        ra = {key(r): r for r in a["rows"]}
        rb = {key(r): r for r in b["rows"]}
        sigb = lambda r: [(c.get("spec") or {}).get("raw_sha256") for c in r["calls"]] + [r["passed"]]
        same = sum(sigb(ra[k]) == sigb(rb[k]) for k in ra)
        print(f"\n## repeat: k3 vs k3-repeat byte-identical (and same verdict) on {same} of {len(ra)}")
        for k in ra:
            if sigb(ra[k]) != sigb(rb[k]):
                print(f"  moved: {k[0]}/{k[1]} {ra[k]['passed']}->{rb[k]['passed']}")

    smokes()
    return 0


def smokes():
    if SMOKES:
        print("\n## smokes (train only; wiring and gate)")
        pairs = [("spec-gate-qwen2.5-coder-1.5b-control-smoke", "spec-gate-qwen2.5-coder-1.5b-selfdraft-smoke"),
                 ("spec-gate-gemma-4-e4b-it-4bit-control-smoke", "spec-gate-gemma-4-e4b-it-4bit-e2b-k3-smoke")]
        for c, s in pairs:
            pc, ps = R / f"{c}.json", R / f"{s}.json"
            if not (pc.exists() and ps.exists()):
                continue
            dc, ds = results.load(pc), results.load(ps)
            for x in S.compare_rows(dc["rows"], ds["rows"]):
                acc = S.acceptance(next(r for r in ds["rows"] if r["case"] == x["case"])["calls"])
                print(f"  {s}: {x['exam']}/{x['case']} {x['status']}"
                      f"{'' if x['status'] == 'fault' else ' at char ' + str(x['first_divergence_char'])}"
                      f"; acceptance {acc[0]}/{acc[1]}")
            print(f"  {s}: gen tok/s {fmt(ds['aggregates']['gen_tps'], 1)} vs control {fmt(dc['aggregates']['gen_tps'], 1)}"
                  f" = x{fmt(ds['aggregates']['gen_tps'] / dc['aggregates']['gen_tps'], 3)}")


if __name__ == "__main__":
    sys.exit(main())
