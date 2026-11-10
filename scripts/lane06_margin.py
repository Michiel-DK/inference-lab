"""Lane 06 post-hoc probe (not pre-registered, ungraded): is a byte divergence a near-tie?

For each listed case, re-sit the CONTROL path (plain greedy, no draft) of the first call and record, per generated token,
the gap between the top-1 and top-2 log-probabilities. Then report the gap at the token where the draft arm's text first
departs from the control's (char offset from the committed results files), and that gap's rank among all the call's gaps.
A near-tie (gap far below the call's typical gap) reads as arithmetic; a large gap at the divergence reads as a broken
state. The re-sit's raw text must equal the control's committed raw (sha), or the probe says so and reports nothing.

    HF_HUB_OFFLINE=1 python3 scripts/lane06_margin.py results/<control>.json results/<arm>.json exam/case [exam/case ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import cases as C  # noqa: E402
from lab import results  # noqa: E402
from lab import spec as S  # noqa: E402
from lab.run import import_wilco_runner, resolve_model  # noqa: E402


class MarginProbe:
    def __init__(self, path):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.model, self.tokenizer = load(path)
        self.trace = None

    def generate(self, messages, model, temperature=0.0, max_tokens=512):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        mx = self.mx
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        segs, gaps = [], []
        for resp in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens,
                                    sampler=make_sampler(temp=0.0)):
            lp = resp.logprobs
            top2 = mx.topk(lp, 2)
            mx.eval(top2)
            hi, lo = sorted(top2.tolist(), reverse=True)
            segs.append(resp.text)
            gaps.append(hi - lo)
        self.trace = (segs, gaps)
        raise StopProbe()           # first call only; the case is not graded


class StopProbe(Exception):
    pass


def main(ctl_path, arm_path, ids):
    ctl, arm = results.load(ctl_path), results.load(arm_path)
    key = lambda r: f"{r['exam']}/{r['case']}"
    crow, srow = {key(r): r for r in ctl["rows"]}, {key(r): r for r in arm["rows"]}
    runner, _ = import_wilco_runner()
    probe = MarginProbe(resolve_model(ctl["model"]["ref"])[0])
    real = runner.adapter_for
    runner.adapter_for = lambda *a, **k: probe
    try:
        for cid in ids:
            exam, case = cid.split("/")
            craw = crow[cid]["calls"][0]["spec"]["raw"]
            sraw = srow[cid]["calls"][0]["spec"].get("raw", "")
            div = S.first_divergence(craw, sraw)
            if div is None:
                print(f"{cid}: first call identical; nothing to probe")
                continue
            doc = json.loads(C.cases_path(exam).read_text())
            c = next(x for x in doc["cases"] if x["id"] == case)
            try:
                runner.run_exam(runner.load_agent(exam), {**doc, "cases": [c]}, provider="mlx", model="probe")
            except StopProbe:
                pass
            segs, gaps = probe.trace
            text = "".join(segs)
            if text != craw:
                print(f"{cid}: re-sit text != committed control raw; probe void")
                continue
            pos, t = 0, None
            for i, sgm in enumerate(segs):
                if pos + len(sgm) > div:
                    t = i
                    break
                pos += len(sgm)
            # detokenizer segments can lag a token; take the smallest gap within one token of the divergence
            window = [j for j in (t - 1, t, t + 1) if j is not None and 0 <= j < len(gaps)]
            g = min(gaps[j] for j in window)
            rank = sum(x <= g for x in gaps) / len(gaps)
            med = sorted(gaps)[len(gaps) // 2]
            print(f"{cid}: divergence char {div} = gen token ~{t} of {len(gaps)}; top1-top2 gap {g:.4f} nats "
                  f"(median gap {med:.3f}; {rank:.3f} of tokens have a gap this small or smaller); "
                  f"control: {craw[max(0, div - 40):div + 25]!r} | arm: {sraw[max(0, div - 40):div + 25]!r}")
    finally:
        runner.adapter_for = real


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
