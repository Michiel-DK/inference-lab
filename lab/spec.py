"""Lane 06: speculative decoding under mlx-lm 0.31.3, metered per call, with the raw text kept so arms can be diffed.

The pre-registered protocol is in predictions/speculative.md (addendum, 10 Nov). In short:
  * a control arm (no draft) and a speculative arm (draft + num_draft_tokens) run through THIS adapter, so both record the
    same fields; lane 02's file has verdicts and token counts but no text, so a byte diff needs a control sat here;
  * `prompt_cache` is never passed: mlx-lm splits one list at len(model.layers) between target and draft, and the e4b
    builds 24 caches for 42 layers (KV sharing), so a shared list would hand the draft the wrong caches (a lab-made break);
  * a generation fault is recorded as a call (`fault`) and re-raised as ValueError, which wilco buckets as `error` and
    continues; the rows script counts those cases as runtime faults, never as model FAILs.

mlx is imported lazily so the tests run with no GPU.
"""
import hashlib
import time

from .metering import MeteredMLX, split_harmony, split_thinking


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def tokenizer_files_sha(path):
    """sha256 of tokenizer.json, so 'the draft shares the target's tokenizer' is a recorded fact, not an assumption."""
    from pathlib import Path
    p = Path(path) / "tokenizer.json"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def cache_classes(caches):
    out = {}
    for c in caches:
        out[type(c).__name__] = out.get(type(c).__name__, 0) + 1
    return out


class FaultError(ValueError):
    """A generation fault under the speculative path. A ValueError on purpose: wilco scores it `error` and moves on."""


class SpecMLX(MeteredMLX):
    """MeteredMLX plus an optional draft model; every call keeps its raw text, its hash and the draft-acceptance count."""

    def __init__(self, model_ref, draft_ref=None, num_draft_tokens=None, empty_answer_exc=None):
        super().__init__(model_ref, empty_answer_exc=empty_answer_exc)
        self.draft = None
        self.num_draft_tokens = num_draft_tokens
        if draft_ref is not None:
            from mlx_lm import load
            self.draft, dtok = load(draft_ref)
            if dtok.vocab_size != self.tokenizer.vocab_size:
                raise SystemExit(f"refused: draft vocab {dtok.vocab_size} != target vocab {self.tokenizer.vocab_size}")

    def generate(self, messages, model, temperature=0.0, max_tokens=512):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        if temperature != 0.0:
            raise ValueError(f"SpecMLX is greedy only; got temperature={temperature}")
        t0 = time.perf_counter()
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        kw = {} if self.draft is None else {"draft_model": self.draft, "num_draft_tokens": self.num_draft_tokens}
        self.mx.reset_peak_memory()
        parts, last, n_draft = [], None, 0
        try:
            for resp in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens,
                                        sampler=make_sampler(temp=0.0), **kw):
                parts.append(resp.text)
                n_draft += bool(resp.from_draft)
                last = resp
        except Exception as exc:
            raw = "".join(parts)
            self.calls.append({"prompt_tokens": len(prompt), "prompt_tps": None, "gen_tokens": None, "gen_tps": None,
                               "peak_memory_gb": float(self.mx.get_peak_memory() / 1e9), "finish_reason": "fault",
                               "wall_s": time.perf_counter() - t0, "content_chars": 0, "reasoning_chars": 0,
                               "spec": {"draft": self.draft is not None, "num_draft_tokens": self.num_draft_tokens,
                                        "fault": f"{type(exc).__name__}: {str(exc)[:400]}",
                                        "tokens_before_fault": last.generation_tokens if last else 0,
                                        "raw_chars": len(raw), "raw_sha256": sha(raw)}})
            raise FaultError(f"speculative-lane generation fault: {type(exc).__name__}: {str(exc)[:200]}") from exc
        wall_s = time.perf_counter() - t0
        raw = "".join(parts)
        tk = self.tokenizer
        starts_in = False
        if tk.has_thinking:
            ptxt = tk.decode(prompt)
            starts_in = ptxt.rfind(tk.think_start) > ptxt.rfind(tk.think_end)
        if self.harmony:
            text, reasoning = split_harmony(raw)
        else:
            text, reasoning = split_thinking(raw, tk.think_start if tk.has_thinking else None,
                                             tk.think_end if tk.has_thinking else None, starts_in)
        call = {"prompt_tokens": last.prompt_tokens, "prompt_tps": float(last.prompt_tps),
                "gen_tokens": last.generation_tokens, "gen_tps": float(last.generation_tps),
                "peak_memory_gb": float(last.peak_memory), "finish_reason": last.finish_reason,
                "wall_s": wall_s, "content_chars": len(text), "reasoning_chars": len(reasoning),
                "spec": {"draft": self.draft is not None, "num_draft_tokens": self.num_draft_tokens,
                         "from_draft_tokens": n_draft, "raw_chars": len(raw), "raw_sha256": sha(raw), "raw": raw}}
        self.calls.append(call)
        meta = {"prompt_tokens": last.prompt_tokens, "completion_tokens": last.generation_tokens,
                "total_tokens": last.prompt_tokens + last.generation_tokens,
                "finish_reason": last.finish_reason, "content_chars": len(text), "reasoning_chars": len(reasoning)}
        if not text.strip() and self.empty_answer_exc is not None:
            raise self.empty_answer_exc(f"model emitted no answer (finish_reason={last.finish_reason!r})",
                                        {"finish_reason": last.finish_reason, "completion_tokens": last.generation_tokens,
                                         "content_chars": len(text), "reasoning_chars": len(reasoning)})
        return text, meta


# ---- pure comparison helpers (scripts/lane06_rows.py and the tests use them)

def is_fault_row(row):
    return any((c.get("spec") or {}).get("fault") for c in row["calls"])


def acceptance(calls):
    """(tokens from the draft, tokens generated) over completed calls."""
    done = [c for c in calls if c.get("gen_tokens")]
    return sum(c["spec"]["from_draft_tokens"] for c in done), sum(c["gen_tokens"] for c in done)


def first_divergence(a, b):
    """Index of the first differing character of two strings (None if equal)."""
    if a == b:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def compare_rows(control, spec):
    """Per (exam, case): did the speculative arm produce the control's bytes, verdict and token counts?

    Both arms are one load each. Returns a list of dicts; a fault row is `fault`, never `differs`.
    """
    key = lambda r: (r["exam"], r["case"])
    ctl = {key(r): r for r in control}
    out = []
    for r in spec:
        c = ctl[key(r)]
        if is_fault_row(r):
            out.append({"exam": r["exam"], "case": r["case"], "split": r["split"], "status": "fault",
                        "fault": next(x["spec"]["fault"] for x in r["calls"] if (x.get("spec") or {}).get("fault"))})
            continue
        ch = [x["spec"]["raw_sha256"] for x in c["calls"]]
        sh = [x["spec"]["raw_sha256"] for x in r["calls"]]
        same = ch == sh
        div = None
        if not same:
            craw = "".join(x["spec"].get("raw", "") for x in c["calls"])
            sraw = "".join(x["spec"].get("raw", "") for x in r["calls"])
            div = first_divergence(craw, sraw)
        out.append({"exam": r["exam"], "case": r["case"], "split": r["split"],
                    "status": "identical" if same else "differs", "first_divergence_char": div,
                    "verdict_control": c["passed"], "verdict_spec": r["passed"],
                    "gen_tokens_control": [x["gen_tokens"] for x in c["calls"]],
                    "gen_tokens_spec": [x["gen_tokens"] for x in r["calls"]]})
    return out
