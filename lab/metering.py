"""Metering per case: tokens/s, peak memory and wall time from mlx-lm itself; Wh from a powermetrics capture (lab/energy.py).

What is recorded per model call (one case can make several: wilco re-asks once on an unparseable answer):
  prompt_tokens, prompt_tps, gen_tokens, gen_tps, peak_memory_gb, finish_reason
      read off the LAST GenerationResponse of `mlx_lm.stream_generate` (mlx-lm 0.31.3 fields of the same names;
      peak_memory is mx.get_peak_memory() / 1e9, reset with mx.reset_peak_memory() before every call)
  wall_s
      time.perf_counter() around the whole call, prompt templating included
Per case, lab/run.py adds start_utc / end_utc (the window the energy integration uses) and the case's wall_s.

mlx is imported lazily so `lab check` and the tests run with no GPU and no mlx installed.
Greedy only: wilco's three exams run at temperature 0; any other temperature is refused rather than silently sampled.
"""
import time

from .energy import METHOD as ENERGY_METHOD, energy_block  # noqa: F401  (re-exported: metering = speed + memory + energy)


class MeteredMLX:
    """A wilco Adapter (generate(messages, model, temperature, max_tokens) -> (text, meta)) over an in-process mlx-lm model."""

    def __init__(self, model_ref, empty_answer_exc=None):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.model, self.tokenizer = load(model_ref)
        self.empty_answer_exc = empty_answer_exc   # wilco's router.TerminationError, so an empty answer fails like it does there
        self.calls = []

    def generate(self, messages, model, temperature=0.0, max_tokens=512):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        if temperature != 0.0:
            raise ValueError(f"MeteredMLX is greedy only; got temperature={temperature}")
        t0 = time.perf_counter()
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        self.mx.reset_peak_memory()
        parts, last = [], None
        for resp in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens,
                                    sampler=make_sampler(temp=0.0)):
            parts.append(resp.text)
            last = resp
        wall_s = time.perf_counter() - t0
        text = "".join(parts)
        call = {"prompt_tokens": last.prompt_tokens, "prompt_tps": float(last.prompt_tps),
                "gen_tokens": last.generation_tokens, "gen_tps": float(last.generation_tps),
                "peak_memory_gb": float(last.peak_memory), "finish_reason": last.finish_reason,
                "wall_s": wall_s}
        self.calls.append(call)
        meta = {"prompt_tokens": last.prompt_tokens, "completion_tokens": last.generation_tokens,
                "total_tokens": last.prompt_tokens + last.generation_tokens,
                "finish_reason": last.finish_reason, "content_chars": len(text), "reasoning_chars": 0}
        if not text.strip() and self.empty_answer_exc is not None:
            raise self.empty_answer_exc(f"model emitted no answer (finish_reason={last.finish_reason!r})",
                                        {"finish_reason": last.finish_reason, "completion_tokens": last.generation_tokens,
                                         "content_chars": len(text), "reasoning_chars": 0})
        return text, meta
