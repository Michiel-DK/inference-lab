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


def split_thinking(text, think_start, think_end, starts_in_thinking=False):
    """(content, reasoning): what mlx_lm.server 0.31.3 puts in message.content vs message.reasoning.

    The server's state machine (server.py: transitions normal -> reasoning on think_start, reasoning -> normal on
    think_end; initial state reasoning when the prompt's last think_start is after its last think_end) applied to the
    finished text. wilco grades `content` only (router.py, TERMINATION-DETECT D2: no fallback to thinking), so an
    unterminated thought leaves content empty, which the adapter turns into wilco's TerminationError.
    """
    if not think_start:
        return text, ""
    content, reasoning, rest, thinking = [], [], text, starts_in_thinking
    while rest:
        mark = think_end if thinking else think_start
        i = rest.find(mark)
        head, rest = (rest, "") if i < 0 else (rest[:i], rest[i + len(mark):])
        (reasoning if thinking else content).append(head)
        if i >= 0:
            thinking = not thinking
    return "".join(content), "".join(reasoning)

HARMONY_FINAL = "<|channel|>final<|message|>"
HARMONY_STOPS = ("<|return|>", "<|end|>", "<|call|>")


def split_harmony(text):
    """(content, reasoning) for OpenAI's Harmony format (gpt-oss): the answer is the `final` channel's message.

    mlx-lm 0.31.3 does not parse Harmony (`tokenizer_utils._infer_thinking` knows `<think>` and Gemma's
    `<|channel>thought` only), so mlx_lm.server would hand wilco the analysis channel as the answer: lane 02's void
    again. Rule (pre-registered in predictions/moe-vs-dense.md, addendum of 9 Nov): content = the text after the last
    `<|channel|>final<|message|>`, up to the first Harmony stop token; everything else is reasoning. Output with no
    `<|channel|>` marker at all is content as-is; channels opened but no final one = empty content (TerminationError).
    """
    if "<|channel|>" not in text:
        return text, ""
    i = text.rfind(HARMONY_FINAL)
    if i < 0:
        return "", text
    content = text[i + len(HARMONY_FINAL):]
    cut = min([j for j in (content.find(s) for s in HARMONY_STOPS) if j >= 0], default=len(content))
    return content[:cut], text[:i]


from .energy import METHOD as ENERGY_METHOD, energy_block  # noqa: F401  (re-exported: metering = speed + memory + energy)


class MeteredMLX:
    """A wilco Adapter (generate(messages, model, temperature, max_tokens) -> (text, meta)) over an in-process mlx-lm model."""

    def __init__(self, model_ref, empty_answer_exc=None):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.model, self.tokenizer = load(model_ref)
        self.harmony = HARMONY_FINAL.split("final")[0] in self.tokenizer.get_vocab()   # gpt-oss: '<|channel|>' is a token
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
        raw = "".join(parts)
        tk = self.tokenizer
        starts_in = False
        if tk.has_thinking:
            ptxt = tk.decode(prompt) if not isinstance(prompt, str) else prompt
            starts_in = ptxt.rfind(tk.think_start) > ptxt.rfind(tk.think_end)
        if self.harmony:
            text, reasoning = split_harmony(raw)
        else:
            text, reasoning = split_thinking(raw, tk.think_start if tk.has_thinking else None,
                                             tk.think_end if tk.has_thinking else None, starts_in)
        call = {"prompt_tokens": last.prompt_tokens, "prompt_tps": float(last.prompt_tps),
                "gen_tokens": last.generation_tokens, "gen_tps": float(last.generation_tps),
                "peak_memory_gb": float(last.peak_memory), "finish_reason": last.finish_reason,
                "wall_s": wall_s, "content_chars": len(text), "reasoning_chars": len(reasoning)}
        self.calls.append(call)
        meta = {"prompt_tokens": last.prompt_tokens, "completion_tokens": last.generation_tokens,
                "total_tokens": last.prompt_tokens + last.generation_tokens,
                "finish_reason": last.finish_reason, "content_chars": len(text), "reasoning_chars": len(reasoning)}
        if not text.strip() and self.empty_answer_exc is not None:
            raise self.empty_answer_exc(f"model emitted no answer (finish_reason={last.finish_reason!r})",
                                        {"finish_reason": last.finish_reason, "completion_tokens": last.generation_tokens,
                                         "content_chars": len(text), "reasoning_chars": len(reasoning)})
        return text, meta
