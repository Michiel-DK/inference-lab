# speculative

Lane 06 (written 6 Nov 2026, before any lane-06 run): where mlx-lm supports a draft model, what tokens/s gain do the exams
see, with pass fraction unchanged?

Rank letters: **A** measured on our cases on this machine; **B** adjacent measurement; **C** read, not run.

## Prediction

**The pair matters more than the method.** mlx-lm 0.31.3 has `draft_model` / `num_draft_tokens` in `generate.py` (rank A). Two
outcomes, by pair:

1. **Gemma 4 target with a Gemma 4 draft (e4b + e2b) on mlx-lm 0.31.3: the harness breaks, and that is the finding.** The
   sandbox ran exactly this through `mlx_lm.server` on 1 Nov (`agent-sandbox/docs/probes/mlx-s1-2026-11-01`, rank A): every
   email-triage call raised `broadcast_shapes` in `scaled_dot_product_attention` once the context passed ~512 tokens, and below
   that the drafted text was wrong and 2× slower. I predict the in-process `generate` path reproduces it: 0 gradable answers on
   ≥ 80 of 83 cases, so no tokens/s number. Probability I put on this if the lane picks the Gemma pair: 0.8.
2. **A full-attention pair with a shared tokenizer (Qwen3-8B-4bit or Qwen3-14B-4bit target + Qwen3-0.6B-4bit draft, 0.3 GB;
   or the cached Llama-3.1-8B-Instruct-4bit + Llama-3.2-1B-Instruct) runs.** Against lane 02's row for the same target
   (`results/dense-<target>.json`):
   - **Outputs.** Byte-identical text on ≥ 80 of 83 cases; 0–3 cases differ in text; **0–1 verdict flips**, and I commit to
     0 held-out flips. I am predicting the text diff is *not* zero: fp16 batched verification of 3–4 positions is not bit-exact
     with one-token decode, and a near-tie argmax can swap. That is numerics, not a broken harness; the brief's "must be zero"
     is right for verdicts and too strict for bytes (see the report).
   - **Speed (`gen_tps`, `num_draft_tokens` 3–4).** email-triage 1.4–1.8× (short JSON, high acceptance); call-fields 1.3–1.6×;
     recap 1.1–1.4× (Dutch prose, lower acceptance). For an 8B target at ~24 tok/s that is 32–43 tok/s on email-triage. If the
     draft is more than a quarter of the target's bytes (e2b drafting e4b is ~60%), the gain is ≤ 1.2× or negative.
   - **Energy (`wh_per_case`, if metered).** 0.7–0.9× the baseline: wall falls by the speed-up, package power rises 10–20%
     with two models resident and the draft running every step.
   - **Memory.** Baseline + the draft's weights + a second KV cache: +0.4–0.7 GB for a 0.6B 4-bit draft.

**Kill clause (copied from the brief):** "speculative decoding must not change outputs at temperature 0; if it does, the
harness is broken, and that is the finding." The lane should state which reading of "outputs" it kills on; I predict it fires on
bytes (1–3 cases) and does not fire on verdicts.

## Mechanism

Speculative decoding runs the small draft for k tokens, then one batched forward of the target over those k positions accepts
the longest prefix the target would itself have produced under greedy decoding; the target's distribution is unchanged by
construction, so verdicts should not move, and tokens per second rise by the mean accepted length divided by the cost of a
draft step plus one target step over k positions. On a bandwidth-bound laptop the batched target step costs about the same as a
single decode step (the weights are read once), so the gain is set by acceptance: JSON with a fixed schema and copied field
values is highly predictable (email-triage, call-fields), free-form Dutch prose less so (recap). The draft's own weight read is
the overhead, which is why a 0.6B draft on an 8B target pays and a 2B-effective draft on a 4B target does not. The exactness
argument holds in exact arithmetic; in fp16 the verify pass computes logits for k positions in one batched matmul with a
different accumulation order than the single-token path, so near-tie tokens can differ, which is why I predict a handful of
byte diffs and no verdict flips. The Gemma failure is architectural, not numerical: the sliding-window cache on 35 layers has to
be rewound by k minus accepted tokens after each verify, and the S1 traceback shows the mask (k wide, prompt-length keys) being
broadcast against a 1,024-position window cache; mlx-lm 0.31.3 does not handle that combination, so the pair cannot run until
the engine does.
