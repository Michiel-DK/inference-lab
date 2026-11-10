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

## Addendum (10 Nov 2026, before any Gemma or held-out sitting)

**The pair is forced, and it is outcome 1.** Lane 02's targets are e2b, e4b and Llama-3.1-8B (4-bit). The HF cache on this
machine holds weights for no Llama-3.2-1B (only its tokenizer and config; `find ~` turns up LoRA adapters, no base) and no
Qwen3. Downloading inside a run is not allowed. The one lane-02 target with a cached same-family draft is the e4b:

- target `mlx-community/gemma-4-e4b-it-4bit@475b9088` (lane 02's `results/dense-gemma-4-e4b-it-4bit.json`);
- draft `mlx-community/gemma-4-e2b-it-4bit@23876752` (vocab 262144 on both; the run stamps both `tokenizer.json` hashes);
- `num_draft_tokens` 3 (mlx-lm's CLI default, inside the band above). One value; no sweep.

So branch 2's rows (Qwen3 / Llama pair) are void for this run, not for the lane.

**Seen before this addendum, and disclosed here** (none of it a Gemma sitting):
1. mlx-lm 0.31.3 source (rank C until run): `speculative_generate_step` refuses only if a cache is not trimmable, and it
   checks the *fresh* caches. `RotatingKVCache.is_trimmable()` is `offset < max_size`, true at offset 0, so the guard
   passes on Gemma 4. Its sliding layers have a 512-token window (`sliding_window: 512` in both configs). Past 512,
   `trim(n)` just lowers `offset` and `_idx`, and the next multi-token verify re-orders the buffer as if it had wrapped. So I
   no longer expect a clean refusal; I expect wrong text or a shape fault once a cache passes 512.
2. Lane 02's e4b token counts (load 0, all 83 cases): every call-fields (32) and recap (23) prompt is over 512 tokens
   (1133–8935, 520–2603). Every email-triage prompt is under (194–436); prompt + generation stays under 512 on 12 of 28.
3. Wiring control (pre-registered here as wiring, not a lane number): `Qwen2.5-Coder-1.5B-bf16` drafting for itself on the
   3-case train smoke, plain `KVCache` (trimmable everywhere). Files `results/spec-gate-qwen2.5-coder-1.5b-{control,selfdraft}-smoke.json`.
   Acceptance 100% (1536 of 2048 tokens per call from the draft = k/(k+1)). 1 of 3 cases byte-identical (support-long-train,
   2 × 4096 tokens); 2 diverge at chars 8268 / 8726, inside a degenerate `<|repo_name|>` repetition, where the loop exits one
   repeat earlier or later. One verdict moved (overdue-invoice PASS → FAIL). Speed 0.58× (self-draft costs a full model).
   So the bytes are not exact even with a trimmable cache and a perfect draft: the batched verify is not the single-token
   path's arithmetic. That is the numerics clause above, now observed (rank A, one model).

**Arms** (one fresh load each, every case of the three exams, 83, train + held-out, through `lab spec`):
1. `spec-gemma-4-e4b-it-4bit-control`: no draft, same adapter (`lab/spec.py`), raw text and its sha256 kept per call.
2. `spec-gemma-4-e4b-it-4bit-e2b-k3`: draft e2b, k = 3.
3. `spec-gemma-4-e4b-it-4bit-e2b-k3-repeat`: arm 2 again, to tell a deterministic engine fault from run-to-run noise.
Before arm 1, the 3-case train smoke on arms 1 and 2 (`spec-gate-*-smoke`), which sits nothing held-out. Order fixed: smoke,
1, 2, 3. A fault never stops an arm: it is recorded per call and wilco scores the case `error`; the rows script counts it as a
runtime fault, never as a model FAIL.

**Predictions on these arms** (the original branch-1 line "0 gradable answers on ≥ 80 of 83" stands and is graded too):
- Control reproduces lane 02 load 0 (verdict and per-call `gen_tokens`, `content_chars`, `reasoning_chars`) on ≥ 80 of 83.
  Below 80, the comparison to lane 02 is void and only the control-vs-draft diff stands.
- Cases whose target cache never passes 512 (≈ 12 email-triage): byte-identical to control on ≥ 9 (short JSON-ish output;
  the Qwen near-tie needed ~2000 tokens of a degenerate loop).
- Cases that start over 512 (55 call-fields + recap): **≥ 45 of 55 differ from control or fault**. Fault vs wrong-text
  split: unsure; I lean wrong text (≥ 30 differ), because the reorder keeps shapes consistent, but the S1 probe saw
  `broadcast_shapes` through the server.
- Email cases that cross 512 during generation (≈ 16): ≥ 10 differ or fault.
- Verdicts, arm 2 vs control: ≥ 5 flips, net negative (wrong context = worse answers), held-out total below the control's
  by ≥ 3.
- Speed on the cases that are byte-identical: spec / control `gen_tps` 0.5–0.9× (the e2b pack is 3.3 GB on disk vs the
  e4b's 4.8 GB, so every draft step costs most of a target step). Acceptance (draft tokens / generated) 0.45–0.70.
- Peak memory: 5.45 → 8.5–9.5 GB, under the 11.45 GB working set.
- Arm 3 vs arm 2: identical bytes on ≥ 80 of 83 (the engine is deterministic; its fault is too).
- Wh: no powermetrics capture is scheduled, so every Wh row is void (`null`), not 0.

**Kill clause, the reading.** It fires on bytes, per case, on any case the draft arm completed without a fault. Every
such id is posted. Verdict flips are reported beside it, against the control and against lane 02's 3-load majority. The
Qwen smoke already shows bytes can move without a broken harness, so a byte diff on a case whose cache stayed under 512 is
numerics; a byte diff past 512 is attributed to the cache rewind only if the divergence point is after the cache passes
512 (the rows keep the raw text, so the script can locate it).
