# context-kv

Lane 05 (written 6 Nov 2026, before any lane-05 run): as prompts grow to 2k, 4k, 8k tokens (padded with the case's own
history, same answer key), what happens to pass fraction and latency, and does the KV-cache type change anything?

Rank letters: **A** measured on our cases on this machine; **B** adjacent measurement; **C** read, not run.

## Prediction

**Arms.** One dense (`gemma-4-e4b-it-4bit`, lane 02's row as baseline, `results/dense-gemma-4-e4b-it-4bit.json`) and lane 03's
MoE (`gpt-oss-20b-MXFP4-Q8`, baseline its lane-03 file). KV-cache types mlx-lm 0.31.3 exposes (`mlx_lm/models/cache.py`, rank A):
the default `KVCache` (fp16), `QuantizedKVCache` via `kv_bits` 8 or 4 with `kv_group_size` 64 and `quantized_kv_start`, and
`RotatingKVCache` via `max_kv_size`. Gemma 4 already uses a 512-token sliding cache on 35 of its 42 layers and shares KV across
18 (`config.json`, rank A), so only its 7 global layers hold the long context.

**Pass fraction vs length (held-out, default KV).** e4b: 2k within ±1 of baseline on every exam; 4k −0 to −2; 8k −1 to −3 per
exam, the largest drop on recap (the padding adds messages the recap must not pick up) then call-fields, email-triage ≤ −1.
Committed numbers at 8k for e4b: email-triage 15/20, call-fields 12/19, recap 10/16. gpt-oss-20b: the same direction, half the
size of the drop (full-attention layers, no 512 window on the path to the answer): 8k −0 to −2 per exam.

**Latency.** p50 wall per case grows by the prefill time, which is linear in prompt length at 250–450 prompt tok/s for the e4b on
this M1 Pro (lane 01 saw 185 tok/s on a 1.5B bf16 warm, 27 cold, rank A; the 4-bit e4b should be faster per token): +15–30 s
at 8k. email-triage p50 goes from ~9 s to 25–40 s (3–4×), p95 from ~14 s to 35–50 s; recap p50 from ~18 s to 35–50 s (2–2.5×).
`gen_tps` itself drops only 5–10% at 8k (attention over 8k on 7 global layers; the sliding layers do not see it).

**KV-cache type.** Memory: on the e4b, `kv_bits` 8 or 4 saves under 0.2 GB at 8k (7 global layers × 2 KV heads × 512 dims
≈ 28 KB per token fp16 → ~230 MB at 8k, halved or quartered); on gpt-oss-20b (24 layers, 8 KV heads × 64, half of them
sliding) the saving is under 0.1 GB. So **the KV-cache type changes memory by under 5% of peak on both models.** Verdicts:
`kv_bits=8`: 0–1 stable flips per model over the 83 cases; `kv_bits=4`: 1–3 stable flips, concentrated on call-fields exact
tokens (M7's pattern on Ollama's q4_0 cache, rank B: held-out call summaries lost an amount and a ticket id). Speed: quantised KV
is 5–15% slower at these lengths (dequantisation on every step, little bandwidth saved). `max_kv_size` (RotatingKVCache): if set
below the padded prompt it is **the num_ctx-style failure**: the oldest tokens, i.e. the system prompt with the label schema,
leave the cache silently and the model still answers; with `max_kv_size=4096` on 8k prompts I predict email-triage held-out
falls to ≤ 10/20 and the output stops being valid JSON on most cases. If `max_kv_size` is left unset, the failure does not
appear: Gemma 4's position limit is 131k and mlx-lm does not truncate.

**The stamp I will recommend every result carries:** mlx and mlx-lm versions; model repo, snapshot sha and quant (`bits`,
`group_size`, `mode`); `kv_bits`, `kv_group_size`, `quantized_kv_start`, `max_kv_size`, `prefill_step_size`; `max_tokens`,
temperature; the chat template's sha; the wilco pin and each case's prompt token count. The sandbox's lesson (gotcha 21, rank A)
is that the version alone cannot see a cache setting that moves verdicts.

**Kill clause.** The brief names none. Proposed, not from the brief: the lane does not post a length effect if the padded
prompts were not shown to keep the gold answer (run the exam's own checker once with the gold answer against each padded case
and commit that it passes); without that, a drop at 8k is a changed question, not a context effect.

## Mechanism

Latency at longer prompts is prefill: every prompt token is a full forward pass batched through the weights, so time grows
linearly with tokens at a rate set by compute, not bandwidth, and on an M1 Pro that is a few hundred tokens per second for a
4B model; decode speed barely moves because Gemma 4's local layers attend to 512 tokens regardless and only 7 global layers pay
for the long context. The same architecture caps the KV footprint, which is why quantising the cache returns almost no memory
here, unlike on a model where every layer is full attention at 32 or 40 layers (Qwen3-14B would show ~160 KB per token). Pass
fraction drops with padding for two reasons the lane can tell apart: distraction (the padding contains messages and facts that a
recap or field extraction can wrongly include, which property checks and exact-match fields punish) and position (the answer-
relevant text sits further from the question; attention spreads over more keys). Distraction predicts recap and call-fields
suffer most and email-triage least, which is the order above. A quantised cache perturbs attention scores, which lands first on
tokens that must be copied exactly (amounts, ids), the mechanism M7 measured on another engine (rank B). A rotating cache that
drops the head of the prompt is Ollama's `num_ctx` behaviour reproduced by hand, and the system prompt is the first thing to go.

## Addendum (9 Nov 2026, before any graded lane-05 run)

Written by the lane-05 job after reading the harness and running one **ungraded** probe
(`scripts/lane05_probe_kv.py`: an ~8.3k-token prompt of wilco *train* recap inputs, 32 greedy tokens, no grading). The
probe's output was seen before this addendum, so every line below that rests on it is marked **(probe)** and is not a blind
prediction. The original text above stands and is graded as written where it still applies.

**Arms change.** "Lane 03's MoE" is the one lane 03 posted: `LiquidAI/LFM2.5-8B-A1B-MLX-4bit@2e92b640` (8.3B total / 1.5B
active; 24 layers, of which 6 are full attention with 8 KV heads × 64 and 18 are short convolutions). gpt-oss-20b failed
lane 03's gate (2.91 tok/s, paging on 16 GB), so its rows above are **void** for this lane. The dense arm is
`mlx-community/gemma-4-e4b-it-4bit@475b9088` as written. Baselines (0 padding) are the committed lane-02 and lane-03 files,
not re-sat: held-out e4b 16 · 10 · 10 = 36/55, LFM 14 · 2 · 5 = 21/55.

**What mlx-lm 0.31.3 actually exposes on these two models (rank A, probe + source).** Both models define `make_cache`,
so `make_prompt_cache` never reads `max_kv_size`: **`max_kv_size` is accepted and silently ignored on both** (probe:
same cache classes with and without it, identical tokens to the default at 8.3k with `max_kv_size=1024`). The
`RotatingKVCache` arm above therefore cannot happen through mlx-lm's API here; the predicted "email-triage ≤ 10/20 at
`max_kv_size=4096`" is **void**, and the lane posts the silent no-op itself as the num_ctx-style finding in mlx form (a
setting taken without error and not applied). No hand-built rotating cache: that would not be what mlx-lm exposes.
`kv_bits` on Gemma 4 **raises** `NotImplementedError: RotatingKVCache Quantization NYI` (its 35 sliding layers are
`RotatingKVCache`; probe). So the e4b's KV arms are **void by the runtime**, posted as such. `kv_bits` 8 and 4 run on LFM
(probe), quantising its 6 attention `KVCache`s; the 18 `ArraysCache` conv states have no `to_quantized` and stay fp16.

**Padding rule (fixed now).** wilco cases carry no per-case history, so "the case's own history" is read as the same agent's
earlier inputs in the same session:
- one extra **user** turn and one extra **assistant** turn are inserted between wilco's system message and the case's own
  user turn (found by identity: the first `user` message whose content equals `case["input"]`; wilco's re-ask keeps the
  same padding at the same place). The user turn is `"Earlier messages from this session, already handled. Do not act on
  them; your task is the next message.\n\n"` + pool text; the assistant turn is `"Understood."`. **No gold answer is in
  the padding** (no few-shot effect).
- pool = the sitting exam's **train** inputs in wilco order, then the other two exams' train inputs (sorted exam order),
  joined by `"\n\n--- earlier message ---\n\n"`; the sitting case is excluded; no held-out input is ever in a pool. The
  pool is ~107k chars, so nothing is cycled.
- target L ∈ {2048, 4096, 8192} **templated prompt tokens of the first call, in that model's own tokenizer**. The pool is cut
  at the longest character prefix that keeps the first call ≤ L. A case whose unpadded prompt is already ≥ L − 64 tokens is
  not padded; the post reports how many cases each length padded.
- the case dict and wilco's system prompt are untouched, so the grader (incl. recap's `check_grounded` / `check_compression`
  on `case["input"]`) is byte-identical. The original kill clause ("run the checker with the gold answer") is vacuous under
  this construction and is **replaced** by a test asserting the system message and the case's user turn are byte-identical
  to wilco's and that no `expected` value of any case is in a pool. Each row records target, pad chars, pad sha256.
- any exception from mlx that is not wilco's empty-answer `TerminationError` aborts the run (never scored as a FAIL).

**Run order, loads, gates.** Held-out 55 cases only; **one load per arm** (lanes 02–04: 0 of 83 unstable on all 9 local
arms at temperature 0), plus a second load of each model's 8k default arm as the stability check. If that check moves
> 3 of 55 on a model, no length delta or KV flip on that model within that count is posted as an effect.
1. gate smoke (train, not posted): e4b and LFM at L = 8192 on `client-quote-request`, `support-long-train`,
   `busy-day-standard`; project the 8k held-out arm as (55 / 3) × smoke wall × 1.5 (lane 04: a gate case at the token cap
   projects low).
2. 8k default e4b, 8k default LFM → 2k and 4k both → LFM `kv_bits=8` at 8k → LFM `kv_bits=4` at 8k → second load of
   8k default (LFM, then e4b). An arm is not started if the job has spent > 5 h of GPU since step 1; it is posted as not sat.

**Predictions for LFM (new; the original only covered gpt-oss).** From 21/55: 2k within ±1 per exam; 8k email −1 to −3,
call-fields 1–3 (it is near the floor at 2), recap 3–6; 8k total 15–21. p50 wall grows by prefill at ~700–800 prompt
tok/s (probe: 769 at 8.3k): +8–12 s per call at 8k. `kv_bits=8` vs 8k default: 0–2 held-out flips; `kv_bits=4`: 1–4,
with tokens differing on most cases (probe: 32-token outputs already differ at both). Memory **(probe, not blind)**:
quantised KV **raises** peak by ~0.9 GB on LFM at 8k (5.89 → 6.81 GB), since the fp16 cache is built during prefill and
converted after it. The original's "KV type changes memory by < 5% of peak" is graded on the measured peak. Speed
**(probe)**: prompt tok/s −8 to −10%, gen tok/s ≈ −1%.

**e4b at 8k against the actual baseline.** The original's absolute 15 · 12 · 10 was anchored on lane 02's predicted
baseline (16 · 14 · 13). Both are graded: the absolute numbers as written, and the deltas (−1 to −3 per exam) against the
actual 16 · 10 · 10 (i.e. 13–15 · 7–9 · 7–9).

**Stamp.** Unchanged from the original list, plus what the lane found it needs: the cache classes *applied* per layer
(read off the cache objects, not the requested kwargs), because the requested `max_kv_size` and the applied cache disagree.
