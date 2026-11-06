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
