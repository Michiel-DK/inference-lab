# dense-baselines

Lane 02 (written 6 Nov 2026, before any lane-02 run): three dense models, same quantisation, on the three wilco exams.
Run names the lane will use: `dense-<model>` per model; this file is the lane's one prediction and covers all three.

Rank letters used below (as `LITERATURE.md` defines them): **A** measured on our own cases on this machine; **B** adjacent
measurement (same exams on another engine or hosted, or a different exam on this engine); **C** read, not run.

## Prediction

**Models.** The brief says "three dense models that fit 16 GB at the same quantisation, from mlx-lm's supported list". At 4-bit
the likely trio, all `mlx-community` packs (sizes from the HF API on 6 Nov, rank C): small `gemma-4-e2b-it-4bit` (3.3 GB, cached),
mid `gemma-4-e4b-it-4bit` (4.8 GB, cached), large `Qwen3-14B-4bit` (8.3 GB). `gemma-4-31b-it-4bit` is 18.4 GB and does not fit.
If the lane picks `Qwen3-8B-4bit` (4.6 GB) or the cached `Meta-Llama-3.1-8B-Instruct-4bit` (4.2 GB) as the large model instead,
the rows marked † apply to it. The predictions below are per named model; a different pick voids that row, not the lane.

**Baseline.** wilco's committed snapshots at the pin (`agent-exam-suite@62fcd449`, `evals/<exam>/snapshot.json`, Ollama 0.40.11,
QAT Q4_0 champions, 3 loads, rank B because the engine differs) and the sandbox's mlx_lm.server run of `gemma-4-e4b-it-4bit` on
email-triage (`agent-sandbox/docs/probes/mlx-s1-2026-11-01`, arm A, 3 loads, 16/20 held-out, 37.6 tok/s, rank A).

**Pass fraction (held-out; all-cases in brackets where I have a view). Metric `pass_rate` per exam.**

| model | email-triage (20 held-out) | call-fields (19) | recap (16) |
|---|---|---|---|
| e2b-4bit | 16/20, range 15–17 [23–25/28] | 11/19, range 9–13 | 11/16, range 10–12 |
| e4b-4bit | **16/20**, range 15–17 (the rank-A number; a flip of 0–1 from in-process vs server) [24/28] | 14/19, range 13–15 | 13/16, range 12–14 |
| Qwen3-14B-4bit | 18/20, range 17–19 | 15/19, range 13–16 | **10/16**, range 8–11, i.e. below e4b |
| † Qwen3-8B-4bit or Llama-3.1-8B-4bit | 17/20 | 13/19 | 9–11/16 |

Direction I am committing to: **recap is not monotone in size** (the 14B lands below the 4B); email-triage is saturated above the
2B (every model within 2 cases of 16/20); call-fields is the only exam where size buys held-out cases (+1 to +3 from e4b to 14B).

**Speed and memory (`gen_tps` median over calls, `peak_memory_gb_max`).**

| model | gen_tps | peak memory GB | mean wall s/case: email-triage · call-fields · recap |
|---|---|---|---|
| e2b-4bit | 55 (48–65) | 3.6–4.0 | 6 · 12 · 13 |
| e4b-4bit | 37 (34–40) | 5.2–5.8 | 9 · 17 · 18 |
| Qwen3-14B-4bit | 14 (12–16) | 9.0–10.5 (the 8k call-fields prompt adds ~1.4 GB of KV) | 25 · 40 · 50 |
| † 8B at 4-bit | 24 (20–28) | 5.0–6.0 | 14 · 25 · 30 |

**Energy (`wh_per_case`, only if the operator starts the powermetrics capture; otherwise null, and this row is void).** Package
power stays in the 24–30 W band of the 1 Nov sandbox run, so Wh per case tracks wall time: e2b 0.045 / 0.09 / 0.10; e4b 0.06 /
0.12 / 0.13; Qwen3-14B 0.18 / 0.28 / 0.35 (email-triage / call-fields / recap). Per 1,000 cases: e4b ≈ 60 / 120 / 130 Wh and
150 / 280 / 300 minutes; Qwen3-14B ≈ 3× both.

**Shape of the cost-vs-pass curve.** Per exam, x = minutes (or Wh) per 1,000 cases, y = held-out passes: a hockey stick lying
down. From e2b to e4b cost rises ~1.5× and buys +0 (email-triage), +3 (call-fields), +2 (recap) cases; from e4b to the 14B cost
rises ~3× and buys +2, +1, −3. The knee is at the 4B. Unsure: whether Qwen3's thinking mode eats wilco's `max_tokens` (2048 /
4096) and truncates the JSON on long call-fields cases; if it does, the 14B's call-fields row falls below e4b and the curve is
monotone-down above the knee.

**Kill clause.** The brief names none. Proposed, not from the brief: the lane does not post if any model's three loads disagree on
more than 3 of the 83 cases (the snapshot rule has nothing stable to compare), or if `lab check` cannot re-derive a posted number.

## Mechanism

Decode on this laptop is bandwidth-bound: tokens per second is roughly the bytes read per token divided by an effective
~85–140 GB/s (lane 01's smoke moved 3.1 GB of bf16 weights at 44–48 tok/s; the sandbox's e4b server run moved ~2.4 GB of
4-bit transformer weights plus a 262k-row lm_head at 37.6 tok/s, lower because 42 small layers are launch-bound). So the 14B at
8.3 GB lands near 14 tok/s, the 8B near 24, and the e2b gains less than 2× over the e4b because its 35 layers cost almost the
same dispatch time. Peak memory is weights plus the per-layer embedding tables plus KV; Gemma 4 keeps 35 of 42 layers on a
512-token sliding window and shares KV across 18 layers, so its KV is tiny even at 8k; Qwen3's 40 full-attention layers at 8
KV heads cost ~160 KB per token, visible on the longest call-fields prompts. Power is flat (24–30 W package, 1 Nov capture,
rank A), so Wh per case is wall time times a constant. On pass fraction the anchors are rank B (same exams, hosted): Gemma-4 31B
20/20 · 26B-A4B 18/20 · Qwen3-14B 17/20 on email-triage; on recap the small Gemma QAT champion (13/16) ties the 26B-A4B (13/16)
and beats Qwen3-14B (9/16) and every Claude (3–8/16), because recap is a Dutch grounded summary scored by property checks that
reward restraint, which the Gemma family has and the thinking models spend tokens against. call-fields is eight exact fields
from a transcript, the one task where a larger model's extraction accuracy should show. e31 (rank A, Ollama) showed that
packaging moves verdicts as much as size (post-hoc Q4_K_M beat QAT Q4_0 by 6 up / 0 down on crm-followup), which is why the
mlx affine 4-bit packs are predicted within ±1 of the QAT champions rather than equal to them.
