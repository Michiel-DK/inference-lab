# quantisation

Lane 04 (written 6 Nov 2026, before any lane-04 run): on the lane-02 models, what does 4-bit vs 8-bit vs bf16 move on the
exams, and what does it give back in memory and Wh?

Rank letters: **A** measured on our cases on this machine; **B** adjacent measurement; **C** read, not run.

## Prediction

**What can actually run (HF API sizes, 6 Nov, rank C; 16 GB machine).** `gemma-4-e2b-it`: 4-bit 3.3 GB, 8-bit 5.9 GB, bf16
10.2 GB, all three fit. `gemma-4-e4b-it`: 4-bit 4.8, 8-bit 8.9 fit; bf16 15.9 GB does not. `Qwen3-14B`: 4-bit 8.3 fits; 8-bit
(~15.7 GB) and bf16 do not. So the full three-way comparison exists for the e2b only; the e4b gives 4 vs 8; the 14B gives
nothing. Baselines: lane 02's `results/dense-<model>.json` (the 4-bit rows).

**Pass fraction (metric `pass_rate`, held-out, per exam).** Direction: no monotone gain from more bits. Size: 0–2 held-out
cases move per exam between 4-bit and 8-bit, in both directions, net within ±1; 8-bit vs bf16 moves 0–1. Across the two
Gemma models and three exams I expect **3–6 distinct case ids to flip** between 4-bit and 8-bit, at least two of them on
call-fields (exact amounts, dates, ticket ids), at most one on email-triage. The held-out totals after the lane: e2b 8-bit
within ±1 of its 4-bit row on every exam; e4b 8-bit within ±1 of 16/20, 14/19, 13/16. Unsure which direction call-fields
nets; if forced: 8-bit +1 on call-fields, 0 elsewhere.

**Speed (`gen_tps`).** 8-bit = 0.55–0.70× the 4-bit figure (e2b 55 → 32–38; e4b 37 → 21–26). bf16 (e2b only) = 0.30–0.40×
(55 → 17–22). Not the naive 0.5× and 0.25×: launch overhead on 35–42 small layers does not scale with bytes.

**Memory (`peak_memory_gb_max`).** e2b: 3.6–4.0 → 6.2–6.7 → 10.5–11.2 GB. e4b: 5.2–5.8 → 9.3–9.9 GB. Roughly weights on
disk plus 0.3–0.6 GB.

**Energy (`wh_per_case`, if metered).** 8-bit = 1.4–1.7× the 4-bit Wh per case; bf16 = 2.3–3.0×. Package power stays 24–30 W,
so the ratio is the inverse of the tok/s ratio, slightly less because prefill is a smaller share of the slower runs.

**The rule I expect to ship:** "4-bit everywhere; 8-bit only for an exam where a stable held-out case fails at 4-bit and passes
at 8-bit on all three loads, and then for that exam only." I predict no exam meets that condition, so the posted rule is
"4-bit, full stop; the flips are symmetric noise, and 8-bit costs 1.5× the energy for them."

**Kill clause.** The brief names none. Proposed, not from the brief: the lane does not post a per-model table if either arm
has more than 3 of 83 cases unstable across its three loads; it does not post the rule if the flipped-case list is not
committed with ids.

## Mechanism

Weight precision moves outputs through small logit perturbations that only matter where the top-2 tokens are close; on these
exams the decisive tokens are a label from three, eight short field values, or a property-checked sentence, so most cases are
far from a tie and a few are near one in both directions. The rank-A anchors are the sandbox's: e31 (same weights, QAT Q4_0 vs
post-hoc Q4_K_M, 6 up / 0 down on crm-followup, 0/38 unstable) shows packaging can move verdicts by several cases, and M7
(KV cache at q8_0 moved 4 stable cases of 70, q4_0 lost held-out cases on exact tokens) shows precision losses land on amounts
and ids first; weight bits at 8 vs 4 are a milder version of the same lever, hence 0–2 per exam. Speed is bandwidth: bytes per
token double at 8-bit and quadruple at bf16, but the per-layer dispatch floor (lane 01 saw ~190 µs per fused op at decode
size in `bench/RESULTS.md`, rank A) does not, so the slowdown is sub-proportional, most visibly on the e2b. Memory is the
weights. Energy is wall time at near-constant package power (1 Nov capture, rank A).
