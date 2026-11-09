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

## Addendum (9 Nov 2026, ~04:20 WET, before any lane-04 load; written after lanes 02 and 03 were known)

The bands above are scored as written, including where lane 02 has since shown their anchors were off. The e4b pass band
uses wilco's champion figures (16/20, 14/19, 13/16), not lane 02's mlx 4-bit row (16 · 10 · 10). The tok/s bands assume a
4-bit e2b at 55 and e4b at 37, where lane 02 measured 69.2 and 41.9. Both bands are scored literally and the mismatch is
noted in the post. This addendum fixes what lane 02 changed, adds the model the prediction did not cover, and settles how
one evening decides which arms sit.

1. **Arms.** All packs come from `mlx-community` in the same conversion family as lane 02's 4-bit packs (HF API, 9 Nov, rank C):

   | model | 4-bit | 8-bit | bf16 |
   |---|---|---|---|
   | gemma-4-e2b-it | lane 02 file (3.58 GB) | `03dcf209` 5.90 GB | `fb0b166b` 10.24 GB |
   | gemma-4-e4b-it | lane 02 file (5.18 GB) | `4255b21b` 8.91 GB | 15.91 GB: **void, does not fit** |
   | Meta-Llama-3.1-8B-Instruct | lane 02 file (4.53 GB) | `142d4280` 8.54 GB | 16.08 GB: **void, does not fit** |

   The 4-bit arms are lane 02's committed results and are not re-sat. mlx 0.32.2, mlx-lm 0.31.3 and wilco pin `62fcd449`
   are the same, and so is the protocol: all 83 cases, three fresh loads, wilco's scorer, majority/stable rule.
   The Qwen3-14B rows are void (lane 02).
2. **Packs fetched before the run, not inside it.** None of the 8-bit or bf16 packs was cached. They are fetched with
   `snapshot_download` at the revisions above before any load, and `refs/main` is written by hand (lane 03's lesson).
   `lab run` resolves with `local_files_only` (unchanged).
3. **Llama-3.1-8B: prediction (new, written now, after its 4-bit row is known).** Held-out pass at 8-bit vs 4-bit
   (16 · 6 · 9):
   - email-triage within ±1 (unsure on direction);
   - call-fields +0 to +2 (8-bit keeps digits and ids better; if forced: +1);
   - recap within ±1.
   - 1–4 distinct case ids flip across the three exams.
   - gen tok/s 19–23, i.e. 0.55–0.65× 35.4. The 8.5 GB pack read per token puts it near this box's bandwidth.
   - Peak memory 9.9–10.6 GB: the pack plus lane 02's +1.77 GB of KV and activations, under the 11.45 GB working set.
   - Minutes per 1,000 cases 1.3–1.6× the 4-bit row. Llama barely thinks (43 tokens per sitting), so prefill is a large
     share and is not halved by bits.
   - Wh void (see 6).
4. **Run order, time gate, memory gate.** Order: Llama-8bit → e2b-8bit → e4b-8bit → e2b-bf16. Rationale: the cheapest
   arm first, then the two 4-vs-8 arms the shipping rule rests on, then the one three-way point. The three-way point is
   last because it is the most memory-risky (10.24 GB pack against 11.45 GB).
   - **Gate smoke.** Before each arm it sits a train-only gate smoke: `--smoke` (two email-triage train cases) plus
     call-fields `support-long-train` (the case that hit e2b's 4096 cap).
   - **Projection.** Projected arm wall = (gate wall / the same three cases' load-0 wall in the 4-bit lane-02 file) × that
     file's 3 × 83 `wall_s_total`.
   - **The arm sits the 3 × 83 run iff** (a) the smoke completes without a Metal allocation failure, (b) smoke peak memory
     is under 11.45 GB, and (c) the projected wall fits before **10:10 WET**. 10:10 is the GPU cut-off: the job's
     420-minute cap ends near 11:05, and the post needs about 50 minutes.
   - **An arm that fails the gate is "not sat (time)" or "not sat (memory)", not dropped.** Its gate smoke file is
     committed as the evidence.
   - An arm still running at 10:10 is stopped and reported not sat. No partial 3 × 83 file is posted.
5. **Flips.** A flip is a case id whose majority verdict differs between two precisions of the same model.
   - Flips are listed by id, split into stable-in-both (all three loads agree at each precision) and touching an unstable
     case.
   - They come from `scripts/lane04_rows.py` over the checked rows.
   - The shipping-rule trigger is as written above: a held-out case stable-fail at 4-bit and stable-pass at 8-bit, on
     that exam.
6. **Kill clause, operational reading.** As proposed above: no per-model table if either arm has more than 3 of 83
   unstable cases. No rule is posted unless the flip ids are committed. Wh: no operator `powermetrics` capture is running
   (checked 04:12), so `wh_per_case` is `null` and every energy row is void, not estimated. The memory and wall deltas
   stand in its place, labelled as what they are.
