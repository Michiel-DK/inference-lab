# moe-vs-dense

Lane 03 (written 6 Nov 2026, before any lane-03 run): does an MoE with N active parameters pass the exams at the cost of a
dense N model, or of its total parameter count?

Rank letters: **A** measured on our cases on this machine; **B** adjacent measurement; **C** read, not run.

## Prediction

**Candidate.** The brief's first choice, Gemma-4 26B-A4B, does not fit this machine at the lane's quantisation: the 4-bit
mlx packs are 15.3 GB (`gemma-4-26b-a4b-it-4bit`) and 15.6 GB (`-qat-4bit`) of weights (HF API, 6 Nov, rank C) on a 16 GB
M1 Pro; the sandbox's 8.6 GB ternary 27B already paged to a 17.2 GB peak and 5.7 tok/s (J11, rank A). The smaller 26B-A4B
variants that would fit are pruned (REAP-14B, 11.2 GB) or abliterated 2.6-bit community packs, which are different models. The
Qwen/DeepSeek MoEs the brief names as fallback do not fit at 4-bit either (`Qwen3-30B-A3B-4bit` 17.2 GB, `Qwen3.5-35B-A3B-4bit`
20.4 GB, Nemotron 3.5 Lightning 30B-A3B 17.8 GB). What fits and mlx-lm 0.31.3 serves (`models/gpt_oss.py`, `lfm2_moe.py`,
`granitemoehybrid.py`, `olmoe.py` exist in the installed package, rank A):

- primary: `gpt-oss-20b-MXFP4-Q8` (12.1 GB; 3.6B active / 21B total). Active-size dense = lane 02's `gemma-4-e4b-it-4bit`;
  total-size dense = lane 02's `Qwen3-14B-4bit` (the nearest dense that fits; 21B dense does not).
- fallback if gpt-oss's Harmony chat format breaks wilco's JSON path: `LFM2.5-8B-A1B-MLX-4bit` (4.8 GB; ~1.5B active / 8.3B
  total). Active-size dense = `gemma-4-e2b-it-4bit`; total-size dense = `Qwen3-8B-4bit` or the cached Llama-3.1-8B 4-bit.

Baseline = lane 02's committed results files for those dense models (`results/dense-<model>.json`).

**Which dense model the MoE resembles, per number (primary candidate).** "Within noise" below means: pass within ±1 held-out
case per exam, gen_tps within ±10%, Wh within ±15%, where the 3-load spread is smaller than that.

| number | prediction | resembles |
|---|---|---|
| `peak_memory_gb_max` | 12.5–13.5 GB; above the 14B dense (9–10.5) and 2.3× the e4b | **neither: total-size, and over it** |
| `gen_tps` | 26–34 tok/s = 0.7–0.9× the e4b's 37; 2× the 14B's 14 | **active-size dense** (discounted by router + expert gather + memory pressure) |
| `wh_per_case` (if metered) | 0.9–1.3× the e4b's per exam | **active-size dense** |
| `pass_rate` email-triage | 17/20 (16–18) | between e4b 16 and 14B 18 |
| `pass_rate` call-fields | 13/19 (11–15) | e4b (14) or below |
| `pass_rate` recap | 8/16 (6–10) | **below both** (Dutch grounded recap; gpt-oss-20b hosted scored 3/10 on the old recap set, rank B) |

Summary: time and energy cost like N (active); memory costs like the total; quality is not better than the active-size dense
on these exams. If the pack pages (peak reported above ~14 GB, or gen_tps under 12), every number turns total-like and the
lane's answer flips to "costs like the total on a 16 GB machine", which is a hardware statement, not a MoE one.

Unsure: whether gpt-oss's reasoning channel spends wilco's `max_tokens` before the final JSON on the long call-fields cases
(the frontier-direct rows show reasoning models burning 2–25k tokens there); if so, call-fields falls to ≤ 10/19 for a
truncation reason, not a capability one, and the lane should report `finish_reason: length` counts beside the pass fraction.

**Kill clause (copied from the brief):** "within noise of the active-size dense model on all numbers = 'costs like N', lane
closes with that post." Note for the lane (also in the report): peak memory can never be within noise of the active-size
dense, because the total parameters must be resident, so as written the kill cannot fire; I read it as "on tok/s, Wh and pass",
with memory reported beside it, and the prediction above says the kill does fire on that reading.

## Mechanism

Per decoded token an MoE reads its attention weights, the router, and top-k experts' weights, so the bytes crossing unified
memory per token are the active parameters' bytes plus the shared ones, not the total; on a bandwidth-bound laptop decode
speed therefore follows active size (PDScope, radar row of 2026-11-06 PARKED, rank C, is the same observation at low batch).
Two discounts: the expert gather is a scattered read (mlx's `gather_qmm`), and at 12–13 GB resident on 16 GB the OS starts
compressing or paging other memory, which J11 showed turns into a 3–5× slowdown once the weights themselves page. Peak memory
is the total weights plus KV: a MoE never gets to leave its idle experts on disk under mlx-lm. Energy is power times wall, and
package power is flat at 24–30 W across model sizes here (1 Nov capture, rank A), so Wh tracks tok/s. On quality the exams are
small-model-friendly: email-triage saturates at the 2B, recap rewards restraint and Dutch grounding where a 4B Gemma ties the
hosted 26B-A4B (13/16 each, rank B) while reasoning-style models underperform, and call-fields wants exact fields where a
larger active capacity would help but gpt-oss's 3.6B active is the e4b's size. Hence "costs like N, scores like N" on this
machine, with the memory column as the one place the total shows.
