# calibrated-4bit

origin: {"kind": "lane", "url": null, "label": "inference literature: calibration-free 4-bit vs calibrated INT4"}

Lane `lab-inf-calibrated-4bit` (written 17 Nov 2026, ~00:30 WET, before any load). Lane 04 found e4b 4-bit → 8-bit 36 → 41
held-out /55 on mlx-community affine packs (no calibration data). Question: does a properly made 4-bit pack of the same model
close half or more of that gap? The decider (operator) is whether the inference cards keep the 8-bit house rule.

Rank letters: **A** measured on our cases on this machine; **B** adjacent measurement; **C** read, not run.

## What the "QAT 4-bit" pack is (checked before this prediction, CPU only, no generation)

`mlx-community/gemma-4-E4B-it-qat-4bit`, snapshot `0f35c6f6`, 6.80 GB (index `total_size` 6,797,955,540 bytes):

- **Not a pure 4-bit pack.** `config.json` `quantization` = affine, group 64, 4 bits, with **126 per-layer overrides to
  8 bits, group 64**: `mlp.gate_proj`, `mlp.up_proj` and `mlp.down_proj` on all 42 decoder layers. Its safetensors headers
  differ from the plain 4-bit pack (`gemma-4-e4b-it-4bit`, `475b9088`) in exactly those 126 weights (same key set). MLP bytes:
  3.51 GB here vs 1.86 GB plain 4-bit; attention 0.33 GB and the language-model embeddings 1.96 GB are the same format in
  both. So about 85 % of the language model's linear-layer bytes are at 8 bits in this pack. Name used everywhere below:
  **QAT pack (4-bit attention / 8-bit MLP, affine g64)**.
- **Different weights.** Dequantised, its tensors sit 0.15–0.53 relative L2 distance from the plain 8-bit pack's, against
  0.09–0.10 for the plain 4-bit pack (`scripts/calib4_prov.py`, layers 0/10/20/41, q/o/gate/down). The same 8-bit MLP format
  differs by 0.16–0.53, so this is a further-trained checkpoint, not a re-pack of the stock weights. The relative L2 distance
  can be inflated by norm/scale folding, so it shows "different", not "how different".
- **Base checkpoint ⚠️ unverified for this pack.** The mlx-community card names no `base_model` and no source. Google's
  `google/gemma-4-E4B-it-qat-q4_0-unquantized` card (HF API, 17 Nov) names `base_model: google/gemma-4-E4B-it` (the instruct
  model lane 04 quantised) and says the QAT targets **Q4_0**. Q4_0 is symmetric, block 32. This pack is affine group 64, so
  even its 4-bit attention is not on the grid QAT trained for. Confound, stated.
- **Template.** `chat_template.jinja` differs from the plain packs by one 9-line block that emits image/audio/video tokens
  inside tool-response bodies. The tokenizer files are identical. `scripts/calib4_prov.py` checks that the rendered prompt
  token ids on the train smoke cases are identical; if they are not, that is a confound and the post says so.

## Arms (e4b only; all 83 cases; three fresh loads; wilco pin `62fcd449`; mlx 0.32.2 / mlx-lm 0.31.3; T = 0)

| arm | weights | format | GB | file |
|---|---|---|---|---|
| plain 4-bit (baseline) | stock | affine 4-bit g64 everywhere | 5.18 | `results/dense-gemma-4-e4b-it-4bit.json` (lane 02, not re-sat) |
| plain 8-bit (ceiling) | stock | affine 8-bit g64 everywhere | 8.91 | `results/quant-gemma-4-e4b-it-8bit.json` (lane 04, not re-sat) |
| **A. QAT pack** | QAT | 4-bit attention / 8-bit MLP, g64 | 6.80 | `results/calib4-gemma-4-e4b-it-qat-mixed.json` |
| **B. QAT pure 4-bit** (derived) | QAT | affine 4-bit g64 everywhere: arm A with its 126 8-bit MLP weights dequantised and re-quantised by `mx.quantize` at 4 bits, group 64 (plain 4-bit's recipe) | ~5.2 | `results/calib4-gemma-4-e4b-it-qat-4bit-requant.json` |
| **C. splice control** (derived) | stock | plain 4-bit pack with the 126 MLP weights taken from the plain 8-bit pack: arm A's layout, stock weights | ~6.8 | `results/calib4-gemma-4-e4b-it-splice-mlp8.json` |

B and C are built locally from cached packs (no download, no bf16 load) by `scripts/calib4_build.py`, which records the
sha256 of every output file. The OptiQ arm from the brief is swapped for C: OptiQ is another mixed-precision pack and would
isolate nothing. Readings: **A vs C** = QAT weights at a fixed layout; **C vs plain 4-bit** = what 8-bit MLPs alone buy;
**B vs plain 4-bit** = QAT weights at the baseline's exact format, which is the decider's "properly made 4-bit pack".
Caveat on B: the MLP weights are rounded twice (QAT → 8-bit by mlx-community → 4-bit here), and none of it is on QAT's Q4_0
grid. A B-result is "QAT weights, mx affine 4-bit", not "Google's Q4_0".

**Order and gates.** A, then B, then C. Before each arm: the lane-04 gate smoke (`--smoke` + call-fields
`support-long-train`). An arm sits 3 × 83 iff (a) the smoke completes, (b) smoke peak + 2.0 GB fits free + inactive +
speculative memory, (c) projected wall = (smoke wall / the same three cases' load-0 wall in the plain 4-bit file) × 4,888 s
ends before **05:15 WET** (job cap 06:08), and (d) swap used has not grown more than 2.0 GB above the value recorded at lane
start (3.04 GB at 00:00; brief's "past 2 GB" read as growth). Swap is logged every 5 minutes; past +2.0 GB the running arm is
stopped and reported "not sat (swap)". An arm that fails a gate is "not sat (<gate>)" and its smoke file is committed.

## Prediction

Held-out /55 email · call · recap. Baselines: plain 4-bit 16 · 10 · 10 = **36**, plain 8-bit 16 · 13 · 12 = **41**.
**Decider threshold: half the gap = 38.5, so a pack "closes half" at ≥ 39/55.** Noise floor: LESSONS row 4 (bf16 below 8-bit
by −4 on e2b) puts a ±4 perturbation floor under any single-model precision claim. A total of 37–40 sits inside that floor
from both 36 and 41, and is posted as "inside the floor", not as a flip of the decider. Only ≥ 41 (at 8-bit, and > 36 + 4)
counts as a clean close; 39–40 meets the operator's threshold but not the floor. Both readings are posted.

- **A (QAT pack):** 38–43, point **40** (email 15–17, call 11–14, recap 10–13). About 85 % of the linear bytes are 8-bit,
  so A should sit near 8-bit whatever QAT does. P(A ≥ 39) ≈ 0.65.
- **B (QAT pure 4-bit):** 34–40, point **37**. QAT's robustness was trained for Q4_0 g32, not affine g64, and the MLPs are
  rounded twice, so I expect only part of Google's claimed bf16 parity to survive. P(B ≥ 39) ≈ 0.3. If forced: B < A.
- **C (splice control):** 37–42, point **39**. Most of the 4 → 8 gap is in the MLPs because that is where most of the
  4-bit rounding error is. If forced: A − C within ±2, i.e. QAT weights add little once the MLPs are 8-bit.
- **Flips** (distinct case ids, majority verdict, all 83) vs plain 4-bit: A 8–18, B 6–16, C 6–16 (lane 04: 16). vs plain
  8-bit: A 6–16.
- **Unstable:** 0–1 of 83 on every arm (lanes 02/04/10: 0, greedy is repeatable across loads).
- **gen tok/s** (plain 4-bit 41.9, 8-bit 29.3): A 30–36, B 39–45, C 30–36. Decode is weight bandwidth, and A/C read
  ~0.33 + 3.51 GB of linear weights per token vs 2.19 (4-bit) and ~4.2 (8-bit).
- **Peak GB** (plain 4-bit 5.45, 8-bit 9.18; peak ≈ pack + 0.27 there): A 6.9–7.4, B 5.3–5.7, C 6.9–7.4.
- **Gen tokens per sitting** (plain 4-bit 686, 8-bit 577): A 520–700, B 560–760, C 540–680. Unsure; lane 04 says the 8-bit
  Gemmas think less, so A and C should sit nearer 577.
- **Wall per 3 × 83** (81.5 / 94.5 min): A 80–100, B 75–90, C 80–100.
- **Wh:** `null` (no operator `powermetrics` capture). Energy rows void.

**Decider reading, written now.**
- A < 39: **strong negative.** Even QAT weights with 8-bit MLPs do not close half the gap, so calibrated GPTQ/AWQ at pure
  4-bit is unlikely to either. The 8-bit house rule stands.
- A ≥ 39 and B ≥ 39: the cards may name a QAT 4-bit (QAT weights, affine 4-bit g64) as an alternative, but within the ±4
  floor. The rule changes only if B ≥ 41.
- A ≥ 39 and B < 39: the close comes from the 8-bit MLPs, not from QAT. The 8-bit rule stands. "QAT 4-bit" on mlx means
  6.8 GB, not 5.2.
- C tells which of the two A carries. If C is not sat, the A ≥ 39 / B < 39 attribution rests on B alone, and the post says so.

**Kill clause.** No per-arm table if an arm has > 3 of 83 unstable cases. An arm is void if mlx-lm does not apply the 8-bit
overrides at load (checked on the loaded model before arm A: `mlp.gate_proj.bits` = 8, `self_attn.q_proj.bits` = 4), or if
the rendered prompts differ from the plain pack's on the smoke cases. No decider reading is posted unless the flip ids are
committed (`scripts/lane04_rows.py`).

## Mechanism

Weight rounding error at 4 bits moves logits where top-2 tokens are near a tie. Lane 04 measured that on these exams it
costs the e4b 5 held-out cases (rank A), on call-fields and recap, and that the 8-bit Gemma thinks ~16 % less. QAT trains the
weights so that rounding to a specific grid costs little. Google claims bf16-level quality for Q4_0 (card, rank C; not at
[F]). The field's calibrated INT4 (Kurtić et al., 2411.02355, [F], `docs/literature-2026-11-16.md` row 1: GPTQ g128 W4A16
recovers 98.7 % of bf16 on Llama-3.1-8B, Table 2) and AWQ ([F], row 2: RTN g128 behind AWQ/GPTQ in perplexity, Table 4) say
a better-made 4-bit loses much less than round-to-nearest. Neither measures pass counts on 55 cases, so neither predicts our
number. What breaks the clean test here is the pack: mlx-community's QAT pack keeps the MLPs at 8 bits, which is most of the
bytes, so arm A alone cannot tell QAT from bits. Arms B and C split the two. The QAT paper or blog behind Google's card is not
read at [F]; its claim is used only as rank C and no published number from it is set beside ours.
