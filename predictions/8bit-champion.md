# 8bit-champion

Lane 10 (written 10 Nov 2026, before any lane-10 load): is the 8-bit e4b good enough to be the lab's default local pack?
One run: `champion-e4b-8bit`, the mlx arm (a). The flow arm (b) in roger3000_automations does not run. No 8-bit gemma4
Ollama tag exists on this machine (see "Not run"), so it has no prediction here or there.

**This is a reproduction, not a first measurement.** Lane 04 already sat this pack on all 83 cases, three loads, at this
wilco pin (`results/quant-gemma-4-e4b-it-8bit.json`, held-out 16/20 · 13/19 · 12/16 = 41/55, 0 of 83 unstable, peak 9.18
GB). I read those numbers, and lane 02's 4-bit row (16 · 10 · 10 = 36/55), before writing this file. SHOWCASE.md already
states the rule ("8-bit, unless its peak does not fit the working set"). This lane re-sits the pack on fresh loads, at a
later lab commit, to check that the rule's main evidence repeats before the rule is written into the docs.

Rank letters as in `predictions/quantisation.md`: **A** measured on our cases on this machine; **B** adjacent; **C** read.

## Prediction

**Run.** `HF_HUB_OFFLINE=1 LAB_WILCO_DIR=<pin clone> python -m lab run mlx-community/gemma-4-e4b-it-8bit --name
champion-e4b-8bit --exams all --loads 3 --prediction 8bit-champion --post`. Every case (train + held-out) of email-triage,
call-fields and recap at wilco pin `62fcd449`, 83 cases, three fresh loads, wilco's scorer, majority/stable rule.
- Pack: `mlx-community/gemma-4-e4b-it-8bit@4255b21b`, cached, the same snapshot as lane 04.
- mlx 0.32.2, mlx-lm 0.31.3.
- Baselines:
  - `results/quant-gemma-4-e4b-it-8bit.json` (lane 04, the run being reproduced);
  - `results/dense-gemma-4-e4b-it-4bit.json` (lane 02, the 4-bit e4b, **not re-sat**).

**Pass fraction (metric `pass_rate`, held-out majority pass, per exam).**

| exam | lane 04 (8-bit) | predicted | band |
|---|---|---|---|
| email-triage | 16/20 | 16/20 | 15–17 |
| call-fields | 13/19 | 13/19 | 12–14 |
| recap | 12/16 | 12/16 | 11–13 |
| **total** | **41/55** | **41/55** | **40–42** |

- **Flips vs lane 04:** 0 of 83 case verdicts flip, train included. **Unstable:** 0 of 83 across the three loads.
- **Why the band is ±1 per exam and not ±4.** The ±4 that lane 04 saw at temperature 0 was between *different* arms
  (e2b 8-bit vs bf16): a weight perturbation. This run changes no weight and no prompt. Re-sits of the *same* arm have
  shown no drift so far (rank A):
  - lane 04: 0 of 83 unstable across three loads on every arm;
  - lane 06: the repeat arm was 83 of 83 identical;
  - lane 06: the control reproduced lane 02's load 0 on 83 of 83.

  So 41 exactly is the point prediction. The ±1 per exam covers only a load-order or runtime effect I have not seen. A
  total outside 40–42 means the harness is not reproducible at a fixed pack, and that is the finding.
- **Speed and memory.**
  - gen tok/s 27–32 (lane 04: 29.3);
  - `peak_memory_gb_max` 9.1–9.3 GB (lane 04: 9.18);
  - wall per 3 × 83 run 85–105 min (lane 04: 94.5).

  Wh is not metered (no operator powermetrics capture), so it is `null`.

**Kill clause (written now).** If the re-sat 8-bit e4b held-out total is **not ≥ 36/55** (lane 02's committed 4-bit e4b
file, not re-sat), the rule change is refused: the docs keep 4-bit as the default and the post says so. A total of 36–39 is
outside the band but does not fire the kill. The rule still ships in that case, and the post records the reproduction as
refuted.

**Memory rule (scheduler rule 7).** Before the pack loads: abort if 9.18 GB (lane 04 peak) + 2.0 GB headroom exceeds
`scheduler.probe.ram_free_gb()`. An abort is a result, and it is recorded.

**Re-derive.**
```sh
P=~/.pyenv/versions/lewagon/bin/python3
$P -m lab check results/champion-e4b-8bit.json results/quant-gemma-4-e4b-it-8bit.json results/dense-gemma-4-e4b-it-4bit.json
$P -m lab diff results/quant-gemma-4-e4b-it-8bit.json results/champion-e4b-8bit.json
$P scripts/lane04_rows.py quant-gemma-4-e4b-it-8bit champion-e4b-8bit       # flip ids between the two 8-bit sits
$P scripts/lane04_rows.py dense-gemma-4-e4b-it-4bit champion-e4b-8bit       # 4 -> 8 flip ids on the re-sit
```

**Not run: the flow arm (b).** It was briefed as "does an 8-bit local model beat `gemma4-e2b-ctx16k` (76/155) on
roger3000_automations' 155 flow sentences". It needs an 8-bit Ollama gemma4 tag.
- `ollama show` on 10 Nov reads Q4_0 or Q4_K_M on every gemma4 tag: `gemma4-e2b-ctx16k`, `gemma4:e2b`,
  `gemma4-e2b-ctx16k-plain`, `gemma4-e4b-ctx16k`, `gemma4:e4b`, `gemma4-e4b-ctx16k-plain`, `gemma4:e4b-it-qat`,
  `gemma4:e2b-it-qat`.
- The Ollama manifests hold only `gemma4/{e2b, e2b-it-qat, e4b, e4b-it-qat}`. So no Q8_0 gemma4 base is on disk to build a
  16k tag from, and the brief allows a build only from a base already on disk.
- The only Q8_0 tags at a 16k window are qwen3 8B. That is a different model, so it would confound model and precision.

The flow arm is therefore not predicted and not run.

## Mechanism

Greedy decoding on a fixed pack, a fixed prompt and a fixed runtime is a deterministic function of the bytes, up to the
order of floating-point reductions in the Metal kernels. On this box that order has not varied between loads (rank A, the
re-sits listed above).

The lab code moved from lane 04's `lab_commit` `410d9c3` to `origin/main` `99fe0d3`. `git diff 410d9c3 99fe0d3 -- lab/run.py
lab/metering.py` shows that every change is behind `context is not None` (lane 05) or `spec is not None` (lane 06). With
both `None`, which is what `lab run --exams` passes, `MeteredMLX.generate` runs the same `apply_chat_template` →
`stream_generate(sampler=make_sampler(temp=0.0))` lines as before. The one new line in the sit loop sets
`adapter.current_case`, which only the lane-05 path reads. So nothing on the default path should move a token, and the
prediction is the lane-04 row.

The kill clause is the brief's. A reproduction that fell below the 4-bit row would mean lane 04's +5 was not a property of
the pack, and the rule would rest on nothing.
