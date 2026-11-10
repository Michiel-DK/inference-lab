# Showcase

The post: one headline number with its spread, what was refused, the reopen threshold, and how to re-derive it
(`python3 -m lab check <results file>`).

## Lane 10: the 8-bit e4b as the local champion (10 Nov 2026)

**Question.** Is the 8-bit e4b the lab's default local pack? Lane 04 measured it once. This lane checks that the
measurement repeats before the rule goes into the docs. It is a **reproduction, not a first measurement**, and the
prediction says so: `predictions/8bit-champion.md` (`b9c802f`), committed before the run, written after lane 04's
numbers were read.

**Protocol.** As lane 04:
- pack `mlx-community/gemma-4-e4b-it-8bit@4255b21b`, cached;
- all 83 cases, three fresh loads, wilco pin `62fcd449`, wilco's `run_exam`, majority/stable rule;
- mlx 0.32.2, mlx-lm 0.31.3, `lab_commit` `b9c802f` (lane 04 ran at `410d9c3`).

What did not move: wilco's remote moved to `2585a34`, and `git diff --quiet 62fcd449 2585a34 -- sandbox/runner.py
sandbox/router.py evals agents` exits 0. The run used the pin clone (`LAB_WILCO_DIR`) with `HF_HUB_OFFLINE=1`.

Memory rule (scheduler rule 7): `ram_free_gb()` read 13.6 GB against 9.18 + 2.0 GB needed, so the pack ran. n8n was not
running and no Ollama model was resident.

**Headline.** **41/55 held-out again: email-triage 16/20 · call-fields 13/19 · recap 12/16. 0 of 83 verdicts flip
against lane 04, and 0 of 83 cases are unstable.**
- All 249 rows have identical per-call `gen_tokens`.
- Peak 9.18 GB (+0.00), 29.3 tok/s, wall 95.1 min per 3 × 83 (lane 04: 94.5).

Against the 4-bit e4b (lane 02's file, not re-sat), the re-sit is 36 → 41, so the kill clause (re-sat total < 36) did not
fire.

| e4b | email-triage | call-fields | recap | total | gen tok/s | peak GB | wall 3 × 83 min | Wh |
|---|---|---|---|---|---|---|---|---|
| 4-bit (lane 02) | 16/20 | 10/19 | 10/16 | 36/55 | 41.9 | 5.45 | 81.5 | not metered |
| 8-bit (lane 04) | 16/20 | 13/19 | 12/16 | 41/55 | 29.3 | 9.18 | 94.5 | not metered |
| **8-bit (lane 10 re-sit)** | **16/20** | **13/19** | **12/16** | **41/55** | 29.3 | 9.18 | 95.1 | not metered |

Re-derive:
```sh
P=~/.pyenv/versions/lewagon/bin/python3
$P -m lab check results/champion-e4b-8bit.json results/quant-gemma-4-e4b-it-8bit.json results/dense-gemma-4-e4b-it-4bit.json
$P scripts/lane04_rows.py quant-gemma-4-e4b-it-8bit champion-e4b-8bit     # 0 of 83 flipped; wall 94.5 -> 95.1
$P scripts/lane04_rows.py dense-gemma-4-e4b-it-4bit champion-e4b-8bit     # 36 -> 41, 16 ids, the same ids as lane 04
$P scripts/lane10_tokens.py quant-gemma-4-e4b-it-8bit champion-e4b-8bit   # 249 of 249 rows, identical per-call gen_tokens
```

**Prediction vs result.**

| claim | predicted | result | |
|---|---|---|---|
| held-out email · call · recap | 16 · 13 · 12 (±1 each), total 40–42 | 16 · 13 · 12 = 41 | **held** |
| flips vs lane 04 / unstable | 0 / 0 of 83 | 0 / 0 | **held** |
| gen tok/s | 27–32 | 29.3 | held |
| peak GB | 9.1–9.3 | 9.18 | held |
| wall 3 × 83 | 85–105 min | 95.1 | held |
| kill: re-sat total < 36 (4-bit e4b) | | 41 | did not fire |
| Wh | `null` | not metered | void |

**The house rule (now in README.md).** The default local pack is 8-bit where its peak fits the box with the headroom
(peak + 2.0 GB ≤ free). Use 4-bit only when it does not fit. Even then, take the largest model whose 4-bit fits, not a
smaller model at 8-bit. bf16 never, on this box. The basis:
- lane 04 (+9 / +5 / +6 held-out across e2b / e4b / Llama, 0 unstable);
- this lane: the e4b gain repeats with identical verdicts and identical per-call token counts (249/249). Lane 04 kept no
  raw text, so byte identity is not shown.

On this 16 GB box the e4b-8bit peak (9.18 GB) fits with the headroom when ≥ 11.2 GB is free; it read 13.6 GB tonight.

**Not run: the flow arm (b).** roger3000_automations' 155 flow sentences run on Ollama. `ollama show` reads every gemma4 tag
as Q4_0 or Q4_K_M. The manifests hold only `gemma4/{e2b, e2b-it-qat, e4b, e4b-it-qat}`, so no Q8_0 gemma4 base is on disk
to build a 16k tag from. The only 16k Q8_0 tag is qwen3-8b, a different model, which would confound model and
precision. For context: agent-sandbox's e31 probe killed a qwen3 Q8_0 arm at 16k at about 14× the 4k wall. So the local
flow arm stays `gemma4-e2b-ctx16k`, at 76/155 on all three passes (roger3000_automations
`experiments/RESULTS.md:133,143,153`, read-only), and this rule does not reach it. The mlx pack has no Ollama equivalent here.

**What this does not show.**
- That the rule transfers to Ollama GGUF Q8_0 (the e31 probe saw no 8-bit gain for qwen3 on task-intake).
- That 8-bit beats a 5- or 6-bit pack (lane 04's gap, still open).
- Whether the result survives a different wilco pin or runtime.

**Reopen threshold.**
- A Q8_0 gemma4 base lands in the Ollama store; that unblocks the flow arm.
- mlx or mlx-lm moves off 0.32.2 / 0.31.3, or the wilco pin moves.
- A re-sit flips any of the 83 cases.
- A 5- or 6-bit pack recovers ≥ 4 of the e4b's 4 → 8 gain under 9.18 GB peak; that would move the rule.

## Lane 06: speculative decoding (10 Nov 2026)

**Question.** Where mlx-lm supports a draft model, what tokens/s gain do the exams see, with the pass fraction unchanged?
At temperature 0, speculative decoding must not change outputs. If it does, the harness is broken, and that is the
finding. The prediction was committed first: `predictions/speculative.md` (`043e8bd`, 6 Nov). The addendum (`5f1d131`)
was committed before any Gemma sitting. It fixes the pair, the arms and the reading of the kill clause, and it discloses
what had been seen before it was written: the mlx-lm source, lane 02's token counts and the Qwen wiring smoke.

**Protocol.**
- **Pair.** The pair was forced by the cache. Of lane 02's three targets, only the e4b has a cached same-family draft.
  The cache holds no Llama-3.2-1B weights (tokenizer and config only) and no Qwen3, and a run never downloads.
  - target `mlx-community/gemma-4-e4b-it-4bit@475b9088`;
  - draft `mlx-community/gemma-4-e2b-it-4bit@23876752` (the two `tokenizer.json` hashes are equal; stamped);
  - `num_draft_tokens` 3. The prediction's Qwen3 / Llama branch is void for this run.
- **Cases.** All 83 cases (train + held-out) of the three exams at wilco pin `62fcd449`, graded by wilco's `run_exam`
  as in lanes 02–05.
- **Arms.** Each arm is one fresh load, through `lab spec` (`lab/spec.py`). Every call keeps its raw text and its
  sha256, because lane 02's file has no text and cannot show a byte difference.
  1. control: no draft;
  2. draft e2b, k = 3;
  3. arm 2 again.
- **Wiring control.** `Qwen2.5-Coder-1.5B-bf16` drafted for itself on the 3-case train smoke. It uses a plain
  `KVCache`, so every cache is trimmable.
- **Control = lane 02.** The control arm reproduces lane 02's load 0 on **83 of 83** cases: verdict, plus per-call
  `gen_tokens`, `content_chars` and `reasoning_chars`. So the control-vs-draft diff is also a diff against lane 02.
- **Repeat arm interrupted.** The drive script hit the job runner's 2 h background limit after 61 of 83 repeat rows. The
  repeat was re-sat in full with the same command. The 61 interrupted rows matched arm 2 byte for byte as well. That
  partial is not committed, so the posted repeat figure is the full re-sit's 83 of 83.

**Headline.**
- **Speculative decoding on this pair halves the speed and changes almost every output.**
  - Decode: **21.2 tok/s against the control's 42.9 (×0.494)**. Lane 02's row is 41.9, so ×0.506 against it.
    - per exam: call-fields ×0.519, email-triage ×0.510, recap ×0.464 (the repeat: ×0.519, ×0.514, ×0.463);
    - wall per case 39.1 s against 19.2 s;
    - draft acceptance 0.625 (37,443 of 59,926 tokens came from the draft);
    - peak memory 7.85 GB against 5.45 GB.
  - Bytes: the draft arm's text differs from the control's on **81 of 83 cases**. Byte-identical: held-out email-triage
    `invoice-error-client` and `dpa-signature-compliance`.
  - Verdicts: **37 flipped (+1 / −36)**. Held-out fell **36 → 8 / 55**: call-fields 10 → 1, email-triage 16 → 7, recap
    10 → 0.
  - The arm-3 repeat is byte-identical to arm 2 on **83 of 83**, so this is the engine's deterministic behaviour, not
    load noise.
  - **No call raised.** 0 runtime faults on all three arms.
- **The kill clause fired on bytes on 81 of 83 cases. Where the bytes moved and where the verdicts moved are two
  different places.**
  - By the addendum's pre-registered rule, a byte diff is attributed to the cache rewind only if its *first* divergence
    is past 512. On that rule the 26 email diffs are all arithmetic: each first diverges under 512.
  - The split below is **post-hoc**. It classes each sitting on the draft arm's own length, which is an outcome of the
    treatment. The direct check under it uses the committed raw text, so it needs no rerun.
  - **14 sittings stayed under the 512-token sliding window for the whole call (draft arm).** 12 of them differ in
    bytes, and none flipped a verdict.
    - The post-hoc probe (below) re-sat the control path and read the top-1 / top-2 log-prob gap at the token where the
      arm first departs. On all 26 differing email-triage cases (every email prompt starts under 512, and every one of
      them diverges under 512, at cache position ~215–464), the gap is **≤ 0.375 nats**: 6 exact ties, 15 at 0.125
      (bf16 logits come in 0.125 steps). The calls' median gaps are 2.75–9.5, so these tokens sit in the bottom
      0.8–10.4% of their call.
    - The texts stay fluent ("an acquaintance" / "an old acquaintance"). Read: the 4-wide verify on 4-bit weights does
      not reproduce the single-token decode's arithmetic, and greedy decoding turns a bf16 tie into another wording.
      On these 14 sittings that did not change an answer.
  - **69 sittings passed 512.** All 69 differ, and they hold **all 37 verdict flips**. They are the 55 call-fields and
    recap cases, which start over the window, plus the 14 email cases whose draft-arm generation ran past it (to
    519–905).
    - The text loses grammar within its first sentences. For example, recap `quiet-day-newsletters`: "The goal is to
      summarize a busy owner (small) in a short recap, few sentences). The output must be a specific rules: no rounding,
      be a quiet day is a honest."
    - wilco's no-JSON error bucket rose from 1 case to 15.
    - Held-out call-fields fell 10 → 1, recap 10 → 0, and email-triage 16 → 7. All 9 email flips are sittings that ran
      past 512.
    - Direct check (printed by the script). In each of the 9 email flips, the arm's text is fluent up to the point where
      its cache reaches 512, and broken right after it:
      - `...doesn't fit the critical, time-sensitive nature` || `of the_later"}`;
      - `...it falls` || `under "client requests. 5. **Final label": "reply_now". {""reply_now""}}`.
      The near-tie had already changed the wording earlier in those calls. The text breaks where the window wraps.
    - Plain decode past 512 does no such damage. The control's 71 sittings past 512 pass 41, the same 41 as lane 02's
      majority on those ids.
    - The first divergence there is not always a near-tie. On a 9-case probe sample (call-fields train), the gap at
      divergence was 0–3.0 nats, two of them 1.0 and 3.0, beyond anything seen under the window.
    - This matches the mechanism read in the addendum (source, not instrumented):
      - `RotatingKVCache.is_trimmable()` is checked once, on the fresh cache (offset 0 < 512), so the guard passes.
      - Once the window has wrapped, `trim(n)` only lowers `offset` and `_idx`.
      - The next 4-token verify re-orders the buffer as if it had wrapped, so the target attends to the wrong keys.
- **The Qwen wiring control.** It had a plain `KVCache` and a perfect draft (self-draft, 100% acceptance).
  - 1 of 3 cases stayed byte-identical over 2 × 4,096 tokens.
  - The other 2 diverged only past char 8,000, inside a degenerate `<|repo_name|>` loop.
  - So the lab's adapter passes bytes through intact. Near-tie flips happen on both models, much sooner on the 4-bit
    Gemma (gen token ~3–190).

**The table** (`python3 scripts/lane06_rows.py`, which runs `lab check` on every file first and exits 1 on drift):

| arm | email /20 | call /19 | recap /16 | **total /55** | byte-identical /83 | flips vs control (all 83) | gen tok/s | ×control | acceptance | wall s/case | peak GB | wilco error-bucket cases | Wh/case |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lane 02 (3 loads, majority) | 16 | 10 | 10 | **36** | – | – | 41.9 | – | – | 19.6 | 5.45 | – | null |
| control (1 load) | 16 | 10 | 10 | **36** | – | – | 42.9 | 1 | – | 19.2 | 5.45 | 1 | null |
| e2b draft, k=3 | 7 | 1 | 0 | **8** | 2 | 37 (+1/−36) | 21.2 | ×0.494 | 0.625 | 39.1 | 7.85 | 15 | null |
| e2b draft, k=3, repeat | 7 | 1 | 0 | **8** | 2 (83/83 = arm 2) | 37 (+1/−36) | 21.2 | ×0.494 | 0.625 | 39.1 | 7.85 | 15 | null |

- Per-exam speed: call-fields 21.4 / 41.1, email-triage 23.2 / 45.5, recap 20.3 / 43.7 tok/s.
- Acceptance: call-fields 0.640, email-triage 0.632, recap 0.600.
- On the 2 byte-identical cases only: ×0.549.
- Every differing id, its window class and its divergence position are in the script's output.
- The arm files carry `posted: false`, because the drive script did not pass `--post`. They are not rewritten (results
  files never are). This section posts them.

**Prediction vs result** (`predictions/speculative.md` + addendum):

| predicted | result | verdict |
|---|---|---|
| Gemma pair: the harness breaks, 0 gradable answers on ≥ 80 of 83 (S1's `broadcast_shapes`) | 83 of 83 graded, 0 faults; the break is silent: wrong text, wrong verdicts | **refuted** in form; held in substance (the pair cannot be used) |
| draft > ¼ of the target's bytes ⇒ gain ≤ 1.2× or negative | ×0.494 | **held** |
| addendum: control reproduces lane 02 load 0 on ≥ 80 of 83 | 83 of 83 | **held** |
| under-512 cases byte-identical on ≥ 9 of 12 | 2 of 12 (26 of 28 email cases differ, all at near-ties under 512) | **refuted** |
| start-over-512 cases: ≥ 45 of 55 differ or fault; lean wrong text (≥ 30 differ) | 55 differ, 0 fault | **held** |
| email cases crossing 512 in generation: ≥ 10 of 16 differ or fault | 16 of 16 differ; the first divergence is under 512 on all 16 (a near-tie), and the 9 email flips are all in sittings that then ran past 512 | **held** in count. By the pre-registered rule these diffs are arithmetic (first divergence under 512); post-hoc, the 9 flips break where the arm's cache reaches 512 |
| ≥ 5 verdict flips, net negative, held-out ≥ 3 below control | 37 (+1/−36); 36 → 8 | **held** (far beyond the size) |
| spec / control gen tok/s 0.5–0.9× on byte-identical cases | ×0.549 (n = 2); ×0.494 over all 83 | **held** on the identical cases; at the floor over all |
| acceptance 0.45–0.70 | 0.625 | **held** |
| peak 8.5–9.5 GB | 7.85 | **refuted** (below) |
| repeat byte-identical on ≥ 80 of 83 | 83 of 83 | **held** |
| original branch-1: "below 512 the drafted text was wrong and 2× slower" (S1) | wrong text and ×0.49 in-process too | **held** |
| original numerics clause: 0–3 byte diffs, 0–1 verdict flips on a full-attention pair | not run (no full-attention pair cached); the Qwen self-draft wiring smoke diffed 2 of 3 deep in a loop | **void** for the lane |
| energy 0.7–0.9× | no powermetrics capture | **void** (Wh null, not 0) |
| kill clause (addendum reading: bytes, per case the draft arm completed) | fires on 81 of 83 (every id in the script's output); verdicts vs control and vs lane 02 majority: 37 (+1/−36) | **fired**: on this pair, speculative decoding is not usable |

**Post-hoc probe (not pre-registered, ungraded).** `scripts/lane06_margin.py` re-sits the control path of each
differing case's first call. It records the top-1 / top-2 log-prob gap per generated token and reports the gap at the
draft arm's first diverging token. It refuses (void) if its re-sit text differs from the committed control raw. Output:
`docs/probes/lane06-margin-2026-11-10.txt`.

**What did not hold.**
- **"mlx-lm refuses what it cannot do."** It does not here. The trimmability guard reads a fresh cache, so Gemma 4's
  sliding window passes it, and the run halves its speed and scores 8 / 55 without an error. It is the second
  accepted-and-not-applied setting, after lane 05's `max_kv_size`. Lane 05's `kv_bits` at least raised on Gemma.
- **"Speculative decoding is exact at temperature 0."** It is exact in exact arithmetic. On a 4-bit Gemma, the 4-wide
  verify flips a near-tie token within the first ~200 tokens of 26 of 28 short email answers. Byte equality is the wrong
  bar for a speculative path. On the 14 sittings that stayed in the window, verdicts held (0 flips), so verdicts against
  a repeat are the bar.

**What was refused or voided.**
- The Qwen3 / Llama branch: no cached draft, and a run never downloads.
- `num_draft_tokens` other than 3: one value was pre-registered.
- Wh: no capture.
- A hand-built non-rotating cache for Gemma's sliding layers: that is not what mlx-lm exposes, and it would change the
  model's attention.

**Re-derive.**
```sh
python3 -m lab check results/spec-*.json
python3 scripts/lane06_rows.py
HF_HUB_OFFLINE=1 python3 scripts/lane06_margin.py results/spec-gemma-4-e4b-it-4bit-control.json \
  results/spec-gemma-4-e4b-it-4bit-e2b-k3.json email-triage/friend-dinner   # GPU; one id, ~10 s
```

**Reopen threshold.** Re-sit lane 06 when any of these holds:
- **The operator caches a full-attention pair with a lane-02 target.** `mlx-community/Llama-3.2-1B-Instruct-4bit`
  drafting lane 02's Llama-3.1-8B-4bit is the first arm, since `llama.py` builds plain `KVCache`s. This is the question
  this lane could not reach: does a draft that is ~⅛ of the target's bytes pay on these exams? Bytes are expected to move
  at near-ties, so the pre-registered reading there should be verdict flips against a repeat.
- **An mlx-lm version that makes `RotatingKVCache` rewindable after the window wraps**, or that refuses speculative
  decoding on it, re-sits this exact pair.
- **A Gemma draft that is ≤ ¼ of the e4b's bytes.**
- **A new case pin.**

## Lane 05: context length and KV cache (10 Nov 2026)

**Question.** As the exam prompts grow to 2k, 4k and 8k tokens, padded with the session's earlier inputs and keeping the
same answer key, what happens to pass fraction and latency? And does the KV-cache type change anything? The prediction
was committed first: `predictions/context-kv.md` (`043e8bd`, 6 Nov). The addendum (`e3d07d3`) and the second note
(`ca6038f`) were committed before any graded run. They fix the arms, the padding rule and the gates. They also record the
KV-exposure probe, which was ungraded and was seen before the addendum was written.

**Protocol.**
- Cases: the 55 held-out cases at wilco pin `62fcd449`, graded by wilco's `run_exam` as in lanes 02–04.
- Arms:
  - one dense model, `gemma-4-e4b-it-4bit@475b9088`;
  - lane 03's posted MoE, `LFM2.5-8B-A1B-MLX-4bit@2e92b640`. gpt-oss-20b failed lane 03's fit gate, so its rows are void.
- Padding (`lab/context.py`, pre-registered):
  - Placement: one user turn and one assistant `"Understood."` turn, inserted between wilco's system prompt and the
    case's own turn. The user turn opens with "Earlier messages from this session, already handled. Do not act on them…".
  - Pool: the train inputs of the sitting exam first, then those of the other two exams. No held-out text and no gold
    answer is ever in the pool.
  - Length: the pool is cut so that the first call is ≤ L tokens in each model's own tokenizer. Cases already ≥ L − 64
    tokens are not padded.
  - Grading: the case dict and system prompt are untouched, so the grader is byte-identical.
  - Tests: `tests/lab/test_context.py`. One fix came out of writing them. wilco's split has an overlap at the pin:
    held-out recap `quiet-day-newsletters` is verbatim inside train recap `compression-heavy`. The pool also drops any
    train input that contains the sitting case.
- Loads: one load per arm, plus a second load of each model's 8k arm as the stability check.
- Baselines: the 0-padding row is the committed lane 02 / 03 file (3 loads, majority). Rows the padding rule left
  unpadded reproduce those files' load 0 exactly: **18 of 18 per model** (printed by the script), summed over the 2k, 4k, 8k and repeat arms
  (verdict and gen tokens per call). So the lane-05 code path is the old path in effect.

**Headline.**
- **Length did not cost the dense model anything up to 8k.** e4b held-out scored **36 → 40 / 36 / 38** at 0 / 2k / 4k / 8k.
  - Each padded arm moves 6–8 ids, both ways (+6/−2, +4/−4, +5/−3).
  - The 8k repeat load is identical (0 of 55 moved), so these moves are the padding, not load noise. But they do not grow
    with length: they are lane 04's ±4 perturbation floor, not a context effect.
- **The MoE lost a third of its passes as soon as any history turn was present, and more length did not add to it.**
  LFM2.5 scored **21 → 14 / 16 / 14**.
  - Recap went 5 → 1 / 2 / 0. Recap sittings that ended with no answer went 4 → 7 / 4 / 11 per load. In every padded
    arm, every no-answer sitting had a length-capped call: thinking ran to `max_tokens`.
  - This is one model, not "MoEs". LFM2.5 differs from the e4b in more than active size: it is hybrid-conv, from
    another family, with another chat template. The inserted assistant turn renders cleanly in both templates
    (`<|im_start|>assistant\nUnderstood.<|im_end|>` / `<|turn>model\nUnderstood.<turn|>`, decoded), but the turn is a
    template-dependent structure, and this lane cannot separate model from template.
  - Twice, both in the `kv_bits` arms, it answered the padding turn itself ("Understood. I will await your next
    message."; wilco's `no JSON` error string). Rows store answer text only in that error string, so how often a
    *graded* answer does this is unmeasured.
- **Latency: on the dense model, p50 grows by about the prefill time.** p50 per held-out sitting:
  - e4b 15.3 → 16.3 / 21.5 / 31.5 s. Extra prompt tokens ÷ prompt tok/s accounts for +2.4 / +6.2 / +14.0 s of the
    measured +1.0 / +6.2 / +16.2 s. Prompt throughput is flat at 529–536 tok/s; decode is −9.7% at 8k.
  - e4b p95 42.8 → 37.9 / 40.3 / 50.7 s is not monotone. Padding removed the baseline's length-capped tails (2 length-capped
    calls per load at baseline and at 2k, 0 at 4k and 8k).
  - LFM 12.4 → 17.6 / 17.7 / 25.7 s grows by more than its prefill (+1.7 / +4.2 / +9.2 s): it thinks longer
    (1166 → 1395 gen tokens per sitting at 8k). p95 31.1 → 29.4 / 32.4 / 53.3 s.
- **The KV-cache type, as mlx-lm 0.31.3 exposes it:**
  - **`max_kv_size` is accepted and silently ignored on both models.** Both define `make_cache`, so
    `make_prompt_cache` never reads it. On the graded train smoke at `max_kv_size=1024` with 8k prompts, verdicts, gen
    tokens and the applied cache classes all equal the default's. This is the num_ctx-style failure in mlx form: the
    setting meant to bound the context took no effect and raised no error. Nothing was truncated, because these caches
    (`KVCache`, `ArraysCache`, and Gemma's fixed 512-token sliding windows) have no context bound to truncate at.
  - **`kv_bits` cannot run on Gemma 4.** `RotatingKVCache Quantization NYI` is raised on its sliding layers. The run
    aborts (lab adapter → `RuntimeError`); it is never scored as a FAIL.
  - **On LFM at 8k, `kv_bits` moves verdicts and costs memory.** kv8 scored 14 (+3/−3 ids, same total), kv4 scored 18
    (+4/−0) against the 8k default's 14, whose repeat moved 0. Peak memory **rose** from 5.90 to 6.94 / 6.91 GB. That is
    ~10× LFM's whole fp16 KV at 8k (6 layers × 8 heads × 64 × 2 × 2 B ≈ 12 KB/token ≈ 0.1 GB), so it is a conversion
    transient, not the cache. The mechanism was not isolated.

**The table** (`python3 scripts/lane05_rows.py`, which runs `lab check` on all 15 files first and exits 1 on drift; its
"Derived figures" blocks print every delta, ratio and count quoted in this section):

| model | arm | email /20 | call /19 | recap /16 | **total /55** | ids moved (vs) | p50 s | p95 s | prompt tok/s p50 | gen tok/s | peak GB | no-answer sittings (call · recap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b | 0 (lane 02) | 16 | 10 | 10 | **36** | – | 15.3 | 42.8 | 514 | 41.9 | 5.45 | 0 · 0 |
| e4b | 2k | 18 | 13 | 9 | **40** | +6/−2 (0) | 16.3 | 37.9 | 536 | 41.7 | 5.45 | 0 · 0 |
| e4b | 4k | 17 | 10 | 9 | **36** | +4/−4 (0) | 21.5 | 40.3 | 535 | 40.1 | 5.45 | 0 · 0 |
| e4b | 8k | 17 | 11 | 10 | **38** | +5/−3 (0) | 31.5 | 50.7 | 529 | 37.8 | 5.45 | 0 · 0 |
| e4b | 8k repeat | 17 | 11 | 10 | **38** | 0/0 (8k) | 31.6 | 50.7 | 529 | 37.8 | 5.45 | 0 · 0 |
| e4b | 8k kv_bits 8 / 4 | void | void | void | **void** | runtime raises NYI | | | | | | |
| LFM2.5 | 0 (lane 03) | 14 | 2 | 5 | **21** | – | 12.4 | 31.1 | 772 | 94.7 | 5.89 | 1 · 4 |
| LFM2.5 | 2k | 13 | 0 | 1 | **14** | +2/−9 (0) | 17.6 | 29.4 | 786 | 90.2 | 5.89 | 1 · 7 |
| LFM2.5 | 4k | 14 | 0 | 2 | **16** | +4/−9 (0) | 17.7 | 32.4 | 795 | 88.1 | 5.89 | 1 · 4 |
| LFM2.5 | 8k | 13 | 1 | 0 | **14** | +2/−9 (0) | 25.7 | 53.3 | 809 | 85.3 | 5.90 | 2 · 11 |
| LFM2.5 | 8k repeat | 13 | 1 | 0 | **14** | 0/0 (8k) | 26.4 | 53.3 | 786 | 83.7 | 5.90 | 2 · 11 |
| LFM2.5 | 8k kv_bits 8 | 14 | 0 | 0 | **14** | +3/−3 (8k) | 26.1 | 43.6 | 741 | 84.6 | 6.94 | 0 · 10 |
| LFM2.5 | 8k kv_bits 4 | 16 | 1 | 1 | **18** | +4/−0 (8k) | 26.3 | 45.4 | 723 | 85.0 | 6.91 | 1 · 6 |

How to read the table:
- Baseline walls are over all 3 loads' held-out sittings (165); the padded arms are over 55.
- Count columns are per load.
- Padded cases per arm (email · call · recap): 20 · 11 · 14 at 2k, 20 · 13 · 16 at 4k, 20 · 18 · 16 at 8k.
- Per-exam p50/p95, gen tokens per sitting, length-capped calls and every moved id are in the script's output.
- One load per padded arm. The two 8k repeats moved 0 of 55 each, so a single-load row is as reproducible as lanes
  02–04's 3-load rows, but this lane cannot show load-to-load stability at the other lengths.

**Prediction vs result** (`predictions/context-kv.md` + addendum):

| predicted | result | verdict |
|---|---|---|
| e4b 2k within ±1 per exam; 4k −0 to −2; 8k −1 to −3, recap worst, email ≤ −1 | 2k +2 · +3 · −1; 4k +1 · 0 · −1; 8k +1 · +1 · 0 | **refuted**: no drop, no order |
| e4b 8k absolute 15 · 12 · 10 | 17 · 11 · 10 | **refuted** (recap held) |
| LFM (addendum) 2k ±1 per exam; 8k email −1 to −3, call 1–3, recap 3–6, total 15–21 | 2k −1 · −2 · −4; 8k 13 · 1 · 0 = 14 | **refuted** (8k email and call held; recap and total below) |
| mechanism: distraction grows with length | the MoE step happens at 2k and does not grow; the dense model shows none | **refuted** |
| e4b latency: +15–30 s at 8k; email p50 ~9 → 25–40 s, p95 ~14 → 35–50; recap p50 ~18 → 35–50; prefill 250–450 tok/s | p50 +16.2 s; email 7.8 → 24.1 / 12.3 → 28.0; recap 20.7 → 34.5; 529 tok/s | **mostly refuted** (the shift is in band; the endpoints and the prefill rate are not) |
| e4b gen tok/s −5 to −10% at 8k | −9.7% | **held** |
| LFM +8–12 s per call at 8k (addendum) | email p50 +10.3, call +8.9, recap +16.5 s | **held** on email and call; recap thinks longer |
| KV type changes memory by < 5% of peak | e4b void; LFM +1.04 / +1.01 GB (+18% / +17%) | **refuted** (the probe had already shown it) |
| LFM kv8 0–2 flips; kv4 1–4, on call-fields exact tokens | kv8 6 (+3/−3); kv4 4 (+4/−0), on email and recap | **refuted** on kv8; kv4 count held, its place and direction refuted |
| quantised KV 5–15% slower | prompt −8% / −11%; decode ≤ −1%; p50 +0.4 / +0.5 s | **held** on prefill only |
| `max_kv_size` below the prompt is the num_ctx failure (email ≤ 10/20) | void: mlx-lm ignores `max_kv_size` on both models | **void**; the silent no-op is the finding |
| stability (addendum gate: > 3 of 55 moved voids small effects) | 0 of 55 on both repeats | did not fire |

**The stamp every result should carry.** Read it off the runtime, not off the request:
- mlx and mlx-lm versions;
- model repo, snapshot sha and quant (`bits`, `group_size`, `mode`);
- the chat template's sha256;
- per call, `prompt_tokens`, plus the padding (target, chars, sha256) when present;
- **the cache classes actually built, per class count** (e.g. `{"KVCache": 4, "RotatingKVCache": 20}` on the e4b, which
  has 42 layers but 18 share KV; `{"ArraysCache": 18, "QuantizedKVCache": 6}` on LFM at kv_bits) next to the requested
  `kv_bits`, `kv_group_size`, `quantized_kv_start`, `max_kv_size`;
- `prefill_step_size` (2048, the `generate_step` default);
- `max_tokens`, temperature, and the wilco pin.

The lane-05 results files carry all of it (`context` block, `calls[].context`). The requested-vs-applied pair is what
the version alone cannot see: today `max_kv_size=1024` was requested and not applied, and nothing failed.

**What did not hold.**
- **"Longer prompts hurt."** Not on these exams with this padding. The e4b's moves at 2k / 4k / 8k are the same size as
  lane 04's precision perturbation and net positive. The padding header tells the model the history is handled; a
  distractor pad without it is a different question, and this lane did not ask it.
- **"The MoE suffers half as much."** It was written for gpt-oss and is void with it. The MoE that did run fell 7 cases
  while the dense model gained.
- **"Quantising the KV cache saves memory."** On LFM it costs ~1 GB at 8k. Its KV is small (6 attention layers ×
  8 heads × 64, ~0.1 GB at 8k), so the transient outweighs the saving.

**What was refused or voided.**
- gpt-oss-20b rows: it does not serve on 16 GB (lane 03).
- e4b `kv_bits` arms: the runtime raises.
- Every `max_kv_size` arm: the runtime ignores it on both models. Not hand-built: a `RotatingKVCache` wired by hand would
  not be what mlx-lm exposes.
- Wh: no powermetrics capture, so it is null.
- Load-to-load stability at 2k and 4k is unmeasured (one load each).
- Probes and smokes are committed and not posted: `results/ctx-gate-*.json`, plus `scripts/lane05_probe_kv.py`, whose
  output is in the addendum. The e4b `kv_bits=8` smoke aborted with no file, as designed (its stderr is in the job
  report).

**Re-derive.**
```sh
python3 -m lab check results/ctx-*.json
python3 scripts/lane05_rows.py
```

**Reopen threshold.** Re-sit lane 05 when any of these holds:
- **an mlx-lm version** that implements `RotatingKVCache.to_quantized` or makes `make_cache` honour `max_kv_size`;
- **a model whose mlx-lm class has no `make_cache`**, e.g. `qwen3`, `qwen2`, `mistral3`, `phi3` in mlx-lm 0.31.3
  (`grep -L "def make_cache" mlx_lm/models/{qwen3,qwen2,mistral3,phi3}.py` lists all four). Lane 02's Llama-3.1-8B is *not* one: `llama.py` defines `make_cache`. On those classes
  `max_kv_size` *does* build `RotatingKVCache(keep=4)`, which silently drops the head of a longer prompt. That is the
  truncation this lane could not reach, and the first arm to run;
- prompts beyond 8k (the e4b's 7 global layers, of which 4 hold their own cache, have not yet been asked to carry more);
- a distractor padding without the "already handled" header;
- a new case pin.

## Lane 09: hosted baseline on the 55 held-out cases (9 Nov 2026)

**Question.** Lanes 02–04 posted what the laptop passes on wilco's three office exams. Lane 09 asks what hosted Claude
passes on the same 55 held-out cases with the same grader, and what a case costs. The models are `claude-haiku-5-5`
(three sits), `claude-sonnet-5-5` (one sit) and `claude-opus-5-5` (one sit). This lane lifts the lab's no-hosted-calls
rule for this one comparison, under a $2.00 ceiling and the operator's go. The prediction was committed alone, before any
call: `predictions/hosted-baseline.md`, commit `1088b19`.

**Protocol.**
- Cases: the 55 held-out ids at wilco pin `62fcd449` (the read-only clone, as in lanes 02–04).
- Grading: wilco's `run_exam` per case, through **wilco's own** `anthropic` provider (`router.get_adapter`, the
  OpenAI-compatible endpoint, `temperature` dropped by wilco).
- Metering: `lab/hosted.py` wraps `requests.post` only to read each billed response's `usage`, `finish_reason` and model;
  no request byte is changed.
- Settings: `max_tokens` from each `agent.yaml`, effort at the API default, no prompt caching, no fallback model.
- Cost: list price × metered tokens. Prices are stored in each file, and `lab check` re-derives every $ cell. Rates per
  MTok in / out: Haiku $0.10 / $0.50, Sonnet $2 / $10, Opus $4 / $20. These come from the claude-api model table cached
  2026-10-06 (rank C), the same rates wilco's 28 Oct probe used.

**Headline.** On these three exams, **hosted Claude is level with the laptop's best row, not above it**:
- **Sonnet 44/55**, **Haiku 40/55** (sits 39 · 38 · 40), **Opus 40/55**, against the 8-bit e4b's **41/55**.
- Two of the three hosted models sit below the laptop. Sonnet is +3.
- The gap is not uniform. It is two opposite gaps that cancel:
  - Hosted wins email-triage by +3 to +4 and call-fields by +1 to +4.
  - Hosted loses recap by 5 to 6 cases (6–7/16 against 12/16). Claude writes past recap's length rule
    (`check_compression`) and calls a quiet day important (`check_abstention`).
- A case costs **$0.00037 (Haiku), $0.0062 (Sonnet), $0.0132 (Opus)**, i.e. $0.37 / $6.15 / $13.24 per 1,000 held-out
  cases. Local cost per case is void: no powermetrics capture, so Wh is null, as in lane 04.
- Lane spend: **$1.128**, all of it in the committed files plus the 3-case smoke. All five sits ran; Opus sat after the
  pre-registered gate.

**The 55-case table** (held-out, majority over loads or sits; `python3 scripts/lane09_rows.py`, which first runs
`lab check` on every file and exits 1 on drift):

| arm | sits | email-triage /20 | call-fields /19 | recap /16 | **total /55** | unstable held-out ids | wall s / held-out case | $ / held-out case | $ / case: email · call · recap |
|---|---|---|---|---|---|---|---|---|---|
| gemma-4-e2b-it 4-bit (local) | 3 loads | 14 | 7 | 2 | **23** | 0 | 11.47 | void (Wh not metered) | void |
| gemma-4-e2b-it 8-bit (local) | 3 loads | 17 | 9 | 6 | **32** | 0 | 13.58 | void (Wh not metered) | void |
| gemma-4-e4b-it 4-bit (local) | 3 loads | 16 | 10 | 10 | **36** | 0 | 19.62 | void (Wh not metered) | void |
| gemma-4-e4b-it 8-bit (local) | 3 loads | 16 | 13 | 12 | **41** | 0 | 22.37 | void (Wh not metered) | void |
| Llama-3.1-8B-Instruct 4-bit (local) | 3 loads | 16 | 6 | 9 | **31** | 0 | 6.99 | void (Wh not metered) | void |
| Llama-3.1-8B-Instruct 8-bit (local) | 3 loads | 17 | 10 | 10 | **37** | 0 | 7.90 | void (Wh not metered) | void |
| claude-haiku-5-5 (hosted) | 3 | 19 | 15 | 6 | **40** | 7 | 2.08 | 0.00037 | 0.00005 · 0.00068 · 0.00039 |
| claude-sonnet-5-5 (hosted) | 1 | 20 | 17 | 7 | **44** | unmeasured (1 sit) | 2.12 | 0.00615 | 0.00112 · 0.01183 · 0.00569 |
| claude-opus-5-5 (hosted) | 1 | 20 | 14 | 6 | **40** | unmeasured (1 sit) | 2.84 | 0.01324 | 0.00275 · 0.02299 · 0.01479 |

How to read the table:
- "Unstable" means different things in the two halves. Local: load-to-load disagreement at temperature 0 (0 everywhere).
  Hosted: sampling disagreement between sits. The endpoint rejects `temperature`, so a hosted sit is a sample.
- Sonnet's and Opus's stability is **unmeasured**, not 0. Haiku moved 7 of 55 ids across three sits, so a one-sit
  hosted row carries about ±2 cases of noise.
- Local wall is mlx on this M-series laptop. Hosted wall is a network round trip, Anthropic's own compute, ~2–3 s.
- The $ cells are list-price arithmetic over metered tokens, not an invoice.

**Detail** (same script; its "Derived figures" and probe sections print every number below):
- **Haiku per sit:** email 19 · 18 · 19; call-fields 13 · 15 · 15; recap 7 · 5 · 6.
  - Majority 40; stable-pass 35/55.
  - The 7 ids that moved between sits are recap `amounts-at-length`, `buried-last-en`, `quiet-day-with-receipt`,
    `rounding-bait`; call-fields `discovery-heldout-long`, `longcall-mixed-heldout-3`; email `intro-no-ask`.
- **Tokens:** Haiku 361,950 in + 49,015 out over 165 sittings. Sonnet 120,650 + 9,686. Opus 120,650 + 12,292.
  - One call per sitting on every arm. 0 unmetered calls, 0 length-capped no-answer sittings.
  - Input is 60–71% of the cost on every model (59.6 · 71.4 · 66.3%), not the hidden thinking. The 5.5 family barely thinks on these exams
    (Sonnet 176 output tokens per case; on the smoke, email cases billed 13).
- **Against the 8-bit e4b, by id** (majority):
  - Every hosted model passes email `cold-sales-injection-urgent`, `newsletter-injection-urgent` and
    `recruiter-fake-urgency`, and call-fields `discovery-heldout-long`, `longcall-renewal-heldout-2` and
    `mixed-heldout-dense`. The laptop fails all six.
  - The laptop passes recap `automated-but-actionable`, `dependency-delivery-vs-install`, `rounding-bait` and
    `two-urgent-among-noise`, and call-fields `support-heldout-long`. All three hosted models fail them.
  - Hosted-only / local-only passes: Haiku 7 / 8, Sonnet 9 / 6, Opus 7 / 8.
- **Against wilco's 28 Oct probe** (same models, same exams, same provider, one pass each):
  - Opus 40 = 40 per exam. By id, 2 recap cases swapped (`meeting-moved-date-decoy`, `rounding-bait`).
  - Haiku 42 → 40 (email and call-fields identical by id; recap 8 → 6, 4 ids moved).
  - Sonnet 41 → 44 (call 18 → 17 on `support-heldout-long`; recap 3 → 7, 4 ids up).
  - Every case id matches between the probe and this lane, and all moves are on recap except Sonnet's one call-fields case.
  - Sonnet's recap moving 4 cases between two days is the sampling noise this table cannot remove with one sit.

**Prediction vs result** (`predictions/hosted-baseline.md`):

| predicted | result | verdict |
|---|---|---|
| Haiku 39–45 (point 42); email 18–20 · call 13–16 · recap 6–10 | 40; 19 · 15 · 6 | **held** |
| Sonnet 38–43 (point 41); 19–20 · 16–18 · 2–6 | 44; 20 · 17 · 7 | **refuted** (total +1 over the band, recap +1 over) |
| Opus 37–43 (point 40); 19–20 · 13–16 · 4–8 | 40; 20 · 14 · 6 | **held** |
| at least one hosted model strictly below 41, most likely Opus; none above 41 by more than 3 | Haiku 40, Opus 40; Sonnet 44 (= +3) | **held** |
| hosted wins email (+3 to +4), loses recap; Sonnet +3 to +5 on call-fields, Haiku and Opus within ±2 | email +3 / +4 / +4; recap −6 / −5 / −6; call Sonnet +4, Haiku +2, Opus +1 | **held** |
| Haiku sit spread 1–4 (point 2); 2–7 ids move, most on recap; majority within ±1 of the median sit | 2; 7 ids, 4 on recap; majority 40 vs median 39 | **held** |
| $ per case: Haiku 0.0003–0.0006, Sonnet 0.004–0.009, Opus 0.009–0.018 | 0.00037 · 0.00615 · 0.01324 | **held** |
| call-fields about 10× email per case; Opus / Sonnet 1.8–2.6× | 12.4× · 10.6× · 8.4×; 2.15× | **held** |
| wall 3–15 s per case, Opus slowest | 2.08 · 2.12 · 2.84 s; Opus slowest | **refuted** (faster than the band) |
| mechanism: output cost is mostly hidden thinking | input is 60–71% of cost; Sonnet 176 output tokens per case | **refuted** |
| kill clauses: unmetered calls; thinking not in `completion_tokens`; > 5 length no-answers per sit | 0 unmetered; the smoke's 468 tokens for a 226-char answer; 0 length no-answers | did not fire |

**What did not hold.**
- **Sonnet's band.** The anchor was its 28 Oct recap 3/16. Today it scored 7/16 on one sit. A single hosted sit moved 4
  recap cases between two days, more than the 0–4 per exam the band allowed for.
- **"Hosted cost is thinking."** On these short-answer exams the 5.5 family mostly does not think. The bill is the prompt.
  Prompt caching of the fixed system prompt, not effort, is the lever. Not tested here: the lane is list price, uncached.
- **Wall.** Hosted is 2.5–10.7× faster per case than the laptop (2.08–2.84 s vs 6.99–22.37 s), the reverse of a cost-per-case story. The laptop's wall is
  its own compute; hosted wall is someone else's.

**What was refused or voided.**
- The local cost per case is void: no operator powermetrics capture, Wh null, as lane 04 posted.
- Sonnet and Opus stability is unmeasured (one sit each, by the brief's budget).
- No train case sat in a posted arm. The gate smoke (`results/hosted-gate-claude-haiku-5-5-smoke.json`, 3 train cases,
  $0.0007) is committed and not posted.
- The effort level was not tuned: the endpoint sends none, so these are the API defaults (Haiku and Opus medium, Sonnet
  high; rank C).
- `tier: local` in the three `agent.yaml` was lifted by the operator's go for this lane. The cases are wilco's synthetic,
  public exams.

**Re-derive.**
```sh
python3 -m lab check results/hosted-claude-haiku-5-5.json results/hosted-claude-sonnet-5-5.json results/hosted-claude-opus-5-5.json
python3 scripts/lane09_rows.py
```

**Reopen threshold.** Re-sit lane 09 when any of these holds:
- **a new case pin**: wilco's pin or any `cases.json` hash moves;
- **a new model tag**: a new Claude model id; a local row is re-posted (a new pack, a new mlx or mlx-lm); or a local
  model reaches ≥ 45/55, which would clear Sonnet's 44 by more than one sit's noise;
- the list prices change, or the endpoint starts honouring `temperature` or `effort`;
- an operator powermetrics capture exists for a local arm, so a local $ per case can sit beside the hosted one;
- the budget allows three sits of Sonnet and Opus. Their one-sit rows are within Haiku's ±2 sit noise of each other, so
  the order Sonnet > Opus = Haiku is not established by this lane.

## Lane 04: quantisation (9 Nov 2026)

**Question.** On lane 02's three models, what does 4-bit vs 8-bit vs bf16 move on wilco's three office exams, and what
does it give back in memory and Wh? Prediction committed first: `predictions/quantisation.md` (commit `043e8bd`, 6 Nov).
A dated addendum was committed before any lane-04 load (`8e12cb7`, 9 Nov). It fixes the arms and voids, the pre-run
fetch, a prediction for the Llama-8B (the original covered only the Gemmas), the run order with time and memory gates,
and the flip definition.

Protocol as in lane 02: all 83 cases at wilco pin `62fcd449`, three fresh loads, wilco's scorer, majority/stable rule, mlx
0.32.2 and mlx-lm 0.31.3. The 4-bit arms are lane 02's committed files, not re-sat. The packs are all `mlx-community`
affine quantisation, group size 64 (`config.json`). The 4-bit files ran at `lab_commit` `27786ced`, the lane-04 arms at `7b8fd34`–`148bb3f`. `git diff 27786ced HEAD -- lab/` touches only `lab/metering.py`, to add `split_harmony`, and that branch runs only when `<|channel|>` is a vocab token. It is absent from all seven tokenizers here (`tokenizer.json`), so the adapter path for Gemma and Llama is unchanged between the two sets. The packs' `model.snapshot` fields match the addendum's revisions.

**Headline.** **8-bit beat 4-bit on every model**:
- held-out passes: e2b **23 → 32/55**, e4b **36 → 41/55**, Llama-3.1-8B **31 → 37/55**;
- **0 of 83 cases unstable** on every arm, so each flip is a stable verdict at both precisions;
- the price: +2.3 to +3.9 GB peak memory and **×1.08 to ×1.16 wall** per 3 × 83 run. Decode is 0.59–0.76× as fast, but
  the Gemmas think less at 8-bit.

bf16, which fits for the e2b only, scored **28/55: below 8-bit, above 4-bit**, at ×1.81 the 4-bit wall and 10.28 GB.
- Read: 4-bit costs real cases (bf16 is +5 over it).
- bf16 vs 8-bit (−4) is the size of the perturbation spread at temperature 0, not a precision gain.

Wh was not metered (no operator powermetrics capture), so every energy figure below is void, and wall time stands in,
labelled as wall.

**Per model** (held-out majority pass; all three loads agreed on every case in every arm):

| e2b (gemma-4-e2b-it) | email-triage | call-fields | recap | total | gen tok/s | peak GB | wall 3 × 83 min | min / 1,000 cases: email · call · recap | tokens per sitting | Wh |
|---|---|---|---|---|---|---|---|---|---|---|
| 4-bit (lane 02) | 14/20 | 7/19 | 2/16 | 23/55 | 69.2 | 3.71 | 52.5 | 59.5 · 321.3 · 241.2 | 804.5 | not metered |
| **8-bit** | **17/20** | **9/19** | **6/16** | **32/55** | 52.9 | 6.03 | 56.9 | 113.4 · 266.8 · 315.8 | 671.9 | not metered |
| bf16 | 16/20 | 8/19 | 4/16 | 28/55 | 31.6 | 10.28 | 95.1 | 185.6 · 452.3 · 523.2 | 694.4 | not metered |

| e4b (gemma-4-e4b-it) | email-triage | call-fields | recap | total | gen tok/s | peak GB | wall 3 × 83 min | min / 1,000 cases: email · call · recap | tokens per sitting | Wh |
|---|---|---|---|---|---|---|---|---|---|---|
| 4-bit (lane 02) | 16/20 | 10/19 | 10/16 | 36/55 | 41.9 | 5.45 | 81.5 | 138.3 · 374.7 · 491.1 | 686.3 | not metered |
| **8-bit** | 16/20 | **13/19** | **12/16** | **41/55** | 29.3 | 9.18 | 94.5 | 168.0 · 403.7 · 602.9 | 576.5 | not metered |
| bf16 | void: 15.91 GB pack, over the 11.45 GB working set | | | | | | | | | |

| Meta-Llama-3.1-8B-Instruct | email-triage | call-fields | recap | total | gen tok/s | peak GB | wall 3 × 83 min | min / 1,000 cases: email · call · recap | tokens per sitting | Wh |
|---|---|---|---|---|---|---|---|---|---|---|
| 4-bit (lane 02) | 16/20 | 6/19 | 9/16 | 31/55 | 35.4 | 6.30 | 29.7 | 21.8 · 213.3 · 106.9 | 43.4 | not metered |
| **8-bit** | **17/20** | **10/19** | **10/16** | **37/55** | 20.9 | 10.24 | 33.8 | 23.2 · 237.5 · 131.1 | 45.3 | not metered |
| bf16 | void: 16.08 GB pack | | | | | | | | | |

**The cases that flipped** (ids; `[t]` train, else held-out; every flip is stable at both precisions):
- **e2b 4 → 8** (16 ids):
  - email up `cold-sales-injection-urgent`, `invoice-error-client`, `newsletter-injection-urgent`;
  - call up `longcall-mixed-heldout-3`, `quiet-checkin-heldout-medium`, `support-heldout-short`, `longcall-quiet-train-1[t]`,
    `longcall-support-train-1[t]`, `quiet-checkin-medium-train[t]`, `support-long-train[t]`;
  - call down `renewal-heldout-medium`;
  - recap up `buried-critical`, `dependency-delivery-vs-install`, `rounding-bait`, `two-urgent-among-noise`,
    `quiet-day-automated[t]`.
- **e2b 8 → bf16** (7 ids):
  - email down `cold-sales-injection-urgent`;
  - call up `longcall-support-heldout-2`;
  - call down `longcall-mixed-heldout-3`, `support-heldout-short`, `longcall-support-train-1[t]`;
  - recap down `rounding-bait`, `two-urgent-among-noise`.
- **e4b 4 → 8** (16 ids):
  - email up `intro-no-ask`, `client-question-no-rush[t]`;
  - email down `paid-workshop-invite`;
  - call up `longcall-pricing-heldout-1`, `quiet-checkin-heldout-medium`, `renewal-heldout-digits`, `renewal-heldout-medium`,
    `longcall-renewal-train-1[t]`, `quiet-checkin-medium-train[t]`, `quiet-checkin-short-train[t]`, `support-long-train[t]`,
    `support-medium-train[t]`;
  - call down `longcall-mixed-heldout-3`, `longcall-support-train-1[t]`;
  - recap up `amounts-at-length`, `quiet-day-with-receipt`.
- **Llama 4 → 8** (14 ids):
  - email up `invoice-error-client`;
  - call up `longcall-discovery-heldout-2`, `longcall-mixed-heldout-3`, `renewal-heldout-digits`, `renewal-heldout-medium`,
    `discovery-long-train[t]`;
  - recap up `amounts-at-length`, `long-quiet-day`, `quiet-day-newsletters`, `quiet-day-with-receipt`;
  - recap down `automated-but-actionable`, `dependency-delivery-vs-install`, `rounding-bait`, `amounts-and-dates[t]`.

Held-out flips, 4 → 8:
- **e2b**: 10 up, 1 down.
- **e4b**: 7 up, 2 down.
- **Llama**: 9 up, 3 down. Its recap is near-symmetric (held-out 4 up, 3 down; plus one train case down) and its call-fields one-way (4 up, 0 down).

Some cases flip in more than one model, all in the 8-bit direction: `invoice-error-client` (e2b and Llama),
`quiet-checkin-heldout-medium` (e2b and e4b), `renewal-heldout-digits` (e4b and Llama) and `amounts-at-length` (e4b and
Llama). Others move opposite ways in different models. `longcall-mixed-heldout-3` goes up in e2b and Llama and down in
e4b. `renewal-heldout-medium` goes up in e4b and Llama and down in e2b.

Re-derive (every number above):
```sh
~/.pyenv/versions/lewagon/bin/python3 -m lab check results/quant-*.json \
  results/dense-gemma-4-e2b-it-4bit.json results/dense-gemma-4-e4b-it-4bit.json results/dense-llama-3.1-8b-instruct-4bit.json
P=~/.pyenv/versions/lewagon/bin/python3
$P scripts/lane04_rows.py dense-gemma-4-e2b-it-4bit quant-gemma-4-e2b-it-8bit quant-gemma-4-e2b-it-bf16
$P scripts/lane04_rows.py quant-gemma-4-e2b-it-8bit quant-gemma-4-e2b-it-bf16
$P scripts/lane04_rows.py dense-gemma-4-e4b-it-4bit quant-gemma-4-e4b-it-8bit
$P scripts/lane04_rows.py dense-llama-3.1-8b-instruct-4bit quant-llama-3.1-8b-instruct-8bit
$P scripts/lane04_gate.py quant-gate-gemma-4-e2b-it-8bit-smoke quant-gate-gemma-4-e2b-it-8bit-call dense-gemma-4-e2b-it-4bit  # likewise per arm
```
Flip ids, tokens per sitting, the wall ratios and the deltas come from `scripts/lane04_rows.py` over the checked rows
(`rows_sha256`). Pass counts, tok/s, peak GB and minutes per 1,000 cases are `lab check` aggregates.

**Memory and Wh deltas.**
- **Memory**, peak GB at 8-bit vs 4-bit: e2b +2.31, e4b +3.73, Llama +3.94. bf16 vs 4-bit on e2b: +6.57.
- **Llama-8bit's 10.24 GB** peak is 1.21 GB under mlx's recommended working set; e2b-bf16's 10.28 GB is 1.17 GB under.
- **Wh: void on every arm** (`wh_per_case: null`). There was no powermetrics capture. The prediction's 1.4–1.7× (8-bit)
  and 2.3–3.0× (bf16) are not scored.
- **What stands in: wall per 3 × 83 run**, ×1.08 / ×1.16 / ×1.14 at 8-bit (e2b / e4b / Llama) and ×1.81 at bf16. At the
  near-constant package power the prediction's mechanism assumed, Wh would track these ratios. That is a mechanism, not a
  measurement.

**What the cost is made of.** Decode slows less than the bytes double: 8-bit runs at ×0.76 (e2b), ×0.70 (e4b) and ×0.59
(Llama) the 4-bit tok/s, and bf16 at ×0.46. The two Gemmas also generate fewer tokens per sitting at 8-bit:
- e2b: 804.5 → 671.9 tokens per sitting. On call-fields, 1192 → 745: lane 02's 4096-capped `support-long-train` stops
  capping and passes.
- e4b: 686.3 → 576.5.

Fewer tokens at a slower decode gives near-equal wall: e2b call-fields at 8-bit is *cheaper* than at 4-bit (266.8 vs 321.3
min per 1,000 cases). Llama barely thinks (43 → 45 tokens per sitting), so its wall follows prefill and decode: ×1.14.

**The rule I would ship.** **8-bit, unless its peak does not fit the working set. In that case, take the largest model
whose 4-bit fits, not a smaller model at 8-bit. bf16 never, on this box.**
- *Why 8-bit:* the pre-registered trigger is a held-out case stable-fail at 4-bit and stable-pass at 8-bit. It fired on
  every exam of every model except e4b email-triage, where it fired once against one case going down (net 0). The gain is
  +5 to +9 held-out cases per model, for ≤ ×1.16 wall.
- *Peak to budget:* the 8-bit pack plus 0.1–1.7 GB (measured: e2b 5.90 → 6.03, e4b 8.91 → 9.18, Llama 8.54 → 10.24).
- *Why the larger 4-bit:* one pair. e4b-4bit scored 36/55 at 5.45 GB; e2b-8bit scored 32/55 at 6.03 GB.
- *Why not bf16:* on the one model where it fits, it scored 4 below 8-bit at ×1.67 the wall and +4.3 GB.
- *Not shown:* that 8-bit beats a 6-bit or 5-bit pack, or the QAT/OptiQ packs. mlx-community ships 5-bit, 6-bit, mxfp4,
  mxfp8, nvfp4 and QAT/OptiQ packs for both Gemmas, and 3-bit for Llama (HF API, 9 Nov). None was sat, because the
  addendum fixed 4/8/bf16.

**Prediction vs result.**

| claim | predicted | result | |
|---|---|---|---|
| 4 → 8 held-out moves per exam | 0–2 each way, net within ±1 | e2b +3 · +2 · +4; e4b 0 · +3 · +2 | refuted (5 of 6 exams net > ±1) |
| distinct ids flipped 4 → 8, both Gemmas | 3–6 | 16 + 16 (all splits) | refuted |
| ≥ 2 flips on call-fields, ≤ 1 on email-triage | yes | call 8 (e2b), 11 (e4b); email 3 · 3 | call held; email refuted |
| e2b 8-bit within ±1 of its 4-bit row, every exam | yes | +3 · +2 · +4 | refuted |
| e4b 8-bit within ±1 of 16/20 · 14/19 · 13/16 (wilco's QAT champions) | yes | 16 · 13 · 12 | **held**, all three (see Literature) |
| 8-bit vs bf16 moves 0–1 per exam (e2b) | 0–1 | −1 · −1 · −2 (bf16 lower) | email, call held; recap refuted |
| 8-bit tok/s = 0.55–0.70× 4-bit (e2b 32–38, e4b 21–26) | ratio and absolute | ×0.76 (52.9), ×0.70 (29.3) | e2b refuted; e4b ratio held at the edge, absolute refuted (anchors were 55/37) |
| bf16 tok/s = 0.30–0.40× (17–22) | | ×0.46 (31.6) | refuted (faster) |
| peak GB e2b 6.2–6.7 · 10.5–11.2; e4b 9.3–9.9 | | 6.03 · 10.28; 9.18 | refuted, all below (pack + 0.04–0.27 GB, not + 0.3–0.6) |
| Wh 8-bit 1.4–1.7×, bf16 2.3–3.0× | | not metered | void |
| rule: "4-bit, full stop; flips symmetric noise" | | 8-bit up on all three models, 0 unstable; Llama recap symmetric, the rest one-way | **refuted** |
| † Llama 8-bit held-out email · call · recap | ±1 · +0 to +2 · ±1 | +1 · +4 · +1 | email, recap held; call refuted |
| † Llama ids flipped | 1–4 | 14 | refuted |
| † Llama tok/s 19–23 · peak 9.9–10.6 GB | | 20.9 · 10.24 | **held**, both |
| † Llama min / 1,000 cases ×1.3–1.6 | | ×1.06 · ×1.11 · ×1.23 | refuted (cheaper) |
| kill: > 3 of 83 unstable on either arm | | 0 on every arm | did not fire |

† = addendum rows, written after lane 02's 4-bit row was known.

**What did not hold.**
- **"The flips are symmetric noise."** Pooled over the three models, 26 held-out flips go 4 → 8 up and 6 go down, all
  stable at both precisions. Only Llama's recap is near-symmetric (4 up, 3 down).
- **More bits is not monotone.** e2b-bf16 sits 4 below e2b-8bit, on 7 stable flips (1 up, 6 down). So a ±4 swing is
  within reach of a perturbation that is not a precision loss. A single model's 4 → 8 gain of +5 is therefore not proof.
  The claim rests on the direction holding across three models (+9, +5, +6) and on bf16 also sitting above 4-bit (+5).
- **The gate's time projection ran low on the Gemmas.** It projected 22 min for e2b-8bit (actual 56.9) and 35 min for
  e2b-bf16 (actual 95.1). The 4-bit baseline's gate case `support-long-train` was length-capped (63 s), so the ratio
  understated the run. Every arm still finished before the 10:10 cut-off, so the gate decided nothing, but the projection
  is not a reliable estimator when a baseline gate case caps.

**What was refused or voided.**
- e4b and Llama bf16 do not fit (15.91 and 16.08 GB packs).
- Wh is void.
- The 14B rows are void (lane 02).
- None of the four lane-04 packs was cached. They were fetched before any load at the revisions in the addendum
  (`03dcf209`, `fb0b166b`, `4255b21b`, `142d4280`), with `refs/main` written by hand. `lab run` stayed `local_files_only`.
- The gate smoke files (`results/quant-gate-*.json`) are committed and not posted. All loads ran on AC power.

**Reopen threshold.** Re-open lane 04 when any of these holds:
- an operator powermetrics capture exists for a re-sit, so the Wh rows can be scored;
- a 6-bit (or mxfp8, or a QAT/OptiQ 4-bit) pack recovers ≥ 4 of a model's 4 → 8 held-out gain at under the 8-bit peak;
  that would move the rule;
- mlx or mlx-lm moves off 0.32.2 / 0.31.3, or wilco's pin moves;
- a re-sit flips any of the 83 cases on any lane-04 arm (all are stable today);
- a machine with ≥ 24 GB, where e4b and Llama bf16 can sit.

## Lane 03: MoE vs dense (9 Nov 2026)

**Question.** Does a mixture-of-experts model with N active parameters pass wilco's three office exams at the cost of a
dense N model, or at the cost of its total parameter count? Prediction committed first: `predictions/moe-vs-dense.md`
(commit `043e8bd`, 6 Nov), with a dated addendum committed before any lane-03 load (`366780e`, 9 Nov). The addendum adds
no numbers. It fixes the comparators after lane 02, the machine gate and the Harmony rule. Case set, loads and grading are
as in lane 02: all 83 cases at wilco pin `62fcd449`, three fresh loads, wilco's scorer, majority/stable rule.

**Headline.** On this 16 GB M1 Pro the answer depends on whether the MoE fits.
- **The MoE that fits is cheaper than its active size per token, and costs its total in memory.** `LFM2.5-8B-A1B`
  (1.5B active / 8.3B total) sat the 3 × 83 run.
  - Decode: **94.7 tok/s**, against 69.2 for the active-size dense e2b and 35.4 for the total-size dense
    Llama-3.1-8B.
  - Peak memory: **5.89 GB**, against e2b's 3.71 and the 8B's 6.30.
  - Per case it does not keep the per-token saving: e2b +0% · +9.5% · +32% in minutes per 1,000 cases (email · call ·
    recap), because it thinks 1.48× longer. It sits far above the 8B, but that comparator does not think.
  - It passes less than either: held-out **21/55** (14 · 2 · 5) vs e2b 23/55 and 8B 31/55.
  - 0 of 83 cases unstable.
- **The MoE in the brief's size class does not run on this box at any cost the lab can pay.** `gpt-oss-20b` (3.6B
  active / 21B total) is a 12.1 GB pack, over mlx's 11.45 GB recommended working set.
  - On the gate smoke it decoded at **2.91 tok/s** against e4b's 43.9 on the same two cases, with a 12.35 GB peak.
  - One call-fields train case did not finish in 40 minutes.
  - It costs like more than its total: that is a 16 GB statement, not a MoE one.

| model (4-bit) | active / total | email-triage held-out | call-fields held-out | recap held-out | gen tok/s | peak GB | tokens per sitting | min / 1,000 cases: email · call · recap | Wh |
|---|---|---|---|---|---|---|---|---|---|
| **LFM2.5-8B-A1B-MLX-4bit (MoE)** | 1.5B / 8.3B | 14/20 | 2/19 | 5/16 | **94.7** | 5.89 | 1190.1 | 59.7 · 351.8 · 318.3 | not metered |
| gemma-4-e2b-it-4bit (lane 02, active-size dense) | dense | 14/20 | 7/19 | 2/16 | 69.2 | 3.71 | 804.5 | 59.5 · 321.3 · 241.2 | not metered |
| Meta-Llama-3.1-8B-Instruct-4bit (lane 02, total-size dense) | dense | 16/20 | 6/19 | 9/16 | 35.4 | 6.30 | 43.4 | 21.8 · 213.3 · 106.9 | not metered |
| gpt-oss-20b-MXFP4-Q8 (MoE), gate smoke only | 3.6B / 21B | 2/2 train (smoke) | did not finish 1 train case in 40 min | not sat | 2.91 | 12.35 | 70.5 (2 cases) | not projectable | not metered |
| gemma-4-e4b-it-4bit (lane 02, gpt-oss's active-size dense) | dense | 16/20 | 10/19 | 10/16 | 41.9 (43.9 on the smoke cases) | 5.45 | 686.3 | 138.3 · 374.7 · 491.1 | not metered |

Re-derive (every aggregate above):
```sh
~/.pyenv/versions/lewagon/bin/python3 -m lab check results/moe-lfm2.5-8b-a1b-mlx-4bit.json \
  results/moe-gate-gpt-oss-20b-smoke.json results/dense-gemma-4-e2b-it-4bit.json \
  results/dense-gemma-4-e4b-it-4bit.json results/dense-llama-3.1-8b-instruct-4bit.json
~/.pyenv/versions/lewagon/bin/python3 scripts/lane03_rows.py moe-lfm2.5-8b-a1b-mlx-4bit moe-gate-gpt-oss-20b-smoke \
  dense-gemma-4-e2b-it-4bit dense-gemma-4-e4b-it-4bit dense-llama-3.1-8b-instruct-4bit
```
Three kinds of figure come from `scripts/lane03_rows.py` over the checked rows, which are covered by `rows_sha256`:
- the row-level reads (tokens per sitting, thinking share, length-capped and empty-answer calls, failing cases);
- the tok/s on the two smoke cases;
- the gate's 2-case projection: gpt-oss 76.5 s / e4b 13.7 s on the same two cases × e4b's 4888 s run = **454 min**,
  against the 300-minute threshold.

The projection supports the gate; it is not the gate. The pre-registered formula also needed the call-fields case, and
that case **did not complete within the 40-minute cap, so it has no results file** (`lab run` writes on completion). Wh is
`null`: there was no operator powermetrics capture. All loads ran on AC power.

**What the cost is made of.** The MoE that fits decodes 1.37× faster per token than the active-size dense (94.7 vs 69.2
tok/s). It also thinks 1.48× longer (1190 vs 804 tokens per sitting, a 0.971 thinking share), and the two cancel: per case
it lands on e2b's cost on email-triage (59.7 vs 59.5 min per 1,000), 9.5% above on call-fields and 32% above on recap. On
recap, 30 of its calls hit wilco's 2048-token cap. That is the prediction's "unsure" item (reasoning spends `max_tokens`
before the answer), here on recap. Lane 02's lesson holds again: on this box a thinking model's cost is its thinking. A
MoE's per-token saving is real (decode follows the active bytes) but small next to how long the model chooses to think.

**Prediction vs result.** The numeric bands were written for gpt-oss-20b. The gate stopped it after the smoke, so most of
its rows are void. LFM2.5 was the named fallback, and only the prediction's summary line applies to it.

| claim | predicted | result | |
|---|---|---|---|
| gpt-oss peak GB | 12.5–13.5 | 12.35 on 2 short train cases, a lower bound | void: a lower bound below the band cannot refute it |
| gpt-oss gen tok/s | 26–34 (0.7–0.9× e4b) | 2.91 (0.07× e4b on the same cases) | refuted |
| "if the pack pages (gen_tps under 12) every number turns total-like: costs like the total on a 16 GB machine" | conditional | 2.91 tok/s; mlx-lm warns 11,516 MB needed vs 10,922 MB recommended | **held** |
| gpt-oss held-out pass email · call · recap | 17 · 13 · 8 | not sat (gate) | void |
| gpt-oss Wh | 0.9–1.3× e4b | not metered | void |
| reasoning spends `max_tokens` (call-fields) | unsure | gpt-oss: void. LFM2.5: 3 capped calls on call-fields, 30 on recap | turned up on recap |
| memory costs like the total (summary) | yes | LFM2.5 5.89 GB vs 8B 6.30 / e2b 3.71 | **held** |
| time costs like N, the active size (summary) | yes | tok/s: 1.37× faster than e2b, not "like" it. Minutes per case: +0% · +9.5% · +32% vs e2b, far above the 8B (which does not think) | per case held on email and call; recap and tok/s refuted |
| quality not better than the active-size dense (summary) | yes | 21/55 vs e2b 23/55; recap +3, call-fields −5 | held overall, not per exam |
| kill: within noise of the active-size dense on tok/s (±10%) and pass (±1 per exam) | fires | tok/s +37%; call −5, recap +3 | **did not fire** |

**What did not hold.**
- **Call-fields collapses on the MoE.** It scored 2/19 held-out against e2b's 7 and the 8B's 6. Its load-0 held-out
  fails are all field mismatches (16) plus one termination. Exact-field extraction is where the 1.5B active budget shows.
- **The fit claim on gpt-oss's card is true and useless here.** The card says the 20b "run[s] within 16GB of memory". It
  loads and answers (2/2 smoke), but at 2.91 tok/s. A 12 GB pack is not usable on a 16 GB laptop running anything else.

**What was refused or voided.**
- gpt-oss-20b's 3 × 83 run was refused by the gate committed before any load: smoke wall over 137 s (the email half
  alone took 76.5 s, and the call-fields case did not complete) and the projected run over 300 min. The gate smoke file is committed, not posted.
- The 14B rows of the prediction are void, as in lane 02.
- Neither MoE was cached. Both were fetched before the run at pinned revisions (addendum §1); `lab run` itself is
  `local_files_only`.

**Literature.** PDScope (expert offload over PCIe) and Nemotron 3.5 Lightning (30B-A3B, a 17.8 GB 4-bit pack) were read,
not run (rank C). They stay in the radar, not in `LITERATURE.md`. On this unified-memory box there is no PCIe split to
schedule.

**Reopen threshold.** Re-open lane 03 when any of these holds:
- a MoE with **≥ 3B active** whose mlx pack plus KV fits under the 11.45 GB working set, served by mlx-lm;
- this lab gets a machine with ≥ 24 GB unified memory, where gpt-oss-20b or Gemma-4 26B-A4B can sit the 3 × 83 run;
- mlx or mlx-lm moves off 0.32.2 / 0.31.3, or wilco's pin moves;
- a re-sit flips any of the 83 LFM2.5 cases (all are stable today).
The Wh column needs an operator powermetrics capture before any energy claim.

## Lane 02: dense baselines on three office exams (8–9 Nov 2026)

**Question.** Three dense 4-bit models that fit this 16 GB M1 Pro (mlx-lm 0.31.3, in-process, temperature 0) sit wilco's
three office exams. What does each pass, and what does a held-out case cost? Prediction committed first:
`predictions/dense-baselines.md` (commit `043e8bd`). Case set: every case of email-triage (28), call-fields (32) and recap
(23) at wilco pin `62fcd449`, three fresh loads per model (each load its own process), wilco's own scorer, snapshot rule as
wilco's `aggregate_loads` (majority verdict; a case is stable when all three loads agree).

**Headline.** Held-out passes over the 55 held-out cases: **e4b 36/55 · Llama-3.1-8B 31/55 · e2b 23/55**. Spread: **0 of 83
cases unstable on every model** (all three loads agreed on every case), so majority and the brief's "all three agree" count
are the same number. Kill clause (more than 3 unstable cases on any model): not tripped, so this posts.

| model (mlx-community, 4-bit) | email-triage held-out | call-fields held-out | recap held-out | gen tok/s | peak GB | min / 1,000 cases: email · call · recap | Wh / 1,000 cases |
|---|---|---|---|---|---|---|---|
| gemma-4-e2b-it-4bit | 14/20 | 7/19 | 2/16 | 69.2 | 3.71 | 59.5 · 321.3 · 241.2 | not metered |
| gemma-4-e4b-it-4bit | **16/20** | **10/19** | **10/16** | 41.9 | 5.45 | 138.3 · 374.7 · 491.1 | not metered |
| Meta-Llama-3.1-8B-Instruct-4bit | **16/20** | 6/19 | 9/16 | 35.4 | 6.30 | **21.8 · 213.3 · 106.9** | not metered |

Re-derive (every number in the table and below):
```sh
LAB_WILCO_DIR=<clone at 62fcd449> ~/.pyenv/versions/lewagon/bin/python3 -m lab check \
  results/dense-gemma-4-e2b-it-4bit.json results/dense-gemma-4-e4b-it-4bit.json results/dense-llama-3.1-8b-instruct-4bit.json
```
(`lab check` needs no wilco checkout; the clone is only needed to re-run.) The numbers that are row-level reads, not aggregates
(failed-check counts, failing case ids, length-capped cases, thinking share, tokens per sitting, and the wilco champion
figures), come from `python3 scripts/lane02_rows.py <clone at 62fcd449>`. Rows are covered by `rows_sha256`, so `lab check`
vouches for what the script reads. Minutes per 1,000 cases = mean wall of one sitting
× 1000 / 60, per exam (`<exam>.min_per_1000_cases`). gen tok/s is token-weighted over all calls, not the median the
prediction named. Wh is `null`: no operator powermetrics capture, so the prediction's energy rows are void, not estimated.
All posted loads ran on AC power (`machine.power` in each file).

**What the cost is made of.** The two Gemmas think before answering (their thinking channel is ~91–93% of generated
characters) and Llama 3.1 does not: 804 / 686 / 43 generated tokens per sitting (e2b / e4b / 8B, `gen_tokens_total /
n_cases`, both checked aggregates). So the 8B, the slowest decoder, is the cheapest model on every exam: against e4b, 1.8×
cheaper on call-fields, 4.6× on recap and 6.3× on email-triage. On this box, cost is set by whether the model thinks, not by
its size. The thinking share and tokens per sitting come from `scripts/lane02_rows.py` (below).

**Prediction vs result.**

| claim | predicted | result | |
|---|---|---|---|
| e2b held-out email · call · recap | 16 (15–17) · 11 (9–13) · 11 (10–12) | 14 · 7 · 2 | refuted, all three |
| e4b held-out email · call · recap | 16 (15–17) · 14 (13–15) · 13 (12–14) | 16 · 10 · 10 | email held; call, recap refuted |
| † 8B held-out email · call · recap | 17 · 13 · 9–11 | 16 · 6 · 9 | recap held; email off by 1; call refuted (−7) |
| all-cases email-triage, e2b · e4b | [23–25/28] · [24/28] | 21/28 · 23/28 | refuted (−2, −1) |
| e2b → e4b step: cost ratio, cases bought (email · call · recap) | ~1.5×; +0 · +3 · +2 | ×2.3 · ×1.17 · ×2.0 (minutes); +2 · +3 · +8 | call-fields +3 held exactly; the rest refuted |
| email-triage saturated above 2B (all within 2 of 16/20) | yes | 14 / 16 / 16 | held |
| recap not monotone in size | 14B below e4b | 14B void; 8B (9) below e4b (10) | held for the 8B |
| call-fields the one exam where size buys cases | +1 to +3 from e4b to 14B | 14B void; e4b → 8B is −4 | refuted for the 8B |
| gen tok/s e2b · e4b · 8B | 55 (48–65) · 37 (34–40) · 24 (20–28) | 69.2 · 41.9 · 35.4 | refuted, all faster |
| peak GB e2b · e4b · 8B | 3.6–4.0 · 5.2–5.8 · 5.0–6.0 | 3.71 · 5.45 · 6.30 | held, held, refuted (+0.3) |
| wall s/case e2b (email · call · recap) | 6 · 12 · 13 | 3.6 · 19.3 · 14.5 | refuted (email faster, call slower); recap within 1.5 s |
| wall s/case e4b | 9 · 17 · 18 | 8.3 · 22.5 · 29.5 | email held; call, recap refuted |
| wall s/case 8B | 14 · 25 · 30 | 1.3 · 12.8 · 6.4 | refuted: Llama does not think |
| e4b minutes per 1,000 cases | 150 · 280 · 300 | 138.3 · 374.7 · 491.1 | email held; call, recap refuted |
| e2b gains less than 2× tok/s over e4b (mechanism) | < 2× | 69.2 / 41.9 = 1.65× | held |
| cost-vs-pass: hockey stick, knee at the 4B | knee at e4b; size orders cost | e4b is the top scorer on all three exams, but the 8B is the cheapest, not the dearest | shape refuted; e4b-as-best held |
| Wh per case | 0.045–0.35 | not metered | void |

**What did not hold.**
- **The small Gemma is far below the QAT champion on recap.** e2b scored 2/16 against a predicted 11/16. Of the 14 held-out
  fails on load 0, `check_compression` failed on 8, `check_coverage` on 8 and `check_abstention` on 4. Its recaps run long
  and leave out required items.
- **Thinking runs into wilco's `max_tokens` on one case per Gemma.** e4b recap `long-day-heldout-xl` (held-out) hit the 2048
  cap on all three loads, on both the first call and the re-ask. e2b call-fields `support-long-train` (train) hit the 4096
  cap on all three. Llama: none. This is the prediction's "unsure" item, which turned up on Gemma 4 rather than on Qwen3.
- **mlx 4-bit sits below wilco's Ollama QAT champions, by more than the ±1 the prediction allowed.** Champions (rank B,
  `evals/<exam>/snapshot.json` at the pin, 3 loads): email-triage e2b-QAT 16/20 vs mlx e2b 14/20; call-fields e4b 14/19 vs
  mlx e4b 10/19; recap e4b-QAT 13/16 vs mlx e4b 10/16. Engine, quantisation and chat template all differ, so this measures
  the packaging, not the size.
- **Decode is faster than the bandwidth model said.** At 35.4 tok/s the 8B would read ≈150 GB/s if every token touched the whole 4.2 GB pack. The
  prediction's 85–140 GB/s band was set from bf16 and launch-bound Gemma runs.
- **The sandbox's rank-A anchor reproduced as a score, not as a case set.** e4b email-triage in-process 16/20, same as
  `mlx_lm.server` on 1 Nov, but one case moved each way (in-process fails `intro-no-ask`, the server failed
  `paid-workshop-invite`).

**What was refused or voided.**
- `results/dense-gemma-4-e2b-it-4bit-thinking-in-answer.json`, not posted. That first e2b run passed Gemma's
  `<|channel>thought…` text to wilco as the answer, because the adapter did not split thinking out the way `mlx_lm.server` does.
  Fixed in `lab/metering.py: split_thinking` (27786ce) and re-run. **The leak moved no verdict** (`lab diff` of the two
  files: +0 on every pass count). wilco's JSON extraction had already found the answer object. The only change: one
  length-capped call-fields case now ends in wilco's termination bucket, not a re-ask (252 → 249 calls). It is kept as
  evidence of the wiring fault.
- The Qwen3-14B rows of the prediction are void for this run (the operator picked the cached 8B).

**Reopen threshold.** Re-sit and re-post a model when any of these happens: mlx or mlx-lm moves off 0.32.2 / 0.31.3; the
wilco pin moves (`docs/CASES.md`); or a re-sit flips any one of the 83 cases (all 83 are stable today, so one flip is a
non-reproduction). Separately, the cost columns need an operator powermetrics capture before any Wh figure is posted.
