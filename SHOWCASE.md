# Showcase

The post: one headline number with its spread, what was refused, the reopen threshold, and how to re-derive it
(`python3 -m lab check <results file>`).

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
