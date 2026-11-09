# hosted-baseline

Lane 09 (written 9 Nov 2026, before any lane-09 call). Three hosted Claude models sit the same 55 held-out cases as lanes
02–04 (email-triage 20, call-fields 19, recap 16), with the same grader. The question is the gap between the laptop rows
and a hosted model, and what a case costs on each. Run names: `hosted-claude-haiku-5-5` (3 sits),
`hosted-claude-sonnet-5-5` (1 sit), `hosted-claude-opus-5-5` (1 sit), plus a train-only gate smoke
`hosted-gate-claude-haiku-5-5-smoke`. This file is the lane's one prediction and covers all four.

Rank letters as in `LITERATURE.md`: **A** measured on our cases on this machine; **B** adjacent measurement; **C** read, not run.

## Prediction

**Anchor (rank B, read before writing this).** wilco's own probe `docs/probes/frontier-direct-2026-10-28/` sat these three
models on these three exams through the same `anthropic` provider, one pass each, 28 Oct. Held-out email · call · recap:
- Haiku 19 · 15 · 8 = 42
- Sonnet 20 · 18 · 3 = 41
- Opus 20 · 14 · 6 = 40

The case files have not changed since (email-triage and recap last changed 27 Oct, call-fields 6 Oct; pin `62fcd449` is
later). This lane is therefore close to a re-sit, under the lab's harness and meter, with Haiku sat three times. The bands
below are the anchor plus the 0–4-verdict rerun noise the probe itself names, not an independent guess.

**Pass count, held-out, out of 55** (Haiku: majority over its 3 sits; Sonnet and Opus: their one sit):

| model | email-triage /20 | call-fields /19 | recap /16 | total /55 | point |
|---|---|---|---|---|---|
| claude-haiku-5-5 | 18–20 | 13–16 | 6–10 | 39–45 | 42 |
| claude-sonnet-5-5 | 19–20 | 16–18 | 2–6 | 38–43 | 41 |
| claude-opus-5-5 | 19–20 | 13–16 | 4–8 | 37–43 | 40 |

- **Below the 8-bit e4b's 41/55?** Yes: at least one hosted model is strictly below 41, most likely Opus. No hosted model
  beats 41 by more than 3.
- **Headline I expect:** on these three exams, the laptop's best local row is level with hosted Claude. The hosted models
  win email-triage (+3 to +4 over the e4b 8-bit's 16/20) and lose recap (the e4b 8-bit's 12/16 against 3–10). The probe
  says why: Claude writes past recap's length rule, and never states a wrong fact. call-fields splits them: Sonnet is
  ahead (+3 to +5 over 13/19), Haiku and Opus are level within ±2.
- **Haiku sit-to-sit spread** (max − min of the three per-sit held-out totals): 1–4, point 2. Distinct held-out case ids
  whose verdict differs between any two sits: 2–7, most of them on recap (length-rule cases near the boundary). Haiku's
  majority total is within ±1 of its median sit. The local rows had 0 of 83 unstable at temperature 0; hosted Haiku will
  not, because this endpoint cannot be run at temperature 0 (see Mechanism).
- **Cost per case** (metered tokens × list price, input + output, thinking billed as output; mean over the 55 held-out
  cases and over every sit):
  - Haiku $0.0003–0.0006 (point $0.0004)
  - Sonnet $0.004–0.009 (point $0.006)
  - Opus $0.009–0.018 (point $0.013)

  Per exam, call-fields is the most expensive (8.5k-token transcripts) and email-triage the cheapest, by about 10× on
  every model. Opus over Sonnet per case is 1.8–2.6× (price ratio 2×, plus Opus thinking more on recap).
- **Wall per case:** unsure. 3–15 s mean, network and thinking included; Opus is the slowest.
- **Local arms' cost per case:** void, as in lane 04. No operator `powermetrics` capture exists for those runs, so Wh is
  `null` and the local rows carry wall time only. The comparison is $ per case (hosted) beside wall per case (local), not
  $ beside $.

**Protocol (fixed now).**
- **Case set.** The 55 held-out ids of the three exams, read from each `cases.json` at wilco pin `62fcd449` (the read-only
  clone `LAB_WILCO_DIR=.claude/worktrees/wilco-pin-62fcd449`, as lanes 02–04). No train case sits in a posted arm.
- **Grader.** wilco's `sandbox/runner.py: run_exam`, once per case. The adapter is wilco's own
  `router.get_adapter("anthropic")`: the OpenAI-compatible `/v1/chat/completions` endpoint, with wilco's
  `drop_body_keys: temperature`. A lab wrapper only records each HTTP response's `usage`, `finish_reason` and wall; no
  request byte changes. `max_tokens` is from each `agent.yaml` (2048 / 4096 / 2048).
- **Effort.** This endpoint sends no effort, so each model runs at its API default: Haiku and Opus `medium`, Sonnet `high`
  (rank C, the claude-api reference). Not tuned.
- **Prices** (stored in each results file; `lab check` re-derives every $ figure from tokens × these): Haiku 5.5 $0.10 in /
  $0.50 out per MTok; Sonnet 5.5 $2 / $10; Opus 5.5 $4 / $20. Source: the claude-api model table, cached 2026-10-06, rank C;
  the same rates wilco's 28 Oct probe used. No caching and no batch discount: list price, standard tier.
- **No fallback.** A refusal or an empty answer is that model's verdict (wilco's `TerminationError` → fail). No
  server-side fallback model, which would put another model's answer in this row.
- **Order and ceiling.**
  1. Gate smoke: Haiku on the two `SMOKE` email-triage train cases plus call-fields `support-long-train`. It checks that
     usage comes back, whether `completion_tokens` includes thinking, and whether any call ends `length` with no answer.
     It is not posted.
  2. Haiku ×3.
  3. Sonnet ×1.
  4. Opus ×1.

  A per-call ledger counts every billed response in the lane, the smoke included. Before each case, the runner refuses if
  spent so far + that case's worst-case cost (prompt + 2 × `max_tokens` at the model's rates) would pass **$2.00**.
- **Opus gate.** Opus sits only if spent so far + (Sonnet's measured sit cost × 2 × 1.25) ≤ $2.00. Otherwise it is
  "not sat (ceiling)", and the post says so. An arm stopped mid-sit by the ceiling is "not sat (ceiling)". No partial file
  is posted.
- **Data tier.** All three `agent.yaml` say `tier: local`. The cases are wilco's synthetic, public exam cases; the
  operator's go for this lane is what lifts the tier, for this comparison only (as wilco's 28 Oct probe stamped
  `tier_breach: true`).

**Kill clause.**
- No row is posted for an arm whose sit has any call with unmetered usage (no `usage` in a billed 200 response), because
  then its cost is a guess.
- If the smoke shows `completion_tokens` excluding thinking, the $ figures are void, not under-counted. The pass counts
  still post.
- If more than 5 of 55 held-out cases in a sit end in `TerminationError` with `finish_reason=length`, that model's row is
  posted as "a verdict on wilco's `max_tokens`, not on the model".

## Mechanism

- **Why near-level.** These three exams reward different things.
  - email-triage is a three-way label with a few judgment calls. Frontier models saturate it (probe: 19–20/20); the local
    e4b has 4 misses that are judgment, not reading.
  - call-fields is 8 exact fields over transcripts of up to 8.5k tokens. Every field must match, and a defensible
    alternative reading scores zero (wilco's `not_verified`). Capability helps, but so does terseness, and a model that
    "improves" a field value loses the case. The probe has Sonnet at 18, with Haiku and Opus at 15 and 14.
  - recap is property-checked against length rules (`max_ratio`, `abstain_max_chars`). A model that writes a fuller,
    correct recap fails it. The 5.5 family does exactly that (probe: 3–8/16 against the e4b's 13/16 at its Ollama
    champion and 12/16 at mlx 8-bit).

  So the hosted models' email and call gains are paid back on recap. The sum lands near 41.
- **Why Haiku moves between sits.** The endpoint rejects `temperature`. The 5.5 family thinks on every call and samples
  as it likes, so the same prompt can produce a different answer: wilco's router labels this path PROBE-GRADE. Three sits
  measure sampling spread, which is a different thing from the local rows' load-to-load determinism at temperature 0. One
  sit each for Sonnet and Opus means their stability is unmeasured, not 0 unstable. Recap carries most of the flips
  because its failures are length overruns near a threshold.
- **Why the costs.** The input per sit is about 80k tokens on the Llama tokenizer: lane 02's held-out prompt sums are
  5.0k email + 56.3k call + 19.3k recap. Claude's tokenizer runs 1–1.35× that. Output is mostly hidden thinking, billed
  as output, with the answer itself only about 50–200 tokens. The probe's per-10k rates on these three exams
  (Haiku $0.54 / $6.72 / $3.85; Sonnet $10.99 / $119.35 / $56.59; Opus $28.39 / $230.80 / $141.34), weighted 20 / 19 / 16,
  give the point estimates above.
