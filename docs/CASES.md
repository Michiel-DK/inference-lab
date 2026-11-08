# Cases

The case set is three exams from **wilco** (`~/code/Michiel-DK/agent-exam-suite`, GitHub `Michiel-DK/wilco`). They are read
in place, never copied or edited, and pinned to one commit and to the sha256 of each `cases.json`. `lab run` refuses if
wilco's HEAD is not the pin, if an exam's `evals/` or `agents/` directory has uncommitted edits, or if a `cases.json` hash
moved (`lab/cases.py: verify_pin`; `tests/lab/test_lab.py` asserts it on a machine with the checkout).

**Pin:** `62fcd449d5cb3ef7359a0a0262f0f403c929062d` (wilco `main`, 6 Nov 2026, "docs(inbox): where this repo sits under
automations-first").

| exam | mode (wilco) | train | held-out | total | cases.json sha256 | cases.json last changed |
|---|---|---|---|---|---|---|
| email-triage | labels: exact `label` | 8 | 20 | 28 | `955dd300…9314bfd` | 589d39f, 2026-10-27 |
| call-fields | fields: 8 fields, case passes only if all match | 13 | 19 | 32 | `16ea5b02…4951af` | 9fd0f37, 2026-10-06 |
| recap | properties: deterministic checks in `evals/recap/properties.py` | 7 | 16 | 23 | `315a6132…aaf120` | 589d39f, 2026-10-27 |

Sizes are wilco's own split, counted from each case's `split` field at the pin (the denominators of wilco's README held-out
scores agree: 16/20, 14/19, 13/16). Full hashes are in `lab/cases.py`. The split is wilco's, drawn there
before this lab existed. **Held-out is never tuned on here.** Every lab tuning or smoke run uses train cases only.

**Pin vs HEAD (8 Nov 2026, lane 02).** wilco's `main` moved to `6b2c7bd` ("refresh 2026-11-08", adds a `wilco grade`
subcommand). `git diff --quiet 62fcd449 6b2c7bd -- sandbox/runner.py sandbox/router.py evals agents` exits 0: the grader and
all case and agent files are byte-identical; only tests and new files changed. The pin did not move (moving it is its own
commit). Lane 02 ran against a read-only clone at the pin, `LAB_WILCO_DIR=.claude/worktrees/wilco-pin-62fcd449`, which is the
`wilco.dir` in its results files; `verify_pin()` returned `[]` on it. Without `LAB_WILCO_DIR` the lab gate's
`test_wilco_pin_holds_and_sizes_match` now fails on the moved HEAD, by design.

## Grading

Deterministic, and wilco's own code. wilco at the pin has **no `grade` subcommand** (its CLI lists new, validate, list, run,
diff, pick, adopt, check, live). The lab calls `sandbox/runner.py: run_exam`, the function `wilco run` scores with, once per
case, in-process and read-only (no bytecode written). Only `adapter_for` is swapped, for the metered mlx-lm adapter. So
prompt assembly (`agents/<exam>/prompt.md`), temperature (0) and max_tokens (from `agent.yaml`), JSON extraction, the one
re-ask, and the labels / fields / properties scorers are wilco's. None of the three exams has a `rubric.md`, so no LLM judge
is loaded.

What wilco itself declares unverified for each exam (`not_verified` in each `cases.json`) applies unchanged. One example:
call-fields grades a defensible alternative reading as zero.

## Smoke set (lane 01)

`email-triage` train cases `client-quote-request`, `overdue-invoice` (`lab/cases.py: SMOKE`). It is the exam with the shortest
inputs (≈210 chars), so the smoke checks wiring, not capability.
