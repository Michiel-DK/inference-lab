# Literature

Only what this lab ran against. Each row gives the paper or repo, the claim tested, the run that tested it, what held, and
a rank letter for how directly the run tested the claim (A: reproduced on our cases; B: partial or adjacent; C: read, not
run, and so not admitted here). The radar's serving-track rows (`repo-radar/mimir/VERDICTS.md`, private) are where to look
first. It holds what exists and its verdicts. This file holds what was reproduced here.

| source | claim tested | run | held? | rank |
|---|---|---|---|---|
| wilco snapshots at `62fcd449` (`evals/<exam>/snapshot.json`; Ollama tags `gemma4:e2b-it-qat`, `gemma4-e4b-ctx16k` (= `gemma4:e4b-it-qat` at a 16k window, per `agents/call-fields/agent.yaml`), `gemma4:e4b-it-qat`; 3 loads, 1 Nov; `scripts/lane02_rows.py <clone>` prints them) | the mlx-community affine 4-bit packs land within ±1 held-out case of the QAT champions (email-triage e2b 16/20, call-fields e4b 14/19, recap e4b 13/16) | lane 02, `results/dense-gemma-4-e{2,4}b-it-4bit.json` | **no**: 14/20, 10/19, 10/16 (−2, −4, −3) | B (same cases, other engine and packaging) |
| agent-sandbox `docs/probes/mlx-s1-2026-11-01` (arm A, `mlx_lm.server`, gemma-4-e4b-it-4bit, 3 restarts) | e4b email-triage 16/20 held-out, 0 unstable, 37.6 tok/s | lane 02, `results/dense-gemma-4-e4b-it-4bit.json` | score yes (16/20, 0 unstable); case set no (one case moved each way); speed 41.9 tok/s in-process vs 37.6 decode-inclusive through the server | A (same cases, same machine, same pack; in-process vs server) |

Lane 01 ran no method from the literature. The radar's serving-track rows (`VERDICTS.md`, 7 Nov) hold no dense-baseline method, so lane 02's rows are its two baselines. The energy method is the sandbox's own instrument, cited in
`lab/energy.py` (agent-sandbox@2806cbd0 `scripts/energy_report.py`, PR #158), not a literature claim.
