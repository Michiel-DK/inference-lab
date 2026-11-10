# inference-lab

Hand-written Metal compute kernels, and a lab that measures inference on one laptop. Apple silicon only (M1 Pro, 16 GB).

## The lab

**The question.** What does a small local model pass, and what does a held-out case cost, on a laptop? Three 4-bit models run
under mlx-lm (in-process, temperature 0, cached models only, no hosted model calls) and sit wilco's three office exams:
email-triage, call-fields and recap, with wilco's own scorer. Each case is metered for tokens/s, peak memory and wall time;
Wh comes only from an operator-started `sudo powermetrics` capture (`lab/energy.py`) and is `null` without one.

**The lab rule.** One question per lane. The prediction is committed before the run, the case set is fixed (wilco pinned in
`docs/CASES.md`), grading is deterministic, negatives stay in, `results/*.json` are committed evidence that is never
regenerated, and a number is posted only with the command that re-derives it.

**Headline.** Held-out passes over 55 cases: e4b 36 · Llama-3.1-8B 31 · e2b 23, with 0 of 83 cases unstable on any model.
Re-derive it (no model, no GPU, no wilco checkout needed):

```sh
~/.pyenv/versions/lewagon/bin/python3 -m lab check results/dense-gemma-4-e2b-it-4bit.json \
  results/dense-gemma-4-e4b-it-4bit.json results/dense-llama-3.1-8b-instruct-4bit.json
```

`lab check` re-derives every posted aggregate from the committed rows and exits 1 on drift. `SHOWCASE.md` is the post,
`LESSONS.md` has prediction-to-result rows.

```sh
P=~/.pyenv/versions/lewagon/bin/python3          # mlx 0.32.2, mlx-lm 0.31.3 (requirements.txt pins)
$P -m pytest -q                                  # lab gate, no GPU, no model (needs wilco at the pin, else one test fails)
$P -m lab diff <a> <b>                           # difference of two results files
$P -m lab run <model> --name <run> --smoke       # refuses without a committed predictions/<run>.md
```

| path | what |
|---|---|
| `lab/` | `cases.py` (pin + hashes), `run.py` (wilco `run_exam` per case, mlx-lm in-process), `metering.py`, `context.py` (lane 05: padding + KV settings, `lab context`), `energy.py`, `results.py` (derive + check) |
| `predictions/` | one file per run, committed before it (format in `predictions/README.md`) |
| `results/` | committed evidence, never regenerated |
| `LESSONS.md` · `SHOWCASE.md` · `LITERATURE.md` | prediction → result rows · the post · only what a lane ran against |

---

# Metal kernels

**Why this repo exists.** An on-device inference posting (2026-10-28) asked for Metal Shading Language, mobile
GPU kernels and measured latency / memory wins. The portfolio's serving story lives in `agent-sandbox`
(quantisation, KV-cache type, prefix cache, a 27B paged on 16 GB, an MLX-served adapter), all of it
*measurement* through an exam runner; none of it is kernel code. This repo is where the kernel craft is
learned, one rung at a time, each rung ending in a number. `docs/posting-gap-2026-10-28.md` has the
honest gap and the sentence each rung earns.

**Why it is not inside agent-sandbox.** That repo owns an instrument (registered predictions, three-load
snapshots, a Linux CI gate where MLX does not exist, a curated public cut). A kernel is engine code,
Mac-only, iterated dozens of times a day. agent-sandbox treats engines as runtimes it stamps and measures
(Ollama, `mlx_lm.server`); this repo is one more such runtime. `docs/projects.md` §"How a kernel reaches
wilco" has the three-step hand-off.

### Layout

```
mk/          one module per kernel: the MSL source, a Python wrapper, a patch_<model>() that swaps it in
tests/       correctness vs the stock op, several shapes/dtypes, run with PYTHONPATH=. python3 tests/<t>.py
bench/       microbenchmarks (naive · mx.compile'd stock · ours) and RESULTS.md with the numbers taken here
scripts/     end-to-end smoke runs on a cached mlx-lm model (one run, never evidence)
docs/        the ladder, the six projects, the posting gap
```

### The ladder (from `docs/projects.md`)

| rung | what | status |
|---|---|---|
| 1 | elementwise fusion (GEGLU in Gemma 4's MLP) | **done** — `mk/geglu.py`, `bench/RESULTS.md` |
| 2 | reductions with threadgroup shared memory (RMSNorm, MaxSim) | next |
| 3 | tiled matmul with dequantisation inside (what every 4-bit engine is made of) | the real job |
| 4 | attention as one kernel (flash-attention prefill) | after 3 |

MLX ships fused `rms_norm`, `rope`, `scaled_dot_product_attention` and quantised matmuls already. A first
kernel at rungs 2–4 will lose to them; the table must show by how much, then close the gap.

### Run

```
PYTHONPATH=. python3 tests/test_geglu.py        # correctness, six shapes
PYTHONPATH=. python3 bench/bench_geglu.py       # three arms, µs and GB/s
PYTHONPATH=. python3 scripts/smoke_gemma.py 48  # patched vs stock Gemma 4 e2b 4-bit, text compared
```

Pins in `requirements.txt` are the versions the first numbers were taken on (M1 Pro, 16 GB, macOS 14.3.1).
No Xcode needed: Command Line Tools only; MLX compiles the MSL body itself.

### Rules

- A kernel is not done until it has a test against the stock op, a benchmark row, and a patch function.
- Benchmark against what mlx-lm *actually runs* (usually `mx.compile`d), never only against separate ops.
- Wall times here are microbenchmarks on a shared laptop. An exam-level claim (tok/s, verdict flips, peak
  memory on a champion) is taken in agent-sandbox under its rules, with this repo's commit hash stamped.
