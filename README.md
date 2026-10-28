# metal-kernels

Hand-written Metal compute kernels, compiled at runtime through MLX (`mx.fast.metal_kernel`), patched into
mlx-lm models, and measured. Apple silicon only. Private while it is an exploration.

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

## Layout

```
mk/          one module per kernel: the MSL source, a Python wrapper, a patch_<model>() that swaps it in
tests/       correctness vs the stock op, several shapes/dtypes, run with PYTHONPATH=. python3 tests/<t>.py
bench/       microbenchmarks (naive · mx.compile'd stock · ours) and RESULTS.md with the numbers taken here
scripts/     end-to-end smoke runs on a cached mlx-lm model (one run, never evidence)
docs/        the ladder, the six projects, the posting gap
```

## The ladder (from `docs/projects.md`)

| rung | what | status |
|---|---|---|
| 1 | elementwise fusion (GEGLU in Gemma 4's MLP) | **done** — `mk/geglu.py`, `bench/RESULTS.md` |
| 2 | reductions with threadgroup shared memory (RMSNorm, MaxSim) | next |
| 3 | tiled matmul with dequantisation inside (what every 4-bit engine is made of) | the real job |
| 4 | attention as one kernel (flash-attention prefill) | after 3 |

MLX ships fused `rms_norm`, `rope`, `scaled_dot_product_attention` and quantised matmuls already. A first
kernel at rungs 2–4 will lose to them; the table must show by how much, then close the gap.

## Run

```
PYTHONPATH=. python3 tests/test_geglu.py        # correctness, six shapes
PYTHONPATH=. python3 bench/bench_geglu.py       # three arms, µs and GB/s
PYTHONPATH=. python3 scripts/smoke_gemma.py 48  # patched vs stock Gemma 4 e2b 4-bit, text compared
```

Pins in `requirements.txt` are the versions the first numbers were taken on (M1 Pro, 16 GB, macOS 14.3.1).
No Xcode needed: Command Line Tools only; MLX compiles the MSL body itself.

## Rules

- A kernel is not done until it has a test against the stock op, a benchmark row, and a patch function.
- Benchmark against what mlx-lm *actually runs* (usually `mx.compile`d), never only against separate ops.
- Wall times here are microbenchmarks on a shared laptop. An exam-level claim (tok/s, verdict flips, peak
  memory on a champion) is taken in agent-sandbox under its rules, with this repo's commit hash stamped.
