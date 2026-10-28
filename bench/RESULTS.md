# Rung 1 — fused GEGLU (2026-10-28, M1 Pro 16 GB, macOS 14.3.1, mlx 0.32.2, mlx-lm 0.31.3)

> **EVIDENCE CELL:** `bench/bench_geglu.py`, 100 timed calls after 5 warm-ups, three arms on the same arrays:
> the two stock ops uncompiled · the stock ops under `mx.compile(shapeless=True)` (what `mlx_lm/models/gemma4_text.py`
> actually runs) · `mk/geglu.py`. **At prefill sizes the fused kernel takes 0.54–0.70× the compiled stock time;
> at a single decode step (1×1×8192) all three arms are ~190 µs and the kernel changes nothing.**
>
> **Not evidence:** the smoke run's wall seconds (one run, shared laptop); anything about end-to-end tok/s.

## Microbenchmark

| shape | dtype | naive µs | compiled µs | fused µs | fused / compiled | GB/s (fused, 3 passes) |
|---|---|---|---|---|---|---|
| (1, 1, 8192) | f16 | 239.1 | 191.3 | 194.1 | 1.01 | 0.3 |
| (1, 128, 8192) | f16 | 457.9 | 418.2 | 289.0 | 0.69 | 21.8 |
| (1, 2048, 8192) | f16 | 2068.5 | 1425.1 | 767.2 | 0.54 | 131.2 |
| (4,194,304,) | f16 | 580.4 | 493.9 | 344.3 | 0.70 | 73.1 |
| (4,194,304,) | f32 | 1282.4 | 957.9 | 585.1 | 0.61 | 86.0 |

Correctness (`tests/test_geglu.py`): six shapes incl. non-multiples of the 256-thread group, f32 / f16 / bf16; max abs diff
1.5e-7 (f32), 2e-3 (f16), 1.6e-2 (bf16) against `nn.gelu_approx(gate) * up`.

## Reading

1. **The honest baseline mattered.** Against the two separate ops the kernel looked like 2×. Against `mx.compile`, which
   already fuses elementwise chains, it is 1.4–1.9× at prefill sizes. The remaining gap is real: the compiled path still
   sits well below the ~130 GB/s the fused kernel reaches at the largest shape (M1 Pro unified memory is nominally 200 GB/s).
2. **Decode size is launch-bound.** 8,192 elements is far too small for a kernel to matter; ~190 µs is dispatch overhead on
   every arm. For the decode tok/s of a 2B–4B model this kernel is irrelevant. The lever at decode is the weight read in the
   matmuls (rung 3), not the activation function.
3. **Patched into the model it is a no-op on text and not faster** (`scripts/smoke_gemma.py`, Gemma 4 e2b 4-bit, 48 tokens,
   greedy): output byte-identical with and without the patch; wall 1.14 s stock vs 1.34 s patched, one run each — within
   noise and in the wrong direction, consistent with (2) and with the patch replacing a compiled function by an uncompiled
   kernel call inside an otherwise compiled graph. Not evidence; recorded so nobody claims a speedup from rung 1.

## What rung 1 bought

The sentence: "I have written one Metal kernel; it beats the compiled stock path by 1.4–1.9× at prefill sizes and does
nothing at decode size, and I can say why." Next rung: a reduction (RMSNorm, with MLX's own fused `mx.fast.rms_norm` as
the bar to clear), then the dequantising matmul for project 1 (`docs/projects.md`).
