"""Three arms: stock ops uncompiled · stock ops under mx.compile (what mlx-lm actually runs) · fused kernel."""
import sys, time
from functools import partial
import mlx.core as mx
import mlx.nn as nn
from mk.geglu import geglu_fused


def naive(gate, up):
    return nn.gelu_approx(gate) * up


compiled = partial(mx.compile, shapeless=True)(naive)


def bench(f, gate, up, k=100):
    for _ in range(5):
        mx.eval(f(gate, up))
    t = time.perf_counter()
    for _ in range(k):
        mx.eval(f(gate, up))
    return (time.perf_counter() - t) / k * 1e6  # µs


def main():
    print(f"mlx {mx.__version__} · device {mx.default_device()}")
    print(f"{'shape':>18} {'dtype':>9} {'naive µs':>10} {'compiled µs':>12} {'fused µs':>10} {'fused/compiled':>15} {'GB/s fused':>11}")
    for shape, dtype in [((1, 1, 8192), mx.float16), ((1, 128, 8192), mx.float16), ((1, 2048, 8192), mx.float16),
                         ((4 << 20,), mx.float16), ((4 << 20,), mx.float32)]:
        gate = mx.random.normal(shape).astype(dtype)
        up = mx.random.normal(shape).astype(dtype)
        n, c, f = bench(naive, gate, up), bench(compiled, gate, up), bench(geglu_fused, gate, up)
        nbytes = 3 * gate.size * gate.dtype.size  # read gate, read up, write out
        print(f"{str(shape):>18} {str(dtype).split('.')[-1]:>9} {n:10.1f} {c:12.1f} {f:10.1f} {f / c:15.2f} {nbytes / f / 1e3:11.1f}")


if __name__ == "__main__":
    main()
