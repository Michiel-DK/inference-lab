"""Correctness: fused kernel vs the stock compiled geglu, several shapes and dtypes."""
import mlx.core as mx
import mlx.nn as nn
from mk.geglu import geglu_fused


def stock(gate, up):
    return nn.gelu_approx(gate) * up


def check(shape, dtype, tol):
    gate = mx.random.normal(shape).astype(dtype)
    up = mx.random.normal(shape).astype(dtype)
    a, b = geglu_fused(gate, up), stock(gate, up)
    mx.eval(a, b)
    diff = float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())
    assert a.shape == shape and a.dtype == dtype, (a.shape, a.dtype)
    assert diff <= tol, f"{shape} {dtype}: max abs diff {diff} > {tol}"
    return diff


def main():
    cases = [
        ((1024,), mx.float32, 1e-5),
        ((1024,), mx.float16, 2e-2),
        ((1, 7, 2048), mx.float16, 2e-2),      # a 7-token prefill at Gemma e2b's intermediate width
        ((3, 1, 8192), mx.bfloat16, 1e-1),     # decode step, batch 3
        ((1, 1, 3), mx.float32, 1e-5),         # smaller than one threadgroup
        ((1, 1, 1000), mx.float32, 1e-5),      # not a multiple of 256
    ]
    for shape, dtype, tol in cases:
        d = check(shape, dtype, tol)
        print(f"[PASS] {str(shape):18} {str(dtype):16} max abs diff {d:.2e}")
    print("ok")


if __name__ == "__main__":
    main()
