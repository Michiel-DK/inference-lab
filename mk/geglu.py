"""Rung 1: elementwise fusion. GEGLU = gelu_approx(gate) * up, one Metal pass.

The stock mlx-lm Gemma 4 text model computes this as `nn.gelu_approx(gate) * x` wrapped in
`mx.compile(shapeless=True)` (mlx_lm/models/gemma4_text.py). mx.compile already fuses
elementwise chains, so the honest baseline is the COMPILED stock function, not two separate ops.
bench/RESULTS.md carries all three numbers.
"""
import mlx.core as mx

_SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i >= gate_shape[0]) return;
    T g = gate[i];
    T inner = T(0.7978845608028654) * (g + T(0.044715) * g * g * g);
    T gelu = g * T(0.5) * (T(1.0) + metal::tanh(inner));
    out[i] = gelu * up[i];
"""

_kernel = mx.fast.metal_kernel(
    name="geglu_fused",
    input_names=["gate", "up"],
    output_names=["out"],
    source=_SOURCE,
)

THREADGROUP = 256


def geglu_fused(gate: mx.array, up: mx.array) -> mx.array:
    """gelu_approx(gate) * up in one kernel. Any shape; runs on the flattened buffer."""
    if gate.shape != up.shape:
        raise ValueError(f"shape mismatch {gate.shape} vs {up.shape}")
    shape = gate.shape
    g, u = gate.reshape(-1), up.reshape(-1)
    (out,) = _kernel(
        inputs=[g, u],
        template=[("T", g.dtype)],
        grid=(g.size, 1, 1),
        threadgroup=(THREADGROUP, 1, 1),
        output_shapes=[g.shape],
        output_dtypes=[g.dtype],
    )
    return out.reshape(shape)


def patch_gemma4():
    """Swap the fused kernel into mlx-lm's Gemma 4 text model (module-level `geglu`).

    Returns a callable that restores the stock function. The MLP calls the module global
    `geglu(...)` at runtime, so rebinding the name is enough; models loaded before or after
    the patch both pick it up.
    """
    import mlx_lm.models.gemma4_text as t

    stock = t.geglu
    t.geglu = geglu_fused

    def restore():
        t.geglu = stock

    return restore
