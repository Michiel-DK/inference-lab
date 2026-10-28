"""metal-kernels: hand-written Metal compute kernels, run through MLX, patched into mlx-lm models.

Every kernel ships with (1) a correctness test against the stock op, (2) a microbenchmark,
(3) a patch function that swaps it into a loaded mlx-lm model so agent-sandbox can serve and
measure it like any other runtime. See README.md.
"""
