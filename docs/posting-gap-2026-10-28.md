# The posting gap, stated once (2026-10-28)

An on-device inference role (fintech company's AI model team; posting text not reproduced) listed, as
"must" / "essential": Metal Shading Language and writing compute shaders from scratch; GPU kernels for
smartphones; low-level kernel optimisation with measured latency / throughput / memory wins on constrained
devices; distributed inference (tensor / pipeline / expert parallelism on clusters); diffusion-model and
ViT math; ideally a PhD with A* publications. "Understanding of": pruning, quantisation, flash attention,
KV cache, speculative decoding (Eagle).

## Checked against every repo (grep, 2026-10-28)

| bucket | lines | portfolio |
|---|---|---|
| screen-out, zero files anywhere | MSL, mobile GPU kernels, kernel optimisation, TP/PP/EP, diffusion/ViT, PhD/pubs | nothing (the `metal` compiler was not even installed; MLX needs none) |
| one bullet where the portfolio is strong | "apply empirical research to serving; design robust evaluation frameworks; iterate" | agent-sandbox: registered predictions, rule-A snapshots, quantisation + KV-cache bake-offs with per-case flips, prefix-cache measurement, a 27B paged on 16 GB with the memory recorded, a LoRA that cut output tokens 85% at held score |
| reading depth only | quantisation, KV cache, flash attention, speculative decoding, pruning | measured the first two, toggled the third, papers only on the last two |

**The overlap is the method, not the craft.** The sentence that survives a check today: *"I have measured
quantisation and KV-cache effects on constrained hardware with a reproducible instrument; I have not
written a kernel."*

## What each rung here changes that sentence to

| after | sentence |
|---|---|
| rung 1 (done) | "…I have written one Metal kernel; it beats the compiled stock path by 1.4–1.9× at prefill sizes and does nothing at decode size, and I can say why." |
| rung 3 on project 1 | "…I wrote the dequantising matmul that let a 27B run in 16 GB without paging; peak memory x → y GB, tok/s a → b, verdicts unchanged on n cases." |
| rung 4 | "…and a flash-attention prefill kernel; prompt tok/s x → y." |

None of these reaches "proven mobile kernel experience". Do not claim it does.

## The two cheapest things that are not kernels (agent-sandbox, one evening each, $0)

Speculative decoding (`mlx_lm.server --draft-model`, e2b drafting for e4b, rule A, prediction "0 flips")
and MLX KV-bits (`--kv-bits 8/4`, same exams as M7). They put a measurement of our own behind the two
"understanding of" items we only have paper rows for. Filed in agent-sandbox
`docs/inbox/2026-10-28-serving-and-inference-what-we-have-what-the-posting-wants-readme-gap.md` §4.
