# Projects, and where each result lands (written 2026-10-28, before any of them started)

The kernels all live here. What integrates into existing repos is the *measurement* of each kernel,
because none of the existing repos owns an inference engine: agent-sandbox, rlvr-codegen and
restaurant-brain all talk to a model over an HTTP endpoint.

| # | project | kernel (rung) | the number at the end | where the measurement lands | exists there already? |
|---|---|---|---|---|---|
| 1 | **the 27B fits in 16 GB without paging** | dequantise-inside-the-matmul for a 2-bit / ternary pack (rung 3) | peak GB and decode tok/s before/after on an exam already scored | agent-sandbox probe: re-sit J11 (`docs/probes/j11-bonsai2-27b-2026-09-30`: 17.2 GB peak, 11 tok/s prompt, 5.7 tok/s decode, pages) | yes — J11 results + the `mlx` provider |
| 2 | **deterministic kernels** | matmul + reductions with a fixed summation order regardless of batch (rung 2–3) | a flip table that reads 0 under concurrent load | agent-sandbox `docs/probes/determinism-batch-invariance-2026-09-28/DESIGN.md` (registered, not fired) | yes — design registered |
| 3 | **faster long-prompt prefill** | chunked flash-attention prefill (rung 4) | prompt tok/s; the long exams J11 could not run, scored on the laptop | agent-sandbox: the long exams (transcript-en, crm-followup, call-fields, recap, reply-draft) | yes — the exams |
| 4 | **late-interaction retrieval (MaxSim)** | fused max-over-token-pair dot products, never materialising the score tensor (rung 2) | queries/s and memory at ~32k companies | mast (IDEAS S84-B3 ColBERT-style); mast is dormant and embeds through a hosted API → demo only | note only |
| 5 | **image generation on the laptop** | fused GroupNorm+SiLU, then UNet/DiT attention (rung 2 → 4) | images/s and peak memory | nowhere yet — new surface; the one project touching diffusion / ViT | no |
| 6 | **speech front end** | mel spectrogram + encoder blocks for a Whisper-class model (rung 2) | wall s per audio minute | agent-sandbox sprint K, arm K2 ("Whisper-class via MLX or whisper.cpp"); enters only if K2 trips its kill ("large cannot finish an evening on 10 calls") — the laptop alternative to renting a GPU (K5) | planned, not run |

**Order:** 1 first (measured problem, visible payoff, probe = a rerun). 5 second for breadth. Speech stays on
sprint K's schedule and does not wait for this repo.

## How a kernel reaches wilco / agent-sandbox

1. Here: the kernel holds its correctness test and beats the `mx.compile`d stock path on the microbenchmark.
2. agent-sandbox gets a probe folder: a small server wrapper (the J11 `proxy.py` pattern) that loads the
   model with `patch_<model>()` applied and serves it on the port `PROVIDERS["mlx"]` already expects; the
   champion is re-sat under rule A (3 loads, per-case flips); this repo's commit hash goes into the runtime
   stamp beside the adapter hash. **That stamp field is the only agent-sandbox code change any of this needs.**
3. The wilco README lists this repo as a runtime it measured, the way it lists Ollama.

## Prerequisite on the agent-sandbox side (not this repo's work)

Every RESULTS file there calls wall seconds and `ollama ps` samples "not evidence". A serving-first claim
needs a metered per-call latency and a peak-resident-memory sample per load in the snapshot's runtime block
(the Q7 "rented box" line in repo-radar `BUILD_TRACK.md`). Until then an exam-level kernel result is
"verdict flips plus an uninstrumented wall time".

## What a custom kernel is, in one paragraph (for the next reader)

Every tensor op in a forward pass runs as a kernel: a small program on the GPU, thousands of threads, each
handling one element or one tile. Frameworks ship kernels for the standard ops. "Custom" means writing that
program yourself, for an op the framework lacks or for a chain it runs as several passes over memory. The
math inside is usually trivial; the craft is memory traffic — how many times activations cross unified
memory, how threads are grouped, how loads are vectorised. At decode time on a laptop the model is
bandwidth-bound, so fewer passes is the lever. Rung 1 here is the smallest instance: `gelu(gate) * up`
in one pass instead of two.
