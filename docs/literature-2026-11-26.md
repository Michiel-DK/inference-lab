# Literature, full-text read: serving on Apple silicon, behind lane 22 (26 Nov 2026)

Companion to [`literature-2026-11-16.md`](literature-2026-11-16.md) and [`literature-2026-11-24.md`](literature-2026-11-24.md). It covers
the one queued lane they do not: lane 22 (`lab-inf-22-vllm-metal`, vLLM on this laptop through vllm-metal). No model was loaded, no GPU
used, nothing installed or run. Reader question C2; this read changes no card on its own. Decider: the operator, whether lane 22 runs as
briefed, with changed arms, or not at all. Written for the operator and for whoever opens this public lab's literature folder.

**Rank rule** (as 16 Nov). [F] = body read: setup, n, metric as written, the table, figure or file line the number comes from. A
published number sits beside ours only at [F]. Pages and PDFs were fetched into `/tmp/lab-inf-20-lit-serving/` and are not committed.
Every quote was matched in the fetched text this session.

## 0. First lines

- **Does an [F] row already report lane 22's number on a comparable setup? Partly, and it points the other way from the stub.** The
  vLLM blog (22 Sep 2026) reports vllm-metal on an **M1 Pro 32 GB** with Gemma 4 E4B 4-bit at concurrency 1 and 8: output throughput
  **18.9 → 23.8 tok/s (x1.26)**, mean TTFT **2.50 → 12.35 s** (§2 row 4, read from the chart's own SVG labels). Lane 22's stub predicts
  x1.8 to x3.5 at concurrency 4 on the same chip family. Different model (E4B, not E2B), 32 GB not 16 GB, c=8 not c=4, mean TTFT not p95.
  So the throughput half of the decider's flip has fired in part (an M-series laptop, a ~4B model, throughput at two concurrencies); the
  **p95 half has not: no source below reports p95 for any request latency** (§2: PagedAttention uses a mean, SiliconBench p50, the blog a mean).
- **The paper the brief names, `2601.19139`, does not measure vllm-metal under concurrency.** It is a different engine (`vllm-mlx`, "shares
  no code with" vLLM), on an M4 Max 128 GB; vllm-metal appears only as one column of the single-stream Table 1.
- **The second flip ("defines throughput or p95 differently from the brief") fires for p95 only on its letter:** nobody defines it, so the
  brief's own definition stands, written down openly. Throughput is defined as the brief defines it by SiliconBench (§2 row 3).
- Result for the operator: lane 22 still has something to find (p95, c=2 and c=4, a 16 GB machine, verdict equality against a control, none
  of which any [F] source reports), but its numeric ranges should be re-anchored on the M1 Pro row before the prediction is committed (§4).

## 1. Sources and final rank

| # | source | rank | why |
|---|---|---|---|
| 1 | `2601.19139` Barrios, *Native LLM and MLLM Inference at Scale on Apple Silicon* (v2, 29 Jan 2026) | **F** | the brief's named paper; `vllm-mlx`, not vllm-metal |
| 2 | `2309.06180` Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (12 Sep 2023) | **F** | the baseline claim and its metric |
| 3 | `2609.19169` Zhang et al., *SiliconBench* (v1, 12 Sep 2026; 75 days old; cites 1 and 2 as Barrios 2026 and Kwon 2023) | **F** | added under brief item 4: the only [F] vllm-metal concurrency sweep on Apple silicon; closed loop, same throughput definition as lane 22 |
| 4 | vLLM blog, *Announcing vllm-metal: Concurrent Serving on Apple Silicon* (22 Sep 2026) | **F** (page text; figure values read from the SVG's text labels) | the only M1 Pro number; vendor post, not a paper |
| 5 | vllm-metal `v0.34.0`: `README.md`, `pyproject.toml`, `install.sh`, `docs/installation.md`, `docs/configuration.md`, `docs/supported_models.md`, `docs/recipes.md`, `docs/benchmarking-macos.md` | **F** (read, nothing run) | the code and docs rows lane 22 installs |
| 6 | `mlx-lm` 0.31.3 `server.py`, `models/cache.py` (the lab env; read, nothing run) | **F** (code) | lane 22's "batching without vLLM" |

Brief item 4 asked for at most two added papers. One qualified and is row 3. A web search for others turned up `2607.00501` (BaseRT, July
2026: older than 90 days, a non-MLX runtime) and `2511.05502` (already ranked on 16 Nov); neither was read for this file and neither is
used. **[F] papers: 3** (rows 1 to 3), plus the code and post rows.

## 2. Source table

### Row 1. `2601.19139` vllm-mlx, [F]

- **Setup.** "All experiments run on Apple M4 Max with 128GB unified memory", 4-bit models (§4 Setup). Text models Qwen3 0.6B to 30B-A3B,
  Llama-3.2 1B and 3B, Gemma 3-4B, Nemotron-30B-A3B. Engine: `vllm-mlx`, "an independent project that draws architectural inspiration from
  vLLM but shares no code with it" (§5); it "wraps mlx-lm" (§3.2). The paper calls vllm-metal "a hybrid MLX and PyTorch approach" (§5).
- **n and repeats.** None stated for any throughput figure. No seeds, no repeats, no variance. Prompt set, output length and concurrency
  procedure are not described.
- **Metrics as written.** "Throughput (tok/s)" in Table 1 and Figure 2a; "Requests/sec" in Figure 2b. **Neither is defined, and no latency
  percentile (p50, p95 or p99) appears anywhere in the paper**; Table 7 gives one TTFT per row (245 ms, 42 ms).
- **Numbers.** Table 1 (single stream, M4 Max): Gemma 3-4B 152.5 (vllm-mlx), 117.0 (vllm-metal), 105.4 (mlx-lm), 123.2 (llama.cpp);
  Qwen3-4B 159.0 / 137.3 / 128.9 / 118.2. Figure 2 and §4.1: Qwen3-0.6B 441 → 1642 tok/s from 1 to 16 concurrent requests (x3.7), Qwen3-8B
  x2.6. The abstract says "4.3x aggregate throughput at 16 concurrent requests"; the text and figure give 3.7x for the same model, so the
  abstract number is not backed by the body.
- **Match needed.** Not stated enough to match.
- **Comparable to lane 22: no.** Other engine, M4 Max 128 GB, no p95, no concurrency number for vllm-metal at all.

### Row 2. `2309.06180` PagedAttention / vLLM, [F]

- **Setup.** OPT-13B on one A100 40 GB (26 GB weights, 12 GB KV, "15.7K" slots, Table 1), OPT-66B on 4 A100, OPT-175B on 8 A100-80GB, plus
  LLaMA-13B. ShareGPT and Alpaca length distributions with **Poisson arrivals** ("we generate request arrival times using Poisson
  distribution with different request rates", §6.1). Baselines: FasterTransformer (with the authors' own dynamic-batching scheduler) and Orca
  **as the authors' re-implementation** ("Orca is not publicly available", §6.1).
- **Metric as written.** "normalized latency of the systems, the mean of every request's end-to-end latency divided by its output length"
  (§6.1). Load is a request **rate** (open loop), not a concurrency level; throughput is read as the rate a system sustains "while
  maintaining similar latencies" (§6.2). **No percentile.** Traces are 1 hour (15 minutes for OPT-175B).
- **Headline.** "2-4× throughput ... with the same level of latency" (abstract); on ShareGPT "1.7×-2.7× higher request rates compared to Orca
  (Oracle) and 2.7×-8× compared to Orca (Max)", "up to 22×" over FasterTransformer (§6.2, Fig. 12). It batches 2.2x more requests than Orca
  (Oracle) at the same time on OPT-13B (Fig. 13a).
- **The mechanism condition that matters for a laptop (§6.2):** where memory is plentiful and sequences short, "the performance of the systems
  becomes compute-bound rather than memory-bound" and vLLM's edge shrinks (OPT-175B on Alpaca, Fig. 12f). The gain comes from fitting more
  requests into a KV pool that fragmentation would otherwise fill. Lane 22's Model A is ~3.6 GB of weights against a ~11 GB working set and
  eight or fewer streams: the pool is not the binding constraint there. **Lane 22 tests continuous batching on a small GPU, not the
  paging claim.** Say so on the page.
- **Comparable to lane 22: no.** A100 and 13B to 175B; open-loop rate against a mean; no Apple silicon; baselines are the authors' own.

### Row 3. `2609.19169` SiliconBench, [F]

- **Setup.** Nine engines on one Apple M5 Pro 64 GB, macOS 26.6 (Table 6: Metal's recommended working set "Hint" 51.84 GB; the same table
  lists an M1 Pro 32 GB at 24.96 GB, used only for the left-pad memory study). Models Qwen3-0.6B (BF16), Qwen3.5-0.8B, Gemma-4-E4B-it;
  larger dense and MoE follow-ups. Two splits of n = 100: chat (21 to 3.5K tokens) and agent (0.3K to 8.9K, ~4.3K mean) (Table 2).
- **Protocol as written (§4).** "three warmup requests, then issues requests **in a closed loop with at most c in flight**. Payloads set
  temperature to zero ... **Output throughput is total generated tokens divided by the measured sweep's wall time.** TTFT runs from HTTP
  submission to the first streamed content or reasoning output. Latency summaries cover completed requests; failed requests still
  contribute to sweep wall time." Concurrency 1, 8, 16 (1, 2, 4 for the 27B). Reports **TTFT p50 and ITL p50, plus means; no p95 or p99**
  (grep of the full text; "ITL p50 is the median of request-level mean decode intervals", App. A).
- **Numbers, Gemma-4-E4B-it agent split (Table 10), vllm-metal:** 21.1 tok/s (c=1), 52.3 (c=8), 63.6 (c=16: "3.0× its
  single-stream rate"); TTFT p50 551 / 905 / 1491 ms; success 93/100 at each level (the paper attributes part of the agent failures to
  tool-call serialization). In the same table mlx_lm completed 56/100 at every level and reaches 23.4 tok/s at c=8: its row is a partial run.
  Qwen3-0.6B chat: vllm-metal x3.65 from c=1 to c=16, agent x2.71 (§6.1).
- **Settings that bear on lane 22.** The harness uses `VLLM_METAL_MEMORY_FRACTION` for the vllm-metal KV budget (App. A); **that variable is
  removed at v0.34.0** (`docs/configuration.md`, row 5), so a SiliconBench setting cannot be copied. vllm-metal "reject[s]" prompts above
  its default context, so the harness raises per-slot context to 16384 on the agent split. No designed repeat pass: "the campaign carries no
  formal run-to-run variance estimate" (Limitations); 8 of 9 incidental repeats moved at most 5.6%.
- **Comparable to lane 22: partly.** Same closed-loop design and the same throughput definition, the same engine and a ~4B Gemma 4. Different
  chip (M5 Pro, 64 GB, with a different GPU generation; the blog below shows M5 scaling far above M1), prompts (~4.3K agent tokens, no
  grading), no verdicts, no p95.

### Row 4. vLLM blog, 22 Sep 2026, [F] (page text; chart values from SVG labels)

- **M1 Pro row (cross-machine chart, `cross-machine.svg`).** "We ran the same Gemma 4 E4B 4-bit workload at concurrency 1 and 8 on four
  Macs." Each run: `vllm bench serve`, 100 Sonnet prompts of about 1,024 input and 128 output tokens, `--gpu-memory-utilization 0.5`, prefix
  caching off, `--request-rate 10 --max-concurrency <c>`, fresh server per level, vllm-metal **v0.29.0** (not v0.34.0). Metrics: "Mean TTFT"
  and "Output token throughput". The SVG's labels, matched to bars by y position (checked this session):

  | machine | TTFT avg c=1 | TTFT avg c=8 | out tok/s c=1 | out tok/s c=8 | c8 / c1 |
  |---|---|---|---|---|---|
  | M1 Pro 32 GB | 2.50 s | 12.35 s | 18.9 | 23.8 | **x1.26** |
  | M1 Max 64 GB | 1.15 s | 5.34 s | 31.4 | 51.1 | x1.63 |
  | M2 Max 64 GB | 1.09 s | 5.15 s | 34.9 | 52.5 | x1.50 |
  | M5 Pro 64 GB | 0.27 s | 1.21 s | 59.9 | 154.4 | x2.58 |

  One run per cell, no spread, no failure counts in the figure (the post says to "keep the successful-request count"; none is printed).
- **Other rows in the post (concurrent-serving claims, all M5 Pro 64 GB):** "vllm-metal has the lowest TTFT and end-to-end latency at concurrency
  2 and 4" on Qwen3.8-27B (SiliconBench agent split, charts only; I did not read values). A table on Qwen3.6-35B-A3B shows batch wall time for
  8 requests: mlx_lm 4.52 s (equal prompts) and 10.99 s (one long prompt), vllm-metal 3.87 s and 3.64 s. mlx_lm's padded batch grows
  +143% when one prompt is longer. These come from the vendor, on a bigger model.
- **Comparable to lane 22: partly (the nearest row there is).** Same chip family, same engine family, Gemma 4. E4B not E2B, 32 GB, c=8 only,
  a mean not p95, release 0.29.0, a vendor post.

### Row 5. vllm-metal `v0.34.0` code and docs, [F]

- **OS and Python.** "macOS 15 (Sequoia) or later" (`README.md:40`; `docs/installation.md:3`). `requires-python = ">=3.12,<3.14"`
  (`pyproject.toml:12`).
- **Pins.** `mlx==0.32.4` exact (`pyproject.toml:34`); mlx-lm at git sha `4cfc6478473b0668ede3eb0cc93f2a9181a15052` (`:37`); mlx-vlm at sha
  `b7bec926…` (`:41`). Against the lab's mlx 0.32.2 and mlx-lm 0.31.3: three versions moved, so lane 22's in-venv control arm is needed.
- **`install.sh` does not install the tag it comes from.** It selects the newest release of a channel ("`--dev` ... the default"; "`--stable`
  ... latest tagged stable release") and takes the wheel URL from GitHub releases (`install.sh:205-232`, `:304-327`); it has no tag argument.
  Running the file at `v0.34.0` therefore installs whatever is latest on the dev channel. Lane 22's "install v0.34.0 by the steps of its
  `install.sh`" has to be done step by step: the vLLM release named in `.github/vllm-release-tag.commit` at that tag, then the `v0.34.0`
  wheel, by `uv pip install`. Defaults: venv `~/.venv-vllm-metal`, overridden by `VLLM_METAL_VENV` (`:289`); Python 3.12 unless
  `VLLM_METAL_PYTHON` (`:223`); `ensure_uv` installs `uv` (the function body is in a `lib.sh` fetched or sourced at run time; I did not read
  it, so where `uv` lands is ⚠️ UNVERIFIED). Run from a source checkout it builds native code (`ensure_metal_toolchain`): do not run it that way.
  Installation also states "`pip install vllm-metal` is not supported" (`docs/installation.md:50`).
- **KV pool sizing (`docs/configuration.md:113-159`).** "The paged KV cache budget follows vLLM's standard `--gpu-memory-utilization` flag
  ... a fraction in `(0, 1]`. The former `VLLM_METAL_MEMORY_FRACTION` override has been removed." Default 0.92: "the pool and the model
  weights together take about the `--gpu-memory-utilization` fraction (default 0.92) of the memory macOS recommends for GPU use (Metal's
  recommended working set)." **The cap flag is `--kv-cache-memory-bytes` (`kv_cache_memory_bytes=` for `LLM()`)**, "to override automatic
  sizing"; it budgets cache storage only, "weights, activations, runtime workspace ... need additional memory"; sizes round down to whole
  blocks; an explicit budget that would need shrinking is rejected. `--num-gpu-blocks-override` takes precedence. "Plan for the KV pool as RAM
  the server holds at its full size, even when requests use little of it." A startup probe tests up to 512 MiB of the pool against the
  machine and can shrink an automatic pool (or stop startup for an explicit one); `VLLM_METAL_KV_COMMIT_PROBE=0` skips it. Both flags apply to Model
  A and B alike.
- **What that means on this 16 GB box.** The lab's recommended working set is 11.45 GB (LESSONS, "moe" row). At the 0.92 default,
  weights + pool plan for 0.92 x 11.45 ≈ **10.5 GB** (arithmetic, not observed). The blog's own recipes use 0.5 (≈ 5.7 GB). For Model A
  (file bytes 3.55 GB, see below) a pool of 2 GiB by `--kv-cache-memory-bytes` leaves the rest for activations and the sidecar; for Model B
  (9.18 GB peak in lane 04) nothing like a useful pool fits under 10.5 GB, which is the stub's own guess. **Recommended brief wording:** "set
  `--kv-cache-memory-bytes` explicitly (start at 2 GiB for Model A) and leave `--gpu-memory-utilization` at a value for which the planned
  total stays under 11.45 GB; read the startup log line for the pool before the first request."
- **Gemma 4 and the sidecar (`docs/supported_models.md:64-90`).** "mlx_lm text backbone + mlx-vlm encoder sidecar, paged generation"; the
  listed checkpoint is `mlx-community/gemma-4-e2b-it-4bit` "(audio, video)". The text says "Gemma 4 keeps its text path exactly as on the
  text-only table ... only `vision_tower` and `embed_vision` are loaded from the checkpoint through mlx-vlm, plus `audio_tower` and
  `embed_audio` on checkpoints whose config has an `audio_config`". The sidecar is on when `VLLM_METAL_MULTIMODAL_MODE=auto` **and** the
  checkpoint is a fully cached local safetensors directory with `vision_tower.*` weights, the HF `Gemma4Processor` builds (the
  `processor_config.json` needs a `video_processor` block with transformers 5.14+), and no speculative decoding is configured; "otherwise the
  model stays text-only and the reason is logged". **Does a text-only request need it? The docs do not say that a text-only request skips
  loading the towers; they say the sidecar loads when its conditions hold.** The docs give no memory figure for it. On disk, in the cached
  `gemma-4-e2b-it-4bit` (`model.safetensors` header read; nothing loaded): vision tensors **0.335 GB**, audio **0.611 GB**, the rest
  **2.604 GB**, 3.55 GB in all (summed from tensor data offsets; file bytes, not resident memory). The checkpoint has `vision_config` and
  `audio_config`, 658 vision and 751 audio tower keys, and a `video_processor` key in `processor_config.json`, so the sidecar's file
  conditions are met here; whether the transformers version in the venv meets "5.14+" is unknown until it is installed. Consequence for the
  prediction: Model A's weights bill is up to 3.55 GB, not the lab's 3.71 GB peak for the text path alone, and the sidecar may add up to
  0.95 GB the lab's rows never carried. Lane 22 should read the log line for the multimodal mode and, if it is on, also set
  `VLLM_METAL_MULTIMODAL_MODE` to its text-only value (name taken from the doc text above; **the accepted values are not in the files I
  read**) or record the mode beside the memory figure.
- **Benchmark guidance the project itself gives (`docs/benchmarking-macos.md`).** Use `vllm bench serve` with
  `--percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99` against a real `vllm serve`; "client concurrency does not establish the
  number of decode rows in an actual scheduler step"; "Output throughput: completed output tokens divided by the benchmark duration,
  including the workload's prefill and queueing"; warm up, alternate arm order, record power mode and background GPU load; "Natural
  generations can diverge even with greedy sampling" and "neither close attention outputs nor matching throughput establishes
  model-quality equivalence". So **p95 is available in the project's own tool and its throughput definition matches lane 22's**.

### Row 6. `mlx-lm` 0.31.3 `server.py` batching, [F] (code read)

- `BatchGenerator` is created per model in `_generate` (`server.py:821-826`) with `completion_batch_size=--decode-concurrency` (default **32**)
  and `prefill_batch_size=--prompt-concurrency` (default **8**), `prefill_step_size` default 2048 (`:1854-1870`). A request is batched only if
  `draft_model is None`, **every cache object from `make_prompt_cache(model)` has a `merge` method** (`:370-373`) and `args.seed is None`
  (`:685-686`). In the lab env, `KVCache`, `RotatingKVCache` and `ArraysCache` have `merge`; `QuantizedKVCache` does not (checked by
  `hasattr`, this session). Gemma 4 uses `KVCache` plus `RotatingKVCache`, so by source its server batches; **not run**, and it matters that a
  quantized-KV flag would silently drop a model to the unbatched path. Contiguous batch-padded KV: the blog says "mlx_lm keeps a contiguous
  `[B, H, T, D]` cache" and pads queries to the longest prompt.
- **Consequence for lane 22.** "Batching without vLLM" exists inside the pinned stack with no installation: `mlx_lm.server` at the lab's
  versions. It is the right third arm (L-mlx) for the load table: it separates what vLLM's scheduler and paged kernel buy from what
  continuous batching alone buys. The lab's lane 02 server probe already ran `mlx_lm.server`; the lane's brief names only a control arm of
  plain generation, not this.

## 3. Comparison with our finished runs

Lane 22 has no result yet, so there is no number of ours to set beside these. The lab's committed anchor for Model A is lane 02
(`results/dense-gemma-4-e2b-it-4bit.json`, 69.2 gen tok/s in process, single stream, peak 3.71 GB; figures as the stub cites them; I did not
re-open the file for this read). Set beside the M1 Pro blog row (18.9 tok/s at c=1 for E4B with ~1K-token prompts), the two are not the same
quantity (output tokens over a sweep that includes prefill and queueing, a different model, 100 prompts): **do not compare lane 22's single-stream
rate with the blog's 18.9.**

## 4. Lane 22 against the bar

**Does an [F] row already report lane 22's number on a comparable setup?** Partly. Throughput at c=1 and c=8 on M1 Pro, Gemma 4 E4B 4-bit
(blog, row 4): yes, one run, no spread. c=2 and c=4: no. p95 of any latency: no source. Verdicts under vLLM (equality with a control): no source
at all; SiliconBench runs a classification fidelity set (GMRID) against an NVIDIA reference, on Qwen3-0.6B and E4B, and reports F1, not
case-by-case equality. A 16 GB Mac: no source (the two Macs with a memory budget are 32 GB and 64 GB). No [F] source defines p95.

**Memory setting on a 16 GB Mac.** None of the papers or the post uses one. SiliconBench's vllm-metal budget variable is removed; the blog uses
`--gpu-memory-utilization 0.5` on 32 to 64 GB machines. The project's documented settings for this lane are `--kv-cache-memory-bytes` (explicit
pool) and `--gpu-memory-utilization` (default 0.92 of the recommended working set, i.e. ≈ 10.5 GB planned at 11.45 GB; arithmetic). Use the
explicit byte cap and read the pool line from the log; do not rely on the fraction.

**Changes to the brief that this read supports (the operator amends openly, with a dated note, before the prediction is committed):**
1. **Re-anchor the load ranges.** The stub's "throughput at c=4 is 1.8 to 3.5 times c=1" and its "c=8 gains under 30% over c=4" sit above or
   beside the only M1 Pro row (c8/c1 = x1.26; M1 Max x1.63; M2 Max x1.50; M5 Pro x2.58 in the same post). A prediction range that the one comparable
   [F] row falls below is a prediction written without reading it. Row 4 is E4B at 32 GB, so say what is carried over (the chip's bandwidth) and
   what is not (model size, memory).
2. **State definitions before the run.** Throughput = total output tokens over the sweep's wall time (SiliconBench §4; `vllm bench serve`).
   p95 is the lane's own number: the load driver must compute it itself from per-request latencies, and `vllm bench serve --metric-percentiles
   50,95,99` gives the same for an independent check. At n = 20 requests per level (the stub's email-triage set, three rounds = 60), a p95 is
   the 57th of 60 sorted values: state n beside it.
3. **Closed loop, not Poisson.** The stub's design (concurrency 1, 2, 4, 8 with N in flight) matches SiliconBench and differs from
   PagedAttention's open-loop request rate. Say which.
4. **Add the third arm (L-mlx): `mlx_lm.server` at the lab's versions on the same load table**, because "what does a paged, batching server buy"
   is answered only against a batching baseline; the blog's own mlx_lm row on a long-prompt batch (+143%) is the effect to look for or not.
   Not queued here; the operator's call.
5. **Record the multimodal mode** (sidecar on or off, and its memory) for Model A, and add 0.95 GB of possible sidecar weights to the memory plan.
6. **The installation step in the brief is not what `install.sh` does** (no tag argument): the lane needs an explicit v0.34.0 wheel plus its
   matching vLLM release, or a written note that the dev channel was installed.
7. **Variance.** No source reports run-to-run spread for vllm-metal on a laptop (SiliconBench: none designed; blog: one run). Lane 22's "three
   rounds each" is already above the published bar; keep it.

**What lane 22 may claim after it runs.** Its own numbers on one 16 GB M1 Pro at the pinned v0.34.0 / mlx 0.32.4 / mlx-lm `4cfc647`, Model A:
throughput and p95 at c = 1, 2, 4, 8 on the email-triage held-out cases, and the count of verdicts that differ from the in-venv control.
**May set beside:** SiliconBench Table 10 and the blog's M1 Pro row as "published, other chip, model, memory, prompts, release and
statistic". **May not:** say "vLLM improves local throughput by x" in general, or "the field reports" anything for a 16 GB Mac; call the
result a test of PagedAttention (the paged pool is not the binding constraint at this model size, §2 row 2); compare the single-stream
rate with the blog's 18.9 tok/s; claim a p95 gain over mlx-lm without the L-mlx arm; or say vLLM "was used on a client's flows" (the lane
measures the exams, not a flow). Anything about CUDA is out of scope, as in the stub.

## 5. What our pages may and may not claim

**May:**
- Quote SiliconBench's c=1, 8 and 16 vllm-metal figures for Gemma-4-E4B-it (21.1, 52.3, 63.6 tok/s; Table 10, M5 Pro 64 GB) and its "3.0× its
  single-stream rate", as their claim on that machine.
- Quote the vLLM blog's M1 Pro 32 GB row (18.9 → 23.8 tok/s; 2.50 → 12.35 s mean TTFT; one run, v0.29.0, E4B) as the vendor's own
  chart value, with the date of the post.
- Quote PagedAttention's "2-4×" with its baselines (re-implemented Orca; FasterTransformer with the authors' scheduler), its metric (mean normalized
  latency, open loop), its hardware (A100) and its own caveat that gains shrink when the workload is compute-bound.
- Say that vllm-mlx (2601.19139) reports 3.7x at 16 concurrent requests for Qwen3-0.6B on an M4 Max, with no metric definition and no n.

**May not:**
- Present vllm-mlx as vllm-metal or as vLLM; the paper itself says it shares no code with vLLM.
- Quote the vllm-mlx abstract's "4.3x" (the body says 3.7x).
- Say that "published results show vLLM scaling on a MacBook" or "on a 16 GB Mac".
- Use any published p95; none was read.
- Treat the blog's M1 Pro values as a table the vendor published: they were read from chart labels (`cross-machine.svg`), not from a data table.

## 6. Not verified

- Whether the vision and audio towers are resident in memory on a text-only request under v0.34.0 (docs silent; needs an install this file did not do).
- `VLLM_METAL_MULTIMODAL_MODE` accepted values; where `ensure_uv` puts `uv`; which transformers version the v0.34.0 install pulls.
- SiliconBench's 27B and 35B charts and the blog's concurrency-2/4 claims (figures only; values not read).
- That `mlx_lm.server` batches Gemma 4 e2b at the pinned versions: by source only.
- The lab's own numbers quoted from the stub (lane 02, 3.71 GB, 69.2 tok/s) and from LESSONS (11.45 GB): not re-opened for this file.
