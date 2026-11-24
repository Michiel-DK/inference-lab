# Literature, full-text read: vendor claims behind lanes 15 and 16 (24 Nov 2026)

Companion to [`docs/literature-2026-11-16.md`](literature-2026-11-16.md). **That file stays the read for lanes 02 to 13.** This one
covers three vendor posts, the `mlx-lm` batching source, and five papers, for two queued lanes (15 `cold-start-worker`, 16
`batch-invariance`) and one lane that is not queued (a divergence lane). No model was loaded, no GPU used, nothing run. Reader
question B1 / C2; decider: the operator, whether lanes 15 and 16 run as briefed, with changed arms, or not at all.

**Rank rule** (as 16 Nov). [F] = body read: setup, n, metric as written, the table or section the number comes from. [A] = abstract
(and intro) only. [S] = snippet only. A published number sits beside ours only at [F]. Pages and PDFs were fetched into `/tmp` and are
not committed. Every quote below was matched in the fetched text this session.

## 0. First lines: does a published [F] result already report a lane's number?

- **Lane 16 (batch invariance under MLX): no.** No [F] source here reports a count of batch-dependent outputs on Apple silicon or under
  MLX. The closest numbers are all NVIDIA (GB10, GH200/H100, L40S/A100, vLLM or transformers). The lane still runs on this evidence.
- **Lane 15 (cold-start stages under MLX): partly.** `2511.05502` (ranked [F] on 16 Nov, re-read here for this one figure) reports an
  MLX **cold-start total** on Apple silicon: "MLC-LLM loaded in ∼11s ..., MLX in ∼31s, llama.cpp in under 0.5s ..." (M2 Ultra,
  Qwen-2.5 family, Figure 2 and the text under it). It gives **no stage split**, no n for that figure, and does not say which Qwen
  size the 31 s is. The brief's stop rule says "a published [F] number for cold-start **stages**"; one total without stages is not
  that, but it is the operator's call whether it changes the lane. Our committed anchor (`load_s` 3.05 to 4.79 s, in a running process)
  is not the same quantity as "process start to model ready".
- **The decider's second flip** ("a source defines the metric differently from the brief") **fired for lane 15**: see Lane 15 below.
  The brief should be amended openly before its prediction is committed. It did **not** fire for lane 16's numbers, but two arm-design
  changes are listed there.

## 1. Sources and final rank

| # | source | rank | why |
|---|---|---|---|
| 1 | Thinking Machines (He), *Defeating Nondeterminism in LLM Inference*, 10 Sep 2025 | **F** | lane 16's origin |
| 2 | `thinking-machines-lab/batch_invariant_ops` README + `deterministic_vllm_inference.py` (MIT) | **F** | the repo's own count and its script |
| 3 | Runpod, *Cut your vLLM cold starts from 5 minutes to 90 seconds* (page: "Updated September 13, 2026") | **F** | lane 15's origin |
| 4 | Runpod, *Cold starts were never the real problem* (FlashBoot / Flash; page: "Updated October 10, 2026") | **F** | what FlashBoot is said to do |
| 5 | Runpod docs: `serverless/overview.md`, `serverless/development/optimization.md`, `serverless/workers/overview.md`, `.../endpoint-configurations.md` (FlashBoot), `.../local-testing.md` (grep only) | **F** (first four read for the cold-start passages) | the cold-start definition |
| 6 | Fireworks, *How Fireworks evaluates quantization precisely and interpretably*, 1 Aug 2024 | **F** | divergence method; no lane queued |
| 7 | `mlx-lm` 0.31.3 source: `generate.py`, `models/cache.py`, `models/base.py`, `models/gemma4_text.py` | **F** (code read, nothing run) | lane 16 by construction |
| 8 | Blagojevich & Sunderland, *The Noise Floor of a Quantization Comparison*, 2611.15671 (11 Nov 2026) | **F** | restart / batch / engine floor vs 8-bit and 4-bit, counted |
| 9 | Yuan et al., *Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference*, 2506.09501 (v2, 24 Oct 2025) | **F** (setup, metrics, Tables 3 and 4) | run-to-run and batch-size variation of greedy outputs, counted |
| 10 | Dutta et al., *Accuracy is Not All You Need*, 2407.09141 (12 Jul 2024) | **F** (metric definitions, headline) | flips and KL against accuracy; the paper Fireworks cites |
| 11 | Lu & Ni, *How Far Does Second-Order Linear Response Predict Quantization Mismatch?*, 2611.12634 (9 Nov 2026) | **F, §9 only** (abstract + §9; the rest not read, no number from it is used) | a KL noise floor of the engine, batch 8 vs 256 |
| 12 | Kasa, Kasa & Negi, *Same Text, Different Prediction*, 2610.09111 (6 Oct 2026, NeurIPS workshops) | **A** (abstract, intro, related work) | batch shape / padding vs labels in classifiers; GPU/CPU |
| — | `2511.05502` (Rajesh et al.), re-read for the cold-start passage only | **F** (16 Nov rank) | the one MLX cold-start figure |

Not fetched or not read: vllm-mlx 2601.19139 (on disk on 16 Nov, never read; still unread), Wu et al. 2608.06564, Li et al.
2610.42148 (concurrency changes answers), Patodiya 2609.04748 (prefix cache), Pape et al. 2605.19537. These are named in 2611.15671's
related-work section; I know them only from that text, so they are not ranked and not used. 2606.19558, 2601.19934 and 2601.06118
turned up in a search or fetch and were dropped for the five-paper cap. arXiv's search API was rate-limited (HTTP 429); papers were
found with web search, and every id above was opened and its title page matched.

## 2. Source table

Columns: setup, n, metric as written, headline with its location, what our run would need to match, comparable to our run?

### 1. Thinking Machines, "Defeating Nondeterminism in LLM Inference" (F)
- **Setup.** `Qwen/Qwen3-235B-A22B-Instruct-2507` behind a vLLM server, GPU kind not stated in the post. Prompt "Tell me about Richard
  Feynman", non-thinking mode, temperature 0, 1000 completions of 1000 tokens each.
- **Result (Experiments, "How nondeterministic are completions?").** "we generate 80 unique completions, with the most common of
  these occuring 78 times." "the completions are actually identical for the first 102 tokens! The first instance of diverging
  completions occurs at the 103rd token. ... 992 of the completions go on to generate 'Queens, New York' whereas 8 of the completions
  generate 'New York City'." "when we enable our batch-invariant kernels, all of our 1000 completions are identical."
- **The cause they name.** Not atomics: "the forward pass lacks 'batch invariance', causing our request's output to depend on the batch
  size of our forward pass". Reductions in three operations: RMSNorm (data-parallel one row per core, breaks when a small batch
  triggers a split reduction), matrix multiplication (split-K and different tensor-core instructions at small batch; stream-k is not
  even batch-position-invariant), attention (split-KV / FlashDecoding; a KV cache handled separately from the current keys; chunked
  prefill and prefix caching). They name **reduction order**, **small-batch parallelism strategy** and **chunking/KV layout**, not
  padding. The batch size itself varies because server load varies: "the load determines the batch size".
- **The sentence lane 16 wants to test.** "This nondeterminism is not unique to GPUs—LLM inference endpoints served from CPUs or TPUs
  will also have this source of nondeterminism." (confirmed word for word.) The post gives **no CPU or TPU measurement** for it; it is
  an assertion.
- **Cost of the invariant mode.** Qwen-3-8B, one GPU, 1000 sequences of 90 to 110 output tokens: vLLM default 26 s, "Unoptimized
  Deterministic vLLM" 55 s, "+ Improved Attention Kernel" 42 s. Batch-invariant matmul about 20% slower than cuBLAS (their Triton
  kernel, not optimised).
- **Limits they state.** Performance not tuned; "FlexAttention integration in vLLM has not been heavily optimized"; the fixed-size
  split-KV needs "internal FlexAttention changes that are not included in our code release". The one-GPU timing is the only cost number.
- **Comparable to lane 16?** No, and the brief should not word it as a replication: their varying quantity is the **server's batch
  size under load**, uncontrolled, over 1000 identical requests; ours would control the batch. 80 distinct strings in 1000 is a count
  of outputs under an uncontrolled mix of batch shapes.

### 2. `batch_invariant_ops` README and script (F)
- README: "Without the upstream PR, we see that out of 1000 random length 100 completions we see 18 unique samples. After the upstream
  PR, there is only one unique sample." **Correction to the scope job:** the README names no model. The **script**
  (`deterministic_vllm_inference.py`) names `Qwen/Qwen3-8B` (`vllm serve Qwen/Qwen3-8B --enforce-eager`), uses a **different prompt**
  ("Generate 1000 random numbers. Go directly into it ...", i.e. not Feynman), `max_tokens` 100, temperature 0, 1000 requests sent
  concurrently, and prints `Total samples ... Unique samples`. It needs PyTorch 2.9+ and vLLM PR 24583. The 18 is the README's
  sentence; no script output is committed.
- Supported ops: `torch.mm`, `torch.addmm`, `torch.log_softmax`, `torch.mean`. Triton/CUDA; MIT (LICENSE read). Does not run on Metal.
- Comparable to lane 16: no (CUDA, vLLM, different prompt, concurrent load).

### 3. Runpod, vLLM cold-start post (F)
- **Setup.** `Qwen/Qwen3-32B-FP8` (~32 GB), `tensor_parallel_size=2` on 2× H200, image `runpod/worker-v1-vllm:v2.25.0` (vLLM 0.27.0),
  weights on a network volume, one region.
- **Metric, as written.** "The job's delayTime, the interval between job submission and job pickup by a worker. When the endpoint has
  no running workers, that interval contains the entire boot: image pull, weight load, compile, graph capture, and warmup. It is an
  end-to-end number, and it is not comparable to snapshot-restore figures, which measure the time to revive an already-warm process."
  Also: "delayTime also captures two costs that have nothing to do with your configuration: time spent waiting for GPU capacity ...
  and image pull on a host that has never seen your image."
- **n.** "Baseline: 331s / 309s / 324s (Median: 324s)" = **3 trials**. "Optimized: 93s / 89s (Median: ~91s)" = **2 trials**. "Optimized,
  first-ever boot: one trial (~284s)". "Treat the 3.5× ratio as directional." (all confirmed.) A separate proxy (Qwen3-8B, 48 GB GPU,
  FlashBoot off) gave four baseline starts "273s, 355s, 387s, and 834s": "A 479-second spread on fixed config".
- **Stages they list** (no per-stage timing is reported in numbers except one: 32 GB read "in roughly 21 seconds" with prefetch):
  image pull; weight load; torch.compile + CUDA-graph capture ("roughly two to four minutes"); KV-cache allocation and warmup.
- **Their caveats.** One model, one GPU configuration, one region; "Image pull ... is included in our medians"; delayTime
  "bundles vLLM boot time together with platform costs we cannot isolate from a sample this size"; FlashBoot left on, trials are "the
  snapshot-miss path"; throttled trials should be discarded.
- **Page date.** The page shows "Updated September 13, 2026"; the 20 Aug date in the brief is not on the page.
- Comparable to lane 15: no. Different hardware, runtime, stages (three of theirs do not exist on a laptop), n of 2 to 3.

### 4. Runpod, FlashBoot / Flash post (F)
- "FlashBoot relies on container pausing rather than stopping. When a worker finishes a job ... we freeze the process at the OS level,
  but the GPU memory allocation and model weights stay loaded in VRAM. ... the daemon calls docker unpause ... skipping image pull,
  container initialization, and model load entirely." "FlashBoot resumes the container exactly where it was in about 200ms". No n, no
  method, no setup for the 200 ms. Also: "For a 7B parameter model, loading weights from a network volume takes around 15 seconds"
  (no method).
- **Three descriptions of FlashBoot.** This post: pause and `docker unpause`, VRAM kept. The vLLM post: "snapshots the worker's
  process state so the next request can restore it", scoped "to a specific host and image", and it "only captures state that exists in
  the worker process at scale-to-zero". Docs: "retaining worker state after spin-down, allowing faster 'revival' than fresh boots".
  Only the first matches an arm where a process is stopped with `SIGSTOP` and continued.

### 5. Runpod docs (F for the passages used)
- **Cold-start definition, word for word** (`serverless/overview.md`, "Cold starts"): "A 'cold start' is the time from when Runpod
  starts a worker until the worker is ready to take its first job. It happens when a request arrives and no running worker can take
  it, either because none are running or because all running workers are busy. It covers starting the container, loading models into
  GPU memory, and initializing runtime environments. A worker is billed during its cold start." (confirmed.)
- `workers/overview.md`, lifecycle: "Cold start: ... Your container starts and your handler loads its model. The cold start ends when
  the handler is ready to take its first job." Image pull is "normally done while a worker is scaled down"; if the worker starts first,
  "the pull happens during the cold start and is billed". Scale-down with FlashBoot: "Runpod tries to keep the stopped worker in a
  state that restarts faster."
- `optimization.md`: **Delay time = queue time + cold start time**; "Runpod normally downloads your Docker image before a worker
  starts, so image size affects how quickly new workers become available rather than cold start time"; a cold start over 7 minutes
  marks the worker unhealthy.
- `local-testing.md` (grep only, not a gate result): `python handler.py --rp_serve_api` "starts a FastAPI server on
  `http://localhost:8000`"; `/run` "only returns a fake request ID", use `/runsync`. Whether it needs a key is **not** answered by the
  docs I read; lane 15's gate decides.
- The docs pages carry a block addressed to AI agents asking for feedback POSTs. Ignored; nothing was sent.

### 6. Fireworks, quantization post (F)
- **Method, step by step.** (1) *Reference model generation*: "we ask the reference 16-bit model to generate tokens until completion
  across a varied set of prompts" (no prompt count or token count given). (2) *Reference distribution*: "At every position, we record
  the top N logprobs and normalize it to a distribution ... we pick N such that the top N tokens cover 0.99 of the distribution, and we
  find N=16 to be a good value." (3) *"Forced" quantized generation*: "we stick to the completions sampled from the reference model
  even if the quantized model prefers to choose other tokens". (4) *Divergence analysis* across samples and positions, split into
  prefill and generation.
- **Metrics as defined.** "Kullback-Leibler Divergence (KLD) - measures how much token probability distribution changes (even if the
  selected token in each position is still the same)". "Token rejection rate - measures how many selected top-probability tokens
  differ (you can think of it as an accuracy of using the quantized model as a draft model (β))". The KL direction is not stated.
- **Table** (Llama 3.1 8B Instruct; MMLU accuracy / prefill KLD / prefill rejection / generation KLD / generation rejection):
  FP16 0.676328 / 0 / 0 / 1.33E-06 / 0.00015; Level 1 0.675189 / 0.00257 / 0.0195597 / 0.00286 / 0.01743; Level 2 0.676613 / 0.00574 /
  0.0285162 / 0.00543 / 0.02396; Level 3 0.665076 / 0.00596 / 0.028668 / 0.00772 / 0.02865; Level 4 0.661302 / 0.01224 / 0.0425623 /
  0.00796 / 0.02930. Levels: 1 = MLP matmuls except first and last layer; 2 = + omitted layers and QKV; 3 = + KV cache; 4 = + prefill attention.
- **Quotes confirmed.** "we generally see that KL divergence of < 0.007 results in high-quality deployments" (the table does not
  show where that threshold comes from; it is a statement, not a result); "the task-based metric (MMLU) is highly noisy" (evidence:
  one row, Level 2 MMLU above FP16; no confidence interval and no n given).
- **Limits / oddities.** No prompt count, no token count, no seeds. The FP16-vs-FP16 generation row is **not zero** (KLD 1.33E-06,
  rejection 0.00015): the post's own repeat floor, which it does not discuss. MMLU "may not match Meta's reported exactly due to
  templating differences".
- Comparable to any run of ours: not yet (no divergence lane exists; lane 04 keeps no logprobs, see the last section).

### 7. `mlx-lm` 0.31.3 batching source (F, code read; nothing run)
Paths under `~/.pyenv/versions/lewagon/lib/python3.10/site-packages/mlx_lm/`.
- `batch_generate` (`generate.py:1887-1964`) builds a `BatchGenerator` (`:1486`, `__init__` `:1497-1520`: `completion_batch_size=32`,
  `prefill_batch_size=8`, `prefill_step_size=2048`, default sampler `mx.argmax`, `kwargs` pass through so `sampler=make_sampler(temp=0.0)`
  works) and returns **decoded texts only** (`:1952`, `BatchResponse(texts, stats, caches)`): no token counts or token ids.
- **Padding.** Prefill is **right-padded** (`_right_pad_prompts`, `:808`; `PromptProcessingBatch.prompt`, `:1124-1170`:
  `c.prepare(lengths=lengths, right_padding=padding)`, then `c.finalize()` after the chunks). `BatchKVCache.finalize`
  (`models/cache.py:980`) rolls the keys so pads become **left** padding and sets `left_padding` per row.
- **Masking.** `BatchKVCache.make_mask` (`cache.py:1011`) calls `create_causal_mask(..., left_padding=...)` (`models/base.py:24-42`):
  `mask & (left_padding <= rinds)`; a `right_padding` term is also supported. Per-sequence `offset` vectors feed RoPE (Gemma 4:
  `gemma4_text.py:256`, `offset = mx.array(cache.offset)`). So pad positions **are masked**, per row.
- **Sliding-window layers** (Gemma 4: `make_cache`, `gemma4_text.py:662-675`, `RotatingKVCache(max_size=sliding_window=512, keep=0)`;
  converted by `_make_cache`, `generate.py:838-867`, to `BatchRotatingKVCache`, `cache.py:1133`). Its code carries `_lengths` "to make
  sure that padding tokens do not evict valid tokens" (`:1155` region, `_update_concat` `:1169`, `finalize` `:1285`, `make_mask` `:1330`):
  right-padded rows are rolled before trimming, and `left_padding` is reduced as the window slides.
- **Batch mates and batch shape.** Sequences are sampled per row (`_step`, `generate.py:1320-1378`). Finished rows are dropped with
  `filter` (`generate.py:1383`; `cache.py:1016`, `:1359`), so the batch **shrinks during generation** and left padding is re-trimmed
  (`BatchKVCache.filter` shifts left by the minimum padding). New prompts can join a running batch (`_next`, `:1769-1845`; `extend`,
  `cache.py:~1040`, `:1380`). Matrix shapes therefore change within one request's lifetime.
- **Verdict.** I found **no place where a row's result depends on its mates by construction**: pads are masked, rolled, and written
  not to evict real tokens. What the code cannot guarantee is **bitwise** equality: attention runs over a padded key length, matmuls
  run at batch B rather than 1, and B changes as rows finish. That is the floating-point route Thinking Machines describes, not a
  masking bug. Not checked (code only): whether a ~1,445-token prompt on Gemma 4 (arm 2, `prompt_tokens` 1445 in
  `results/quant-gemma-4-e4b-it-8bit.json`) behaves under a 512 window with right padding in one prefill chunk (below 2048), which I
  could read but not run. Whether `batch_generate` runs on the Gemma 4 class at all was not tried.

### 8. 2611.15671, Noise Floor (F)
- **Setup.** Cloudflare's Clef-flash (9B) and Clef (27B), Qwen3.5 hybrid (Gated DeltaNet); **one NVIDIA GB10**; transformers 5.16.1,
  vLLM 0.29.0, llama.cpp b11671. 2,400 public decisions (first 200 of each of 12 tasks); "decision models": top-1 over a fixed option set
  from one forward pass, **not generated text**. Paired bootstrap 95% intervals (2,000 resamples).
- **Metric.** Top-1 flip rate; mean total-variation distance; "the share of records that are bit-identical".
- **Headline (§4.1 table).** llama.cpp BF16 restart: 0 flips on both models (single sequence, no batching). vLLM restarts (BF16 / FP8 /
  NVFP4): Clef-flash 0 / 5 / 20, Clef 4 / 0 / 22. "Batch size 1 to 8, native" (transformers): **8 flips (0.33%) and 15 (0.62%)** of 2,400.
  Engine change at BF16: 6 to 16. Q8_0 21 and 25 (0.88%, 1.04%); FP8 51 and 39; Q4_K_M 139 and 84; NVFP4 139 and 136.
- **Limits they state (§6).** Two models, one family; one device; "Few restart replicates" (one pair per vLLM build); "Confounded
  rerun. ref-r2 combines a rerun with a batch-size change. The attribution to batch size rests on the bit-for-bit reproduction at batch
  1 on Clef-flash, and was not checked on the 27B." The batching mechanism ("Batching pads every sequence to the longest in its group")
  is "an inference, not a measurement". Dated 11 Nov 2026: **9 days old and a preprint**.
- Comparable to lane 16: partly (a counted batch effect on decision flips, GB10); not to MLX or generation.

### 9. 2506.09501, Yuan et al. (F)
- **Setup.** DeepSeek-R1-Distill-Qwen-7B and -Llama-8B, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct; vLLM (HF Transformers in appendix);
  12 runtime configurations = 2 GPU types (L40S, A100) × 2 GPU counts (2, 4) × 3 batch sizes (8, 16, 32); greedy; AIME'24, MATH500,
  LiveCodeBench; max output 32,768 (reasoning) or 2,048 tokens.
- **Metrics.** Std@Acc (std of accuracy across the 12 configs), Avg_Std@Output_Length, Div_Index ("the index of that position" where
  responses first differ), Avg_Std@top1_prob.
- **Headline (Tables 3, 4).** BF16 Std@Acc 9.15% (R1-Distill-Qwen-7B, AIME'24), 1.04% MATH500; output length std 9,189.53 tokens
  (AIME'24); FP32: 0 on AIME'24. Non-reasoning Qwen2.5-7B-Instruct BF16 Std@Acc 1.71% (AIME'24), 0.83% (MATH500).
- Comparable to lane 16: no (NVIDIA, long reasoning outputs, benchmark accuracy). It shows the effect needs length and BF16.

### 10. 2407.09141, Dutta et al. (F, definitions and headline)
- **Metrics as written.** "Flips - distance metric: measures the % of answers that changed from correct → incorrect between baseline
  and quantized model for all tasks" (incorrect → incorrect is excluded for generation tasks; the paper says including it raises MCQ
  flips by 20 to 40%). KL divergence: mean over samples of the KL between the baseline's and the quantized model's token distribution
  over the options.
- **Headline.** "up to 13.6%" flips with accuracy near unchanged; "Spearman correlation on the MMLU benchmark is 0.981" between KL and
  flips; flips correlate with MT-Bench.
- Comparable to ours: our "flip" in LESSONS counts changes of the **exam verdict**, not correct → incorrect on answers; the
  definitions differ and a page must say which.

### 11. 2611.12634, Lu & Ni (F, §9 only)
- **§9 (GH200, H100).** vLLM batch-invariant mode "ran at 0.55× the throughput" at 32 concurrent requests (0.81× in a teacher-forced
  setting). Floor measured on Qwen3-8B, 256 samples of 768 tokens: "every comparison in which the reduction order differs ... gave
  0.17–0.22 nats per sequence"; "reruns that reuse the kernel selection were bitwise identical". **"The batch size itself barely
  matters: reading the same autotune cache, 8 and 256 concurrent sequences agree on 99.3% of the token log-probabilities (KL 0.000)."**
  Teacher-forced, not free generation. This is evidence against expecting a large batch effect, in a different runtime.
- Comparable to lane 16: no (GPU, teacher-forced).

### 12. 2610.09111, Kasa et al. (A)
- Abstract: 180 text classifiers; "Changing only the batch shape changes no labels across fp32 comparisons, yet under bf16 it moves up
  to 56.7 percentage points of predicted probability mass, with label changes concentrated at small margins"; intro: "Changing the
  padded length alone changes about 10% of predicted labels". Classifiers, GPU and CPU. Not read past the intro, so no number from it
  is used below.

### Re-read: 2511.05502 (F, cold-start passage)
- Mac Studio, Apple M2 Ultra, MLX / MLC-LLM / llama.cpp / Ollama / PyTorch MPS, Qwen-2.5 family. "Cold start time: model load +
  initialization latency, distinguished from warm-start runs." Procedure: "Each model loaded once, one warmup prompt executed ... cold-
  start and warm-start results separated"; "N=10 trials" per framework × model × quantisation × prompt type × length (whether the
  cold-start figure has 10 starts is not stated). Figure 2: "First run after process start; one-time TVM compile amortized for
  MLC-LLM"; text: "MLX in ∼31s". No stage split, no model size or file cache state for that number.

## 3. Lane 15 against the bar

**Published [F] number on Apple silicon under MLX?** One total, no stages: 2511.05502, MLX "∼31s" cold start, M2 Ultra (see section 0).
It does not answer "where does a cold start go", and nothing published answers "how much does pause-and-resume save" on a laptop.

**Departures between the brief and the sources, with the concrete change:**
1. **Definition.** Runpod's docs: a cold start is "from when Runpod starts a worker until the worker is ready to take its first job".
   The brief's "fresh start to first token" is stages 1 to 4 and **includes the first request**; Runpod's cold start ends before it
   (stages 1 to 3). Change: call stages 1 to 3 "cold start (Runpod's definition)" and 1 to 4 "fresh start to first token"; never use
   "cold start" for the sum. Stage 4 maps to what Runpod's post calls "warmup" only partly.
2. **The metric is not the one the brief opens with.** `delayTime` = queue time + cold start (docs, `optimization.md`) and, in the
   post, also image pull and capacity wait. Our run has none of those; the post says so. The brief already says "not a reproduction";
   keep that and add that `delayTime` has no laptop equivalent.
3. **n.** The post's 324 s → 91 s rests on 3 and 2 trials ("Treat the 3.5× ratio as directional"). The brief's "Runpod took a vLLM cold
   start from 324 s to 91 s" should carry "median of 3 vs 2". Our n of 10 per condition is larger, which is a strength; the page should
   say the sources are not comparable in n.
4. **FlashBoot is described three ways** (section 2, rows 4 and 5). Arm C (`SIGSTOP`/`SIGCONT` of a ready process) models the "pause"
   description only. Change: name arm C a stand-in for the Flash post's "container pausing", not for FlashBoot in general, and do not
   write "5× faster than a fresh start" as a FlashBoot claim: their only number is "about 200ms" without n or method.
5. **The 31 s figure (2511.05502).** The brief's prediction for A and B (median 5 to 12 s) sits far below it. Different hardware and
   model, so no conflict is proven, but the prediction note should cite it and say why ours might differ (4-bit e4b, warm file cache).
6. **Anchor.** The committed `load_s` (3.05 to 4.79 s) is the lab adapter's load in a running process; the page must not call it a
   cold start.

## 4. Lane 16 against the bar

**Published [F] count of batch-dependent outputs on Apple silicon or under MLX?** No (section 0). The closest: GB10 batch 1 → 8
(2611.15671: 8 and 15 flips of 2,400 decisions, confounded, a forward pass not generation); GH200 batch 8 vs 256 (2611.12634 §9:
99.3% agreement of token log-probabilities, teacher-forced); L40S/A100 (2610.09111 is [A], not used; 2506.09501 Std@Acc, reasoning).

**By construction (item 4).** `batch_generate` does **not** make an answer depend on its batch mates by construction: pads are masked
and rolled out of the window (section 2, row 7). So the brief's conditional ("padding without a mask") does not fire. **Neither arm is
decided before it runs; keep both at the stub sizes.** The remaining route to "the bytes differ" is floating-point, the one Thinking
Machines names, and it is an open measurement. Caveat: code read, not run.

**Departures, with the concrete change:**
1. **"Unique completions".** Theirs counts distinct strings among 1000 identical requests whose batch mix is uncontrolled (80, most
   common 78); the brief's 60 are a designed mix (4 solo, 56 in controlled batches). The two counts are not the same quantity. Change:
   say so in the prediction and the page; report distinct completions **per configuration** next to the pooled count.
2. **Their 1000-token, 103rd-token divergence** vs the brief's 200 new tokens. The stub's "first departure after token 20" has no
   source; their first divergence is at 103 of 1000, and a 200-token run may not reach it. Change: add that a no-difference result at
   200 tokens does not test their claim at 1000.
3. **Solo is a different code path.** The lab's adapter generates through `stream_generate` with a plain `KVCache` / `RotatingKVCache`
   (`lab/metering.py:107`, `:131`). `batch_generate` builds batch caches with array masks even at batch size 1. Change: add a
   **batch-of-1 `batch_generate`** sitting as a second reference, so "solo vs batched" is not mixed with "plain cache vs batch cache".
4. **The batch shrinks as mates finish** (`filter`, `generate.py:1383`), so "batch of 8" is not one shape; the page may only say
   "started as a batch of 8".
5. **`batch_generate` returns text only** (`generate.py:1952`); the rows need token counts. Use `BatchGenerator` directly or re-encode.
6. **Prefill batching.** `prefill_batch_size` defaults to 8, `completion_batch_size` to 32; batches of 2, 4 and 8 prefill together by
   default, but the brief should set both explicitly and write them in the rows.
7. **The "CPUs or TPUs" sentence is an assertion**, not a measurement in the post. Our run, if it shows a difference, is an Apple-GPU
   data point for it; if it shows none, the page may say "not seen here" and no more. Their mechanism is varying load, which the
   brief says it does not test.
8. **Floor.** Lane 16's kill (1) uses `results/champion-e4b-8bit.json` as the floor; the Noise Floor paper's lesson is that one repeat
   can understate restarts "by more than five times" (vLLM). Ours were fresh processes on mlx-lm only; the claim stays "0 of 83
   unstable" at those n.

## 5. A divergence lane (not queued)

- A laptop version of Fireworks' method needs, from `mlx-lm`: the reference model's generated token ids and **per-token top-16 log-probs**
  (`stream_generate` yields `GenerationResponse.logprobs`, a full-vocabulary vector per token, `generate.py:270-297`), then the
  quantised model run over the same ids with a single forward pass (teacher forcing; no generation), and KL / rejection computed on the
  top-N after renormalising. No `generate_step` change is needed; the lab's adapter does not record logprobs (no `logprob` in `lab/`).
- 16-bit reference: Gemma 4 e4b bf16 does not fit on this machine (LESSONS lane 04: "e4b and Llama bf16 (do not fit)"); e2b bf16 does.
  The reference for e4b would be its 8-bit pack, which changes what "divergence from the reference" means.
- **Lane 04 results keep verdicts and counts, not text or logprobs.** Confirmed on all four `results/quant-*.json` (249 rows each):
  row keys `exam, case, split, load, passed, failed_checks, start_utc, end_utc, wall_s, calls`; call keys `prompt_tokens, prompt_tps,
  gen_tokens, gen_tps, peak_memory_gb, finish_reason, content_chars, reasoning_chars, wall_s`. A divergence comparison against lane
  04 would compare **verdicts and token counts only**; any text or logprob comparison needs a fresh run.
- Before it is queued: Dutta's flips and Fireworks' rejection rate are different things from our verdict flips; the page must pick one
  name per metric.

## 6. What our pages may and may not claim

**May:**
- Quote Thinking Machines' count (80 unique of 1000, the most common 78; first divergence at token 103 of 1000; Qwen3-235B-A22B, one
  prompt) and their mechanism (batch-size dependence of reductions under varying load), as their claim, with the page's date.
- Quote Runpod's definition (docs) and median-of-3 and median-of-2 for their 324 s / 91 s, with "directional" in their words.
- Say that 2511.05502 reports an MLX cold start of about 31 s on an M2 Ultra with no stage split, if the operator decides lane 15 runs.
- Say that two preprints from Nov 2026 measure batch and restart effects on NVIDIA GPUs and find small (0.3 to 0.6% of decisions) to
  negligible (99.3% log-prob agreement) batch effects in a decision model and a teacher-forced run.
- Say Fireworks reports "KL divergence of < 0.007" as a rule of thumb, "generally", without a table behind it.

**May not:**
- Say batch invariance, or its absence, "holds on Apple silicon", or that "the field reports" anything for MLX: no [F] source does
  (the only MLX [F] number is a cold-start total).
- Present our count as a replication of Thinking Machines (different model, prompt, length, hardware, controlled batch).
- Call stages 1 to 4 "a cold start", or call any laptop number comparable to `delayTime`, to Runpod's 324 s or 91 s, or to a
  FlashBoot figure.
- Use the Fireworks or Noise Floor thresholds as pass marks for our packs: other models, other hardware, and the Noise Floor paper is
  one device and two models.
- Cite Kasa et al. or Lu & Ni beyond §9 for a number (ranks [A] and [F, §9 only]).
