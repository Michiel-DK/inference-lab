# Literature, full-text read (16 Nov 2026)

Lane `lab-inf-lit-full-read`. No model call, no GPU, no run; every committed result is read, none regenerated. Reader question C2;
decider: the operator, on which local model / quantisation / runtime the inference cards recommend.

**Rank rule.** [F] = PDF body read (setup, case count, metric as written, the table or figure the number comes from).
[A] = abstract only. [S] = snippet or secondary only. A published number sits beside ours only at [F]. Every number below
was read from the PDF text this session (`pdftotext`, saved outside the repo); arXiv ids were checked against the PDF's own title page.
Numbers marked **(computed)** are my arithmetic on a read formula and our posted figures, not a published number.

Before this read, `LITERATURE.md` held no paper at all: its rows are mlx-lm source, wilco snapshots, vendor cards and agent-sandbox
probes, and the closest papers were named only in `predictions/*.md` mechanisms ("rank C, not read") or not at all.

## 1. Papers (final rank)

| # | paper (arXiv, date) | rank | why it is on the list |
|---|---|---|---|
| 1 | Kurtić et al., *"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization*, 2411.02355 (v4, 26 May 2026) | **F** | the closest published 4 / 8 / bf16 study; Llama-3.1-8B is one of our three models |
| 2 | Lin et al., *AWQ*, 2306.00978 | **F** | the standard 4-bit baseline; its round-to-nearest (RTN) arm is what mlx's affine pack is |
| 3 | Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, 2211.17192 (ICML 2023) | **F** | lane 06's method and its walltime formula |
| 4 | Li et al., *EAGLE*, 2401.15077 (v3, Mar 2025) | **F** | reports vanilla draft-model speculative sampling as a baseline, per target size |
| 5 | Cai et al., *Medusa*, 2401.10774 | **F** | the other standard speculative baseline; says which variants are lossless |
| 6 | Xiao et al., *StreamingLLM*, 2309.17453 (ICLR 2024) | **F** | explains why a rotating cache with `keep` = 4 (lane 13) collapses or survives |
| 7 | Liu et al., *KIVI*, 2402.02750 | **F** | the standard KV-cache quantisation study; lane 05's kv8 / kv4 arms |
| 8 | Jiang et al., *Mixtral of Experts*, 2401.04088 | **F** | the primary source for "active parameters = compute, total = memory" (lane 03) |
| 9 | Adhinarayanan & Jayasena, *The qs Inequality*, 2603.08960 (9 Mar 2026) | **F** | the recent MoE-at-inference-cost paper; an analytic model, kept to show where it stops |
| 10 | Rajesh et al., *Production-Grade Local LLM Inference on Apple Silicon* (MLX, MLC-LLM, Ollama, llama.cpp, PyTorch MPS), 2511.05502 | **F** | the only arXiv report of MLX on Apple silicon with a stated method |

Fetched and **not read**, so not ranked and not used: GPTQ 2210.17323, Chen et al. speculative sampling 2302.01318, vllm-mlx 2601.19139
(its text is on disk, I did not read it; a search snippet says it compares against mlx-lm and llama.cpp on an M4 Max, not enough for any claim).
Not found / not attempted: a 2025–26 low-bit study on Gemma-class models, an Apple-silicon MoE study, a MLX-vs-GGUF *accuracy* comparison. The
absence is a finding for the card ("no [F] source covers this"), not proof none exists.
Not checked: that no paper in the last 90 days cites two of the above and reports the same metric (one standard search per topic was run; none surfaced that I read).

## 2. Paper table

Metric definitions are the paper's. "Comparable" is judged against our nearest run.

### 1. 2411.02355 (F)
- **Setup.** Llama-3.1-Instruct 8B / 70B / 405B (plus DeepSeek-R1-Distill Llama and Qwen, §4.3). vLLM 0.6.4.post1 on A6000 / A100 / H100. Formats: W8A8-FP (RTN, dynamic per-token activations), W8A8-INT (GPTQ weights + dynamic activations, SmoothQuant at 70B), W4A16-INT (GPTQ with MSE-optimal clipping, group 128, calibrated on OpenPlatypus). Calibration data matters: random tokens "degrade accuracy" at INT4 (§3.2).
- **n.** "Over 500,000 evaluations": Open LLM Leaderboard V1 (8 tasks) and V2 (6), Arena-Hard-Auto-v0.1 (500 prompts, LLM judge, two runs averaged), HumanEval/+, RULER; reasoning sets sampled at T = 0.6, 20 responses per query. Text-similarity study (ROUGE-1/L, BERTScore, STS) at greedy on Arena-Hard prompts.
- **Metric.** "Recovery %" = quantized average / BF16 average.
- **Numbers.** Table 2, Llama-3.1-8B, V1 average: BF16 74.06, W8A8-FP 73.55, W8A8-INT 74.29, W4A16-INT 73.11 (98.72 % recovery). Table 3, 8B, V2 average: 27.6 / 27.9 / 28.0 / 26.5 (W4A16 96.1 %); Arena-Hard 25.8 / 26.8 / 27.2 / 24.0; HumanEval 67.3 / 67.3 / 67.1 / 67.1. Table 1: GPTQ vs AWQ 4-bit on 8B, real-world average 52.3 vs 49.4. Fig. 1 / §4.4: 8B quantized output vs BF16 ROUGE-1 0.62, ROUGE-L 0.46 (larger models 0.7 / 0.56): same meaning, different words. §5, Tables 5–6: W4A16 is the cheapest per query synchronously, W8A8 under continuous batching (A6000 / A100 / H100, seven use cases).
- **To match.** Same quantiser (calibrated GPTQ g128), same engine, benchmark averages over thousands of items.
- **Comparable to our lane 02 / 04 rows?** **No for the number, yes for the question.** Different quantiser (ours has no calibration, see 2), different engine and hardware, aggregate over thousands of items vs 55 held-out cases with exact-field scoring.

### 2. 2306.00978 AWQ (F)
- **Setup.** Weight-only grouped quantisation, group 128; primary baseline "vanilla round-to-nearest (RTN)… actually quite strong when using a small group size like 128"; calibration set from the Pile; metric WikiText-2 perplexity.
- **Numbers.** Table 4 Llama-2-7B INT4-g128: FP16 5.47, RTN 5.73, GPTQ 5.69, AWQ 5.60; INT3-g128: RTN 6.66, AWQ 6.24. Table 5 Mistral-7B-Instruct INT4-g128 4.30 vs FP16 4.14; Mixtral-8x7B 6.05 vs 5.94.
- **What our packs are (our side, checked in this env).** `mx.quantize(weight, group_size, bits, mode="affine")` takes no calibration data (mlx 0.32.2 `QuantizedLinear.__init__`), so the mlx-community affine packs are round-to-nearest-class. Group size of the specific packs was **not** checked (mlx-lm's `convert` recipes default 64).
- **Comparable to lane 02/04?** **Partly.** Same quantiser class as AWQ's RTN arm; its metric is perplexity on one model family, ours is pass count.

### 3. 2211.17192 Leviathan (F)
- **Setup.** T5-XXL 11B target; T5-small 77M / base 250M / large 800M as draft; EnDe translation and CNN/DM; batch 1, one TPU-v4; temperature 0 and 1. LaMDA 137B and a 97M GPT-like for α only.
- **Definitions.** α = E[min(p, q)] per token; c = ratio of the time of one draft step to one target step; expected walltime improvement (1 − α^(γ+1)) / ((1 − α)(γc + 1)) (Thm 3.8). In their runs "c was always less than 0.05".
- **Numbers.** Table 2, T = 0, EnDe: small γ=7 α=0.75 3.4×, base 0.80 2.8×, large 0.82 1.7×. CNN/DM T = 0: α 0.65 / 0.73 / 0.74, 3.1× / 3.0× / 2.2×. T = 1 is lower (2.6×, 2.4×, 1.4× on EnDe). The 800M draft is 7 % of the target.
- **Guarantee.** "identical outputs", an algorithm-level (distribution) guarantee; the paper says nothing about a cache that cannot rewind.
- **Comparable to lane 06?** **Formula yes, numbers no.** Our draft is ~70 % of the target's bytes, the paper's best is 1 %–7 %.

### 4. 2401.15077 EAGLE (F)
- **Setup.** Vicuna 7B/13B/33B, LLaMA2-Chat 7B/13B/70B, Mixtral-8x7B-Instruct; MT-bench, HumanEval, GSM8K, Alpaca; batch 1 on RTX 3090 / A100; the draft head is trained (ShareGPT, 68k dialogues).
- **Definitions.** Walltime speedup vs vanilla decoding; τ = mean tokens accepted per target forward pass; α measured as n-α.
- **Numbers.** §4.1: vanilla speculative sampling with a 7B draft: **no speed-up on the 13B target, 1.12× on 33B, 1.88× on 70B** (draft/target size 0.54 / 0.21 / 0.10). EAGLE itself 2.7×–3.5× at T = 0 on the dense models; Mixtral 1.5× ("modest… complexity of accelerating MoE", §4.2).
- **Guarantee.** "evaluating the quality of EAGLE's generated results is both unnecessary and meaningless" given the distribution guarantee (§3.2): no downstream-quality check was run.
- **Comparable to lane 06?** **No for the method** (trained head), **yes for the draft-size effect** (the 13B row).

### 5. 2401.10774 Medusa (F)
- **Setup.** Vicuna 7B/13B/33B and Zephyr-7B; MT-Bench judged by GPT-4; batch 1 ("for simplicity"); trained decoding heads, tree attention.
- **Numbers.** Table 2 (Vicuna-7B, quality = MT-Bench score with GPT-4 as judge): baseline 6.17; Medusa-1 (frozen backbone, "lossless") 6.23 at 2.18×; Medusa-2 (backbone fine-tuned with heads) 6.18 at 2.83×; direct fine-tuning 5.925. Typical-acceptance sampling is explicitly not distribution-preserving (§2.3.1); at T = 0 it reverts to greedy decoding. I did not read their SpecDecoding-vs-Medusa speed column (Table 1) closely enough to cite it.
- **Comparable to lane 06?** **No.** Needs trained heads; mlx-lm has no such path; nothing of ours ran it.

### 6. 2309.17453 StreamingLLM (F)
- **Setup.** Llama-2-7/13/70B, MPT, Falcon, Pythia; perplexity on PG19 (65K-token book; 400K concatenated for Table 2), streaming up to 4M tokens; StreamEval retrieval (Fig. 9, Table 7, 100 samples per point); ARC concatenated question stream (Table 5).
- **Definitions.** "Window attention" = only the most recent L tokens' KV; "cache config x+y" = x initial + y recent tokens.
- **Numbers.** Table 1, Llama-2-13B on 65K tokens: window 0+1024 PPL 5158.07; 4+1020 5.40; 4 "\n" tokens in place of the original four 5.60. Table 2: 0+4096 on Llama-2-7B 3359.95; 4+4092 9.59. One or two initial tokens do not fully recover, four do (their models were not trained with one fixed start token).
- **Their stated limit (Limitations; Table 7).** "does not extend the models' context window or enhance their long-term memory"; StreamEval accuracy is good when the query-answer distance is within the cache and "drops to zero when it surpasses the cache size".
- **Comparable to lane 13?** **Mechanism yes, setup no.** Same cache shape (4 initial + recent window); different task (perplexity on a continuous text vs an instruction at the head of a prefilled prompt).

### 7. 2402.02750 KIVI (F)
- **Setup.** Llama-2-7B/13B, Falcon-7B, Mistral-7B; lm-eval CoQA (exact match), TruthfulQA (BLEU), GSM8K (exact match); LongBench; NIAH. HF Transformers + CUDA; group size 32, residual full-precision window R = 128; A100 80 GB for Fig. 5.
- **Numbers.** Table 3, Llama-2-7B: 16-bit CoQA 63.88 / TruthfulQA 30.76 / GSM8K 13.50; 4-bit per-token (K,V) 64.82 / 29.85 / 12.28; KIVI-2 63.05 / 33.95 / 12.74. Their claim "2.6× less peak memory (including model weight)", "up to 4× larger batch", 2.35×–3.47× throughput (Fig. 5), where the KV cache dominates (their OPT-175B example: b = 512, l = 512 gives a KV cache 3.8× the weights).
- **Comparable to lane 05?** **No on setup** (fake-quant per-token, group 32, CUDA, MHA Llama; ours mlx-lm `kv_bits` group 64, one hybrid-conv model, batch 1).

### 8. 2401.04088 Mixtral (F)
- **Setup.** 8 experts, 2 per token per layer; 47B total, 13B active; compared with Llama-2 7B/13B/70B on standard benchmarks (Table 2, Fig. 3).
- **Statement we use (§3, "Size and Efficiency").** Active parameter count "is directly proportional to the inference compute cost, but does not consider the memory costs and hardware utilization. The memory costs for serving Mixtral are proportional to its sparse parameter count, 47B"; the routing adds overhead and the MoE layer is "more suitable for batched workloads".
- **Comparable to lane 03?** **Yes as a statement of the claim**, no speed number to compare (the paper reports none).

### 9. 2603.08960 qs inequality (F)
- **Setup.** An analytic decode-cost model (no measurement). DeepSeek-V3, Qwen3-235B-A22B, Grok-1, Switch-C against quality-matched dense baselines, 128k context, a reference cluster with 8 TB/s HBM; "All throughput results are evaluated at batch size B = n_eff,max" (the largest batch the KV budget allows).
- **Definitions.** s = active fraction; q = factor by which a dense model must grow to match the MoE's quality (taken from Table 1 as input, not measured); "reuse fragmentation": expert-local batch ≈ B·k/E.
- **Headline.** 4.5× throughput for a quality-matched dense model on DeepSeek-V3 at 128k.
- **Comparable to lane 03?** **No.** Cluster HBM, saturated batches, assumed q; says nothing about batch 1 on one 16 GB device, and its own scope sentence is "HBM-centric".

### 10. 2511.05502 Apple-silicon runtimes (F)
- **Setup.** One Mac Studio M2 Ultra, 192 GB. MLX v0.26, MLC-LLM, llama.cpp b5963, Ollama 0.10.1, PyTorch MPS. Qwen-2.5-Coder-3B primary, Qwen-2.5-7B-Instruct; prompts 1k–100k tokens; N = 10 trials per cell.
- **Metrics.** TTFT (client-side), decode throughput = generated tokens / decode time after the first token, end-to-end throughput, p50/p90/p99 latency, peak RSS.
- **Numbers.** Throughput ranking: MLX ~230 tok/s, MLC-LLM ~190, llama.cpp ~150 (short context only), Ollama 20–40, PyTorch MPS 7–9 (Fig. 1 and "Comparative Ranking"). AWQ 4-bit 3B on MLC: ~1.6 GB, ~210 vs ~190 tok/s FP16, with "no noticeable accuracy loss" claimed.
- **Gaps in the paper.** No accuracy or pass metric is measured anywhere, so "no noticeable accuracy loss" is asserted, not shown; one machine, one model family, no energy figure.
- **Comparable to our rows?** **No.** M2 Ultra / 3B / 230 tok/s against an M1 Pro / 16 GB / Gemma e4b / 29–42 tok/s; throughput only.

## 3. Our results beside the field

Our numbers are from `LESSONS.md` / `SHOWCASE.md` and are re-derivable with `$P -m lab check results/<run>.json`; I re-derived none (no run was needed, and the rule is that a lane reads them).

**Lane 02 / 04, 4-bit vs 8-bit vs bf16** (`dense-*`, `quant-*`; held-out /55, 1 pass at T = 0, 0 of 83 unstable). Ours: 4 → 8 on e2b 23 → 32, e4b 36 → 41, Llama-3.1-8B 31 → 37; 16 / 16 / 14 flipped ids, held-out pooled 26 up / 6 down; e2b bf16 28.
Field [F]: Llama-3.1-8B W4A16 (GPTQ g128, calibrated) recovers 98.72 % of BF16 on 8 tasks (Table 2) and 96.1 % on V2 (Table 3); W8A8 about 99–100 %; INT4 is "competitive… rivaling 8-bit" (abstract); AWQ Table 4 puts RTN g128 behind AWQ and GPTQ.
Differences: quantiser (no calibration vs calibrated GPTQ), engine and hardware, metric (exact-field pass on 55 cases vs thousands of items), instruction-tuned Gemma 4 not in any paper.
*Can claim:* on mlx-community affine packs under mlx-lm, at T = 0, on these 55 cases, 8-bit passes more than 4-bit on all three models (a two-sided exact sign test on the 32 pooled discordant ids gives p ≈ 0.0005 **(computed)**, but the ids are pooled over three models and so not independent). *Cannot claim:* that "4-bit costs 5–9 cases" or "INT4 is lossy" for 4-bit generally, that GPTQ/AWQ 4-bit would behave the same, or that our gap contradicts the field's ~1–4 % (the two measure different quantisers and metrics). The lab's own "bf16 below 8-bit" (−4 on e2b) is likewise one model at n = 55 against a field that finds 8-bit within ±1 % of BF16 (Table 2): say "not above 8-bit here", not "worse".

**Lane 05, KV-cache quantisation** (`ctx-lfm2.5-…-8k-kv{8,4}`; LFM2.5-8B-A1B, 55 held-out; kv8 14 (+3/−3), kv4 18 (+4/−0), repeat 0 moved; peak 5.90 → 6.94 / 6.91 GB).
Field [F]: KIVI Table 3: 4-bit per-token KV on Llama-2-7B changes CoQA +0.94, TruthfulQA −0.91, GSM8K −1.22 vs 16-bit; memory saved at large batch × long context where KV dominates (Fig. 5).
Differences: fake-quant vs real, group 32 vs mlx's 64 (`kv_group_size` default, `generate.py`), only 6 of LFM's 24 layers are attention (the 18 conv caches have no `to_quantized`), batch 1, benchmark averages vs one-sitting verdicts.
*Can claim:* quantised KV at 4 or 8 bits moved a handful of verdicts (4–6 of 55) against a repeat that moved none, on one hybrid model; that is the size of effect KIVI reports at 4-bit. *Cannot claim:* anything about 2-bit KIVI, about memory savings (KIVI's saving needs a KV-dominated regime; ours is not one, and our peak rose), or a direction (our +4/−0 and +3/−3 are 55 cases).

**Lane 05 / 13, `max_kv_size`** (`ctx-qwen3-4b-…-maxkv4096`; control 27 → 0/55 held-out, 39 of 39 email + call sittings fail with wilco's "no JSON object found", 0 of 55 moved on the repeat; cache stamped `RotatingKVCache` × 36, `keep` = 4 by mlx-lm source).
Field [F]: StreamingLLM: evicting the initial tokens collapses window attention (Table 1: PPL 5158 vs 5.40 with 4 + 1020); four initial tokens restore stability; "does not extend… long-term memory"; retrieval drops to zero past the cache (Table 7).
Differences: perplexity on a continuous stream vs an instruction placed at the head of a prefilled prompt; the paper's `keep` = 4 retains the sink's *position* (Table 1, "\n" tokens work as well), not the system prompt's content.
*Can claim:* our result is the paper's stated limit, not a contradiction of its sink result: a 4-token sink keeps the model stable, the instructions are gone, and the replies are prose without the JSON contract (consistent with, not proof of, the mechanism; no attention was inspected). *Cannot claim:* that the sink fraction explains the failure, or that the class (not Qwen3-4B) is general (lane 13's own scope).

**Lane 06, speculative decoding** (`spec-*`; Gemma 4 e4b-4bit target, e2b draft, k = 3; ×0.494 tok/s, 81 of 83 outputs differ, held-out 36 → 8, 37 verdict flips all past 512 tokens; draft ≈ 70 % of target bytes; Qwen2.5-Coder self-draft at 100 % acceptance ran ×0.575).
Field [F]: Leviathan Thm 3.8 and Table 2 (the best draft is 1–7 % of the target, 1.7×–3.4× at T = 0); EAGLE §4.1: a 7B draft for a 13B target gives no speed-up, 1.12× for 33B, 1.88× for 70B.
A formula check **(computed)**: with c = 0.7 (our byte ratio standing in for the time ratio) and γ = 3, Thm 3.8 gives 0.60 / 0.70 / 0.82 / 0.95 / 1.11× for α = 0.5 / 0.6 / 0.7 / 0.8 / 0.9. If our "acceptance 0.625" (share of output tokens that came from the draft, SHOWCASE lane 06) is read as 1.67 accepted draft tokens plus one bonus token per round of 3, it gives 2.67 / 3.1 = 0.86×. We measured 0.494, so the 0.86× is an upper bound the measurement misses by ~0.57×; that is the "per-round overhead beyond the steps" that the lane's own Qwen self-draft control (×0.575 at 100 % acceptance, formula ≈ 1.0) already showed. Caveat: our acceptance is not the paper's α and c is a proxy, so the 0.86 is an illustration, not a prediction.
*Can claim:* a draft at ~70 % of the target's bytes does not pay on this box, which both papers' own formula and EAGLE's 13B/7B row also say; and the output-exactness guarantee did not hold *in mlx-lm 0.31.3 with a sliding-window cache past its window* (our evidence). *Cannot claim:* that speculative decoding fails on Apple silicon (one pair, one engine, no draft ≤ ¼ of the target ran; the lab's own list), or that the papers' guarantee is wrong (it is a distribution-level guarantee of the algorithm; none of the papers checks a cache-rewind implementation or fixed-answer verdicts). The Medusa / EAGLE speed-ups need trained heads that mlx-lm does not run, so none of those numbers is a baseline for us.

**Lane 03, MoE vs dense** (`moe-lfm2.5-…`, `moe-gate-gpt-oss-…`; LFM2.5-8B-A1B: tok/s 94.7 vs e2b 69.2, peak 5.89 GB vs 3.71 and 6.30, held-out 21 vs 23 and 31; gpt-oss-20b 2.91 tok/s, 12.35 GB over the 11.45 GB working set).
Field [F]: Mixtral §3: active parameters track compute, memory tracks total, routing adds overhead (statement, no speed number); 2603.08960: analytic only, HBM clusters at saturated batch.
*Can claim:* "memory like the total, per-case cost like the active size" is the Mixtral paper's own caveat and matched on our two MoEs. *Cannot claim:* anything for or against the qs inequality (its regime is not ours), or that MoEs are slower or faster on Apple silicon in general (two models, one is a different family with a thinking channel).

**Lane 09 hosted, lane 10 champion, flow arm.** No [F] paper in this list measures hosted Claude or a Q8_0 vs Q4_K_M GGUF flow task, so nothing here is compared. The champion's house rule (8-bit where it fits) rests on lane 04 only; the field's "INT4 rivals 8-bit" (Kurtić) was measured on calibrated GPTQ, so the README's scope line ("measured on mlx-community affine packs under mlx-lm only") is the right size and should stay.

**Throughput figures.** Our 29–42 tok/s for 4B-class models on an M1 Pro is not comparable to the ~230 tok/s of 2511.05502 (M2 Ultra, 3B, other bandwidth), and that paper measures no accuracy; the README makes no cross-machine throughput claim and none should be added.

## 4. Open-question check

Open or queued items for this lab (`queues/labs.json`, cwd inference-lab) with no result: **lab-inf-07-rented-box** (held by the operator on 16 Nov; `predictions/rented-box.md`, no results file). Lanes 11 (thinking off) and 12 (speculative on a Llama pair) were dropped on 12 Nov and are not re-opened by anything here.

- **lab-inf-07-rented-box.** Not answered by a published result: no [F] paper covers an L4 with these exams. One design point falls below the Kurtić bar (§5, Tables 5–6, A6000 / A100 / H100): W4A16 is cheapest per query for *synchronous* serving, but W8A8 (FP8 on Ada / Hopper) wins under continuous batching, which is exactly the "16–32 concurrent" regime the prediction's throughput claim rests on. The prediction names AWQ/GPTQ 4-bit for the 14B and bf16 for the e4b and no FP8 arm. Concrete change for the operator to make openly, with a dated note, before any go: add an FP8 arm (L4 is Ada, which has FP8 support; ⚠️ not checked against the vLLM release the lane would use) or state why not, and report a synchronous and a concurrent cost line separately, as the paper does.
- **A dropped lane's decider.** Lane 12's brief (spec decoding on a Llama pair) would now start from EAGLE's 13B/7B no-gain row: a 1B draft for an 8B target is 0.125 of the bytes, within the range where papers report gains (EAGLE 33B/7B 0.21 → 1.12×; 70B/7B 0.10 → 1.88×). It is dropped, so this is a note only: nothing published already answers it for mlx-lm, and the paper rows above do not substitute for the run.
- **Design below the field bar, for any future lane (not edited here).** (a) Quantisation lanes are single-pass on 55 cases with one quantiser class; the field varies the quantiser (GPTQ vs AWQ vs RTN, Table 1 here and AWQ Table 4) and reports confidence intervals (Kurtić Table 4, ±). (b) KV lanes: KIVI's full-precision residual window (R = 128) is the part that matters for GSM8K-type tasks (§4.1 there) and mlx-lm's `quantized_kv_start` is the nearest knob, never varied in lane 05. (c) Speculative: the field reports α in the paper's definition and acceptance length τ; ours reports the share of output tokens from the draft. A lane that wants to set its numbers beside Leviathan's formula needs per-round accepted counts.

## 5. What our pages may and may not claim

**May:**
- On mlx-community affine 4-bit and 8-bit packs under mlx-lm 0.31.3, at T = 0, on the three pinned wilco exams, 8-bit passes more held-out cases than 4-bit on all three models measured (lane 04 numbers, `lab check`).
- Our affine packs are calibration-free (round-to-nearest class); the field's near-lossless 4-bit results are for calibrated GPTQ/AWQ with group 128, so the two are not the same experiment.
- Quantised KV at 4–8 bits moved a few verdicts on one hybrid model; no more than that.
- `max_kv_size` on a class without `make_cache` evicts the head of the prompt, and the StreamingLLM paper's own limitation (a sink keeps stability, not content) matches the observed failure.
- A draft that is ~70 % of the target's bytes did not pay (×0.494); EAGLE's own baseline for a 13B target with a 7B draft is also no gain.
- "Memory like the total, speed like the active size" for MoEs on 16 GB, with Mixtral's text for the same distinction.

**May not:**
- "The field reports 4-bit loses N %", "state of the art", "matches" or "beats" with any paper number, except as in section 3 where setups are named.
- That our 4 → 8 gap refutes or confirms Kurtić et al.; different quantiser, metric and engine.
- Any statement about speculative decoding, KV quantisation or MoE on Apple silicon as a class: each is one model pair or one family.
- A throughput comparison with 2511.05502, or "MLX is fastest" as our finding (we ran mlx-lm only; that paper runs one machine and no accuracy).
- That a published speculative speed-up (Medusa, EAGLE) is available in mlx-lm.
- That the speculative output-exactness guarantee is false; only that this implementation broke it past a sliding window.
