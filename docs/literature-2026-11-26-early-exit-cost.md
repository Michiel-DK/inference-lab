# Literature, full-text read: early exit and cost per correct answer, behind lanes 18 and 19 (26 Nov 2026)

**Who this is for, and where it is used.** The operator, for one decision: whether `lab-inf-18-early-exit-settle` and
`lab-inf-19-cost-per-correct` run as briefed, with changed arms, or not at all. It sits in this public lab's `docs/` beside
[`literature-2026-11-16.md`](literature-2026-11-16.md) and [`literature-2026-11-24.md`](literature-2026-11-24.md), which have no row on early
exit, layer-wise prediction or cost per correct answer. Nobody outside asked for it; the lab's rule of 16 Nov puts the full read before a
prediction. Reader question C2 (lane 18) and B5 (lane 19). No model was loaded, no GPU used, no key used, nothing run but `curl`,
`pdftotext`, `gh api` read calls, and a few lines of arithmetic (section 5). This read changes no card.

**Rank rule** (as 16 Nov). [F] = body read, setup and metric as written, the table or section the number comes from. [A] = abstract only.
[S] = snippet only. A published number sits beside ours only at [F]. PDFs and repo files were fetched into `/tmp/lab-inf-17-lit-early-exit-cost/`
and are not committed. Every quotation below in straight double quotes was matched against the fetched text this session (whitespace
and typographic apostrophes normalised; the PDF text of TIDE has a spurious space in "T IDE").

## 0. First lines: does a published [F] result already report a lane's number?

- **Lane 18 (settle layer and oracle saving, Gemma 4 e2b, MLX): no.** No [F] row here reports a settle layer, an oracle layer saving or an
  early-exit rate on Apple silicon, on Gemma 4, or on a ~2B model on short structured tasks. The closest numbers are CALM's 8-layer T5 on
  TPUv3 (1.53 layers per token for an oracle), TIDE's 8B models on an A100 (16 of 322 prefill tokens exit before the last layer), and the
  tuned lens's Pythia 12B. The stop rule in the brief ("an [F] number for settle layers on a Gemma 4 or a ~2B model on short structured
  tasks") **does not fire**. Limit: the six papers named plus the code rows were read; TIDE cites `2412.01455` (Shan et al., early exit
  in pretrained LLMs without joint training), which was **not** read and is not ranked.
- **Lane 18, the brief's own corrections: two fire in part.** (a) **TIDE's "98 to 99% of tokens exit early" is a count of exits at the
  last layer** (section 2.1): it skips no layer. (b) **The lane's `cos(h_l, h_34)` rule is TIDE's training label, not TIDE's exit rule, and not
  CALM's "state saturation"** (sections 2.1, 2.2). The decider's second flip ("a source defines the metric differently from the brief")
  fires in its soft form: the brief should be amended openly, with a dated note, before its prediction is committed. Details and the
  concrete changes in section 3.
- **Lane 19 (hosted against local per correct answer with a review cost): no.** No [F] row prices a wrong answer or compares a hosted
  model with a local one per correct answer. The flip does not fire. Neither paper defines "cost per correct answer", so the second
  flip does not fire either; the brief's `r*` is a quantity of its own and should be named that way (section 4).

## 1. Sources and final rank

| # | source | rank | why |
|---|---|---|---|
| 1 | Jaber & Jaber, *TIDE*, `2603.21365` v1 (22 Mar 2026) | **F** | lane 18's origin |
| 2 | `RightNow-AI/TIDE`, README, `python/TIDE/{config,calibrate,runtime,router,scheduler,utils,__init__}.py`, `adapters/{base,universal,auto}.py` at default-branch head `8bdc187` (Apache-2.0) | **F** (code read, nothing run) | the label, spacing, threshold, KV handling |
| 3 | Schuster et al., *CALM*, `2207.07061` (v2, 25 Oct 2022; submitted 14 Jul 2022) | **F** | the standard early-exit baseline and its confidence measures |
| 4 | Elhoushi et al., *LayerSkip*, `2404.16710` (v4, 18 Oct 2024; submitted 25 Apr 2024) | **F** | why early exit without training degrades |
| 5 | Belrose et al., *Tuned Lens*, `2303.08112` (**v6, 11 Nov 2025**; submitted 14 Mar 2023) | **F** | what the plain logit lens understates |
| 6 | Kapoor et al., *AI Agents That Matter*, `2407.01502` v1 (1 Jul 2024) | **F** | cost-controlled evaluation, accuracy-cost Pareto |
| 7 | Chen, Zaharia & Zou, *FrugalGPT*, `2305.05176` v1 (9 May 2023) | **F** | cascades, cost against accuracy |
| 8 | `mlx-lm` 0.31.3 `models/gemma4_text.py` and the cached `mlx-community/gemma-4-e2b-it-8bit` `config.json` | **F** (code read, nothing run) | the KV-sharing boundary |
| 9 | `RightNow-AI/inference-cost-truth`, README, `LICENSE-DATA` (head `a61c501`) | **F** (README and licence file only; the data files were not read) | what a row is, its date, its licence |

All six papers are [F]: **6 of 6**, the brief asks for 5. Nothing was unfetchable. `curl -sL https://arxiv.org/pdf/<id>` returned each PDF;
page counts and text were checked by `pdftotext`. The tuned lens file read is v6 (11 Nov 2025), not the 14 Mar 2023 version; numbers below
are v6's.

## 2. Source table

### 2.1 TIDE, `2603.21365` (F) and its repository (F)

- **Setup.** Routers (two-layer MLP, bottleneck 128, ~0.5M parameters, Eq. 2) at every 4th layer, `{c-1, 2c-1, ...}` with `c = 4`
  (§3.1), calibrated on 2,000 WikiText-103 texts with label `y = 1[cos(h_k, h_L) > τ]`, `τ = 0.98` (§3.1, Eq. 1). One A100-SXM4-40GB,
  bf16, PyTorch 2.10, transformers 5.3. DeepSeek R1 Distill 8B (32 layers) and Qwen3 8B (36 layers); GPT-2 (12 layers) only in the
  calibration count. Evaluation on "16 prompts: 8 reasoning/math and 8 general knowledge" (§5), latency over 20 runs after 3 warmups.
- **What the system does at inference, in the paper's own words (§3.3, §6).** "all layers run on every step, so the KV cache is always
  fully populated." The router scores are evaluated after the forward pass and the logits come from the earliest layer whose score passes
  θ. The limitation section says it plainly: it "does not achieve wall-clock layer skipping", and "A true skip mode (physically
  bypassing layers) would require managing cache discontinuities and is a target for future work." **TIDE as published skips no
  layer.** Its latency gain is credited to something else: "The improvement comes from the post-hoc output selection: exited tokens
  bypass the final-layer normalization path" (§5.2).
- **Headline numbers, the abstract page against the full text.**

  | abstract-page claim | full text |
  |---|---|
  | prefill latency −7.2% | Table 4: −7.2% is **θ = 0.50** (39.08 → 36.26 ms); at the default **θ = 0.85 it is −5.5%** (36.94 ms), and −5.7% on Qwen3. 20 runs (README: a single reasoning prompt). |
  | throughput +6.6% single batch | Table 4: DeepSeek R1 BS=1, 973 → 1,037 tok/s = +6.6%. The README says +6.5% for the same row. |
  | +8.1% at batch 8 | Table 4: **Qwen3 8B** only (1,781 → 1,926 tok/s). **DeepSeek R1 at batch 8 is −16.3%** (8,668 → 7,252 tok/s); the abstract omits it, §5.2 says "likely because the output_hidden_states overhead grows superlinearly with batch size". |
  | "98–99% of tokens exit early" in decoding | Table 5: exit rate 98.4% (θ 0.85), 99.2% (0.70), 99.6% (0.50), **exit layer L31 in every row**. With 32 layers numbered 0 to 31 (checkpoints 3, 7, ..., 31), **L31 is the last layer**. |
  | "100% prefill exit rate (5% of tokens exit at layer 11, the remaining at layer 31)" | Table 3: DeepSeek 16 of 322 tokens at L11, 306 at L31, at both θ. Qwen3 8B at θ 0.85: all 155 tokens at **L35 = its last layer**; at θ 0.50: 11 at L11, 5 at L23, 139 at L35. |
  | one math example as the quality check | §5.3, Table 5: one word problem, 256 tokens, temperature 0, one model. "Unique Tokens" 99 (off) against 95 (θ ≤ 0.85); "Correct" ✓ in every row. No baseline text comparison, no benchmark number. |

- **The last-layer finding (my reading of §3.1, §5.5 and `calibrate.py`; code read, not run).** `calibrate.py:35` builds
  `checkpoint_layers = range(c-1, n_layers, c)`, which for 32, 36 and 12 layers ends at the **last** layer index (31, 35, 11).
  `calibrate.py:65` reads a checkpoint as `all_hidden[layer_idx + 1]` and `calibrate.py:73` the "final" state as `all_hidden[-1]`; for the last checkpoint
  these are the same tensor, so its cosine to the final state is 1 by construction. That is how §5.5 can report "100% of tokens converge at
  the penultimate checkpoint in every model tested" (the paper's "penultimate checkpoint" is the last layer). **Consequence: every decode
  "exit" in Table 5, and the 306 of 322 and 155 of 155 prefill exits at the last layer, skip zero layers. The only exits before the last
  layer in the paper are 16 of 322 prefill tokens (5.0%) on DeepSeek at both θ, and on Qwen3 at θ 0.50 only 16 of 155 (10.3%); there are
  none in decoding, and none on Qwen3 at the default θ.** Table 5's "95 unique tokens against 99" appears with every exit at the last
  layer, so the output changed with no layer skipped; the paper does not explain it. ⚠️ UNVERIFIED (not run here): `runtime.py:288`
  applies `_final_norm` to the exit layer's `hidden_states` entry, and for the last layer that entry may already be the normed state.
- **Code rows, at head `8bdc187` (committed 18 Mar 2026).**
  - Label: `config.py:19` `convergence_threshold = 0.98`; `calibrate.py:89-100` `labels = (cos(h_l, h_final) > threshold)`. Matches the README
    ("cosine_sim(layer_8, layer_32) > 0.98?").
  - Spacing: `config.py:11` `checkpoint_interval = 4`; `calibrate.py:35` as above.
  - **Exit rule at inference is the router, not the cosine:** `config.py:12` `exit_threshold = 0.85` is a router probability
    (`runtime.py:174`, `scores > exit_threshold`). The cosine 0.98 is only the training label. The README FAQ blurs the two: "the router only
    exits tokens whose hidden state is >98% similar to what the final layer would produce"; no code makes that a guarantee.
  - **`config.py:13` `min_layers = 8` and `runtime.py:169, 277`: checkpoints below index 8 are skipped**, so checkpoints 3 and 7 never
    exit at default settings. On 32 layers the first live checkpoint is 11.
  - **KV cache of an exited token in later layers: nothing happens, because the layers run.** `runtime.py` `_generate_with_skipping` calls
    the whole model each step (`output_hidden_states=True`) and only picks which layer's state feeds the LM head; the cache is complete.
    `config.py:20` `kv_cache_strategy = "zero_pad"` and `runtime.py:195` `_pad_kv_cache` (zero-fills later layers' K/V for a skipped token)
    belong to the future skip mode. **`_pad_kv_cache` has no call site in the 11 files fetched** (`config`, `calibrate`, `runtime`, `router`,
    `scheduler`, `utils`, `__init__`, `adapters/{base,universal,auto}`); `adapters/{llama,mistral,qwen}.py` were not fetched.
  - The repo ships `benchmarks/modal/bench_quality.py` (MMLU, GSM8K, ARC at thresholds 0.7 to 0.95, 500 samples by default) but neither the
    paper nor the README reports a number from it.
  - Hygiene: Apache-2.0 per the GitHub API. The API's `pushed_at` (18 Sep 2026) is later than the default-branch head's commit date
    (18 Mar 2026); the repository has a second branch `wip/TIDE-2026-09-17`, which may be the push. Not resolved further. The PDF footer
    reads "37th Conference on Neural Information Processing Systems (NeurIPS 2023)" on a 2026 preprint: a template leftover; no venue is
    cited here.
- **Comparable to lane 18: no.** A100, 8B models, CUDA, 16 prompts, post-hoc, no layer skipped. TIDE's number is an exit count at a label
  that is trivially true at the last layer; lane 18's oracle is a different quantity (section 3).

### 2.2 CALM, `2207.07061` (F)

- **Setup.** 8-layer T5 1.1 encoder-decoder, trained with a weighted early-exit loss (§3.4), TPUv3; CNN/DM (output 82 tokens), WMT15 EN-FR
  (39), open-book SQuAD 1.1 (5) (Table 1). Calibrated risk guarantees at 95% (§4, §6.1). Metric (§5): "Our main efficiency metric is the average
  number of decoder layers used per output token"; speedup is a TPUv3 estimate over 200 examples, batch size one (App. C.1).
- **Three confidence measures (§3.5).** Softmax response: top-1 minus top-2 softmax margin at layer i. **Hidden-state saturation:**
  "we take the cosine similarity" between layer i's state and layer i−1's for the same token (`sim(d_t^i, d_t^{i-1})`, i > 1). Early-exit
  classifier: a learned linear classifier on the state. **Saturation is cosine to the previous layer, not to the final layer.**
- **State propagation (§3.3.1), the line that matters for Gemma.** With an oracle that exits at the earliest layer whose top-1 agrees with the final
  layer's: "while only using an average of 1.53 layers per token" with ROUGE-L 38.24 against 38.32 for the full model, "by up to ×5.2". Copying the
  exited layer's hidden state into skipped layers' keys and values is fine; copying the projected K and V "results in a significant drop in
  performance to 23.02 ROUGE-L".
- **Results (Table 2, Figure 3).** Softmax response gives the largest layer cut, "up to about three times faster than running through all the
  model's layers"; on state saturation the paper says it "often falls bellow the static baseline".
- **Comparable to lane 18: no.** An 8-layer T5, trained for early exit, TPUv3, different tasks, a first-agree oracle. Lane 18's settle layer
  is not CALM's oracle (section 3).

### 2.3 LayerSkip, `2404.16710` (F)

- **Setup.** Llama2 7B and 13B (continual pretraining, Table 1 and 3), Llama2 1.5B and 7B (pretraining from scratch on 26B tokens, Table 4),
  Llama1 7B on code (Table 5), Llama 1.5B on TOPv2 (Table 6); H100 and A100; greedy, 512 new tokens. Training adds layer dropout (low
  rate early, high rate late) and an early-exit loss with a shared head.
- **Why exit without training degrades (Table 1, Llama2 7B, last layer then middle layer 16, baseline against LayerSkip).** Perplexity of the
  baseline's middle layer is 1,900 (Wikipedia), 4,390 (Books), 968 (The Stack), against 8.12, 6.53, 2.99 with LayerSkip; GSM8K at the middle
  layer is 0 for the baseline against 2.05. On TOPv2 (§6.1): "when removing layers from the baseline model, the model is not able to
  generate complete and accurate parses resulting in 0 EM", where LayerSkip reaches 77% at layer 12. The final layer pays a small price (the
  paper reports a 3% regression on TOPv2).
- **Speedups (abstract; Tables 3 to 6).** "speedups of up to 2.16× on summarization for CNN/DM documents, 1.82× on coding, and 2.0× on TOPv2
  semantic parsing task", for **self-speculative decoding with verification**, not unverified exit.
- **Comparable to lane 18: no, and it explains the sign of the bias.** It does not read a Gemma, and every number needs a model trained
  for early exit. What it says for lane 18: a pretrained model not trained for exit does not decode sensibly at intermediate layers
  through the final head.

### 2.4 Tuned lens, `2303.08112` v6 (F)

- **Setup.** GPT-Neo (125M to 2.7B), Pythia (to 12B), GPT-NeoX-20B, OPT, BLOOM; a learned affine probe per layer trained with KL to the
  final layer's distribution (Eq. 9). Metrics are perplexity, KL, a bias measure, and (§5.4) prediction depth.
- **The plain lens is biased and unreliable (§2, Figure 3, Figure 14).** For GPT-Neo 2.7B "around 4 to 5 bits for most layers" of KL bias
  relative to the final layer; the logit lens "struggles" on BLOOM and OPT 125M, whose top-1 is often the input token. Tuned-lens perplexity
  is lower at every layer (Figures 4, 5).
- **§5.4 defines the lane's quantity.** "prediction depth" is "the number of layers after which a model's top-1 prediction for x stops
  changing", computed with the tuned lens; Table 2 compares its Spearman correlation with the iteration learned for the tuned and the logit
  lens: the tuned lens correlates better in 8 of 11 tasks (Pythia 12B, twelve multiple-choice tasks).
- **What it does not give.** No settle-layer distribution for either lens, no top-1 agreement rate with the final layer, no model at 2B or
  Gemma. **The size of the bias on "settled layer" is not published.** Direction only: the plain lens understates early agreement.
- **Comparable to lane 18: no on numbers, yes on the definition.** Lane 18's settle layer ("the smallest l from which the lens top-1 equals the
  final top-1 at every layer up to 34") is §5.4's prediction depth, measured with the plain lens.

### 2.5 AI Agents That Matter, `2407.01502` (F)

- **Setup.** Re-evaluates three public HumanEval agents (LDB, LATS, Reflexion) against simple baselines (GPT-3.5, GPT-4 zero shot; retry,
  warming, escalation) on 164 problems, five runs each, dollar cost measured (§2.2, Figure 1, Table A1); a DSPy joint-optimisation study on
  HotPotQA (§3.2); a downstream-evaluation argument (§4); holdouts and reproducibility.
- **Frame lane 19 may borrow (§2, §4).** "AI agent evaluations must be cost-controlled"; results as a Pareto curve of accuracy against dollars;
  "downstream evaluations of agents should include input/output token counts in addition to dollar costs", so a reader can re-price. Table A1
  (mean of five runs, min to max): GPT-4 baseline 89.6% at $1.93, warming (GPT-4) 93.2% at $2.45, retry 92.0% at $2.51, **escalation 85.0%
  at $0.27** ("We start with a cheap model (Llama-3 8B) and escalate to more expensive models ... if we encounter a test case failure",
  §2.2, which needs a pass/fail signal per call).
- **What it does not give.** No price on a wrong answer. §5.2 says of human-in-the-loop evaluation that it "is costly and tricky" and "an
  important direction for future research". No local-against-hosted comparison on short office tasks.
- **Comparable to lane 19: no** (HumanEval, 2024 prices, five repeats), but the **method** (dollars with tokens, Pareto, repeats with a
  range) is the bar lane 19 sits under.

### 2.6 FrugalGPT, `2305.05176` (F)

- **Setup.** Twelve hosted APIs from five providers at 2023 prices (Table 1); a cascade of length 3 with a DistilBERT scoring function
  trained on labelled queries, on HEADLINES (10,000), OVERRULING (2,400) and COQA (7,982) (Table 2). Each dataset split into train and test.
- **Headline (abstract; Table 3, "Cost to reach the same accuracy").** "FrugalGPT can match the performance of the best individual LLM (e.g.
  GPT-4) with up to 98% cost reduction": HEADLINES 33.1 → 0.6 (98.3%), OVERRULING 9.7 → 2.6 (73.3%), COQA (best individual GPT-3) 72.5 →
  29.6 (59.2%). Case study (Figure 3): budget one fifth of GPT-4's, cost −80% with accuracy +1.5%.
- **Limits it states (§5).** "To train the LLM cascade strategy in FrugalGPT, we need some labeled examples", from the test distribution.
  Every model is hosted; the cheapest is "GPT-J" on a hosting service. No local model, no review cost.
- **Comparable to lane 19: no.** It reports cost at equal accuracy, hosted only, 2023 prices; it is the baseline for "cheap first, hosted on
  doubt", which lane 19 does **not** test (a cascade needs a scoring function and calls; lane 19 loads no model).

### 2.7 `mlx-lm` 0.31.3, Gemma 4 e2b (F, code read, nothing run)

Scope's reading (35 layers, `num_kv_shared_layers` 20, layers 15 to 34 read KV and write none) is **correct, with one off-by-one in the
safe boundary**:

- `config.json` of `mlx-community/gemma-4-e2b-it-8bit` (cached): `num_hidden_layers` 35, `num_kv_shared_layers` 20, `sliding_window` 512,
  `use_double_wide_mlp` true; `layer_types` is five-periodic with `full_attention` at indices 4, 9, 14, 19, 24, 29, 34.
- `gemma4_text.py:183` `has_kv = layer_idx < num_hidden_layers - num_kv_shared_layers`: **layers 0 to 14 own a KV cache**, 20 layers do not.
  `make_cache` (`:662-675`) builds exactly 15 caches. `DecoderLayer.__call__` (`:333-389`) returns `(h, shared_kv, offset)`.
- `Gemma4TextModel.previous_kvs` (`:434-442`): for layers 15 to 34, the KV is that of the **last layer of the same type among layers 0 to 14**:
  sliding layers read layer **13**, full-attention layers read layer **14**. The loop (`:549-573`) passes `intermediates[prev_idx]` to
  each shared layer.
- **Consequence for an exit.** A token that exits at layer **14 or later** has run layers 0 to 14, so it has written every KV entry any later
  token will read; no entry is missing. A token that exits at layer 13 or earlier leaves a hole (layer 14's full-attention K and V for that
  position, and for layers below 13 their own). Lane 18's boundary "layer 15 or later" is safe but conservative by one: **layer 14 is the
  first exit with no hole.** For exits at 14 or later, real skipping would produce the same cache as running every layer; what changes later
  is only which token id was chosen. (Code read; not run.)
- `gemma4_text.py:105-106`: with `use_double_wide_mlp`, layers 15 to 34 have a **double-width MLP**. A layer is not an equal share of
  compute, so "(34 − settle) / 35" is a share of layers, not of FLOPs; the shared block holds 20 of 35 layers and each has more MLP and no
  K/V projection. The post should say "layers".
- **Comparable to TIDE: n/a.** This is the reason an exit rule might be cheaper to build here (no KV hole from layer 14), as lane 18's
  mechanism says; no [F] source reports it.

### 2.8 `inference-cost-truth` README (F, README and licence file)

- **What a row is.** A price per million tokens or a GPU rental rate, each with `source_url` and `retrieved_on` ("Prices are read from
  official vendor pages only"); three mechanical gates before a number is published; dated immutable snapshots in `data/snapshots/`. The
  GitHub repository description gives "874 verified price rows across 24 providers, 378 GPU rental rates, 395 cited throughput
  datapoints"; **the README itself does not state 874** (a grep for the number found none). Maintained by a vendor that sells GPU kernel
  optimisation; the README says so.
- **Is anything in it a cost-per-correct number? No.** "Not a quality benchmark. Nothing here says which model is better." The grep for
  "per correct" in the README returns 0. It has a self-hosting break-even model (fixed monthly bill against variable API price) which
  excludes engineer time.
- **Dating.** README: verified on 2026-07-31; the default-branch head `a61c501` was committed 2026-07-31, although the README plans a refresh on
  2026-08-31; the API's `pushed_at` is 18 Sep 2026. At 26 Nov the data is almost four months old.
- **Licence.** The GitHub API reports NOASSERTION because the licence file is named `LICENSE-DATA`; it holds "Creative Commons Attribution
  4.0 International (CC BY 4.0)" and the README says "Licensed CC BY 4.0" for the data files. The scripts carry no code licence.
- **Lane 19 takes the question, not the data** (as briefed). Its price column is the repo-radar ledger and the lab's own metered cost.

## 3. Lane 18 against the bar

**Does an [F] row already report the lane's number on a comparable setup? No.** (Section 0.) The lane may run on this evidence. What each
published row lets it say, and what it may not:

| lane 18 element | closest [F] row | the lane **may** | the lane **may not** |
|---|---|---|---|
| "TIDE's cosine rule" `cos(h_l, h_34) > 0.98` at checkpoints 3, 7, ..., 31 | TIDE §3.1, `calibrate.py:89-100` | call it **TIDE's convergence label** (the training target of its routers) | call it "TIDE's exit rule": the exit rule is a router score > 0.85 (`runtime.py:174`); say "an oracle version of TIDE's label". Also note `min_layers = 8` (`config.py:13`): checkpoints 3 and 7 never exit at defaults, so the first live checkpoint is 11. |
| "cosine" as CALM's state saturation | CALM §3.5 | say its label **differs** from CALM's saturation (`cos(d^i, d^{i−1})`, to the previous layer) | call it CALM's measure or set it beside CALM's saturation results ("often falls bellow the static baseline") |
| settle layer = smallest l from which lens top-1 = final top-1 at every later layer | tuned lens §5.4 "prediction depth" (tuned lens there, plain lens in the lane) | name it **prediction depth (§5.4), measured with the plain logit lens** | call it CALM's oracle exit layer: CALM's oracle exits at the **first** layer whose top-1 agrees, even if a later layer disagrees (§3.3.1) |
| oracle saving, (34 − settle) / 35 | CALM §3.3.1 (1.53 of 8 layers, ×5.2), TIDE Table 3/5 | report it as a **share of layers**, an upper bound | set it beside CALM's 1.53 layers or ×5.2 (8-layer T5, trained, TPUv3), or beside TIDE's "98 to 99%" (last-layer exits); call it compute (layers 15 to 34 have a double-width MLP) or speed (no layer is skipped; TIDE's own post-hoc mode is not skipping either) |
| bias of the plain lens on settle layers | tuned lens Figure 3 (4 to 5 bits KL, GPT-Neo 2.7B) | say the plain lens **understates** early agreement, so settle layers are an upper bound on lateness | attach a size to that bias on Gemma: **no [F] row gives one** |
| "settled at layer 15 or later: no KV hole" | `gemma4_text.py:183, 434-442, 662-675` | say **layer 14 or later** leaves no hole (code read) | say it of a real exit run: lane 18 reads only the full-depth pass; a skipped layer changes later tokens' context |
| a mis-exit rate against TIDE's label | TIDE Table 5 (one prompt, "correct") | report it as ours, n = our tokens | say TIDE's quality was measured: the paper has one math prompt, no benchmark number, and its decode "exits" skip no layer |

**Concrete changes tied to rows (for the operator to amend openly, dated, before the prediction is committed):**

1. Rename the cosine arm "TIDE's convergence label", not "TIDE's cosine rule"; the decider's third clause ("TIDE's cosine rule exits")
   should read "the label fires before layer 31 on at least X% of tokens" or be dropped. (TIDE rows 2.1.)
2. Post **two settle definitions from the same pass**: first-agree (CALM §3.3.1) and last-disagree-then-stay (prediction depth, §5.4).
   One extra column; it keeps the lane from being set beside the wrong paper's number. (2.2, 2.4.)
3. KV boundary: "layer 14 or later" (2.7). Anything stated at 15 stays true; the share past the boundary only grows.
4. Checkpoints: report 11, 15, ..., 31 as the live ones (`min_layers = 8`) and 3, 7 as extra. At 35 layers the last checkpoint is 31,
   not 34, so on Gemma the label is not trivially 1 at the last checkpoint, unlike TIDE's three models (2.1).
5. "Oracle saving" is **layers**; say so in the first paragraph, as the brief already says it measures no speed (2.7).

**What lane 18 may not claim from TIDE:** that TIDE saves layers (it skips none), that "98 to 99% exit early" (a last-layer count), or a
throughput gain on Apple silicon (+6.6%, +8.1% and −16.3% are one A100 and 16 prompts).

## 4. Lane 19 against the bar

**Does an [F] row already report hosted against local per correct answer with a review cost on short tasks? No** (2.5, 2.6, 2.8). The lane
runs on this evidence. Definitions that exist, and what the lane may and may not claim:

- **Metric names.** AI Agents That Matter reports accuracy and **total dollars** (Table A1) as a Pareto pair, with token counts advised.
  FrugalGPT reports **cost to reach the same accuracy** (Table 3). Neither defines cost per correct answer or prices a miss.
- **The lane's `r*`, by our own algebra (checked numerically here, not a source's definition).** With N cases, P passes and C dollars,
  and a review cost r per miss, total cost is `C + r (N − P)`. Hosted is cheaper in **total cost** when `r > (C_h − C_l) / (P_h − P_l)`.
  The lane's `r* = C_h / Δpass` sets `C_l = 0`; since the local cost is unpriced and `≥ 0`, **`r*` is an upper bound on the true
  crossover**. Per **correct answer** (total cost divided by passes) the crossover with `C_l = 0` is `C_h P_l / (N Δpass) = r* · P_l / N`,
  lower by the factor `P_l / N`. Both only lower `r*`, so a reading of "r* below €0.50" is unchanged, but **the lane may not call `r*` "cost per
  correct answer"**: name it "the review cost per miss above which the hosted model costs less in total (local cost left unpriced)".
- **Concrete changes tied to rows (for the operator):**
  1. Name the quantity as above and show `P_l / N` beside it if the title keeps "per correct". (2.5, 2.6.)
  2. Post **input and output token counts with the dollars** (AI Agents That Matter §4), so the table re-prices when prices move. The lab's
     results files already hold them.
  3. Show the accuracy-against-dollars pair (the Pareto view) next to `r*`; AI Agents That Matter reports means over five runs with a
     range. Lane 19's Sonnet and Opus rows are **single sits** (Haiku has three), and the post should say so beside every Δpass. (2.5.)
  4. The repo-radar ledger is the lane's price source; `inference-cost-truth` is dated 31 Jul and carries no accuracy, so it adds no column.
  5. A cascade ("cheap first, hosted on doubt") is FrugalGPT's and AI Agents That Matter's baseline; lane 19 does not test one. The post
     should say it is not tested, not that it was considered and found worse.
- **May not:** claim a published precedent for pricing a wrong answer (AI Agents That Matter §5.2 leaves it to future work); call the
  €30 an hour anything but our assumption; set `r*` beside FrugalGPT's 98.3% or 59.2% (hosted only, accuracy held fixed, 2023 prices);
  quote `inference-cost-truth` for any number about quality.

## 5. Checks run this session

- Fetch and extract: `curl -sL https://arxiv.org/pdf/<id> -o <id>.pdf; pdftotext -layout <id>.pdf <id>.txt` for the six ids: 524, 1332,
  1794, 1906, 1727 and 842 lines. The tuned lens and LayerSkip were also extracted without `-layout` for quote matching.
- Quote match: a `python3 -I` script checked 36 quoted source passages against the fetched text (whitespace, apostrophes and dashes normalised): 35 match; the one that does not, the repository description's "874 verified price rows ...", comes from the GitHub API, not from a file.
- Algebra of section 4: a five-trial numeric check that total-cost and per-correct crossovers are `C_h/Δ` and `C_h·P_l/(N·Δ)`, ratio `P_l/N`.
- Code: `gh api repos/RightNow-AI/TIDE/contents/python/TIDE/<file>.py` for the files named in 2.1 (head `8bdc187`);
  `~/.pyenv/versions/lewagon/lib/python3.10/site-packages/mlx_lm/models/gemma4_text.py` (version printed `0.31.3`) and the cached e2b-8bit
  `config.json` for 2.7.

## 6. What our pages may and may not claim

**May:**
- Quote TIDE's tables as its own claims, on its setup (one A100, 8B models, 16 prompts), with "post-hoc: no layer is skipped" beside any
  speed number, and "prefill latency −5.5% at the default θ, −7.2% at θ = 0.50".
- Say TIDE's convergence label is cosine to the final layer at 0.98 and its exit rule is a router score above 0.85; that the router checkpoints
  are every fourth layer from the fourth and exits start at index 8.
- Say CALM's hidden-state saturation is cosine to the previous layer and that CALM's own results rank it below softmax response.
- Say the plain logit lens is biased and understates early agreement (tuned lens §2, Figure 3: 4 to 5 bits KL, GPT-Neo 2.7B), as direction only.
- Say, from code read on the pinned mlx-lm 0.31.3, that on Gemma 4 e2b the 15 caches belong to layers 0 to 14 and a token that leaves at
  layer 14 or later leaves no missing KV entry.
- Say AI Agents That Matter asks for dollars with token counts, Pareto curves and repeats, and that FrugalGPT reports cost at equal
  accuracy for hosted models only.

**May not:**
- Say "the field reports" a settle layer, an oracle early-exit saving, or an exit rate for Apple silicon, Gemma 4 or a ~2B model: no [F]
  row does.
- Present "98 to 99% of tokens exit early" as early exit, or TIDE as saving layers, wall-clock time or energy.
- Put the lane's oracle layer share beside CALM's 1.53 layers, ×5.2 or ×3, LayerSkip's 2.16×, or TIDE's percentages.
- Give a size to the plain lens's bias on settle layers.
- Call lane 19's `r*` "cost per correct answer", or claim a published precedent for pricing a miss.
- Use `2412.01455` or any paper not ranked in section 1.

## 7. Not read, not ranked

`2412.01455` (Shan et al.) and TIDE's other citations (EE-LLM, SkipDecode, SpecEE, ADEPT) were seen only as names in TIDE's text.
`adapters/{llama,mistral,qwen}.py` in TIDE and the data files of `inference-cost-truth` were not read.
`LITERATURE.md` gets no row: no run of ours is set against any of these papers yet. Lane 18 and lane 19 add theirs when they run.
