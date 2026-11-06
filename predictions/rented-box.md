# rented-box

Lane 07 (written 6 Nov 2026, before any lane-07 run; the lane needs an operator go because it costs money): the same three
exams served by vLLM on one rented GPU: throughput and p95 per task, cost per 1,000 cases in euros, against lanes 02–06.

Rank letters: **A** measured on our cases on this machine; **B** adjacent measurement; **C** read, not run.

## Prediction

**Box and price.** One 24 GB L4-class card. Prices the radar read on 2 Oct 2026 (`repo-radar/data/prices/gpu-hours.tsv`, rank
C, will have moved): RunPod L4 secure $0.49/h, Scaleway `L4-1-24G` €0.79/h, OVH L4 $1.00/h; an AWS `g6.xlarge` (L4) is in the
same band (⚠️ UNVERIFIED, from memory, around $0.80/h on-demand). I predict the lane picks Scaleway or RunPod and the posted
hourly rate is **€0.45–0.80**. Total spent for the lane: **€2–6** (2–4 box-hours: image pull, two model loads, three exams × two
models, teardown). Unsure on AWS; if Terraform on AWS, €4–10.

**Models.** The laptop's mlx affine 4-bit packs do not exist for vLLM, so "the same quantised models where available" becomes:
`gemma-4-e4b-it` in bf16 (~16 GB weights, fits a 24 GB L4 with a small KV budget) or a GPTQ/AWQ 4-bit if one exists;
`Qwen3-14B` AWQ/GPTQ 4-bit (~9 GB). Pass fractions therefore differ from the laptop by packaging, not by hardware: within ±2
held-out cases per exam of lane 02's rows, and bf16 e4b slightly above the 4-bit laptop row (+0 to +2 on call-fields).

**Single-stream speed.** L4 memory bandwidth is ~300 GB/s (vendor spec, rank C, not re-read today), so one request at a time: Qwen3-14B 4-bit 25–32 tok/s (laptop 14);
e4b bf16 15–20 tok/s, **slower per stream than the laptop's 4-bit 37**. p50 per task single-stream: email-triage 10–15 s on
e4b bf16, 8–12 s on the 14B.

**Throughput with concurrency (the point of the box).** At 16–32 concurrent requests vLLM's continuous batching gives
300–600 aggregate output tok/s on the 14B 4-bit and 400–800 on the e4b bf16: **10–20× the laptop's single stream**. p95 per
task at that concurrency: email-triage 15–30 s, call-fields 30–60 s (the 8k prompts), recap 25–45 s, i.e. per-task latency no
better than the laptop; throughput is.

**Cost per 1,000 cases (metric: euros, from the box's hourly rate × wall, serving at concurrency; exchange rate stated on the
day).** Using ~330 / 400 / 550 output tokens per case (email-triage / call-fields / recap, from the sandbox's e4b runs, rank A)
and 400 aggregate tok/s at €0.60/h: **email-triage €0.12–0.20, call-fields €0.15–0.30, recap €0.20–0.35 per 1,000 cases**,
excluding the fixed ~15 minutes of setup. Against the laptop (lane 02: ~150 / 280 / 300 minutes per 1,000 cases on the e4b),
the box does 1,000 cases in 10–25 minutes. Energy is not comparable (no package meter on a rented card; say so in the post).

**Kill clause.** The brief names none; it names the reopen threshold, copied: "a provider price change that moves the cost per
1,000 cases by a quarter." Proposed kill, not from the brief: the lane does not post a cost line unless the box's total spend is
read from the provider's invoice or usage page and the per-case wall is the runner's own clock on that box (L3's rule, rank A:
a modelled line can never carry `metered`).

## Mechanism

A single L4 has about 1.5× the memory bandwidth of an M1 Pro's unified memory, so one request at a time it decodes a given
number of weight bytes only modestly faster, and a bf16 model moves four times the bytes of the laptop's 4-bit pack, which is
why single-stream e4b is predicted slower on the card than on the laptop. The card's advantage is that a batched decode step
reads the weights once for every request in the batch, so aggregate tokens per second grow almost linearly with concurrency
until compute or KV memory binds; with 24 GB and short outputs that is 16–32 requests for these models. Cost per case is
hourly rate divided by aggregate throughput, which is why the per-1,000 numbers are cents, and why the fixed setup time
dominates any run under a few thousand cases. The sandbox's L3 sheet (rank A for the arithmetic, modelled for the walls)
reached the opposite conclusion at serial concurrency: a card costs more than hosted per token when it serves one request at a
time; batching is the unmeasured lever that sheet named, and this lane measures it. Pass fractions move by packaging (bf16 vs
4-bit, AWQ vs affine) rather than by hardware, the same lever e31 measured on the laptop (rank A), so a ±2 band is the honest
prediction, not equality.
