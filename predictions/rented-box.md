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

## Amendment 16 Nov 2026 (before any run)

Written before any lane-07 run (none has happened; no `results/rented-box*.json` exists), after the full-text literature read
(`docs/literature-2026-11-16.md` §2.1, §4). It adds to the prediction above and changes nothing in it; the brief change it rests
on is proposed in the scheduler (`jobs/lab-inf-07-rented-box.proposed.md`) and is the operator's to approve or reject.

**Why.** Kurtić et al. 2411.02355 v4 §5 (rank C here: read, not run; their setup is 8B on 1×A6000 48 GB, ours is a 24 GB L4)
report the cost order of 4-bit vs 8-bit flipping with the serving mode. Table 5 (synchronous), docstring generation: BF16 1,395,
INT8 2,157, INT4 3,290 queries per USD. Table 6 (max throughput, setups tuned for BF16), same cell: BF16 25.1k, INT8 34.6k, INT4
24.0k. The prediction above has one cost line, taken at 16–32 concurrency, and no 8-bit-activation arm. So it could not see that
flip.

**Added arms.** W8A8 (8-bit weights and activations) for both models, beside the original e4b bf16 and Qwen3-14B AWQ/GPTQ 4-bit.
⚠️ UNVERIFIED: whether the pinned vLLM runs FP8 W8A8 on the L4 or only INT8, and which checkpoints exist. Both must be named here
before the go.

**Added serving cases.** Synchronous (concurrency 1) and max-throughput (sweep to saturation). Each gets its own cost line, and
the 16–32 point is kept separate from saturation.

**Predicted per-stream decode, synchronous** (bandwidth-bound: ~300 GB/s over the weight bytes, rank C): e4b bf16 15–20 tok/s
(as above), e4b W8A8 (~8 GB) 25–30, 14B AWQ 4-bit (~9 GB) 25–32 (as above), 14B W8A8 (~15 GB) 14–18.

**Predicted cost order.**

| model | synchronous (cheapest first) | max-throughput (cheapest first) |
|---|---|---|
| e4b | W8A8 < bf16 | W8A8 < bf16 (the gap is wider than in sync: W8A8 frees ~8 GB of KV on a 24 GB card) |
| Qwen3-14B | AWQ 4-bit < W8A8 | email-triage: W8A8 ≤ AWQ (short prompts, the batch fits); call-fields: AWQ < W8A8 (KV-bound) |

So the Kurtić flip is predicted for the 14B on short prompts only. On the 8k call-fields prompts the predicted order is the same
in both modes. The reason: ~15 GB of W8A8 weights leave about 6 GB of KV at `--gpu-memory-utilization 0.9`, against about 12 GB
for AWQ. At ~0.16 MB per token (⚠️ UNVERIFIED Qwen3-14B shape, 40 layers × 8 KV heads × 128 × 2 × 2 bytes, not read from a config
today) that caps W8A8 at ~4 concurrent 8.5k-token requests, against ~8 for AWQ. The paper's 48 GB card never meets that cap. For
the e4b there is no flip to see, because the lane has no 4-bit e4b on vLLM unless a public checkpoint exists. Confidence: low on
the 14B email-triage order (a near-tie is the likeliest outcome), moderate on the others.

**Predicted cost per 1,000 cases, synchronous** (mean output ~427 tokens, plus 1–2 s prefill, at €0.45–0.80/h): e4b bf16
€2.8–6.8, e4b W8A8 €1.9–4.2, 14B AWQ €1.8–4.2, 14B W8A8 €3.1–7.2. That is 10–30× the concurrent line above (€0.09–0.24 at
300–800 aggregate tok/s, €0.60/h). Command:

```sh
python3 -c "
o=(330+400+550)/3
for r in (.45,.8):
  for a,(lo,hi) in {'e4b bf16':(15,20),'e4b W8A8':(25,30),'14B AWQ4':(25,32),'14B W8A8':(14,18)}.items():
    print(r,a,round(r*(o/hi+1)/3.6,2),round(r*(o/lo+2)/3.6,2))"
```

**Total spent, revised.** It was 2–4 box-hours, €2–6. Now it is **6–10 box-hours, €3–8**: 4 arms × (249 synchronous cases + a
sweep + a load), with a ceiling of €12 for the go note. Command:
`python3 -c "[print(w,round(4*(0.1+249*w/3600+0.25)+0.5,1)) for w in (16,30)]"`.

**Kill clause, added.** No max-throughput cost line is posted without the saturation bound named (KV preemption or a tok/s
plateau) and the replay marked timing-only.
