# cold-start-worker

Lane 15, written 24 Nov 2026 by the scope job `companies-tech-scope-2026-11-24`, before any model was loaded and before the Runpod
SDK was installed. Reader question C2. Origin:

origin: {"kind": "inbox", "url": "https://www.runpod.io/blog/cut-vllm-cold-starts-runpod-serverless", "label": "operator 24 Nov 2026: a smaller version of Runpod's own tech; Runpod's vLLM cold-start post (20 Aug 2026) and its FlashBoot posts (rank C until lab-inf-14 reads them in full)"}

Who this is for: the operator (decider below) and, as application evidence, a reader at Runpod. It is not a client offer.

## Prediction
**What runs.** One worker script: a Runpod serverless handler (their Python SDK in local mode) that loads an MLX model at module
level and answers one fixed prompt with at most 64 new tokens at temperature 0. A driver starts the worker as a fresh process, waits
until it accepts a request, sends one request, then a second, and kills it. Every trial writes time stamps for five stages:
(1) process start to imports done, (2) weights loaded, (3) server accepting requests, (4) first request sent to its first token,
(5) first token to the 64th. "Fresh start to first token" is the sum of 1 to 4.
**Conditions,** ten trials each, in one fixed interleaved order: A `mlx-community/gemma-4-e4b-it-4bit`, B
`mlx-community/gemma-4-e4b-it-8bit` (the lab's champion), both with the weight files read once in full beforehand (a warm file cache);
C the worker from A or B left ready, stopped with `SIGSTOP` for 60 s, continued with `SIGCONT`, then one request (the laptop's
stand-in for pausing a container). D, a cold file cache, runs only if `sudo -n purge` succeeds; otherwise it is "not measured".
**Numbers.** Anchor, from committed files: the lab's adapter took 3.05, 3.22 and 3.24 s to load the 4-bit e4b
(`results/dense-gemma-4-e4b-it-4bit.json`, `model.loads[].load_s`) and 4.65, 4.71 and 4.79 s for the 8-bit
(`results/champion-e4b-8bit.json`), inside a process that was already running. The rest are the scope job's guesses.
- A, fresh start to first token: median between 5 and 12 s. Imports 1.5 to 5 s, weights 2.5 to 4 s, first request to first token 0.5 to 3 s.
- B: the weights stage takes 1.3 to 1.6 times A's (4 to 5.5 s); fresh start to first token is 1 to 4 s above A's.
- C: first token within 1.0 s at the median, and at least 5 times faster than A's fresh start.
- First request against second request in the same worker: time to first token at least 1.5 times longer on the first. Unsure.
- Spread: the slowest of ten warm trials is at most 1.5 times the fastest, in A and in B.
**Kills.** (1) On the median trial the five stages do not add up to within 10% of the driver's own end-to-end time: the instrument
is broken, nothing is posted but that gap. (2) The slowest warm trial is more than 2 times the fastest in both A and B: there is no
stable breakdown; the post is the spread and its cause if found, with no stage table.
**What this cannot say.** Nothing about GPU allocation, image pull, vLLM compilation or a network volume: the stages Runpod's own
324 s is mostly made of. Nothing about `delayTime`. One laptop, one runtime version, one model family.

## Mechanism
A fresh worker pays for the interpreter and its imports, for mapping the weight file, and for the first request, where a lazily
loaded model is first evaluated on the GPU. The 8-bit pack is about twice the bytes, so its weight stage should scale with size when
the file cache is warm. A stopped process keeps its memory, so resuming it skips all three: that is the difference Runpod names
between an end-to-end cold start and reviving a warm process.

---

## Pre-run notes (26 Nov 2026, lane 15 job, before the Runpod SDK is installed and before any model is loaded)

Everything above this line is the scope job's stub, unchanged except that "Origin: the object in the brief" now shows the object.
The only "unsure" in it (first request against second) is left unsure: no committed results file times a second request in the
same process. This section applies lane 14's corrections (`docs/literature-2026-11-24.md`, section 3) and fixes the design choices
the stub leaves open. It changes no predicted number.

**Step 0 outcome.** Lane 14 found no published [F] number for cold-start *stages* under MLX, so the brief's stop rule did not fire.
It found one MLX cold-start *total* on Apple silicon: 2511.05502, "MLX in ∼31s" (M2 Ultra, Qwen-2.5 family, no stage split, no n,
model size not stated). The stub's A band (5 to 12 s) sits far below it. Why ours may differ: a 4-bit / 8-bit e4b (5.1 / 8.9 GB on
disk), a warm file cache, and a different, unstated quantity on their side. No conflict is proven either way.

**Words (lane 14, correction 1).** Runpod's docs end a cold start when the worker "is ready to take its first job". Here:
- stages 1 to 3 = **cold start (Runpod's definition)**;
- stages 1 to 4 = **fresh start to first token** (the decider's quantity). The sum 1 to 4 is never called a cold start.
- `delayTime` (queue + cold start, and in their post image pull and capacity wait) has **no laptop equivalent** (correction 2).
- Their 324 s → 91 s is a **median of 3 against 2** trials ("Treat the 3.5× ratio as directional"); our n is 10 per condition. The
  two are not comparable in n (correction 3).
- **Arm C stands in for the Flash post's "container pausing" only**, not for FlashBoot in general. "5 times faster" is our
  threshold, not a FlashBoot claim; their only number ("about 200ms") has no n or method (correction 4).
- The committed `load_s` is the adapter's load inside a running process. It is an anchor for stage 2, not a cold start (correction 6).

**Source facts that bear on the mechanism (code read, nothing run).**
- `mlx_lm.load` in 0.31.3 defaults to `lazy=False` and runs `mx.eval(model.parameters())` (`mlx_lm/utils.py:417-418`). So the weights
  are evaluated inside stage 2. Any extra cost on the first request (stage 4) is kernel and graph warmup and prompt processing, not
  weight evaluation. The stub's "where a lazily loaded model is first evaluated on the GPU" does not hold for a default load; the
  text above is left as written and the result is read against this note. The worker uses the default `load()`, as a real handler would.
- Weight files on disk: 4-bit 5,146,800,534 bytes (one file); 8-bit 4,906,172,928 + 3,974,804,036 = 8,880,976,964 bytes. The ratio is
  1.73, not "about twice". The stub's B band (1.3 to 1.6 times A's weight stage) stands as written.

**Design, fixed before the run.**
- **Prompt.** email-triage TRAIN case `overdue-invoice` (the first id of `lab/cases.py: SMOKE`), assembled as wilco's
  `run_plain_case` does at the pin: system = `agents/email-triage/prompt.md`, user = the case `input`, read-only from the pinned
  clone. The tokenizer's chat template with `add_generation_prompt=True`. Greedy (`make_sampler(temp=0.0)`), `max_tokens=64`. Nothing is
  graded.
- **Stamps.** The worker writes `time.time()` and `time.perf_counter()` at each stamp; stages that cross the two processes use
  `time.time()`. The driver passes its spawn time in an environment variable. Stamps: spawn (driver) → `import runpod` done →
  `import mlx_lm` done (= imports done, end of stage 1) → `load()` returned (end of stage 2) → first successful poll of the local
  server (driver, poll every 20 ms; end of stage 3) → request sent (driver) → first token (handler, first `stream_generate` response) →
  64th token (handler, last response; end of stage 5) → response received (driver). Stages 4 and 5 are stamped in the handler because
  `/runsync` returns only after the job is done.
- **Kill 1 is not true by construction.** The gap between "server accepting" and "request sent", and the gap between the last token
  and "response received", are recorded as their own intervals. Kill 1 compares the sum of stages 1 to 5 with the driver's own
  spawn → response received, on the median trial.
- **Arm C rides on every A and B trial**: after request 2 the driver sends `SIGSTOP` to the worker's process group, waits 60 s, sends
  `SIGCONT`, then request 3. So C has n = 10 on A (C-A) and n = 10 on B (C-B). C's number is `SIGCONT` → first token of request 3.
  The stub's "5 times faster than A's fresh start" is read on C-A; the decider is read on C-B against B. Also posted: request 3
  against request 2 in the same process, i.e. whether the pause itself costs anything.
- **Order.** A, B, A, B, ... (trial index 0 to 9 each, 20 workers in all), each preceded by `cat <that pack's weight files> > /dev/null`
  with the cat's duration recorded (both packs together are 14 GB on a 16 GB machine, so the cache is re-warmed before each trial
  rather than once per block). A gate trial of each comes first and is kept in the rows as `gate`, outside the ten.
- **Hygiene.** The worker starts in its own session; signals and the final kill go to the process group. `ollama ps` must list no
  loaded model before the timed block. Swap (`sysctl vm.swapusage`) is read before and after every trial; the lane stops if it grows
  more than 2 GB above the reading at the start of the block.
- **D (cold file cache)** is not measured: `sudo -n purge` needs a password on this machine (brief, checked 24 Nov).
