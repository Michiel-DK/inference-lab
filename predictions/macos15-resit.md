# macos15-resit

Lane 21 (stub written 25 Nov 2026 by the scope job; this file committed 27 Nov 2026 by the lane, before any model is
loaded): the OS moved under the pinned runtime. Does the champion still give the same verdicts, token for token? Reader
question B1 ("Does it give the same answer twice?"), audit path, publish level.

origin: {"kind": "lane", "url": null, "label": "lab-inf-10-8bit-champion (results/champion-e4b-8bit.json, 10 Nov 2026): the baseline re-sat on the upgraded OS"}

**Environment, recorded before the load (27 Nov 2026).** The brief says "macOS 15"; the machine reads macOS **27.2.1**
(Apple's year-numbered releases), which is "15 or later", so the lane runs.

```
$ sw_vers
ProductName:		macOS
ProductVersion:		27.2.1
BuildVersion:		26B131
$ uname -a
Darwin MacBook-Pro-van-Michiel.local 27.2.0 Darwin Kernel Version 27.2.0: Thu Oct 15 00:02:46 PDT 2026; root:xnu-13432.41.4~25/RELEASE_ARM64_T6000 arm64
$ ~/.pyenv/versions/lewagon/bin/python3 -c "import mlx.core as mx, mlx_lm; print(mx.__version__, mlx_lm.__version__); print(mx.metal.device_info())"
0.32.2 0.31.3
{'device_name': 'Apple M1 Pro', 'max_recommended_working_set_size': 12713115648, 'memory_size': 17179869184, 'architecture': 'applegpu_g13s', 'max_buffer_length': 9534832640, 'resource_limit': 499000}
$ ls ~/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-8bit/snapshots/
4255b21bd9a9d3fc807ef7abd80373f5e3a52a73
```

- Pins hold: mlx 0.32.2, mlx-lm 0.31.3 (`requirements.txt`).
- Snapshot `4255b21b` = lane 10's `model.snapshot` in `results/champion-e4b-8bit.json`.
- Lane 10's `machine.macos` reads `14.3.1`. Swap before the run: 7.7 GB of 9.2 GB used (`sysctl vm.swapusage`), so
  "swap flat" means growth against that, not the absolute figure.

**Baseline, load 0 of `results/champion-e4b-8bit.json`** (re-derived with `lab.results.derive` on the 83 load-0 rows;
these numbers settle nothing in the stub, they only name what the stub compares against): 62 of 83 cases pass, held-out
41/55, gen tok/s 29.27 (all three loads: 29.26, the stub's 29.3), peak 9.18 GB, 47,853 gen tokens, wall 31.7 min.

## Prediction

> # macos15-resit
> Written 25 Nov 2026 by the scope job, before the upgrade. Reader question B1. Origin: lane 10.
> **What runs.** One load (load 0 only) of `mlx-community/gemma-4-e4b-it-8bit`, the cached files lane 10 used (record the snapshot
> sha and compare it with lane 10's), the lab's own `lab run` path, the 83 cases, unchanged pins. Compared per case and per call with
> load 0 of `results/champion-e4b-8bit.json`.
> **Numbers (unsure).** Verdicts: 83 of 83 identical. Per-call `gen_tokens`: identical on at least 80 of 83 cases (a Metal kernel change
> can move a near-tie, LESSONS "spec" row). gen tok/s within ±10% of 29.3; peak within ±0.3 GB of 9.18.
> **Kill.** More than 3 of 83 verdicts move: the committed numbers in this lab do not carry over to the new OS without a re-sit, and
> the post says which lanes' rows that puts in doubt (all of them, until re-sat). A fired kill is a result.
> **What this cannot say.** Anything about Ollama, other packs or other models; whether the old OS would also have moved on a re-sit
> today (lane 10 says it did not on 10 Nov).

**How the one load runs.** `lab run --exams` refuses `--loads` below 3 (wilco's majority rule needs an odd count of at
least 3). The brief asks for one load. `scripts/lane21_drive.py` applies the CLI's own gates (committed prediction,
`verify_pin()`, no overwrite). It then calls the same `lab.run.run_loads` that `lab run --exams all` calls, with
`n_loads=1`, `context=None` and `spec=None`, so the `lab sit` child runs the default path that lane 10 ran. With one load,
"majority" and "stable" are the single verdict; the comparison is per case against lane 10's load 0, not against a majority.

**Gate first** (`macos15-gate-gemma-4-e4b-it-8bit-call`): one train case, call-fields `support-long-train` (the longest
train sitting), through `lab run --exam call-fields --cases support-long-train`. Go on only if peak < 11 GB and swap
grows by < 2 GB. During the full load, stop it if swap grows past 2 GB over the pre-run figure.

**Compare.** `scripts/lane21_compare.py` (stdlib) writes the per-case verdict diff and the per-call `gen_tokens` diff
to `docs/probes/macos15-resit/diff.txt`. With `--check` it re-derives every count posted in LESSONS and exits 1 on a
mismatch.

## Mechanism

Greedy decoding at a fixed pack, prompt and runtime is a deterministic function of the bytes, up to the order of the
floating-point reductions in the Metal kernels. On macOS 14.3.1 that order did not vary across processes, loads or lab
commits (LESSONS "champion-e4b-8bit": 249 of 249 rows identical per-call `gen_tokens`). mlx 0.32.2 ships its own
compiled `mlx.metallib` (present in the env's `site-packages/mlx/lib/`), so the kernel source does not change with the OS.
What can change (rank C, read, not checked here): the OS's Metal stack that turns the metallib into GPU code at load, and
how the driver schedules threadgroups. A different reduction order moves logits by ulps. That changes a greedy token only where the top two
logits are within that distance, a near-tie. The first moved token then changes everything after it in that sitting.
So the expected outcome is either nothing moves at all, or a few long sittings drift in token count, with verdicts
moving on few of them. More than 3 verdicts moving would mean the change is not a near-tie effect but a systematic one.

Step 1b (design check, before any load):
(a) Bearing rows: LESSONS "champion-e4b-8bit" (identical tokens across 3 processes and 2 lab commits, same OS); LESSONS
"spec" (control = lane 02 load 0 on 83 of 83, same OS); LESSONS "calib4" (B identical across 3 loads). All on 14.3.1.
(b) No arm is decided by construction: every bearing row holds the OS fixed, and the OS is the one thing that changed. One
load and not three, because three loads were identical on 10 Nov, so loads 1 and 2 add nothing the kill needs.
(c) If the prediction holds, the lab's committed numbers carry over to macOS 27.2.1 at these pins without a re-sit, and
later lanes on this laptop can be compared with them.
