# context-kv-truncation
origin: {"kind": "lane", "url": "https://github.com/Michiel-DK/inference-lab/pull/9", "label": "lab-inf-05-context-kv"}

Lane 13 (written 11 Nov 2026, before any lane-13 sitting): lane 05's reopen, first arm. Model
`mlx-community/Qwen3-4B-4bit@4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` (mlx-lm class `qwen3`, no `make_cache`:
`max_kv_size` builds `RotatingKVCache`, where on lane 05's two models it was accepted and not applied:
`applied_cache_classes` `{KVCache: 4, RotatingKVCache: 20}` on the e4b (`results/ctx-gemma-4-e4b-it-4bit-8k.json`; with
`max_kv_size=1024` the same classes in lane 05's ungraded probe, `predictions/context-kv.md` addendum) and
`{ArraysCache: 18, KVCache: 6}` on LFM with and without `max_kv_size=1024` (`results/ctx-lfm2.5-8b-a1b-mlx-4bit-8k.json`,
`results/ctx-gate-lfm2.5-8b-a1b-mlx-4bit-8k-maxkv1024-smoke.json`)). The 4B, not the 8B (4.62 GB): the question is about
the cache class, not the model, and the smaller pack leaves KV room for two arms in one evening. Case set: the 55 held-out
cases at wilco pin `62fcd449`, padded to 8,192 tokens with lane 05's rule (`lab/context.py`: the session's earlier inputs
between wilco's system prompt and the case turn, "already handled" header, 64 slack tokens; lane 05's 8k prompts ran
8,191–8,935 tokens on the e4b tokenizer and 8,192–9,296 on LFM's). Baselines: lane 05's 8k rows,
`results/ctx-gemma-4-e4b-it-4bit-8k.json` (38/55, call-fields 11/19, peak 5.45 GB, prompt 529 tok/s, 32.2 s per case,
repeat moved 0 of 55) and `results/ctx-lfm2.5-8b-a1b-mlx-4bit-8k.json` (14/55, call-fields 1/19, peak 5.90 GB, 809 tok/s,
27.0 s per case), where the setting did nothing; and this lane's own control arm. Arms, one fresh load each, after the two
train-only smokes (`ctx-gate-qwen3-4b-4bit-8k-smoke`, `ctx-gate-qwen3-4b-4bit-8k-maxkv4096-smoke`):
1. `ctx-qwen3-4b-4bit-8k` (control: 8k padding, default cache);
2. `ctx-qwen3-4b-4bit-8k-maxkv4096` (`--max-kv-size 4096`, below every prompt);
3. `ctx-qwen3-4b-4bit-8k-maxkv4096-repeat` (arm 2 again, a fresh load; lane 05's stability gate).

## Prediction

**The cache is applied.** Arm 2's stamp `context.applied_cache_classes` lists `RotatingKVCache` on all 36 layers (arm 1:
`KVCache` on 36); that reading, not the kwarg sent, is the finding lane 05 asked for.

**Control (arm 1), held-out /55:** unsure of the level, 28–40 (a new model with no 0-pad row of its own; the 4B sits
between lane 05's e4b at 38 and the LFM at 14; recap, the longest Dutch answers, the weakest exam). Thinking on:
`reasoning_chars` > 0 on ≥ 50 of 55 sittings, 300–1,500 generated tokens per sitting, `finish_length` on 0–5 recap
sittings at the 2,048 cap. Prompt 400–700 tok/s on this M1 Pro (36 full-attention layers over 8.2–9.3k tokens; slower per
token than the e4b's 7 global layers); gen 35–55 tok/s; p50 wall 35–60 s per case; peak **4.0–4.8 GB** (2.26 GB weights +
36 × 2 × 8 × 128 × 2 bytes ≈ 144 KB per token of KV, 1.2–1.35 GB at these prompts, + working memory).

**Truncated (arm 2) vs control: the num_ctx failure, in mlx form.** `RotatingKVCache(max_size=4096, keep=4)` keeps the
first 4 tokens and the last ~4,092 of an 8.2–9.3k prompt: wilco's system prompt (schema, rules, the answer format) and the
padding header are evicted; the case's own turn, at the end, stays. So: held-out total **≤ 19/55**, at least **15 of 55
below the control**; email-triage **≤ 10/20** (lane 05's committed number), call-fields ≤ 5/19, recap ≤ 4/16; the format
or schema check in wilco's `failed_checks` fails on **≥ 25 of the 39 email + call sittings** (the model answers the last
user turn without the system prompt's JSON contract: prose, a wrong schema, or a reply to the padding's last "earlier
message"). Peak **0.5–0.9 GB below** the control (the one thing `max_kv_size` is for); gen tok/s ×1.05–1.3 the control's
(attention over 4k instead of 8–9k on every layer); prompt tok/s within ±15 % (prefill still runs over the whole prompt).

**Repeat (arm 3 vs arm 2):** verdicts identical on ≥ 52 of 55 (lane 05's 8k repeats moved 0 of 55 on both models at
temperature 0; Qwen3's thinking is longer, so a few near-tie flips are possible; unsure). **Wh**: null (no powermetrics
capture), not 0.

**Kill clause** (registered here): (i) arm 2's stamped `applied_cache_classes` list no `RotatingKVCache` (the runtime
ignored `max_kv_size` on this class too: the probe cannot reach truncation on mlx-lm 0.31.3 at all, and that is the post);
(ii) the runtime raises on more than 5 of 55 sittings in arm 1 or arm 2 (a fault, not a graded failure; counted per call,
never as a model FAIL); (iii) the repeat differs from arm 2 in verdict on more than 3 of 55 (lane 05's gate: small effects
are void). If it fires, the row says **killed** and says which. A truncated arm that stays within 3 of 55 of the control
with the cache applied is **refuted, not killed**: then eviction of the system prompt does not break these exams, the
num_ctx shape on mlx is milder than on Ollama, and the post says so against lane 05's prediction.

## Mechanism

mlx-lm's `make_prompt_cache(model, max_kv_size)` defers to the model's `make_cache` when it exists (Gemma 4, LFM2, Llama:
their own sliding or hybrid caches, which ignore the argument); otherwise it builds a `RotatingKVCache(max_size, keep)` per
layer, which keeps the first `keep` tokens, fills to `max_size`, then overwrites the oldest positions. Prefill still attends
over the whole prompt chunk by chunk (`prefill_step_size` 2048), but once the cache is full the generation steps see only
`keep` + the last `max_size − keep` positions: the head of the prompt, where wilco puts the system prompt, is gone when the
answer is written. That is Ollama's `num_ctx` failure in mlx form: a context setting accepted, applied this time, and the
model answers from a prompt whose instructions it can no longer see. Lane 05 could not reach it because both its models own
their cache construction. Reopen: a second no-`make_cache` class (`qwen2`, `mistral3`, `phi3`) to show it is the class, not
Qwen3; `max_kv_size` 2048 and 6144 for the shape of the drop; the 8B pack; an mlx-lm whose `make_cache` models honour
`max_kv_size`; a wilco pin move; a re-sit flipping > 3 of 55.

## Addendum (11 Nov 2026, step 0, before any lane-13 sitting; source and cache readings only, no model generation)

- **No `make_cache` on `qwen3`** (mlx-lm 0.31.3, mlx 0.32.2, lab env):
  `grep -c "def make_cache" …/site-packages/mlx_lm/models/qwen3.py` → `0`; the same count on `llama.py`, `gemma4.py`,
  `lfm2.py` → `1` each.
- **`make_prompt_cache`** (`mlx_lm/models/cache.py:15-40`), read verbatim:
  `if hasattr(model, "make_cache"): return model.make_cache()` … `if max_kv_size is not None: return
  [RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)]` `else: return [KVCache() for _ in
  range(num_layers)]`. Read alongside it (`RotatingKVCache._update_concat`): a multi-token prefill chunk is concatenated
  after trimming the cache to `keep` + the last `max_size − 1` positions, so from the third 2048-token chunk on, prefill
  itself sees only `keep` + ~4,095 earlier positions plus its own chunk. "Prefill still attends over the whole prompt" in
  the mechanism is therefore loose: the head is already gone for the later prefill chunks, not only at generation. Nothing in
  the prediction changes.
- **Pack.** HF API (11 Nov 00:45 UTC): `mlx-community/Qwen3-4B-4bit` main = `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`
  (last modified 2025-04-28), 11 files, 2,278,972,183 bytes (`model.safetensors` 2,263,022,529). Fetched once with
  `snapshot_download(..., revision="4dcb3d10…")`; on disk 2.1 GiB (`du -shL` of the snapshot). config: `qwen3`, 36 layers,
  32 heads / 8 KV heads, `head_dim` 128, `max_position_embeddings` 40960, `sliding_window` null, 4-bit group 64, vocab
  151936, tied embeddings. A revision fetch writes no `refs/main`, so the lab's offline `resolve_model` could not find the
  pack by repo id; `refs/main` was written by hand with the same sha (main's head per the API read above). From here on
  `HF_HUB_OFFLINE=1`.
- **Thinking.** mlx-lm's tokenizer wrapper on this pack: `has_thinking` True, `<think>` / `</think>`; the generation prompt
  ends `<|im_start|>assistant\n` (no `<think>` in the prompt, so `split_thinking` starts in content). Sat as it ships.
- **Faults.** `lab context` as committed turns any mlx exception into a `RuntimeError` that ends the load (lane 05's
  rule); the partial file is kept under `.lab-scratch/`. Kill (ii) is read from that: an arm that aborts is reported with
  the sitting it stopped on and its stderr, never scored.
  Lane 05's e4b 8k arm already ran this same `RotatingKVCache._update_concat` trim path (its 20 sliding layers) on 55
  sittings at 8k with no fault, so the code is not changed mid-lane. An arm that aborts is reported as kill (ii) fired:
  the sitting it stopped on, the partial rows and the stderr.
- **What "the format or schema check fails" counts (fixed now, before any sitting).** At the pin, wilco's email-triage and
  call-fields have no `format` check. A non-JSON answer files as `{"bucket": "quality", "check": "error"}`, with wilco's
  `no JSON object found in model output: …` kept in the row's `wilco_error` (`sandbox/runner.py:2153-2155`); valid JSON
  with the wrong keys files as `labels` / `fields` (`_fields_matched`: a missing key is a mismatch), the same entry as a
  wrong value. So a sitting counts toward the "≥ 25 of 39" when its `failed_checks` hold any entry with bucket `format`
  or check `error`. A `termination` entry (no answer, e.g. thinking to the token cap) does **not** count and is reported
  on its own line. A wrong schema in valid JSON is not visible in the rows and is not counted, so this count is a lower
  bound on the predicted failure; the prediction is graded on it as written.
