# batch-invariance

Lane 16, written 24 Nov 2026 by the scope job `companies-tech-scope-2026-11-24`, before any model was loaded. Reader question B1.
Origin: the object in the brief. Committed 26 Nov 2026 by `lab-inf-16-batch-invariance`, still before any model was loaded, with
the corrections and departures that lane 14 (`docs/literature-2026-11-24.md` §2 rows 1, 2, 7 and §4) asks for, each marked
"(lane 14)". The scope job's numbers are unchanged.

origin: {"kind": "inbox", "url": "https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/", "label": "operator 24 Nov 2026: a smaller version of Thinking Machines' own tech; 'Defeating Nondeterminism in LLM Inference' (10 Sep 2025; rank C until lab-inf-14 reads it in full)"}

Source rows leaned on, all [F] in `docs/literature-2026-11-24.md`: row 1 (the post: 80 unique of 1000, most common 78, first
divergence at token 103 of 1000, Qwen3-235B-A22B behind vLLM, varying server load), row 2 (`batch_invariant_ops`: **correction
(lane 14)**, the README's "18 unique samples" out of 1000 names no model; the script uses `Qwen/Qwen3-8B` and a different prompt,
"Generate 1000 random numbers ...", at 100 tokens), row 7 (`mlx-lm` 0.31.3 batching source: pads are masked and rolled; nothing
makes a row depend on its mates by construction; bitwise equality is not guaranteed).

## Prediction
**What runs.** `mlx-community/gemma-4-e4b-it-8bit` (the lab's champion), temperature 0, mlx 0.32.2 / mlx-lm 0.31.3, cached files
only. Two arms.
**Arm 1, their test at laptop size.** Their prompt, "Tell me about Richard Feynman", 200 new tokens. The prompt is answered alone
4 times (4 fresh processes), and in batches of 2, 4 and 8: in every slot of the batch, once with copies of itself as batch mates and
once with fixed other prompts of other lengths, each configuration in 2 fresh processes. That is 4 + 56 = 60 completions of the same
prompt. Reported: how many of the 60 are distinct, the first token at which a batched completion leaves the solo one, and the split
by same-length against other-length mates.
**Arm 2, our exams.** The 28 email-triage and 32 call-fields cases at the wilco pin (60 cases, train and held-out; nothing is tuned).
Each case's first prompt, exactly as wilco assembles it, is answered alone, and in batches of 4 and of 8 under two fixed shuffles
each (4 batched sittings). wilco's own grader scores every answer. Reported per batched sitting: answers that differ in bytes from
the solo answer, and verdicts that differ. A case whose solo answer makes wilco ask again is counted and left out of both numbers.
**Numbers (the scope job's guesses).**
- Arm 1: the 4 solo completions are byte-identical. Among the 60, between 3 and 15 distinct completions; chance that all 60 are
  identical about 1 in 4. More distinct completions with other-length mates than with copies. First departure after token 20.
- Arm 2: per batched sitting, 2 to 15 of 60 answers differ in bytes, and 0 to 2 of 60 verdicts differ. Across the 4 sittings at
  least one verdict moves: unsure, about even.
**Kills.** (1) The 4 solo completions of arm 1 are not byte-identical, or the solo sitting of arm 2 does not give the verdicts of
`results/champion-e4b-8bit.json` (load 0) on the cases compared, more than 2 apart: the floor is not what the lab has posted, and
nothing about batches is read until that is explained. (2) `batch_generate` does not run on this model class and on no other cached
model: the lane posts that, as a smoke.
**What this cannot say.** Nothing about a server under load (their setting: the batch size itself varies with traffic), about
Ollama or llama.cpp, or about any other machine. Two exams, one model, 60 cases: a count, not a rate.

### Definitions and departures, fixed before the run (lane 14, and this lane's design check)
- **"Differ in bytes" means the generated token ids differ** (thinking included, the stop token excluded on both paths). Secondary,
  reported beside it: the graded `content` (after the lab's `split_thinking`) differs. The decider's "more than 10% of answers
  differ in bytes" is read on the token ids.
- **"Solo"** is generated exactly as `lab/metering.py: MeteredMLX.generate` does: `apply_chat_template(messages,
  add_generation_prompt=True)`, `stream_generate`, `make_sampler(temp=0.0)`, no prompt cache passed. **(lane 14)** `BatchGenerator`
  builds batch caches with array masks even at batch size 1, so a **batch-of-1 `BatchGenerator`** sitting is added as a second
  reference on both arms (2 completions on arm 1, one per solo process of the first two; one sitting of 60 on arm 2). It is reported
  apart and is not part of arm 1's 60 or arm 2's 4 sittings. "Solo vs batch of N" and "batch of 1 vs batch of N" are both posted.
- **Batched** completions use `mlx_lm.generate.BatchGenerator` directly (the loop of `batch_generate`, which returns text only,
  rewritten to keep the token ids) with `prefill_batch_size = completion_batch_size = N` set explicitly and written in every row, and
  the tokenizer's EOS ids as stop tokens. **(lane 14)** Finished rows leave the batch (`filter`), so the page says "started as a
  batch of N".
- **(lane 14)** Arm 1's distinct count is a count over a designed mix (4 solo, 56 in controlled batches), not over 1000 requests
  under uncontrolled load: it is posted pooled and **per configuration** (batch size × mate set), and is not their quantity. Their
  first divergence was at token 103 of 1000; a null at 200 tokens does not test their claim at 1000. The stub's "after token 20" has
  no source.
- **Arm 1 is not their "non-thinking mode".** The lab's Gemma 4 template is used as the lab uses it, and the champion thinks
  (lane 10 load 0, first email call: 1,268 reasoning chars), so most of the 200 tokens may be thought, not prose.
- **Re-asks.** A case whose solo answer makes wilco ask again is `reask`, counted, and left out of both numbers (stub). A **batched**
  answer that makes wilco ask again, when the solo did not, is counted as a verdict difference (the flow would take a different path)
  and its count is posted apart. The replay adapter has no second answer to give, so no re-asked verdict is invented.
- **Prompt sha256** is over the templated token ids (`json.dumps(ids)`), not the raw messages.
- **Kill (1), read tighter as well.** The solo sitting's per-case verdicts are compared with `results/champion-e4b-8bit.json` load 0
  (60 rows, 47 pass, 0 re-asks); its per-case `gen_tokens` are compared too and posted, but only verdicts fire the kill.
- **Batch mates and shuffles** are fixed lists in `scripts/lane16_batch.py`, committed before any model is loaded.

## Mechanism
Their argument: a kernel may add the same numbers in a different order when the batch has a different shape; floating-point sums
depend on order; at temperature 0 a difference in the last bits flips the choice between two near-tied tokens, and every later
token follows. On MLX the same should hold wherever a batched matrix product or a padded attention row is computed differently from
the single-request one. A short JSON label has few near-ties, so exam verdicts should move far less often than 200 tokens of prose.

## Addendum, 26 Nov 2026 23:30 (after the 8-bit gates, before any 4-bit load)
- **The 8-bit champion did not clear gate (b).** Gate (a) passed on `gemma-4-e4b-it-8bit` (`batch_generate` returned non-empty
  text for 2 train prompts, peak 9.09 GB, swap +770.5 MB). Gate (b) (one batch of the 8 longest train prompts, 200 tokens) was
  killed by the drive's swap stop after 28.7 s at swap +2092.8 MB over the lane's start (2208.5 MB), with no rows written
  (`docs/probes/batch-invariance/drive-8bit.log`, `raw-8bit/gate-a.jsonl`). The brief's fallback for gate (b) is the 4-bit e4b.
- **So both arms run on `mlx-community/gemma-4-e4b-it-4bit`** (snapshot `475b9088d29754a3379866cf5aeb6b41acd313c2`, the pack of
  lane 02's `dense-gemma-4-e4b-it-4bit`), gates (a), (b) and (c) re-sat on it. Both arms on one model; arm 1 is not left on 8-bit.
- **Kill (1)'s floor moves** to `results/dense-gemma-4-e4b-it-4bit.json` load 0 on the same 60 cases (39 pass, 0 re-asks,
  read 26 Nov), verdicts and per-call `gen_tokens` compared as in the definitions above.
- **The swap budget is the lane's**: stop at 2208.5 + 2048 MB used, whatever drive run is active. If the 4-bit gate (b) trips
  it too, the lane stops at the gates with both attempts posted; no third set-up (smaller prefill steps, fewer long cases) is tried,
  because each would change the design (chunked prefill is one of the causes the source names).
- The scope job's numbers are unchanged; they now read for the 4-bit pack.
