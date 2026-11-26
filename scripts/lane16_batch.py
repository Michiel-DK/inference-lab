"""Lane 16 (batch-invariance): does an answer change with the batch it shares, under mlx-lm at temperature 0?

Prediction: predictions/batch-invariance.md (committed before this script). Every fixed list the run depends on (the arm-1
batch mates, the arm-2 shuffles, the process plan) is in this file, committed before any model is loaded.

Subcommands (run with the lab env, HF_HUB_OFFLINE=1, LAB_WILCO_DIR at the pin):
  capture                      wilco's own prompt assembly, no model: docs/probes/batch-invariance/prompts.jsonl
  gen --job <id> --out <file>  ONE fresh process: load the model, generate the job's completions, one JSON line each
  grade                        replay every arm-2 answer through wilco's run_exam; writes rows.jsonl (arm 1 rows pass through)
  plan                         print the job ids in run order (the drive loop reads it)

Nothing under lab/ is changed: the solo path below is lab/metering.py MeteredMLX.generate's, line for line in effect
(apply_chat_template, stream_generate, make_sampler(temp=0.0), no prompt cache), and lab.metering.split_thinking is imported.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "probes" / "batch-invariance"
MODEL = os.environ.get("LANE16_MODEL", "mlx-community/gemma-4-e4b-it-8bit")
EXPECT_SNAPSHOT = {"mlx-community/gemma-4-e4b-it-8bit": "4255b21bd9a9d3fc807ef7abd80373f5e3a52a73"}
EXAMS = ("email-triage", "call-fields")

# ---- arm 1: their prompt, and fixed mates of other lengths (English, chosen before any run; lengths differ on purpose)
ARM1_PROMPT = "Tell me about Richard Feynman"
ARM1_TOKENS = 200
ARM1_MATES = [
    "What is 7 times 8?",
    "Write a haiku about rain on a tin roof.",
    "Explain in three sentences why the sky is blue, as you would to a ten-year-old who has just asked you at the beach.",
    "List five things a small bakery should check before it signs a lease for a new shop, with one line on each.",
    "Summarise the plot of Romeo and Juliet in about one hundred words, then name the two families and say where the play is set.",
    "A customer writes: 'I ordered two boxes of printer paper on Monday and only one arrived today. The invoice says two. Can you "
    "sort this out before Friday? We have a big print job.' Draft a short, polite reply from the supplier.",
    "You are helping a team plan a two-day offsite for twelve people in Lisbon in March. Propose an agenda with morning and "
    "afternoon blocks for each day, one dinner suggestion per evening, a rough budget per person in euros, and three questions "
    "the organiser should answer before booking anything. Keep it under three hundred words and use plain headings.",
]
ARM1_SIZES = (2, 4, 8)
ARM1_REPS = 2          # each configuration in 2 fresh processes
ARM1_SOLO_PROCS = 4    # solo: 4 fresh processes; the first 2 also sit the batch-of-1 BatchGenerator reference

# ---- arm 2: the 60 cases, two fixed shuffles per batch size
ARM2_SIZES = (4, 8)
ARM2_SHUFFLE_SEEDS = {0: 1601, 1: 1602}


def arm1_jobs():
    jobs = [f"a1-solo-p{p}" for p in range(ARM1_SOLO_PROCS)]
    for rep in range(ARM1_REPS):
        for b in ARM1_SIZES:
            jobs.append(f"a1-b{b}-copies-r{rep}")
            for slot in range(b):
                jobs.append(f"a1-b{b}-others-s{slot}-r{rep}")
    return jobs


def arm2_jobs():
    return ["a2-solo", "a2-b1"] + [f"a2-b{b}-sh{sh}" for b in ARM2_SIZES for sh in ARM2_SHUFFLE_SEEDS]


GATE_JOBS = ["gate-a", "gate-b", "gate-c-p0", "gate-c-p1"]


def plan():
    return GATE_JOBS + arm1_jobs() + arm2_jobs()


def arm1_batch(job):
    """(list of prompt texts, slot of the target or None for copies, mate set) for an arm-1 batched job."""
    parts = job.split("-")            # a1-b8-others-s3-r0 | a1-b4-copies-r1
    b = int(parts[1][1:])
    if parts[2] == "copies":
        return [ARM1_PROMPT] * b, None, "copies"
    slot = int(parts[3][1:])
    mates = ARM1_MATES[: b - 1]
    texts = mates[:slot] + [ARM1_PROMPT] + mates[slot:]
    return texts, slot, "others"


def case_key(p):
    return f"{p['exam']}/{p['case']}"


def shuffled(prompts, sh):
    keys = sorted(case_key(p) for p in prompts)
    random.Random(ARM2_SHUFFLE_SEEDS[sh]).shuffle(keys)
    return keys


# ---- wilco plumbing (read-only, as lab/run.py does it)

def wilco():
    from lab.run import import_wilco_runner
    from lab import cases as C
    problems = C.verify_pin()
    if problems:
        raise SystemExit(f"refused: wilco pin: {problems}")
    return import_wilco_runner()


class _Recorder:
    def __init__(self):
        self.seen = []

    def generate(self, messages, model, temperature=0.0, max_tokens=512):
        self.seen.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        return '{"label": "reply_later"}', {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                                            "finish_reason": "stop", "content_chars": 25, "reasoning_chars": 0}


def capture():
    runner, _ = wilco()
    from lab import cases as C
    rec = _Recorder()
    real = runner.adapter_for
    runner.adapter_for = lambda *a, **k: rec
    out = []
    try:
        for exam in EXAMS:
            agent = runner.load_agent(exam)
            doc = json.loads(C.cases_path(exam).read_text())
            for case in doc["cases"]:
                n0 = len(rec.seen)
                runner.run_exam(agent, {**doc, "cases": [case]}, provider="mlx", model=MODEL)
                calls = rec.seen[n0:]
                if len(calls) != 1:
                    raise SystemExit(f"{exam}/{case['id']}: the recorder was called {len(calls)} times, expected 1")
                out.append({"exam": exam, "case": case["id"], "split": case.get("split", "train"), **calls[0]})
    finally:
        runner.adapter_for = real
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "prompts.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"captured {len(out)} first prompts -> {OUT / 'prompts.jsonl'}", file=sys.stderr)


def load_prompts():
    return [json.loads(x) for x in (OUT / "prompts.jsonl").read_text().splitlines() if x.strip()]


# ---- generation (one fresh process per job)

class Gen:
    def __init__(self):
        import mlx.core as mx
        from mlx_lm import load
        from lab.run import resolve_model
        self.mx = mx
        path, self.snapshot = resolve_model(MODEL)
        if MODEL in EXPECT_SNAPSHOT and self.snapshot != EXPECT_SNAPSHOT[MODEL]:
            raise SystemExit(f"refused: {MODEL} snapshot {self.snapshot} != {EXPECT_SNAPSHOT[MODEL]}")
        t0 = time.perf_counter()
        self.model, self.tok = load(path)
        self.load_s = time.perf_counter() - t0

    def template(self, messages):
        return self.tok.apply_chat_template(messages, add_generation_prompt=True)

    def solo(self, ids, max_tokens):
        """lab/metering.py MeteredMLX.generate's path: stream_generate, greedy sampler, no prompt cache."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        self.mx.reset_peak_memory()
        toks, parts, last = [], [], None
        t0 = time.perf_counter()
        for r in stream_generate(self.model, self.tok, ids, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)):
            parts.append(r.text)
            if r.finish_reason is None:
                toks.append(int(r.token))
            last = r
        return [{"tokens": toks, "stream_text": "".join(parts), "finish_reason": last.finish_reason,
                 "wall_s": time.perf_counter() - t0, "peak_memory_gb": self.mx.get_peak_memory() / 1e9}]

    def batch(self, ids_list, max_tokens_list, n):
        """batch_generate's loop (mlx_lm/generate.py:1887-1964) with the token ids kept; prefill and completion batch = n."""
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.sample_utils import make_sampler
        self.mx.reset_peak_memory()
        t0 = time.perf_counter()
        gen = BatchGenerator(self.model, stop_tokens=[[t] for t in self.tok.eos_token_ids], sampler=make_sampler(temp=0.0),
                             prefill_batch_size=n, completion_batch_size=n)
        uids = gen.insert(ids_list, max_tokens_list)
        got = {u: [] for u in uids}
        fin = {}
        while responses := gen.next_generated():
            for r in responses:
                if r.finish_reason != "stop":
                    got[r.uid].append(int(r.token))
                if r.finish_reason is not None:
                    fin[r.uid] = r.finish_reason
        gen.close()
        wall = time.perf_counter() - t0
        peak = self.mx.get_peak_memory() / 1e9
        return [{"tokens": got[u], "finish_reason": fin.get(u), "wall_s": wall, "peak_memory_gb": peak} for u in uids]


def sha_ids(ids):
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def content_of(tok, prompt_ids, text):
    from lab.metering import split_thinking
    starts_in = False
    if tok.has_thinking:
        ptxt = tok.decode(prompt_ids)
        starts_in = ptxt.rfind(tok.think_start) > ptxt.rfind(tok.think_end)
    return split_thinking(text, tok.think_start if tok.has_thinking else None,
                          tok.think_end if tok.has_thinking else None, starts_in)


def gen_job(job, out_path):
    g = Gen()
    pid = os.getpid()
    rows = []

    def row(arm, path, n, slot, mates, ids, res, **extra):
        text = g.tok.decode(res["tokens"])
        content, reasoning = content_of(g.tok, ids, text)
        r = {"arm": arm, "job": job, "model": MODEL, "snapshot": g.snapshot, "path": path, "batch_size": n, "slot": slot,
             "mates": mates, "process_id": pid, "prefill_batch_size": n if path == "batch" else None,
             "completion_batch_size": n if path == "batch" else None, "prompt_tokens": len(ids),
             "prompt_sha256": sha_ids(ids), "tokens": res["tokens"], "n_tokens": len(res["tokens"]),
             "text": text, "content": content, "reasoning_chars": len(reasoning),
             "finish_reason": res["finish_reason"], "wall_s": res["wall_s"], "peak_memory_gb": res["peak_memory_gb"], **extra}
        if "stream_text" in res:
            r["stream_text_equals_decode"] = res["stream_text"] == text
        rows.append(r)

    msg = lambda t: [{"role": "user", "content": t}]
    if job.startswith("a1-solo-"):
        p = int(job.split("-p")[1])
        ids = g.template(msg(ARM1_PROMPT))
        row("arm1", "solo", 1, 0, "solo", ids, g.solo(ids, ARM1_TOKENS)[0])
        if p < 2:
            row("arm1", "batch", 1, 0, "b1-ref", ids, g.batch([ids], [ARM1_TOKENS], 1)[0])
    elif job.startswith("a1-b"):
        texts, slot, mates = arm1_batch(job)
        ids_list = [g.template(msg(t)) for t in texts]
        res = g.batch(ids_list, [ARM1_TOKENS] * len(texts), len(texts))
        for s, (t, ids, r) in enumerate(zip(texts, ids_list, res)):
            if t != ARM1_PROMPT:
                continue                   # mates' own completions are not kept: the question is the target's answer
            row("arm1", "batch", len(texts), s, mates, ids, r, rep=int(job.rsplit("-r", 1)[1]))
    elif job.startswith("a2-") or job.startswith("gate-"):
        prompts = {case_key(p): p for p in load_prompts()}
        if job == "a2-solo":
            for k in sorted(prompts):
                p = prompts[k]
                ids = g.template(p["messages"])
                row("arm2", "solo", 1, 0, "solo", ids, g.solo(ids, p["max_tokens"])[0], exam=p["exam"], case=p["case"],
                    split=p["split"], max_tokens=p["max_tokens"])
        elif job == "a2-b1":
            for k in sorted(prompts):
                p = prompts[k]
                ids = g.template(p["messages"])
                row("arm2", "batch", 1, 0, "b1-ref", ids, g.batch([ids], [p["max_tokens"]], 1)[0], exam=p["exam"],
                    case=p["case"], split=p["split"], max_tokens=p["max_tokens"])
        elif job.startswith("a2-b"):
            b, sh = int(job.split("-")[1][1:]), int(job.split("-sh")[1])
            order = shuffled(prompts.values(), sh)
            for bi in range(0, len(order), b):
                chunk = [prompts[k] for k in order[bi: bi + b]]
                ids_list = [g.template(p["messages"]) for p in chunk]
                res = g.batch(ids_list, [p["max_tokens"] for p in chunk], len(chunk))
                for s, (p, ids, r) in enumerate(zip(chunk, ids_list, res)):
                    row("arm2", "batch", len(chunk), s, f"sh{sh}", ids, r, exam=p["exam"], case=p["case"], split=p["split"],
                        max_tokens=p["max_tokens"], batch_index=bi // b, started_as=b)
        else:                                                    # gates: train cases only
            train = [prompts[k] for k in sorted(prompts) if prompts[k]["split"] == "train"]
            if job == "gate-a":                                   # the library function itself, on 2 train prompts
                from mlx_lm import batch_generate
                from mlx_lm.sample_utils import make_sampler
                two = [train[0], train[-1]]
                ids_list = [g.template(p["messages"]) for p in two]
                br = batch_generate(g.model, g.tok, ids_list, max_tokens=200, sampler=make_sampler(temp=0.0))
                for p, ids, t in zip(two, ids_list, br.texts):
                    rows.append({"arm": "gate", "job": job, "model": MODEL, "case": p["case"], "exam": p["exam"],
                                 "prompt_sha256": sha_ids(ids), "text": t, "nonempty": bool(t.strip()),
                                 "peak_memory_gb": br.stats.peak_memory})
            elif job == "gate-b":                                 # the 8 LONGEST train prompts, one batch, 200 tokens
                ids_by = sorted(((len(g.template(p["messages"])), p) for p in train), key=lambda x: -x[0])[:8]
                chunk = [p for _, p in ids_by]
                ids_list = [g.template(p["messages"]) for p in chunk]
                res = g.batch(ids_list, [200] * 8, 8)
                for s, (p, ids, r) in enumerate(zip(chunk, ids_list, res)):
                    row("gate", "batch", 8, s, "gate-b", ids, r, exam=p["exam"], case=p["case"], split=p["split"])
            else:                                                 # gate-c: one train prompt solo, in two processes
                p = train[0]
                ids = g.template(p["messages"])
                row("gate", "solo", 1, 0, "gate-c", ids, g.solo(ids, 200)[0], exam=p["exam"], case=p["case"],
                    split=p["split"])
    else:
        raise SystemExit(f"unknown job {job}")
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.write(json.dumps({"_footer": {"job": job, "process_id": pid, "load_s": g.load_s, "n_rows": len(rows)}}) + "\n")


# ---- grading: replay each arm-2 answer through wilco's own run_exam

class Reask(Exception):
    """wilco called generate a second time for the case (its re-ask): there is no second answer to replay."""


class _Replay:
    def __init__(self, content, finish_reason, n_tokens, reasoning_chars, termination_exc):
        self.content, self.fr, self.n, self.rc, self.exc = content, finish_reason, n_tokens, reasoning_chars, termination_exc
        self.calls = 0

    def generate(self, messages, model, temperature=0.0, max_tokens=512):
        self.calls += 1
        if self.calls > 1:
            raise Reask()
        meta = {"prompt_tokens": None, "completion_tokens": self.n, "total_tokens": None, "finish_reason": self.fr,
                "content_chars": len(self.content), "reasoning_chars": self.rc}
        if not self.content.strip():                 # as lab/metering.py: an empty answer fails like wilco's router does
            raise self.exc(f"model emitted no answer (finish_reason={self.fr!r})",
                           {"finish_reason": self.fr, "completion_tokens": self.n, "content_chars": 0,
                            "reasoning_chars": self.rc})
        return self.content, meta


def grade():
    runner, router = wilco()
    from lab import cases as C
    docs = {e: json.loads(C.cases_path(e).read_text()) for e in EXAMS}
    agents = {e: runner.load_agent(e) for e in EXAMS}
    raw = []
    for f in sorted((OUT / "raw").glob("*.jsonl")):
        lines = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if not lines or "_footer" not in lines[-1]:
            raise SystemExit(f"{f}: no footer (interrupted job); not graded")
        raw += lines[:-1]
    real = runner.adapter_for
    out = []
    try:
        for r in raw:
            if r["arm"] == "arm2" or (r["arm"] == "gate" and "content" in r):   # gate-a keeps text only: not graded
                case = next(c for c in docs[r["exam"]]["cases"] if c["id"] == r["case"])
                rep = _Replay(r.get("content", ""), r.get("finish_reason"), r.get("n_tokens"), r.get("reasoning_chars", 0),
                              router.TerminationError)
                runner.adapter_for = lambda *a, **k: rep
                try:
                    res = runner.run_exam(agents[r["exam"]], {**docs[r["exam"]], "cases": [case]}, provider="mlx",
                                          model=MODEL)
                    g = res["cases"][0]
                    r = {**r, "reask": False, "passed": bool(g["passed"]), "failed_checks": g.get("failed_checks", [])}
                    if isinstance(g.get("got"), dict) and "error" in g["got"]:
                        r["wilco_error"] = str(g["got"]["error"])[:300]
                except Reask:
                    r = {**r, "reask": True, "passed": None, "failed_checks": []}
            out.append(r)
    finally:
        runner.adapter_for = real
    with open(OUT / "rows.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"graded -> {OUT / 'rows.jsonl'} ({len(out)} rows)", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("capture")
    sub.add_parser("plan")
    sub.add_parser("grade")
    g = sub.add_parser("gen")
    g.add_argument("--job", required=True)
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "capture":
        capture()
    elif a.cmd == "plan":
        print("\n".join(plan()))
    elif a.cmd == "grade":
        grade()
    else:
        gen_job(a.job, a.out)
