"""Lane 05: pad a case's prompt with the session's earlier inputs to a token target, and pass KV-cache settings.

The rule is pre-registered in predictions/context-kv.md (addendum, 9 Nov). In short:
  * one user turn (HEADER + pool prefix) and one assistant turn (ACK) are inserted between wilco's system message and the
    case's own user turn, found by identity (the first user message whose content == case["input"]); wilco's re-ask keeps
    the same padding in the same place;
  * pool = the sitting exam's TRAIN inputs (wilco order), then the other exams' train inputs (sorted), minus the sitting
    case and any train input containing its text; no held-out input and no gold answer is ever in a pool;
  * the pool is cut at the longest character prefix keeping the FIRST call's templated prompt <= target tokens in the
    model's own tokenizer; a case already >= target - SLACK tokens is not padded.
The case dict and the system prompt are never touched, so wilco's grader sees exactly what it sees unpadded.
"""
import hashlib
import json

from . import cases as C

HEADER = ("Earlier messages from this session, already handled. Do not act on them; your task is the next message.\n\n")
ACK = "Understood."
SEP = "\n\n--- earlier message ---\n\n"
SLACK = 64
KV_KEYS = ("kv_bits", "kv_group_size", "quantized_kv_start", "max_kv_size")


def pool_text(exam, case=None, exams=None):
    """Train inputs, own exam first. Excluded: the case itself and any train input that CONTAINS the case's input (wilco
    at the pin has one: held-out recap `quiet-day-newsletters` is verbatim inside train recap `compression-heavy`)."""
    order = [exam] + sorted(e for e in (exams or C.EXAMS) if e != exam)
    items = []
    for e in order:
        for c in json.loads(C.cases_path(e).read_text())["cases"]:
            if c["split"] != "train" or (case is not None and (c["id"] == case["id"] or case["input"] in c["input"])):
                continue
            items.append(c["input"])
    return SEP.join(items)


def find_case_turn(messages, case_input):
    for i, m in enumerate(messages):
        if m.get("role") == "user" and m.get("content") == case_input:
            return i
    raise ValueError("refused: the case's own user turn is not in the messages (padding has nowhere to go)")


def insert_padding(messages, case_input, pad):
    """A NEW list; wilco's list is not mutated. pad == "" returns the messages unchanged (a copy)."""
    if not pad:
        return list(messages)
    i = find_case_turn(messages, case_input)
    return messages[:i] + [{"role": "user", "content": HEADER + pad}, {"role": "assistant", "content": ACK}] + messages[i:]


def fit_padding(count_tokens, messages, case_input, pool, target):
    """(pad, n_tokens): the longest pool prefix with count_tokens(padded messages) <= target; ("", base) if no room."""
    base = count_tokens(messages)
    if base >= target - SLACK:
        return "", base
    lo, hi = 0, len(pool)
    if count_tokens(insert_padding(messages, case_input, pool[:1] or " ")) > target:
        return "", base
    while lo < hi:                                  # largest k with tokens(pool[:k]) <= target
        mid = (lo + hi + 1) // 2
        if count_tokens(insert_padding(messages, case_input, pool[:mid])) <= target:
            lo = mid
        else:
            hi = mid - 1
    pad = pool[:lo]
    return pad, count_tokens(insert_padding(messages, case_input, pad))


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def check_settings(ctx):
    bad = [k for k in ctx if k not in ("pad_to",) + KV_KEYS]
    if bad:
        raise ValueError(f"unknown context settings {bad}")
    if ctx.get("pad_to") is not None and int(ctx["pad_to"]) <= 0:
        raise ValueError("pad_to must be positive")
    return ctx
