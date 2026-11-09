"""Lane 05 padding: the construction the addendum pre-registers (predictions/context-kv.md, 9 Nov). No GPU, no model."""
import json

import pytest

from lab import cases as C
from lab import context as X

HAVE_WILCO = (C.WILCO_DIR / "evals").exists()


def _msgs(case_input):
    return [{"role": "system", "content": "SYSTEM PROMPT\nlabels..."}, {"role": "user", "content": case_input}]


def test_padding_goes_between_system_and_case_turn_and_leaves_both_byte_identical():
    m = _msgs("the case")
    out = X.insert_padding(m, "the case", "earlier stuff")
    assert out[0] == m[0] and out[-1] == m[1]                      # system and the case's own turn: same objects' content
    assert out[1] == {"role": "user", "content": X.HEADER + "earlier stuff"}
    assert out[2] == {"role": "assistant", "content": X.ACK}
    assert m == _msgs("the case")                                  # wilco's list is not mutated
    assert X.insert_padding(m, "the case", "") == m                # zero pad = unchanged


def test_reask_keeps_padding_in_the_same_place():
    m = _msgs("the case") + [{"role": "assistant", "content": "garbage"}, {"role": "user", "content": "JSON only please"}]
    out = X.insert_padding(m, "the case", "P")
    assert [x["content"] for x in out] == ["SYSTEM PROMPT\nlabels...", X.HEADER + "P", X.ACK, "the case", "garbage",
                                           "JSON only please"]


def test_missing_case_turn_refuses():
    with pytest.raises(ValueError):
        X.insert_padding(_msgs("x"), "not there", "P")


def test_fit_is_the_longest_prefix_under_target():
    count = lambda ms: sum(len(x["content"]) for x in ms) // 4     # fake tokenizer: 4 chars per token
    m, pool = _msgs("c" * 400), "abcdefghij" * 2000
    pad, n = X.fit_padding(count, m, "c" * 400, pool, 2048)
    assert n <= 2048 and count(X.insert_padding(m, "c" * 400, pool[:len(pad) + 4])) > 2048
    assert pad == pool[:len(pad)]
    big = "c" * 4 * 3000                                           # already over the target: not padded
    assert X.fit_padding(count, _msgs(big), big, pool, 2048) == ("", count(_msgs(big)))


@pytest.mark.skipif(not HAVE_WILCO, reason="no wilco checkout on this machine")
def test_pool_is_train_inputs_only_and_never_contains_the_sitting_case():
    docs = {e: json.loads(C.cases_path(e).read_text())["cases"] for e in C.EXAMS}
    for exam in C.EXAMS:
        for case in docs[exam]:
            pool = X.pool_text(exam, case=case)
            want = [c["input"] for e in [exam] + sorted(e for e in C.EXAMS if e != exam) for c in docs[e]
                    if c["split"] == "train" and c["id"] != case["id"] and case["input"] not in c["input"]]
            assert pool.split(X.SEP) == want                       # exactly train inputs, in the pre-registered order
            assert case["input"] not in pool                       # the case never sees its own text as "history"
            assert '{"label"' not in pool and "must_mention" not in pool   # no gold answer object


@pytest.mark.skipif(not HAVE_WILCO, reason="no wilco checkout on this machine")
def test_the_one_split_overlap_at_the_pin_is_excluded():
    recap = {c["id"]: c for c in json.loads(C.cases_path("recap").read_text())["cases"]}
    case = recap["quiet-day-newsletters"]
    assert case["input"] in recap["compression-heavy"]["input"]    # wilco's own overlap (held-out inside train)
    assert recap["compression-heavy"]["input"] not in X.pool_text("recap", case=case).split(X.SEP)
