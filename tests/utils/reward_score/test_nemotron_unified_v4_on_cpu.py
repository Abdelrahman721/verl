# Copyright 2024 Bytedance Ltd. and/or its affiliates
import json

import pytest

from verl.utils.reward_score import nemotron_pivot_judge, nemotron_unified_v4 as v4

CALL = '<tool_call>\n{"name": "get_order", "arguments": {"order_id": "A12"}}\n</tool_call>'
CALL2 = '<tool_call>\n{"name": "get_order", "arguments": {"order_id": "B34"}}\n</tool_call>'
THINK = "<think>\nreasoning\n</think>\n\n"

GT_CALL = json.dumps({"type": "function_call", "name": "get_order", "arguments": json.dumps({"order_id": "A12"})})
GT_BATCH = json.dumps(
    {
        "type": "function_call_batch",
        "calls": [
            {"type": "function_call", "name": "get_order", "arguments": json.dumps({"order_id": "A12"})},
            {"type": "function_call", "name": "get_order", "arguments": json.dumps({"order_id": "B34"})},
        ],
    }
)
GT_MSG = json.dumps({"type": "message", "content": "Hello"})


@pytest.fixture
def judge(monkeypatch):
    calls = []

    def fake(user, cand):
        calls.append(cand)
        return 1.0, ""

    monkeypatch.setattr(nemotron_pivot_judge, "judge_prose", fake)
    return calls


@pytest.mark.parametrize(
    "response",
    [
        THINK + CALL,
        THINK + CALL + "\n",
        "<think></think>" + CALL,
    ],
)
def test_valid_single_call(response, judge):
    out = v4.compute_score(response, GT_CALL)
    assert out["score"] == 1.0 and out["is_call"] == 1.0 and out["format_error"] == 0.0
    assert judge == []


def test_valid_batch():
    out = v4.compute_score(THINK + CALL + "\n" + CALL2, GT_BATCH)
    assert out["score"] == 1.0 and out["is_call"] == 1.0


@pytest.mark.parametrize(
    "tail",
    [
        CALL[: -len("</tool_call>")],  # unclosed
        CALL + "\n" + CALL2[: -len("</tool_call>")],  # last of two unclosed
        CALL + "\n</tool_call>",  # stray close
        "Sure.\n" + CALL,  # text before
        CALL + "\nDone.",  # text after
        CALL + "\natemala\n" + CALL2,  # junk between
        '<tool_call>\n{"name": "get_order", "arguments": "{\\"order_id\\": \\"A12\\"}"}\n</tool_call>',  # string args
        '<tool_call>\n{"name": "get_order"}\n</tool_call>',  # missing arguments
        '<tool_call>\n{"name": "get_order", "arguments": {}, "id": 1}\n</tool_call>',  # extra key
        '<tool_call>\n{"name": "get_order", "name": "x", "arguments": {}}\n</tool_call>',  # duplicate key
        '<tool_call>\n{"name": "", "arguments": {}}\n</tool_call>',  # empty name
        '<tool_call>\n{"name": "get_order", "arguments": {"n": NaN}}\n</tool_call>',  # non-finite
        '<tool_call>\n{"name": "get_order", "arguments": {"order_id": "A12"}}}\n</tool_call>',  # bad JSON
        "<tool_call><tool_call>" + CALL,  # nested open
    ],
)
def test_malformed_markup_is_prose(tail):
    c = v4.classify(THINK + tail)
    assert c["think_ok"] and c["calls"] == [] and c["format_error"]
    out = v4.compute_score(THINK + tail, GT_CALL)
    assert out["score"] == -1.5 and out["is_call"] == 0.0 and out["format_error"] == 1.0


def test_malformed_markup_on_prose_row_goes_to_judge(judge):
    tail = CALL[: -len("</tool_call>")]
    out = v4.compute_score(THINK + tail, GT_MSG, last_user_message="hi")
    assert out["judged"] == 1.0 and out["format_error"] == 1.0 and out["score"] == 1.0
    assert judge == [tail.strip()]


@pytest.mark.parametrize(
    "response",
    [
        CALL,  # no think block
        "reasoning\n</think>\n" + CALL,  # no opening think
        " <think>x</think>" + CALL,  # leading space
        "<think>x" + CALL,  # never closed
        "<think>x</think>y</think>" + CALL,  # two closes
        "<think>x</think><think>y</think>" + CALL,  # two blocks
        THINK + CALL + "\n<think>\n" + CALL2,  # stray think tag between calls
    ],
)
@pytest.mark.parametrize("gt,floor", [(GT_CALL, -1.5), (GT_BATCH, -1.5), (GT_MSG, -1.0)])
def test_bad_think_gets_row_floor(response, gt, floor, judge):
    out = v4.compute_score(response, gt, last_user_message="hi")
    assert out["score"] == floor and out["think_error"] == 1.0 and out["judged"] == 0.0
    assert judge == []


def test_prose_row(judge):
    out = v4.compute_score(THINK + "Hello there!", GT_MSG, last_user_message="hi")
    assert out["score"] == 1.0 and out["judged"] == 1.0 and judge == ["Hello there!"]


def test_valid_call_on_prose_row_skips_judge(judge):
    out = v4.compute_score(THINK + CALL, GT_MSG, last_user_message="hi")
    assert out["score"] == -1.0 and out["is_call"] == 1.0 and judge == []


@pytest.mark.parametrize("gt", [GT_CALL, GT_BATCH])
def test_prose_on_call_row(gt):
    out = v4.compute_score(THINK + "I will look that up.", gt)
    assert out["score"] == -1.5 and out["format_error"] == 0.0


def test_any_valid_call_outranks_prose_on_call_row():
    wrong = '<tool_call>\n{"name": "cancel_order", "arguments": {"x": 1, "y": 2}}\n</tool_call>'
    many_wrong = "\n".join([wrong] * 5)
    for tail in (wrong, many_wrong):
        assert v4.compute_score(THINK + tail, GT_CALL)["score"] == -1.0 > v4.PROSE_ON_CALL_ROW_SCORE


def test_judge_unreachable_uses_fallback(monkeypatch):
    monkeypatch.setattr(nemotron_pivot_judge, "judge_prose", lambda u, c: (None, "HTTP 500"))
    out = v4.compute_score(THINK + "Hello", GT_MSG)
    assert out["score"] == 0.0 and out["judge_error"] == 1.0


def test_key_set_is_constant(judge):
    keys = {
        tuple(sorted(v4.compute_score(r, gt, last_user_message="hi")))
        for r in [THINK + CALL, CALL, THINK + "hi", THINK + CALL[:-3], ""]
        for gt in [GT_CALL, GT_MSG, "not json"]
    }
    assert len(keys) == 1
