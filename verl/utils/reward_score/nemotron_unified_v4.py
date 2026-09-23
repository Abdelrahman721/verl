# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
v4 reward for data_source "nemotron_pivot": v3 scoring behind a strict format gate.

v3 (nemotron_pivot_judge.py) inherited vLLM hermes-parser leniency from nemotron_pivot.
extract_action: an unclosed trailing <tool_call> was still parsed, text between and around
blocks was ignored, and a response that never closed </think> was scanned whole. The late
v3 dumps show the policy exploiting every one of those - junk tokens between blocks, stray
<think> tags after the reasoning, and ~8% of rollouts never closing </think> - while still
collecting full tool reward. v4 accepts only the exact layout the SFT data renders and never
tries to recover anything else.

Format gate, checked in this order:

  1. Think block. The response must start with "<think>" and contain exactly one "<think>"
     and exactly one "</think>". Otherwise the rollout gets its row's floor - -1.5 on tool
     rows, -1.0 on prose rows - and the judge is not called (think_error=1).

  2. Tool-call markup, on the text after </think>. If that text contains no "<tool_call>" and
     no "</tool_call>", it is prose. If it contains either tag, it must be, in full:

         ws <tool_call> BODY </tool_call> (ws <tool_call> BODY </tool_call>)* ws

     where ws is whitespace only and every BODY is a JSON object with exactly the keys
     "name" (non-empty string) and "arguments" (JSON object), no duplicate keys, and no
     NaN/Infinity. Any deviation - unclosed or stray tag, text before, between or after the
     blocks, arguments given as a string or omitted, extra keys - means the response is NOT
     a tool call: it is prose with format_error=1. Nothing is salvaged; one bad block voids
     the others.

Scoring after the gate (identical to v3 except the prose-on-tool-row penalty):

  * expected TOOL CALL -> valid calls: nemotron_pivot_judge.smooth_call_score, in [-1, 1].
                          prose (including every format_error): PROSE_ON_CALL_ROW_SCORE, -1.5.
  * expected PROSE     -> valid calls: -1.0, no judge call.
                          prose (including format_error): LLM judge on the text after
                          </think>, +1 pass / -1 fail, NEMOTRON_JUDGE_FALLBACK when unreachable.
                          A reply carrying broken tool markup is judged like any other prose.

Why -1.5: on the v3 sft737 run, tool calls on single-call rows fell from 43% to 26% of
rollouts over 19 steps. Prose and a wrong call both scored -1 there, so a group that was all
prose or all wrong calls had zero variance and was filtered, and prose kept winning on prose
rows. Pricing prose below the worst possible call (-1) means any call attempt now outranks
prose inside a group. A broken think block takes the same floor on tool rows; at -1 it would
become the cheaper way out of calling.

Judge configuration and env vars are shared with v3 (see nemotron_pivot_judge.py).

Logged keys (always all present, always float): score, think_error, format_error, is_call,
type_match, name_match, judged, judge_score, judge_error.
"""
from __future__ import annotations

import json
import re
from typing import Any

from . import nemotron_pivot, nemotron_pivot_judge

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_CALL_OPEN = "<tool_call>"
_CALL_CLOSE = "</tool_call>"
_TAG_SPLIT_RE = re.compile(r"(<tool_call>|</tool_call>)")

PROSE_ON_CALL_ROW_SCORE = -1.5


class _Invalid(Exception):
    pass


def _reject_constant(token: str) -> Any:
    raise _Invalid(f"non-finite JSON constant {token}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise _Invalid(f"duplicate key {k!r}")
        out[k] = v
    return out


def _parse_body(body: str) -> dict:
    """One <tool_call> body -> {"type": "function_call", "name", "arguments": JSON string}."""
    try:
        obj = json.loads(body, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
    except json.JSONDecodeError as e:
        raise _Invalid(f"body is not JSON: {e}") from None
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
        raise _Invalid("body must be an object with exactly the keys name and arguments")
    if not isinstance(obj["name"], str) or not obj["name"]:
        raise _Invalid("name must be a non-empty string")
    if not isinstance(obj["arguments"], dict):
        raise _Invalid("arguments must be a JSON object")
    return {
        "type": "function_call",
        "name": obj["name"],
        "arguments": json.dumps(obj["arguments"], ensure_ascii=False),
    }


def _parse_calls(tail: str) -> list[dict]:
    """Strictly parse the post-think text as a sequence of tool-call blocks, or raise _Invalid.

    re.split with a capturing group yields [text, tag, text, tag, ..., text], so a well-formed
    response is exactly: ws, open, body, close, ws, open, body, close, ..., ws.
    """
    parts = _TAG_SPLIT_RE.split(tail)
    tags = parts[1::2]
    texts = parts[0::2]
    if not tags or len(tags) % 2:
        raise _Invalid("unbalanced tool_call tags")
    for i, tag in enumerate(tags):
        if tag != (_CALL_OPEN if i % 2 == 0 else _CALL_CLOSE):
            raise _Invalid("tool_call tags out of order")
    # texts[k] sits before tags[k]: even k is outside a block, odd k is a block body.
    calls = []
    for k, text in enumerate(texts):
        if k % 2 == 0:
            if text.strip():
                raise _Invalid("text outside tool_call blocks")
        else:
            calls.append(_parse_body(text))
    return calls


def classify(text: str) -> dict:
    """-> {"think_ok": bool, "tail": str, "calls": list[dict], "format_error": bool}.

    `calls` is non-empty only for a fully valid tool-call response; everything else is prose.
    """
    text = text or ""
    if not (text.startswith(_THINK_OPEN) and text.count(_THINK_OPEN) == 1 and text.count(_THINK_CLOSE) == 1):
        # format_error stays tool-markup-only so the two failure metrics never overlap.
        return {"think_ok": False, "tail": "", "calls": [], "format_error": False}
    tail = text.split(_THINK_CLOSE, 1)[1]
    if _CALL_OPEN not in tail and _CALL_CLOSE not in tail:
        return {"think_ok": True, "tail": tail, "calls": [], "format_error": False}
    try:
        calls = _parse_calls(tail)
    except _Invalid:
        return {"think_ok": True, "tail": tail, "calls": [], "format_error": True}
    return {"think_ok": True, "tail": tail, "calls": calls, "format_error": False}


def _as_action(calls: list[dict]) -> dict:
    if len(calls) == 1:
        return calls[0]
    return {"type": "function_call_batch", "calls": calls}


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    last_user_message: str = "",
    **kwargs: Any,
) -> dict:
    del extra_info, kwargs
    out = {
        "score": -1.0,
        "think_error": 0.0,
        "format_error": 0.0,
        "is_call": 0.0,
        "type_match": 0.0,
        "name_match": 0.0,
        "judged": 0.0,
        "judge_score": 0.0,
        "judge_error": 0.0,
    }
    try:
        expected = json.loads(ground_truth or "")
        expected_type = expected["type"]
    except (json.JSONDecodeError, KeyError, TypeError):
        out["score"] = 0.0
        return out

    c = classify(solution_str)
    out["format_error"] = float(c["format_error"])
    if not c["think_ok"]:
        out["think_error"] = 1.0
        if expected_type != "message":
            out["score"] = PROSE_ON_CALL_ROW_SCORE
        return out
    out["is_call"] = float(bool(c["calls"]))

    if expected_type != "message":
        if not c["calls"]:
            out["score"] = PROSE_ON_CALL_ROW_SCORE
            return out
        out["type_match"] = 1.0
        exp_names = {e["name"] for e in nemotron_pivot._calls_of(expected)}
        out["name_match"] = float(any(a["name"] in exp_names for a in c["calls"]))
        out["score"] = nemotron_pivot_judge.smooth_call_score(expected, _as_action(c["calls"]))
        return out

    if c["calls"]:
        return out
    out["type_match"] = 1.0
    score, err = nemotron_pivot_judge.judge_prose(last_user_message, c["tail"].strip())
    out["judged"] = 1.0
    if score is None:
        out["judge_error"] = 1.0
        out["score"] = float(nemotron_pivot_judge._env("NEMOTRON_JUDGE_FALLBACK", "0.0"))
        out["judge_score"] = out["score"]
        if nemotron_pivot_judge._env("NEMOTRON_JUDGE_VERBOSE", ""):
            print(f"[nemotron_v4] judge failed: {err}")
        return out
    out["judge_score"] = score
    out["score"] = 1.0 if score == 1.0 else -1.0
    return out
