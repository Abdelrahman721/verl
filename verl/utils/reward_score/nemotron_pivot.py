# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# The comparison rules below are a port of NVIDIA NeMo Gym's
# resources_servers/single_step_tool_use_with_argument_comparison/common/
# (verification_utils.py, response_utils.py), Apache-2.0, Copyright NVIDIA Corporation.
"""
Reward for data_source "nemotron_pivot": one decision step against an expert action.

The policy's raw text is classified into an action the way Gym's `extract_action` does:
tool-call blocks win over prose, several blocks become a batch, otherwise the text is a message.
That action is compared with the ground-truth action:

  * expected message   -> 1.0 for any prose reply, 0.0 for a tool call. Content is never compared.
  * expected call      -> 0.0 for prose; else name must match exactly and the arguments must match
                          recursively: same JSON type, same object keys, same list length, floats
                          within 1e-6, strings exact when either side is under two words and by
                          word-count overlap >= WORD_COUNT_SIMILARITY_THRESHOLD otherwise.
  * expected batch     -> unordered maximum bipartite matching of calls (no such rows in the
                          shipped dataset; kept for parity).

Call count follows Gym's canonical config for this dataset, where `parallel_tool_call_rewarding`
is off: a response scores 1.0 if every expected call is matched by some emitted call, and surplus
calls cost nothing. Note this is a reward-hacking vector under GRPO (spray calls, one lands). Set
NEMOTRON_PIVOT_STRICT_CALL_COUNT=1 to require the emitted call count to equal the expected count
(Gym's `parallel_tool_call_rewarding=true, allow_subset=False, allow_superset=False, binary_strict`).

Parsing notes: verl decodes responses with skip_special_tokens=True, but <think>, </think>,
<tool_call> and </tool_call> are added tokens flagged non-special in Qwen3, so they survive.
Everything up to the last </think> is discarded before classification. A <tool_call> block whose
body is not a JSON object with a "name" makes the whole output count as prose, exactly as vLLM's
hermes parser (which Gym relies on) falls back to; `format_error` flags it in the logged metrics.

Returns a dict so the reward manager logs the components: score, is_call, type_match, name_match,
format_error.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

WORD_COUNT_SIMILARITY_THRESHOLD = 0.1  # Gym's canonical config for this dataset
FLOAT_TOLERANCE = 1e-6

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_OPEN_TAG_RE = re.compile(r"<tool_call>")


# --------------------------------------------------------------------------- response -> action

def extract_action(text: str) -> dict:
    """Return {"type": "message"|"function_call"|"function_call_batch", ..., "format_error": bool}.

    Mirrors vLLM's hermes tool parser, which is what Gym's rows pass through: an unclosed
    <tool_call> is still parsed if its remainder is JSON, and any block that fails to parse makes
    the parser give up and return the WHOLE output as assistant text — i.e. a message action.
    `format_error` records that fallback for logging; it does not change the score.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = _TOOL_CALL_RE.findall(text)
    n_open = len(_OPEN_TAG_RE.findall(text))
    if n_open > len(blocks):  # unclosed trailing block: vLLM's `<tool_call>(.*)` fallback
        blocks = blocks + [text.rsplit("<tool_call>", 1)[1]]
    if not blocks:
        return {"type": "message", "content": text.strip(), "format_error": False}
    calls = []
    for body in blocks:
        try:
            obj = json.loads(body.strip())
        except json.JSONDecodeError:
            obj = None
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            return {"type": "message", "content": text.strip(), "format_error": True}
        args = obj.get("arguments", {})
        calls.append({"type": "function_call", "name": obj["name"],
                      "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)})
    if len(calls) == 1:
        calls[0]["format_error"] = False
        return calls[0]
    return {"type": "function_call_batch", "calls": calls, "format_error": False}


# --------------------------------------------------------------------------- argument comparison

def _arguments_match(expected: Any, actual: Any) -> bool:
    if not isinstance(actual, type(expected)):
        return False
    if isinstance(expected, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(_arguments_match(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return False
        return all(_arguments_match(e, a) for e, a in zip(expected, actual))
    if isinstance(expected, float):
        return abs(actual - expected) < FLOAT_TOLERANCE
    if isinstance(expected, str):
        ew, aw = Counter(expected.strip().lower().split()), Counter(actual.strip().lower().split())
        et, at = ew.total(), aw.total()
        if et < 2 or at < 2:
            return expected == actual
        return (ew & aw).total() / (et + at) >= WORD_COUNT_SIMILARITY_THRESHOLD
    return expected == actual


def _call_matches(expected: dict, actual: dict) -> tuple[bool, bool]:
    """-> (name_match, full_match)."""
    if expected["name"] != actual["name"]:
        return False, False
    try:
        ea = json.loads(expected["arguments"])
        aa = json.loads(actual["arguments"])
    except (json.JSONDecodeError, TypeError):
        return True, False
    return True, _arguments_match(ea, aa)


def _max_matching(candidates: list[list[int]]) -> int:
    """Kuhn's algorithm; candidates[i] = actual indices that match expected i."""
    match_of_actual: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in candidates[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match_of_actual or augment(match_of_actual[j], seen):
                match_of_actual[j] = i
                return True
        return False

    return sum(1 for i in range(len(candidates)) if augment(i, set()))


def _calls_of(action: dict) -> list[dict]:
    if action["type"] == "function_call":
        return [action]
    if action["type"] == "function_call_batch":
        return list(action["calls"])
    return []


# --------------------------------------------------------------------------- score

def compute_score(solution_str: str, ground_truth: str, extra_info: dict | None = None, **kwargs: Any) -> dict:
    del extra_info, kwargs
    out = {"score": 0.0, "is_call": 0.0, "type_match": 0.0, "name_match": 0.0, "format_error": 0.0}
    try:
        expected = json.loads(ground_truth or "")
    except json.JSONDecodeError:
        return out
    actual = extract_action(solution_str or "")
    out["format_error"] = 1.0 if actual.get("format_error") else 0.0
    out["is_call"] = 0.0 if actual["type"] == "message" else 1.0

    if expected["type"] == "message":
        if actual["type"] == "message":
            out["type_match"] = 1.0
            out["score"] = 1.0
        return out

    exp_calls, act_calls = _calls_of(expected), _calls_of(actual)
    if not act_calls:
        return out
    out["type_match"] = 1.0

    strict_count = os.getenv("NEMOTRON_PIVOT_STRICT_CALL_COUNT", "0") == "1"
    if strict_count and len(act_calls) != len(exp_calls):
        out["name_match"] = float(any(_call_matches(e, a)[0] for e in exp_calls for a in act_calls))
        return out

    candidates, any_name = [], False
    for e in exp_calls:
        row = []
        for j, a in enumerate(act_calls):
            name_ok, full_ok = _call_matches(e, a)
            any_name |= name_ok
            if full_ok:
                row.append(j)
        candidates.append(row)
    out["name_match"] = float(any_name)
    if _max_matching(candidates) == len(exp_calls):
        out["score"] = 1.0
    return out
