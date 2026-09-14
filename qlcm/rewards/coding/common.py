"""Shared helpers for the medical-coding reward suite.

Mirrors (but does not import from) the pattern in `qa_openrouter_bench.py`:
chat-template format penalty, post-`</think>` answer extraction, the union
reward-dict shape, the `_serialize_gold` shim. Lives here so the coding
package is fully self-contained — no cross-domain imports.
"""

from __future__ import annotations

import json
import re

# ============================================================================
# Format tags  (must match the rest of the repo)
# ============================================================================
THINKING_START = "<think>"
THINKING_END = "</think>"


# ============================================================================
# Reward-dict union — superset of every key any coding scorer can emit.
# DataProto.concat insists every row in a heterogeneous batch shares one key
# set; padding to this union here lets us mix any of the four coding modes
# (plus retention rows from the higher-level mix dispatcher) without
# tripping the concat assert.
# ============================================================================
_REWARD_UNION_DEFAULTS: dict = {
    # Always present
    "score":                     0.0,
    "reward/eval_mode":          "",
    "reward/gold":               "",      # serialized ground truth (for dump)
    "reward/raw_score":          0.0,
    "reward/judge_score":        0.0,     # telemetry alias; mirrors raw_score
    "reward/format_penalty":     0.0,
    "reward/reasoning_length":   0,
    "reward/answer_length":      0,
    "reward/accuracy":           0.0,     # 10-scale projection of raw
    # Multilabel scorers (ICD + SNOMED ML) — set-based P/R/F1
    "reward/precision":          0.0,
    "reward/recall":             0.0,
    "reward/f1":                 0.0,
    # SNOMED IF — EvalResult-derived fields
    "reward/instruction_score":  0.0,
    "reward/accuracy_score":     0.0,
    "reward/judges_enabled":     0,       # 1 iff IF_USE_LLM_JUDGES=1 at the time of the call
}


def _empty_score_dict(eval_mode: str, format_penalty: float,
                      reasoning_len: int, answer_len: int) -> dict:
    """Per-scorer starting shell — populates the always-present fields.

    Mirrors `qa_openrouter_bench._empty_score_dict` but with the coding
    union's defaults. Scorers extend the shell with mode-specific fields.
    """
    out = dict(_REWARD_UNION_DEFAULTS)
    out["reward/eval_mode"]        = eval_mode
    out["reward/format_penalty"]   = format_penalty
    out["reward/reasoning_length"] = reasoning_len
    out["reward/answer_length"]    = answer_len
    return out


def _serialize_gold(ground_truth) -> str:
    """Stringify any ground_truth shape so it survives the rollout dump.

    The fully-async dump path can't recover `gts` from the TQ; the dispatcher
    injects this string into `reward/gold` so the dump is self-contained.
    """
    if ground_truth is None:
        return ""
    if isinstance(ground_truth, str):
        return ground_truth
    try:
        return json.dumps(ground_truth, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(ground_truth)


def _normalize_reward_dict(out) -> dict:
    """Pad `out` to the union key set; preserve existing values; never raise."""
    if not isinstance(out, dict):
        return dict(_REWARD_UNION_DEFAULTS)
    padded = dict(_REWARD_UNION_DEFAULTS)
    padded.update(out)
    return padded


# ============================================================================
# Format / answer extraction
# ============================================================================
_MATCH_FORMAT_PERFECT = re.compile(
    rf"^\s*{re.escape(THINKING_START)}.+?{re.escape(THINKING_END)}\s*.+\s*\Z",
    flags=re.DOTALL,
)
_EXTRACT_AFTER_THINK = re.compile(
    rf"{re.escape(THINKING_END)}\s*(.+)", flags=re.DOTALL,
)
_THINK_BLOCK = re.compile(
    rf"{re.escape(THINKING_START)}(.+?){re.escape(THINKING_END)}",
    flags=re.DOTALL,
)


def _extract_answer(response: str) -> str | None:
    """Return text after `</think>` (the model's actual answer)."""
    if not response:
        return None
    m = _EXTRACT_AFTER_THINK.search(response)
    return m.group(1).strip() if m else None


def _compute_format_penalty(response: str) -> float:
    """Mirror `qa_openrouter_bench._compute_format_penalty`. ≤ 0 always."""
    if not response:
        return -0.5
    ts_count = response.count(THINKING_START)
    te_count = response.count(THINKING_END)
    if ts_count == 1 and te_count == 1 and _MATCH_FORMAT_PERFECT.search(response):
        return 0.0
    penalty = 0.0
    if ts_count == 0:
        penalty -= 0.25
    elif ts_count > 1:
        penalty -= 0.5 * (ts_count - 1)
    if te_count == 0:
        penalty -= 0.25
    elif te_count > 1:
        penalty -= 0.5 * (te_count - 1)
    return max(penalty, -0.5)


def _reasoning_and_answer_lengths(response: str) -> tuple[int, int]:
    """Word counts for the <think>…</think> body and the post-think answer."""
    if not response:
        return 0, 0
    think = _THINK_BLOCK.search(response)
    reasoning_len = len(think.group(1).split()) if think else 0
    answer = _extract_answer(response) or ""
    return reasoning_len, len(answer.split())
