"""SNOMED CT instruction-following scorer.

Port of hazem's `snomed_if_reward.py` with two fixes:
  1. Returns a dict (not a float) — verl needs the dict shape.
  2. Imports from the EMBEDDED `instruction_following` framework under this
     package (not the original repos-relative path) so the coding suite is
     self-contained.

The IF framework's `evaluate_row(rule, response, original_codes, processed_codes, ...)`
returns an `EvalResult(instruction_score, accuracy_score, model_response)`.
The combination rule (mirrors hazem's `_combine_instruction_accuracy`):
  - both numeric → average
  - exactly one None → take the non-None one
  - both None → 0.0
"""

from __future__ import annotations

import os
from pathlib import Path

from .common import _empty_score_dict
from .instruction_following.rules_eval import (
    CodesNotInLabelsError,
    V17InputError,
    evaluate_row,
)


def _combine_instruction_accuracy(
    instruction_score: float | None,
    accuracy_score: float | None,
) -> float:
    """Verbatim from snomed_if_reward.py."""
    if instruction_score is not None and accuracy_score is not None:
        return (float(instruction_score) + float(accuracy_score)) / 2.0
    if instruction_score is not None:
        return float(instruction_score)
    if accuracy_score is not None:
        return float(accuracy_score)
    return 0.0


def _judges_enabled_at_call_time() -> int:
    """1 iff IF_USE_LLM_JUDGES is set to truthy at the moment of scoring.
    Logged into the reward dict so post-hoc analysis can tell whether a
    given row's IF score was deterministic-only or judge-enhanced."""
    v = os.getenv("IF_USE_LLM_JUDGES", "").strip().lower()
    return 1 if v in ("1", "true", "yes", "on") else 0


def score(eval_mode: str,
          solution_str: str,
          ground_truth,
          extra_info,
          format_penalty: float,
          reasoning_len: int,
          answer_len: int,
          answer_to_grade: str) -> dict:
    """Verl-shape SNOMED instruction-following scorer.

    `ground_truth` is a dict (parsed upstream) with:
      - `rule`              — one of the 17 v01..v17 rule names
      - `original_codes`    — list of SNOMED IDs (gold for most rules)
      - `processed_codes`   — list of SNOMED IDs (target for v17, "shown"
                              set for v08/v09/v11/v12)
    `extra_info` may carry:
      - `user_prompt`       — full text the user saw, needed by v10
      - `labels_path`       — override for the labels JSON (otherwise uses
                              INSTRUCTION_FOLLOWING_LABELS_CSV or the
                              embedded default)
    """
    out = _empty_score_dict(eval_mode, format_penalty, reasoning_len, answer_len)

    gt = ground_truth or {}
    rule            = gt.get("rule")
    original_codes  = gt.get("original_codes")
    processed_codes = gt.get("processed_codes")
    if rule is None or original_codes is None or processed_codes is None:
        out["score"] = max(0.0, 0.0 + format_penalty)
        out["reward/judges_enabled"] = _judges_enabled_at_call_time()
        return out

    oc = [str(x) for x in original_codes]
    pc = [str(x) for x in processed_codes]

    # Labels-path resolution: only per-row extra_info override goes through
    # here. Passing a non-None labels_path to evaluate_row defeats the
    # framework's labels cache (utils.get_labels only caches when called with
    # None), so reading the env var here would re-load the 3 MB JSON on
    # EVERY row — ~23 ms per call, measured. Leave the env-var fallback to
    # the framework: default_labels_path() reads INSTRUCTION_FOLLOWING_LABELS_CSV
    # the first time get_labels(None) is called, and the resulting mapping
    # is cached for the lifetime of the process.
    labels_path: Path | None = None
    if isinstance(extra_info, dict) and extra_info.get("labels_path"):
        labels_path = Path(str(extra_info["labels_path"])).expanduser()

    user_prompt: str | None = None
    if isinstance(extra_info, dict):
        up = extra_info.get("user_prompt")
        if isinstance(up, str) and up.strip():
            user_prompt = up

    try:
        result = evaluate_row(
            str(rule),
            solution_str or "",   # framework parses </think> itself
            oc,
            pc,
            prompt=user_prompt,
            labels_path=labels_path,
        )
    except (CodesNotInLabelsError, V17InputError, KeyError, ValueError):
        # Same fail-open behaviour as snomed_if_reward.py — known data
        # issues yield 0 rather than crashing the rollout.
        result = None

    if result is None:
        raw = 0.0
        out["reward/instruction_score"] = 0.0
        out["reward/accuracy_score"]    = 0.0
    else:
        raw = _combine_instruction_accuracy(result.instruction_score,
                                            result.accuracy_score)
        out["reward/instruction_score"] = float(result.instruction_score) if result.instruction_score is not None else 0.0
        out["reward/accuracy_score"]    = float(result.accuracy_score)

    out["reward/raw_score"]      = raw
    out["reward/judge_score"]    = raw
    out["reward/accuracy"]       = raw * 10.0
    out["reward/judges_enabled"] = _judges_enabled_at_call_time()
    out["score"] = max(0.0, raw + format_penalty)
    return out
