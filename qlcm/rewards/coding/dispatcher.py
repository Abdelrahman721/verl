"""Per-row dispatcher for the coding domain.

Routes by `extra_info.task_type` to the right scorer, applies the standard
format-penalty + answer-extraction prologue, injects `reward/gold`, and pads
the output to the coding union shape via `_normalize_reward_dict`.

Supported task_types:
  icd10_multilabel             → icd_scorer.score
  icd10_instruction_follow     → icd_scorer.score   (same scorer per spec)
  snomed_multilabel            → snomed_ml_scorer.score
  snomed_instruction_follow    → snomed_if_scorer.score
  icd_multi_label              → icd_scorer.score
                                 (stage-4 legacy; ICD multilabel data, same
                                  list-of-codes ground_truth shape as
                                  icd10_multilabel — pulled in as stage-5
                                  coding retention)
  snomed_single_label          → snomed_ml_scorer.score
                                 (stage-4 legacy; SNOMED single-label data,
                                  gold dict carries `sct_id` — adapted to a
                                  singleton `gt_sct_ids` list so the ML
                                  scorer can grade it; F1 collapses to
                                  exact-match. Pulled in as stage-5 coding
                                  retention.)

Anything else is a misconfiguration — raises ValueError. The top-level mix
dispatcher (`compute_score_coding_mix.py`) handles routing non-coding rows
to the retention scorer; this module assumes its inputs are coding rows.
"""

from __future__ import annotations

import json
from typing import Any

from . import icd_scorer, snomed_if_scorer, snomed_ml_scorer
from .common import (
    _compute_format_penalty,
    _extract_answer,
    _normalize_reward_dict,
    _reasoning_and_answer_lengths,
    _serialize_gold,
)


_TASK_TYPES = frozenset({
    "icd10_multilabel",
    "icd10_instruction_follow",
    "snomed_multilabel",
    "snomed_instruction_follow",
    # Stage-4 legacy task_types — kept around so stage-4 coding-unmastered
    # rows can be folded in as retention without losing their reward signal.
    "icd_multi_label",
    "snomed_single_label",
})


def _adapt_snomed_sl_gold(ground_truth):
    """Stage-4 SNOMED-SL gold is a single-concept dict
    ``{"id": ..., "sct_id": "<digits>", "label": ...}``.
    The SNOMED-ML scorer reads ``ground_truth["gt_sct_ids"]`` (a list), so we
    promote the singleton id into that shape. F1 then collapses to
    exact-match — exactly what the original stage-4 SL scorer did."""
    if isinstance(ground_truth, dict) and "sct_id" in ground_truth and "gt_sct_ids" not in ground_truth:
        sid = ground_truth.get("sct_id")
        if sid is not None:
            adapted = dict(ground_truth)
            adapted["gt_sct_ids"] = [str(sid)]
            return adapted
    return ground_truth


def _route(task_type: str, eval_mode: str,
           solution_str: str, ground_truth, extra_info,
           format_penalty: float, reasoning_len: int, answer_len: int,
           answer_to_grade: str) -> dict:
    if task_type in ("icd10_multilabel", "icd10_instruction_follow", "icd_multi_label"):
        return icd_scorer.score(
            eval_mode, solution_str, ground_truth,
            format_penalty, reasoning_len, answer_len, answer_to_grade,
        )
    if task_type == "snomed_multilabel":
        return snomed_ml_scorer.score(
            eval_mode, solution_str, ground_truth,
            format_penalty, reasoning_len, answer_len, answer_to_grade,
        )
    if task_type == "snomed_single_label":
        return snomed_ml_scorer.score(
            eval_mode, solution_str, _adapt_snomed_sl_gold(ground_truth),
            format_penalty, reasoning_len, answer_len, answer_to_grade,
        )
    if task_type == "snomed_instruction_follow":
        return snomed_if_scorer.score(
            eval_mode, solution_str, ground_truth, extra_info,
            format_penalty, reasoning_len, answer_len, answer_to_grade,
        )
    raise ValueError(
        f"coding dispatcher saw unknown task_type {task_type!r}; "
        f"expected one of {sorted(_TASK_TYPES)}"
    )


def _score_single(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    """One-row entry point. Returns the normalised reward dict."""
    raw_gt = ground_truth   # preserved for reward/gold dump injection
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            ground_truth = {}

    extra_info = extra_info or {}
    task_type = extra_info.get("task_type") if isinstance(extra_info, dict) else None
    if not task_type:
        raise ValueError(
            "coding dispatcher requires extra_info.task_type to be set; "
            f"got data_source={data_source!r}, extra_info={extra_info!r}"
        )
    # eval_mode mirrors task_type so the rollout dump labels each row clearly
    eval_mode = task_type

    format_penalty = _compute_format_penalty(solution_str)
    reasoning_len, answer_len = _reasoning_and_answer_lengths(solution_str)
    answer_to_grade = _extract_answer(solution_str) or ""

    out = _route(task_type, eval_mode, solution_str, ground_truth, extra_info,
                 format_penalty, reasoning_len, answer_len, answer_to_grade)
    out["reward/gold"] = _serialize_gold(raw_gt)
    return _normalize_reward_dict(out)


def _score_batch(data_sources, solution_strs, ground_truths, extra_infos) -> list[dict]:
    """Row-wise batch entry point. CPU-bound scorers; no need for asyncio."""
    n = len(solution_strs)
    if extra_infos is None:
        extra_infos = [None] * n
    if data_sources is None:
        data_sources = [None] * n
    return [
        _score_single(data_sources[i], solution_strs[i], ground_truths[i], extra_infos[i])
        for i in range(n)
    ]


# ============================================================================
# Public entry point — same dual signature as the other compute_score modules
# ============================================================================
def compute_score(
    data_source=None, solution_str=None, ground_truth=None, extra_info=None,
    data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None,
    **_kwargs: Any,
):
    """Dual-signature entry point.

    Single mode: pass data_source / solution_str / ground_truth / extra_info.
    Batch mode : pass data_sources / solution_strs / ground_truths / extra_infos.
    """
    if solution_strs is not None:
        return _score_batch(data_sources, solution_strs, ground_truths, extra_infos)
    return _score_single(data_source, solution_str, ground_truth, extra_info)
