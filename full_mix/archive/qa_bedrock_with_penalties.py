"""Glue: ``qa_bedrock`` + ``medical_penalty_judge``.

Wraps the unmodified ``qa_bedrock.compute_score`` with a second, OpenRouter-
based behavioural-penalty judge (``medical_penalty_judge.score_penalties``)
and combines their outputs into a single normalised dict suitable for the
mixed-domain training pipeline.

Per-sample pipeline:

    qa_score = qa_bedrock.compute_score(...)           # untouched
    pen      = medical_penalty_judge.score_penalties(...)
    final    = max(0.0, qa_score["score"] - pen["penalty_total"])

The returned dict adds these keys on top of every qa_bedrock key:

    reward/penalty_self_id               float (0.00–0.30)
    reward/penalty_disclaimer_intrusion  float (0.00–0.30)
    reward/penalty_over_conservative     float (0.00–0.30)
    reward/penalty_total                 float (0.00–0.90)
    reward/penalty_judge_ok              int   (1 = penalty judge succeeded; 0 = failed-open)
    reward/score_pre_penalty             float (qa_bedrock's original score)

The original ``score`` key is REPLACED with the post-penalty value. The
pre-penalty qa_bedrock score is preserved as ``reward/score_pre_penalty``
so you can see exactly how much was deducted in rollout dumps.

The batch path issues penalty-judge calls concurrently across the sample
list using a ThreadPoolExecutor capped at ``FULL_MIX_JUDGE_CONCURRENCY``
(the same env var the chat/safety/identity judges use).

qa_bedrock.py is NOT modified — the wrapper imports it as-is.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from full_mix.rewards import qa_bedrock as _qa
from full_mix.rewards import medical_penalty_judge as _pen


logger = logging.getLogger(__name__)


# =============================================================================
# Output-dict construction
# =============================================================================

# Keys produced by score_penalties() (see medical_penalty_judge.empty_result).
# Names we expose on the reward dict.
_PENALTY_OUTPUT_KEYS: dict[str, str] = {
    "penalty_self_id":              "reward/penalty_self_id",
    "penalty_disclaimer_intrusion": "reward/penalty_disclaimer_intrusion",
    "penalty_over_conservative":    "reward/penalty_over_conservative",
    "penalty_total":                "reward/penalty_total",
}


def _normalise_penalty_into(qa_result: dict, penalty: dict) -> dict:
    """Merge penalty fields onto ``qa_result``. Pure (returns new dict).

    Always emits the full penalty key set, with sentinel 0 values when the
    penalty judge failed or was disabled. Sets ``reward/score_pre_penalty``
    to the original qa_bedrock score, replaces ``score`` with the
    post-penalty value (floored at 0), and sets ``reward/penalty_judge_ok``
    to 1/0.
    """
    out = dict(qa_result) if isinstance(qa_result, dict) else {"score": 0.0}
    pre_penalty = float(out.get("score", 0.0))
    total_penalty = float(penalty.get("penalty_total", 0.0)) if penalty else 0.0
    out["reward/score_pre_penalty"] = pre_penalty
    out["score"] = max(0.0, pre_penalty - total_penalty)
    out["reward/penalty_judge_ok"] = int(bool(penalty.get("judge_ok", False))) if penalty else 0
    for src, dst in _PENALTY_OUTPUT_KEYS.items():
        out[dst] = float(penalty.get(src, 0.0)) if penalty else 0.0
    return out


# =============================================================================
# Reference / question extraction from qa_bedrock's inputs
# =============================================================================

def _coerce_gt(ground_truth) -> dict:
    """qa_bedrock accepts either a struct (dict) ground_truth or a JSON string
    (after medical_qa_compat). Either way, get a dict back so we can read
    gold_response / latest_user from it.
    """
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            return json.loads(ground_truth)
        except json.JSONDecodeError:
            return {}
    return {}


def _resolve_user_prompt_and_reference(
    qa_result: dict, ground_truth, extra_info,
) -> tuple[str, str]:
    """Returns (user_prompt, reference_response) for the penalty judge.

    qa mode:
      user_prompt    : extra_info["question"]   (the raw user question)
      reference      : ground_truth["gold_response"]  (or gold_answer fallback)
    conversation mode:
      user_prompt    : ground_truth["latest_user"]  (precomputed by the dataset)
      reference      : ground_truth["gold_response"]

    The selection is driven by qa_bedrock's reported reward/eval_mode so
    behaviour stays in lockstep with whatever qa_bedrock used internally.
    """
    gt = _coerce_gt(ground_truth)
    eval_mode = qa_result.get("reward/eval_mode") if isinstance(qa_result, dict) else None

    if eval_mode == "conversation":
        user_prompt = str(gt.get("latest_user", "") or "")
    else:
        # qa mode (default).
        if isinstance(extra_info, dict):
            user_prompt = str(extra_info.get("question", "") or "")
        else:
            user_prompt = ""

    reference = str(gt.get("gold_response") or gt.get("gold_answer") or "")
    return user_prompt, reference


# =============================================================================
# Per-item path
# =============================================================================

def _score_one(data_source, solution_str, ground_truth, extra_info) -> dict:
    """One sample → final dict (qa_bedrock fields + penalty fields)."""
    qa_result = _qa.compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    if not isinstance(qa_result, dict):
        # qa_bedrock should always return a dict; if not, fail open on penalty
        # and surface whatever it gave us.
        return _normalise_penalty_into(
            {"score": float(qa_result) if qa_result is not None else 0.0},
            _pen.empty_result(judge_ok=False),
        )

    pre_penalty = float(qa_result.get("score", 0.0))
    if pre_penalty <= 0.0:
        # qa_bedrock already gave us 0 (empty answer, judge failed, etc).
        # Subtracting more wouldn't change anything; skip the penalty call.
        return _normalise_penalty_into(qa_result, _pen.empty_result(judge_ok=True))

    user_prompt, reference = _resolve_user_prompt_and_reference(
        qa_result, ground_truth, extra_info
    )
    penalty = _pen.score_penalties(user_prompt, reference, solution_str or "")
    return _normalise_penalty_into(qa_result, penalty)


# =============================================================================
# Batch path
# =============================================================================

def _concurrency() -> int:
    raw = os.environ.get("FULL_MIX_JUDGE_CONCURRENCY", "32")
    try:
        n = int(raw)
    except ValueError:
        n = 32
    return max(1, n)


def _score_batch(data_sources, solution_strs, ground_truths, extra_infos) -> list[dict]:
    n = len(data_sources)
    if extra_infos is None:
        extra_infos = [None] * n

    # Step 1: run qa_bedrock on the whole batch. qa_bedrock has its own
    # internal asyncio batching; we leave that alone.
    qa_results = _qa.compute_score(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )

    # Step 2: penalty judge calls fan out concurrently. Skip the judge when
    # qa_bedrock already returned 0.
    penalties: list[Optional[dict]] = [None] * n

    def _penalty_for(i: int) -> Optional[dict]:
        qa_r = qa_results[i] if isinstance(qa_results[i], dict) else {"score": float(qa_results[i] or 0.0)}
        if float(qa_r.get("score", 0.0)) <= 0.0:
            return _pen.empty_result(judge_ok=True)
        user_prompt, reference = _resolve_user_prompt_and_reference(
            qa_r, ground_truths[i], extra_infos[i],
        )
        return _pen.score_penalties(user_prompt, reference, solution_strs[i] or "")

    if _pen.is_enabled():
        conc = min(_concurrency(), n)
        with ThreadPoolExecutor(max_workers=conc, thread_name_prefix="med-penalty") as pool:
            futures = {pool.submit(_penalty_for, i): i for i in range(n)}
            for fut in futures:
                i = futures[fut]
                try:
                    penalties[i] = fut.result()
                except Exception:
                    logger.exception("medical penalty worker raised at index %d", i)
                    penalties[i] = _pen.empty_result(judge_ok=False)
    else:
        # Toggle is off — skip every call, emit disabled-state sentinels.
        for i in range(n):
            penalties[i] = _pen.empty_result(judge_ok=False)

    # Step 3: merge per-sample.
    out: list[dict] = []
    for i in range(n):
        qa_r = qa_results[i] if isinstance(qa_results[i], dict) else {"score": float(qa_results[i] or 0.0)}
        out.append(_normalise_penalty_into(qa_r, penalties[i]))
    return out


# =============================================================================
# Public entry point — same dual signature as qa_bedrock.compute_score
# =============================================================================

def compute_score(
    data_source=None, solution_str=None, ground_truth=None, extra_info=None,
    data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None,
    **kwargs,
):
    """Same dual signature as qa_bedrock.compute_score; same call patterns."""
    if solution_strs is not None:
        return _score_batch(data_sources, solution_strs, ground_truths, extra_infos)
    return _score_one(data_source, solution_str, ground_truth, extra_info)
