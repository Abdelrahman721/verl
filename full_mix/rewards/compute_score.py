"""Multi-domain reward dispatcher for VERL's BatchRewardManager.

Routes by ``data_source`` to the appropriate domain scorer. Chat + safety use
an LLM-as-judge; those calls are parallelized with a ThreadPoolExecutor so the
whole training batch fans out to the judge endpoint concurrently. Rule-based
scorers (math / gsm8k / ifeval) run inline.

All responses are passed through ``strip_think`` before scoring so that
``<think>...</think>`` traces never contaminate the grade.

Configure via env vars:
    FULL_MIX_JUDGE_API_BASE / _API_KEY / _MODEL  — judge connection
    FULL_MIX_JUDGE_CONCURRENCY                   — thread pool size (default 32)

Expected wiring in VERL:
    reward.reward_manager=batch
    reward.custom_reward_function.path=full_mix/rewards/compute_score.py
    reward.custom_reward_function.name=compute_score
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Make `full_mix` importable whether we're loaded as a script (via
# load_extern_object) or as a package.
_HERE = os.path.dirname(os.path.abspath(__file__))                      # .../full_mix/rewards
_FULL_MIX_DIR = os.path.dirname(_HERE)                                  # .../full_mix
_REPO_ROOT = os.path.dirname(_FULL_MIX_DIR)                             # .../verl
for _p in (_REPO_ROOT, _FULL_MIX_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from full_mix.common.think import strip_think  # noqa: E402
from full_mix.rewards import chat as _chat     # noqa: E402
from full_mix.rewards import gsm8k as _gsm8k   # noqa: E402
from full_mix.rewards import math_boxed as _math_boxed  # noqa: E402
from full_mix.rewards import safety as _safety          # noqa: E402
from full_mix import ifeval_reward as _ifeval           # noqa: E402

logger = logging.getLogger(__name__)


_MATH_BOXED_SOURCES = {
    "local/dolci-math-7b",
    "HuggingFaceH4/MATH-500",
}
_GSM8K_SOURCES = {
    "openai/gsm8k",
}
_IF_SOURCES = {
    "local/dolci-ifeval-32b",
    "google/IFEval",
    "allenai/IFBench_test",
}
_CHAT_SOURCES = {
    "local/dolci-chat-32b",
}
_SAFETY_SOURCES = {
    "local/safety-dpo-reference",
}

_RULE_BASED = _MATH_BOXED_SOURCES | _GSM8K_SOURCES | _IF_SOURCES
_JUDGE_BASED = _CHAT_SOURCES | _SAFETY_SOURCES


def _score_one(data_source: str, solution_str: str, ground_truth, extra_info) -> float:
    cleaned = strip_think(solution_str) if solution_str else solution_str

    try:
        if data_source in _MATH_BOXED_SOURCES:
            return _math_boxed.compute_score(cleaned, ground_truth)
        if data_source in _GSM8K_SOURCES:
            return _gsm8k.compute_score(cleaned, ground_truth)
        if data_source in _IF_SOURCES:
            return float(
                _ifeval.compute_score(
                    data_source=data_source,
                    solution_str=cleaned,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            )
        if data_source in _CHAT_SOURCES:
            return _chat.compute_score(cleaned, ground_truth, extra_info)
        if data_source in _SAFETY_SOURCES:
            return _safety.compute_score(cleaned, ground_truth, extra_info)
    except Exception:
        logger.exception("scorer error for data_source=%r", data_source)
        return 0.0

    logger.warning("no scorer registered for data_source=%r; returning 0.0", data_source)
    return 0.0


def _concurrency() -> int:
    raw = os.environ.get("FULL_MIX_JUDGE_CONCURRENCY", "32")
    try:
        n = int(raw)
    except ValueError:
        n = 32
    return max(1, n)


def _compute_score_batch(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos=None,
) -> list:
    """Batch path: called once for the whole rollout (BatchRewardManager style).

    Rule-based scorers run inline; judge-based (chat/safety) fan out via a
    ThreadPoolExecutor so judge calls issue concurrently against the remote
    vLLM endpoint.
    """
    n = len(data_sources)
    if extra_infos is None:
        extra_infos = [None] * n
    scores: list[float] = [0.0] * n

    judge_indices: list[int] = []
    for i, ds in enumerate(data_sources):
        if ds in _JUDGE_BASED:
            judge_indices.append(i)
        else:
            scores[i] = _score_one(ds, solution_strs[i], ground_truths[i], extra_infos[i])

    if judge_indices:
        conc = min(_concurrency(), len(judge_indices))
        with ThreadPoolExecutor(max_workers=conc, thread_name_prefix="judge") as pool:
            futures = {
                pool.submit(
                    _score_one,
                    data_sources[i],
                    solution_strs[i],
                    ground_truths[i],
                    extra_infos[i],
                ): i
                for i in judge_indices
            }
            for fut in futures:
                i = futures[fut]
                try:
                    scores[i] = fut.result()
                except Exception:
                    logger.exception("judge worker raised for index %d", i)
                    scores[i] = 0.0

    return scores

def compute_score_helper(*args, **kwargs):
    """Dual-signature entry point.

    VERL dispatches reward scoring differently depending on the configured
    reward manager. Two signatures can reach this function:

      Per-item (experimental RewardLoopManager's ``naive`` / ``dapo``):
        compute_score(data_source=..., solution_str=..., ground_truth=...,
                      extra_info={}, **router_kwargs) -> float

      Batched (``BatchRewardManager``):
        compute_score(data_sources=[...], solution_strs=[...],
                      ground_truths=[...], extra_infos=[...]) -> list[float]

    We detect which one the caller used and route accordingly.
    """
    if "data_sources" in kwargs or (args and isinstance(args[0], (list, tuple))):
        assert False
        data_sources = kwargs.get("data_sources", args[0] if args else None)
        solution_strs = kwargs.get("solution_strs", args[1] if len(args) > 1 else None)
        ground_truths = kwargs.get("ground_truths", args[2] if len(args) > 2 else None)
        extra_infos = kwargs.get("extra_infos", args[3] if len(args) > 3 else None)
        return _compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos)

    data_source = kwargs.get("data_source", args[0] if args else None)
    solution_str = kwargs.get("solution_str", args[1] if len(args) > 1 else None)
    ground_truth = kwargs.get("ground_truth", args[2] if len(args) > 2 else None)
    extra_info = kwargs.get("extra_info", args[3] if len(args) > 3 else None)
    return _score_one(data_source, solution_str, ground_truth, extra_info)

def compute_score(*args, **kwargs):
    score = compute_score_helper(*args, **kwargs)
    return {
        "score": score,
    }