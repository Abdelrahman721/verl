"""IFEval-only reward dispatcher.

Mirrors ``full_mix.rewards.compute_score`` but routes EVERY (suffixed or bare)
IFEval-base data_source to the quality-gated IFEval scorer in
``ifeval_quality_gated``. Anything else returns 0.0 and logs a warning — by
design, this dispatcher is for the IFEval-only experiment.

Wire in the training script with:
    REWARD_FN_PATH=/workspace/verl/full_mix/rewards/compute_score_ifeval.py
    REWARD_FN_NAME=compute_score
"""

import logging
import os
import sys

# Make `full_mix` importable whether we're loaded as a script (via
# load_extern_object) or as a package.
_HERE = os.path.dirname(os.path.abspath(__file__))                      # .../full_mix/rewards
_FULL_MIX_DIR = os.path.dirname(_HERE)                                  # .../full_mix
_REPO_ROOT = os.path.dirname(_FULL_MIX_DIR)                             # .../verl
for _p in (_REPO_ROOT, _FULL_MIX_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from full_mix.curriculum.encoding import (  # noqa: E402
    DATASET_ID,
    UID_DATASET_STRIDE,
    base_data_source,
    encode_data_source,
)
from full_mix.common.think import strip_think  # noqa: E402
from full_mix.rewards import ifeval_quality_gated as _ifeval_gated  # noqa: E402

logger = logging.getLogger(__name__)


_IF_SOURCES = {
    "local/dolci-ifeval-32b",
    "google/IFEval",
    "allenai/IFBench_test",
}


def _score_one(data_source: str, solution_str: str, ground_truth, extra_info) -> float:
    cleaned = strip_think(solution_str) if solution_str else solution_str
    base = base_data_source(data_source)

    if base in _IF_SOURCES:
        try:
            return float(
                _ifeval_gated.compute_score(
                    data_source=base,
                    solution_str=cleaned,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            )
        except Exception:
            logger.exception("ifeval_quality_gated raised for data_source=%r", data_source)
            return 0.0

    logger.warning(
        "compute_score_ifeval received non-IFEval data_source=%r; returning 0.0", data_source
    )
    return 0.0


def _compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None):
    n = len(data_sources)
    if extra_infos is None:
        extra_infos = [None] * n
    return [
        _score_one(data_sources[i], solution_strs[i], ground_truths[i], extra_infos[i])
        for i in range(n)
    ]


def _wrap(score: float, data_source: str, extra_info) -> dict:
    """Build the per-sample return dict (same shape as the main dispatcher).

    Keys are int/float only (the rollout dump path serializes via numpy
    column-cast and chokes on string arrays). prompt_uid groups the N
    rollouts of one prompt; data_source_code carries dataset+shard tag.
    """
    info = extra_info or {}
    if "prompt_uid" in info:
        prompt_uid = int(info["prompt_uid"])
    else:
        base = base_data_source(data_source)
        prompt_uid = DATASET_ID.get(base, 0) * UID_DATASET_STRIDE + int(info.get("index", 0))
    return {
        "score": float(score),
        "prompt_uid": prompt_uid,
        "data_source_code": encode_data_source(data_source),
    }


def compute_score(*args, **kwargs):
    is_batch = "data_sources" in kwargs or (args and isinstance(args[0], (list, tuple)))
    if is_batch:
        data_sources = kwargs.get("data_sources", args[0] if args else None)
        solution_strs = kwargs.get("solution_strs", args[1] if len(args) > 1 else None)
        ground_truths = kwargs.get("ground_truths", args[2] if len(args) > 2 else None)
        extra_infos = kwargs.get("extra_infos", args[3] if len(args) > 3 else None)
        scores = _compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos)
        if extra_infos is None:
            extra_infos = [None] * len(data_sources)
        return [_wrap(s, ds, ei) for s, ds, ei in zip(scores, data_sources, extra_infos)]

    data_source = kwargs.get("data_source", args[0] if args else None)
    solution_str = kwargs.get("solution_str", args[1] if len(args) > 1 else None)
    ground_truth = kwargs.get("ground_truth", args[2] if len(args) > 2 else None)
    extra_info = kwargs.get("extra_info", args[3] if len(args) > 3 else None)
    score = _score_one(data_source, solution_str, ground_truth, extra_info)
    return _wrap(score, data_source, extra_info)
