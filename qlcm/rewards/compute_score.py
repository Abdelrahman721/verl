"""Multi-domain reward dispatcher for VERL's BatchRewardManager.

Routes by ``data_source`` to the appropriate domain scorer. Chat + safety use
an LLM-as-judge; those calls are parallelized with a ThreadPoolExecutor so the
whole training batch fans out to the judge endpoint concurrently. Rule-based
scorers (math / gsm8k / ifeval) run inline.

All responses are passed through ``strip_think`` before scoring so that
``<think>...</think>`` traces never contaminate the grade.

Configure via env vars:
    QLCM_JUDGE_API_BASE / _API_KEY / _MODEL  — judge connection
    QLCM_JUDGE_CONCURRENCY                   — thread pool size (default 32)

Expected wiring in VERL:
    reward.reward_manager=batch
    reward.custom_reward_function.path=qlcm/rewards/compute_score.py
    reward.custom_reward_function.name=compute_score
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Make `qlcm` importable whether we're loaded as a script (via
# load_extern_object) or as a package.
_HERE = os.path.dirname(os.path.abspath(__file__))                      # .../qlcm/rewards
_QLCM_DIR = os.path.dirname(_HERE)                                  # .../qlcm
_REPO_ROOT = os.path.dirname(_QLCM_DIR)                             # .../verl
for _p in (_REPO_ROOT, _QLCM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qlcm.per_source_metrics  # noqa: E402,F401  -- rebinds compute_data_metrics for per-source wandb curves
from qlcm.common.think import strip_think  # noqa: E402
from qlcm.curriculum.encoding import (  # noqa: E402
    DATASET_ID,
    UID_DATASET_STRIDE,
    base_data_source,
    encode_data_source,
)
from qlcm.rewards import chat as _chat     # noqa: E402
from qlcm.rewards import gsm8k as _gsm8k   # noqa: E402
from qlcm.rewards import identity as _identity     # noqa: E402
from qlcm.rewards import math_boxed as _math_boxed  # noqa: E402
from qlcm.rewards import safety as _safety          # noqa: E402
from qlcm.rewards.identity_judges import check_global_identity_violation  # noqa: E402
from qlcm import ifeval_reward as _ifeval           # noqa: E402

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
_IDENTITY_SOURCES = {
    "local/avey-identity",
}

_RULE_BASED = _MATH_BOXED_SOURCES | _GSM8K_SOURCES | _IF_SOURCES
_JUDGE_BASED = _CHAT_SOURCES | _SAFETY_SOURCES | _IDENTITY_SOURCES


def _score_one(data_source: str, solution_str: str, ground_truth, extra_info) -> float:
    """Compute the final reward for one sample.

    Two-step pipeline:
      1) Domain reward via the per-source scorer (existing).
      2) GLOBAL IDENTITY GATE (OFF by default; QLCM_IDENTITY_GATE=1 enables):
         a judge call that runs on EVERY non-identity sample. If the candidate
         positively claims a non-Avey identity, the domain reward is multiplied
         by 0. Identity samples skip the gate
         because their reference-aligned scorer already evaluates identity
         alignment across every dimension (including this one) — no double-apply.

    The gate is also skipped when the domain reward is already <= 0 (nothing
    to gate), and when the candidate is empty / lacks any identity-adjacent
    vocabulary (no claim possible). Gate failures FAIL OPEN — a flaky judge
    must not silently zero rewards across the board.
    """
    cleaned = strip_think(solution_str) if solution_str else solution_str
    base = base_data_source(data_source)

    # ---- 1) domain reward ----
    try:
        if base in _IDENTITY_SOURCES:
            # Reference-aligned identity scorer already incorporates the
            # global-gate concerns plus richer alignment dimensions. Return
            # directly without re-applying the gate.
            return float(_identity.compute_score(cleaned, ground_truth, extra_info))
        if base in _MATH_BOXED_SOURCES:
            domain_score = float(_math_boxed.compute_score(cleaned, ground_truth))
        elif base in _GSM8K_SOURCES:
            domain_score = float(_gsm8k.compute_score(cleaned, ground_truth))
        elif base in _IF_SOURCES:
            domain_score = float(
                _ifeval.compute_score(
                    data_source=base,
                    solution_str=cleaned,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            )
        elif base in _CHAT_SOURCES:
            domain_score = float(_chat.compute_score(cleaned, ground_truth, extra_info))
        elif base in _SAFETY_SOURCES:
            domain_score = float(_safety.compute_score(cleaned, ground_truth, extra_info))
        else:
            logger.warning("no scorer registered for data_source=%r; returning 0.0", data_source)
            return 0.0
    except Exception:
        logger.exception("scorer error for data_source=%r", data_source)
        return 0.0

    # ---- 2) global identity gate (disabled by default) ----
    if not _identity_gate_enabled():
        return domain_score
    if domain_score <= 0.0:
        return domain_score  # nothing to gate

    user_prompt = ""
    if isinstance(extra_info, dict):
        user_prompt = (extra_info.get("user_prompt") or "").strip()
    try:
        violation = check_global_identity_violation(user_prompt, cleaned or solution_str)
    except Exception:
        logger.warning("global identity gate raised; failing open", exc_info=True)
        return domain_score
    if violation:
        return 0.0
    return domain_score


# The global identity gate is DISABLED by default.
#
# It issued one extra judge call per non-identity sample, and on
# deepseek-v4-flash it was the dominant source of judge failures: the gate uses
# json_object mode (no schema), and providers in the pinned pool returned
# degenerate objects with the key names stripped —
#   {": false, ": false, "brief_reason": "..."}
# instead of {"false_model_identity": false, ...}. Those fail _is_gate_verdict,
# burn all four attempts, then fail open anyway.
#
# Disabling it is close to a no-op behaviourally: the gate already failed open
# on every error, so a broken gate and an absent gate score identically. What
# changes is that identity violations on NON-identity rows are no longer caught
# and zeroed. `local/avey-identity` rows are unaffected — they are scored by
# reference_aligned_identity_score, which evaluates identity alignment directly
# and never went through this gate.
#
# Set QLCM_IDENTITY_GATE=1 to turn it back on.
def _identity_gate_enabled() -> bool:
    return os.environ.get("QLCM_IDENTITY_GATE", "0").strip().lower() not in ("0", "false", "no", "")


def _concurrency() -> int:
    raw = os.environ.get("QLCM_JUDGE_CONCURRENCY", "32")
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

    Every sample now potentially issues a judge call (the global identity
    gate runs on non-identity samples too), so fan ALL samples out via the
    ThreadPoolExecutor instead of the prior rule-based-inline + judge-pool
    split. Rule-based domain scorers still complete in microseconds; the
    pool overhead is negligible compared to the judge-call latency.
    """
    n = len(data_sources)
    if extra_infos is None:
        extra_infos = [None] * n
    scores: list[float] = [0.0] * n
    if n == 0:
        return scores

    conc = min(_concurrency(), n)
    with ThreadPoolExecutor(max_workers=conc, thread_name_prefix="judge") as pool:
        futures = {
            pool.submit(
                _score_one,
                data_sources[i],
                solution_strs[i],
                ground_truths[i],
                extra_infos[i],
            ): i
            for i in range(n)
        }
        for fut in futures:
            i = futures[fut]
            try:
                scores[i] = fut.result()
            except Exception:
                logger.exception("scorer worker raised for index %d", i)
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

def _wrap(score: float, data_source: str, extra_info) -> dict:
    """Build the per-sample return dict. All values must be int/float (the
    rollout dump path serializes via numpy column-cast and chokes on string
    arrays).

    Key naming avoids collisions with verl-managed ``non_tensor_batch`` keys
    that ``DataProto.union`` deep-compares:
      - ``prompt_uid`` (NOT ``uid``): verl assigns per-rollout-instance UUIDs
        to ``non_tensor_batch["uid"]`` for GRPO grouping. Reusing that key
        causes the union to assert with mismatched arrays.
      - ``data_source_code`` (NOT ``data_source``): the parquet's
        ``data_source`` string column is already in ``non_tensor_batch`` and
        used by the reward dispatcher / metric grouping. Returning a same-
        named int here triggers the same union assertion.

    Curriculum runs populate ``extra_info["prompt_uid"]`` and a ``#shardN``
    suffix on ``data_source``. Non-curriculum / validation samples lack both
    — fall back to a synthetic prompt_uid built from the legacy
    ``extra_info["index"]`` and a shard-less encoded data_source_code so the
    dump still carries recognizable values and the return dict's shape is
    uniform across train and val (required by reward_loop's reward_extra_keys
    logic).
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
        extra_infos = kwargs.get("extra_infos", args[3] if len(args) > 3 else None)
        scores = compute_score_helper(*args, **kwargs)
        if extra_infos is None:
            extra_infos = [None] * len(data_sources)
        return [_wrap(s, ds, ei) for s, ds, ei in zip(scores, data_sources, extra_infos)]

    score = compute_score_helper(*args, **kwargs)
    data_source = kwargs.get("data_source", args[0] if args else None)
    extra_info = kwargs.get("extra_info", args[3] if len(args) > 3 else None)
    return _wrap(score, data_source, extra_info)