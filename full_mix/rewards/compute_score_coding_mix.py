"""Top-level reward dispatcher for stage-5 medical-coding training (+retention).

Three-branch routing by ``data_source``:

  CODING — data_source ∈ {ml, if, snomed_ml, snomed_if, sl, snomed_sl}
      → full_mix.rewards.coding.compute_score
        Self-contained coding suite. Sub-routes on extra_info.task_type to:
          icd10_multilabel + icd10_instruction_follow + icd_multi_label
              → icd_scorer (regex-based P/R/F1 on ICD-10 codes)
          snomed_multilabel + snomed_single_label
              → snomed_ml_scorer (regex-based P/R/F1 on SCT IDs;
                                  snomed_single_label adapts its gold
                                  dict to a singleton gt_sct_ids list)
          snomed_instruction_follow
              → snomed_if_scorer (uses the embedded instruction-following
                                  framework; consults a DeepSeek judge via
                                  OpenRouter when IF_USE_LLM_JUDGES=1).
        Returns dict normalised to the 15-key coding union.

  MEDICAL / BENCH — data_source ∈ {medical_qa, medical_conv,
                                    medical_benchmark_mcq,
                                    medical_benchmark_numeric,
                                    medical_benchmark_text,
                                    medical_benchmark_medec}
      → full_mix.rewards.qa_openrouter_bench.compute_score
        OpenRouter LLM judge for medical_qa / medical_conv (qa + conversation
        eval_modes) plus the four deterministic benchmark scorers
        (mcq / numeric / medec_hybrid / text_judge_conv). Returns dict
        normalised to qa_openrouter_bench's 23-key reward union.
        Same path stage-4's compute_score_medical_mix.py used.

  GENERAL RETENTION — every other data_source (typically
                                    "local/dolci-ifeval-32b",
                                    "local/dolci-chat-32b",
                                    "local/safety-dpo-reference",
                                    "local/avey-identity")
      → full_mix.rewards.compute_score.compute_score
        Existing dispatcher: routes to ifeval / chat / safety / identity
        reward fn, applies the global identity gate, returns dict with
        `score`, `prompt_uid`, `data_source_code`.

Per-sample return dicts have DIFFERENT key sets across families. We pad
every output up to a triple-union of:
  - coding._REWARD_UNION_DEFAULTS                  (15 keys)
  - qa_openrouter_bench._REWARD_UNION_DEFAULTS     (23 keys, overlaps with coding's)
  - {prompt_uid: 0, data_source_code: 0}           (retention sentinel)
so every per-prompt non_tensor_batch row carries the same key set —
satisfies DataProto.concat's first-dict-is-superset assertion.

Both per-item and batch signatures are supported. In batch mode the samples
are partitioned by branch, each branch is scored by its native scorer (so
coding's row-wise loop, qa_openrouter_bench's asyncio batching, and
full_mix's threadpool batching all run concurrently on their own subsets),
and results are reassembled in original order.

Wire in the training script with:
    REWARD_FN_PATH=/workspace/verl/full_mix/rewards/compute_score_coding_mix.py
    REWARD_FN_NAME=compute_score
"""

from __future__ import annotations

import logging
import os
import sys

# Make `full_mix` importable whether we're loaded as a script (via
# load_extern_object) or as a package — same shim the other dispatchers use.
_HERE = os.path.dirname(os.path.abspath(__file__))
_FULL_MIX_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_FULL_MIX_DIR)
for _p in (_REPO_ROOT, _FULL_MIX_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from full_mix.rewards import coding as _coding  # noqa: E402
from full_mix.rewards import compute_score as _full_mix_cs  # noqa: E402
from full_mix.rewards import qa_openrouter_bench as _qa  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================================
# Source classification
# ============================================================================
_CODING_SOURCES = frozenset({
    "ml",          # ICD-10 multilabel (stage-5)
    "if",          # ICD-10 instruction-following (stage-5)
    "snomed_ml",   # SNOMED CT multilabel (stage-5; also stage-6 real-data)
    "snomed_if",   # SNOMED CT instruction-following (stage-5)
    # Stage-4 legacy data_sources — stage-4 coding-unmastered rows that
    # arrive as stage-5 retention still get a real reward signal
    # (icd_multi_label / snomed_single_label task_type routes in
    # coding/dispatcher.py).
    "sl",          # ICD multilabel (stage-4, effectively single-label data)
    "snomed_sl",   # SNOMED single-label (stage-4)
    # Stage-6 new ICD sources — real clinical-note data. Both stamped
    # task_type=icd10_multilabel so they route to the existing ICD scorer.
    "mlb",         # ICD-10 multilabel-binary (stage-6, real)
    "slb",         # ICD-10 single-label-binary (stage-6, real)
})

# Medical QA + medical benchmarks — routed via qa_openrouter_bench.
# Same set stage-4's compute_score_medical_mix used MINUS the coding sources
# (which we handle in the coding suite now).
_MEDICAL_SOURCES = frozenset({
    "medical_qa", "medical_conv",
    "medical_benchmark_mcq", "medical_benchmark_numeric",
    "medical_benchmark_text", "medical_benchmark_medec",
})


def _is_coding(data_source) -> bool:
    return data_source in _CODING_SOURCES


def _is_medical(data_source) -> bool:
    return data_source in _MEDICAL_SOURCES


# ============================================================================
# Padding defaults — TRIPLE UNION of every branch's reward dict shape:
#   1. coding._REWARD_UNION_DEFAULTS                (15 keys)
#   2. qa_openrouter_bench._REWARD_UNION_DEFAULTS   (23 keys, overlaps with #1)
#   3. {prompt_uid, data_source_code}               (retention sentinels)
# Importing _REWARD_UNION_DEFAULTS by name keeps the union in lockstep with
# each scorer — adding a new reward/* key in either module propagates here
# on the next import.
# ============================================================================
_PADDING_DEFAULTS: dict = {
    **_coding._REWARD_UNION_DEFAULTS,
    **_qa._REWARD_UNION_DEFAULTS,
    "prompt_uid":       0,
    "data_source_code": 0,
}


def _pad(result):
    """Pad a per-sample dict up to the union key set; preserve existing values."""
    if not isinstance(result, dict):
        return result
    out = dict(_PADDING_DEFAULTS)
    out.update(result)
    return out


# ============================================================================
# Public entry point
# ============================================================================
def compute_score(*args, **kwargs):
    """Dual-signature entry point — same shape as the other mix dispatchers."""
    is_batch = (
        "data_sources" in kwargs
        or (args and isinstance(args[0], (list, tuple)))
    )
    if is_batch:
        return _compute_batch(*args, **kwargs)
    return _compute_single(*args, **kwargs)


def _compute_single(*args, **kwargs):
    data_source  = kwargs.get("data_source",  args[0] if args else None)
    ground_truth = kwargs.get("ground_truth", args[2] if len(args) > 2 else None)
    if _is_coding(data_source):
        # Coding suite already injects reward/gold itself.
        out = _pad(_coding.compute_score(*args, **kwargs))
    elif _is_medical(data_source):
        # qa_openrouter_bench already injects reward/gold itself.
        out = _pad(_qa.compute_score(*args, **kwargs))
    else:
        # General retention: chat / ifeval / safety / identity.
        out = _pad(_full_mix_cs.compute_score(*args, **kwargs))
    # Retention branch doesn't set reward/gold; fill it here so rollout
    # dumps carry the ground truth uniformly across all three branches.
    if isinstance(out, dict) and not out.get("reward/gold"):
        out["reward/gold"] = _coding._serialize_gold(ground_truth)
    return out


def _compute_batch(*args, **kwargs):
    data_sources  = kwargs.get("data_sources",
                               args[0] if len(args) > 0 else None)
    solution_strs = kwargs.get("solution_strs",
                               args[1] if len(args) > 1 else None)
    ground_truths = kwargs.get("ground_truths",
                               args[2] if len(args) > 2 else None)
    extra_infos   = kwargs.get("extra_infos",
                               args[3] if len(args) > 3 else None)

    if data_sources is None or solution_strs is None or ground_truths is None:
        raise ValueError(
            "compute_score_coding_mix batch call missing "
            "data_sources/solution_strs/ground_truths"
        )

    n = len(data_sources)
    if extra_infos is None:
        extra_infos = [None] * n

    coding_idx: list[int]  = []
    medical_idx: list[int] = []
    other_idx:   list[int] = []
    for i, ds in enumerate(data_sources):
        if _is_coding(ds):
            coding_idx.append(i)
        elif _is_medical(ds):
            medical_idx.append(i)
        else:
            other_idx.append(i)

    results: list = [None] * n

    def _scatter(branch_out, branch_idx):
        for j, i in enumerate(branch_idx):
            results[i] = _pad(branch_out[j])

    if coding_idx:
        _scatter(_coding.compute_score(
            data_sources  = [data_sources[i]  for i in coding_idx],
            solution_strs = [solution_strs[i] for i in coding_idx],
            ground_truths = [ground_truths[i] for i in coding_idx],
            extra_infos   = [extra_infos[i]   for i in coding_idx],
        ), coding_idx)

    if medical_idx:
        _scatter(_qa.compute_score(
            data_sources  = [data_sources[i]  for i in medical_idx],
            solution_strs = [solution_strs[i] for i in medical_idx],
            ground_truths = [ground_truths[i] for i in medical_idx],
            extra_infos   = [extra_infos[i]   for i in medical_idx],
        ), medical_idx)

    if other_idx:
        _scatter(_full_mix_cs.compute_score(
            data_sources  = [data_sources[i]  for i in other_idx],
            solution_strs = [solution_strs[i] for i in other_idx],
            ground_truths = [ground_truths[i] for i in other_idx],
            extra_infos   = [extra_infos[i]   for i in other_idx],
        ), other_idx)

    # Retention branch doesn't set reward/gold itself; fill it here so the
    # rollout dump carries the ground truth uniformly. Coding + medical rows
    # already have it set; this is a no-op for them.
    for i, r in enumerate(results):
        if isinstance(r, dict) and not r.get("reward/gold"):
            r["reward/gold"] = _coding._serialize_gold(ground_truths[i])

    return results
