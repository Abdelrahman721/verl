"""Mixed reward dispatcher for the medical_qa training stage with retention data.

Routes per sample by ``data_source``:

  data_source ∈ {medical_qa, medical_conv, medical_benchmark_mcq,
                 medical_benchmark_numeric, medical_benchmark_text,
                 medical_benchmark_medec, snomed_sl, sl}
      → full_mix/rewards/qa_openrouter_bench.compute_score
        OpenRouter judge for medical_qa / medical_conv (qa + conversation
        eval_modes), plus deterministic bench scorers for the four benchmark
        eval_modes (mcq / numeric / medec_hybrid / text_judge_conv) AND the
        two coding eval_modes (snomed_single_label / icd_multi_label).
        Returns dict normalised to the 23-key reward union.

  any other data_source (typically "local/dolci-ifeval-32b",
                                    "local/dolci-chat-32b",
                                    "local/safety-dpo-reference",
                                    "local/avey-identity"
                                    — possibly with `#shardN` suffix)
      → full_mix/rewards/compute_score.compute_score
        Existing dispatcher: routes to ifeval / chat / safety / identity
        reward fn, applies the global identity gate to non-identity samples,
        returns dict with `score`, `prompt_uid`, `data_source_code`.

Both per-item and batch signatures are supported. In batch mode the samples
are partitioned by source family, each family is scored by its native scorer
(so qa_bedrock's asyncio batching and full_mix's threadpool batching both
get to run on their own subsets concurrently), then results are re-ordered
into the original positions.

Per-sample return dicts have DIFFERENT key sets across families. The
trainer's `reward_loop.compute_rm_score` already takes the UNION of keys
across the batch (after the small fix we made earlier), so columns for
keys only present in one family will be populated for those samples and
None elsewhere — rollout dumps look correct either way.

Wire in the training script with:
    REWARD_FN_PATH=/workspace/verl/full_mix/rewards/compute_score_medical_mix.py
    REWARD_FN_NAME=compute_score
"""

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

from full_mix.rewards import qa_openrouter_bench as _qa  # noqa: E402
from full_mix.rewards import compute_score as _full_mix_cs  # noqa: E402

logger = logging.getLogger(__name__)


# Every data_source that should route to qa_openrouter_bench's compute_score.
# The bench dispatcher inside qa_openrouter_bench then sub-routes on
# extra_info.task_type / extra_info.helm_scenario to pick the actual scorer.
_MEDICAL_SOURCES = frozenset({
    # Stage-1 / stage-2 medical_qa data
    "medical_qa", "medical_conv",
    # Benchmark sources (stage-3 onward)
    "medical_benchmark_mcq", "medical_benchmark_numeric",
    "medical_benchmark_text", "medical_benchmark_medec",
    # Stage-4 coding sources
    "snomed_sl",   # SNOMED single-label
    "sl",          # ICD multi-label  (data_source string from the source parquet)
    "slb",         # ICD-10 SLB variant (same scorer as `sl` — routed via task_type="icd_multi_label")
})


def _is_medical(data_source) -> bool:
    return data_source in _MEDICAL_SOURCES


# ---------------------------------------------------------------------------
# Key-set unification.
#
# Each branch's reward fn returns its own dict shape:
#   qa_bedrock          : {"score", "reward/judge_score", "reward/raw_score",
#                          "reward/format_penalty", "reward/accuracy",
#                          "reward/completeness", "reward/clarity",
#                          "reward/behavior", "reward/compliance",
#                          "reward/reasoning_length", "reward/answer_length",
#                          "reward/eval_mode"}
#   full_mix compute_score: {"score", "prompt_uid", "data_source_code"}
#
# Each per-sample dict becomes one row in a per-batch ``non_tensor_batch``
# in the rollout DataProto. Within a single rollout call, my earlier fix to
# ``reward_loop.compute_rm_score`` unions the keys across samples so that
# call's batch is internally consistent.
#
# But when the trainer pulls MULTIPLE such DataProtos out of the queue (one
# per prompt's rollouts), it concats them via DataProto.concat, which calls
# list_of_dict_to_dict_of_list across the per-DataProto non_tensor_batch
# dicts. That function asserts the FIRST dict's key set is a superset of
# every later dict's — i.e. all DataProtos in the queue must share keys.
# If a medical-source DataProto and a curated-source DataProto land in the
# same queue pull, that assert fires:
#   AssertionError: Key 'prompt_uid' is not present in the keys of the
#                   first dictionary in the list.
#
# Fix: every per-sample dict the dispatcher returns gets padded to the
# UNION of the two branches' key sets. Sentinel defaults are 0 / 0.0 / "" —
# safe to fan out into numpy arrays for the non_tensor_batch columns.
# Downstream analysis distinguishes "real" rows from "padded" rows via
# data_source / data_source_code / reward/eval_mode.
# ---------------------------------------------------------------------------

# Padding defaults = UNION of:
#   1. qa_openrouter_bench's 23-key reward union (already covers every key
#      emitted by the medical / benchmark / coding branches).
#   2. The retention branch's prompt_uid / data_source_code.
# Importing `_REWARD_UNION_DEFAULTS` directly keeps this dispatcher in
# lockstep — any new reward/* key added to qa_openrouter_bench is picked up
# automatically the next time this module is imported.
_PADDING_DEFAULTS: dict = {
    **_qa._REWARD_UNION_DEFAULTS,
    # full_mix compute_score keys (retention branch only):
    "prompt_uid":         0,
    "data_source_code":   0,
}


def _pad(result):
    """Return a copy of ``result`` augmented to the union key set.

    No-ops if ``result`` isn't a dict (defensive — qa_bedrock and full_mix
    both always return dicts in normal operation).
    """
    if not isinstance(result, dict):
        return result
    out = dict(_PADDING_DEFAULTS)
    out.update(result)
    return out


def compute_score(*args, **kwargs):
    """Dual-signature entry point — same shape as the other full_mix dispatchers."""
    is_batch = (
        "data_sources" in kwargs
        or (args and isinstance(args[0], (list, tuple)))
    )
    if is_batch:
        return _compute_batch(*args, **kwargs)
    return _compute_single(*args, **kwargs)


def _compute_single(*args, **kwargs):
    data_source = kwargs.get("data_source", args[0] if args else None)
    ground_truth = kwargs.get("ground_truth", args[2] if len(args) > 2 else None)
    if _is_medical(data_source):
        # qa_openrouter_bench already injects reward/gold in its return path.
        out = _pad(_qa.compute_score(*args, **kwargs))
    else:
        out = _pad(_full_mix_cs.compute_score(*args, **kwargs))
    # Retention branch doesn't set reward/gold itself; fill it here so the
    # rollout dump carries the ground truth for both branches uniformly.
    if not out.get("reward/gold"):
        out["reward/gold"] = _qa._serialize_gold(ground_truth)
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
            "compute_score_medical_mix batch call missing data_sources/solution_strs/ground_truths"
        )

    n = len(data_sources)
    if extra_infos is None:
        extra_infos = [None] * n

    medical_idx: list[int] = []
    other_idx:   list[int] = []
    for i, ds in enumerate(data_sources):
        (medical_idx if _is_medical(ds) else other_idx).append(i)

    results: list = [None] * n

    if medical_idx:
        med_out = _qa.compute_score(
            data_sources  = [data_sources[i]  for i in medical_idx],
            solution_strs = [solution_strs[i] for i in medical_idx],
            ground_truths = [ground_truths[i] for i in medical_idx],
            extra_infos   = [extra_infos[i]   for i in medical_idx],
        )
        for j, i in enumerate(medical_idx):
            results[i] = _pad(med_out[j])

    if other_idx:
        other_out = _full_mix_cs.compute_score(
            data_sources  = [data_sources[i]  for i in other_idx],
            solution_strs = [solution_strs[i] for i in other_idx],
            ground_truths = [ground_truths[i] for i in other_idx],
            extra_infos   = [extra_infos[i]   for i in other_idx],
        )
        for j, i in enumerate(other_idx):
            results[i] = _pad(other_out[j])

    # Retention branch doesn't set reward/gold itself; fill it here so the
    # rollout dump carries the ground truth uniformly. Medical rows already
    # have it set by qa_openrouter_bench; this is a no-op for them.
    for i, r in enumerate(results):
        if isinstance(r, dict) and not r.get("reward/gold"):
            r["reward/gold"] = _qa._serialize_gold(ground_truths[i])

    return results
