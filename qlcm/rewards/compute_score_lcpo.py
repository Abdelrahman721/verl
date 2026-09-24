"""Reward entry point for qlcm LCPO training (stage 9).

A thin wrapper over `compute_score_coding_mix.compute_score` that exists for
three reasons:

  1. Validation rows carry a reporting tag on data_source ("mlb@b00256",
     "medical_qa@free", ...) so verl emits one metric series per budget rung.
     The general branch already tolerates the tag through
     `qlcm.curriculum.encoding.split_variant`, but `compute_score_coding_mix`
     routes coding / medical rows by exact set membership and would send a
     tagged "mlb@b00256" row to the general branch, where it scores 0 with a
     warning. The tag is stripped here, before routing.
  2. Per-source training curves. On the agent-loop reward path the dataset's
     `data_source` column never reaches the trainer batch; only numeric keys in
     the reward's extra info do. `src_id` is a fixed numeric code per source so
     `full_mix/per_source_metrics.py` can group by it.
  3. The qlcm linear length cost (`length_penalty.py`) is read from the
     environment at import time inside the reward workers. Under LCPO the
     budget term replaces it, and leaving it on would double-charge length.
     Importing this module with the cost enabled is an error, not a warning.

The padded 32-key dict from the mix dispatcher is returned unchanged apart
from the added `src_id`. `LCPORewardManager` copies only int/float/bool keys,
so the two string keys (reward/eval_mode, reward/gold) drop out and the
extra-info key set stays fixed across rows.

Wire in with:
    REWARD_FN_PATH=/workspace/verl/qlcm/rewards/compute_score_lcpo.py
    REWARD_FN_NAME=compute_score
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_QLCM_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_QLCM_DIR)
for _p in (_REPO_ROOT, _QLCM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qlcm.curriculum.encoding import split_variant  # noqa: E402
from qlcm.rewards import compute_score_coding_mix as _mix  # noqa: E402
from qlcm.rewards import length_penalty as _length_penalty  # noqa: E402

if _length_penalty.ENABLED:
    raise RuntimeError(
        "qlcm.rewards.length_penalty is enabled in this process. LCPO's budget term "
        "replaces the linear word cost, so export QLCM_LEN_PENALTY_ENABLE=0 BEFORE "
        "sourcing qlcm/load_runtime_env.sh so it reaches the Ray reward workers."
    )

# Numeric source codes for per-source training curves. Fixed on purpose: a
# code must never depend on which sources happen to be in a batch.
SRC_ID: dict[str, int] = {
    "mlb": 1, "ml": 2, "if": 3, "snomed_ml": 4, "snomed_if": 5,
    "sl": 6, "snomed_sl": 7, "slb": 8,
    "medical_qa": 9, "medical_conv": 10,
    "local/dolci-ifeval-32b": 11, "local/dolci-chat-32b": 12,
    "local/safety-dpo-reference": 13, "local/avey-identity": 14,
    # validation-only sources
    "google/IFEval": 21, "openai/gsm8k": 22, "HuggingFaceH4/MATH-500": 23,
}


def base_source(data_source) -> str:
    return split_variant(str(data_source))[0]


def _stamp(result, base: str):
    if isinstance(result, dict):
        result["src_id"] = float(SRC_ID.get(base, 0))
    return result


def compute_score(*args, **kwargs):
    """Same dual signature as compute_score_coding_mix.compute_score."""
    is_batch = "data_sources" in kwargs or (args and isinstance(args[0], (list, tuple)))
    if is_batch:
        if "data_sources" in kwargs:
            bases = [base_source(d) for d in kwargs["data_sources"]]
            kwargs["data_sources"] = bases
        else:
            bases = [base_source(d) for d in args[0]]
            args = (bases,) + tuple(args[1:])
        out = _mix.compute_score(*args, **kwargs)
        for r, b in zip(out, bases):
            _stamp(r, b)
        return out

    if "data_source" in kwargs:
        base = base_source(kwargs["data_source"])
        kwargs["data_source"] = base
    else:
        base = base_source(args[0])
        args = (base,) + tuple(args[1:])
    return _stamp(_mix.compute_score(*args, **kwargs), base)
