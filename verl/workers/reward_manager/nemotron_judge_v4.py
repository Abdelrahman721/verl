# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Reward manager for the v4 reward (verl/utils/reward_score/nemotron_unified_v4.py).

Same run_single plumbing as the v3 NemotronJudgeRewardManager - last user message from
raw_prompt, judge calls on a bounded thread pool, numeric extras coerced to float - with the
v4 scorer bound instead.

The scorer is wrapped so it never raises: the base manager's exception fallback returns the
v3 key set, which lacks think_error, and reward_extra_info is stacked across items, so one
short dict would desync the metric arrays.

Select it with:
  reward.reward_manager.source=importlib
  reward.reward_manager.name=NemotronJudgeV4RewardManager
  reward.reward_manager.module.path=verl/workers/reward_manager/nemotron_judge_v4.py
"""
from __future__ import annotations

from verl.utils.reward_score import nemotron_unified_v4
from verl.workers.reward_manager.nemotron_judge import NemotronJudgeRewardManager

_FAILED = {
    "score": 0.0,
    "think_error": 0.0,
    "format_error": 0.0,
    "is_call": 0.0,
    "type_match": 0.0,
    "name_match": 0.0,
    "judged": 0.0,
    "judge_score": 0.0,
    "judge_error": 1.0,
}


def _safe_compute_score(**kwargs) -> dict:
    try:
        return nemotron_unified_v4.compute_score(**kwargs)
    except Exception as e:  # never let one row kill a training step
        print(f"[nemotron_judge_v4] scoring failed: {type(e).__name__}: {e}")
        return dict(_FAILED)


class NemotronJudgeV4RewardManager(NemotronJudgeRewardManager):
    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None,
                 reward_model_tokenizer=None, **kwargs) -> None:
        super().__init__(config, tokenizer, compute_score, reward_router_address,
                         reward_model_tokenizer, **kwargs)
        # The parent binds the v3 scorer unconditionally; replace it after that.
        self.compute_score = _safe_compute_score
