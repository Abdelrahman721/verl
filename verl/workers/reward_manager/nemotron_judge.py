# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Reward manager for data_source "nemotron_pivot" with an LLM judge on prose rows.

Implements the EXPERIMENTAL reward-loop interface (RewardManagerBase.run_single), not the
classic AbstractRewardManager.__call__ one. With rollout.mode=async, rewards are computed
by RewardLoopWorker actors that await `run_single` on one item at a time, so that is the
contract that has to be met.

Why a custom manager at all: the judge needs the last USER message, and the stock managers
pass only (data_source, solution_str, ground_truth, extra_info) to compute_score. This one
reads the message list off `raw_prompt` - the same field RewardLoopWorker itself asserts on
- so no parsing of the rendered prompt is needed.

Judging is network-bound, so the blocking call is pushed onto a bounded thread pool via
run_in_executor rather than blocking the worker's event loop. Total concurrency against
OpenRouter is reward.num_workers (default 8) x NEMOTRON_JUDGE_WORKERS (default 4).

Select it with:
  reward.reward_manager.source=importlib
  reward.reward_manager.name=NemotronJudgeRewardManager
  reward.reward_manager.module.path=verl/workers/reward_manager/nemotron_judge.py
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import nemotron_pivot_judge


def _last_user_message(raw_prompt: Any) -> str:
    """Last genuine user turn. Tool outputs are user-role messages in this template, so a
    message that is only a <tool_response> wrapper is skipped - matching the
    `multi_step_tool` scan in the chat template."""
    if raw_prompt is None:
        return ""
    out = ""
    for m in list(raw_prompt):
        if isinstance(m, dict):
            role, content = m.get("role"), m.get("content")
        else:
            role, content = getattr(m, "role", None), getattr(m, "content", None)
        if role != "user" or not isinstance(content, str):
            continue
        s = content.strip()
        if s.startswith("<tool_response>") and s.endswith("</tool_response>"):
            continue
        out = s
    return out


class NemotronJudgeRewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None,
                 reward_model_tokenizer=None, **kwargs) -> None:
        super().__init__(config, tokenizer, compute_score)
        # load_reward_manager always injects default_compute_score when no custom_reward_function
        # is configured, so `compute_score or ...` would silently select the wrong scorer.
        # This manager exists to run the judge; bind it unconditionally.
        self.compute_score = nemotron_pivot_judge.compute_score
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self._pool = ThreadPoolExecutor(
            max_workers=int(os.getenv("NEMOTRON_JUDGE_WORKERS", "4")),
            thread_name_prefix="nemotron-judge",
        )

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        item = data[0]

        response_ids = item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(item.non_tensor_batch.get("extra_info") or {})
        extra_info["num_turns"] = item.non_tensor_batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = item.non_tensor_batch.get("reward_scores", {})
        last_user = _last_user_message(item.non_tensor_batch.get("raw_prompt"))

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        def score_it():
            try:
                return self.compute_score(
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    last_user_message=last_user,
                )
            except Exception as e:  # never let one row kill a training step
                print(f"[nemotron_judge] scoring failed: {type(e).__name__}: {e}")
                # Full key set: reward_extra_info is stacked across items, so a partial
                # dict here would desync the metric arrays.
                return {"score": 0.0, "is_call": 0.0, "type_match": 0.0, "name_match": 0.0,
                        "format_error": 0.0, "judged": 0.0, "judge_score": 0.0, "judge_error": 1.0}

        result = await self.loop.run_in_executor(self._pool, score_it)

        reward_extra_info = {}
        if isinstance(result, dict):
            score = result["score"]
            reward_extra_info.update(result)
        else:
            score = result
            reward_extra_info["acc"] = score
        reward_extra_info["last_user_len"] = len(last_user)

        # _dump_generations json.dumps these values directly, and they arrive as numpy
        # scalars: an int list becomes int64, which json cannot encode. Every key that
        # survives that path is a float, so coerce numerics to float here.
        reward_extra_info = {
            k: (float(v) if isinstance(v, (int, float, bool)) and not isinstance(v, str) else v)
            for k, v in reward_extra_info.items()
        }
        return {"reward_score": float(score), "reward_extra_info": reward_extra_info}
