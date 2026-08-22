"""Reward manager for LCPO budget training.

Registered through config, so verl core is never patched:

    reward.reward_manager.source=importlib
    reward.reward_manager.module.path=/workspace/verl/full_mix/rewards/lcpo_reward_manager.py
    reward.reward_manager.name=LCPORewardManager

Why a manager and not the plain `compute_score` hook: `compute_score` only
receives the DECODED response string. Counting reasoning tokens would mean
re-tokenizing that string, and decode->encode is not exactly reversible around
the added <think>/</think> tokens. A content-dependent drift of a few tokens is
not acceptable when the target is ~3% length accuracy. `run_single` gets the raw
token ids, so we count by scanning for a token id and never re-tokenize.

Task scoring still goes through `full_mix/rewards/compute_score.py` unchanged;
this class only adds the length/format logic on top.

The extra-info key set is FIXED. verl's streaming path takes the key list from
the first row and then indexes those exact keys into every other row, so a row
with a different key set raises KeyError partway through a run. Every key here
is initialized to a numeric default and then filled in — never added
conditionally. Validation rows that predate the budget columns fall back to
`free` mode with budget -1, which keeps the key set identical on the val path.
"""

import inspect
import os
import sys

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

_HERE = os.path.dirname(os.path.abspath(__file__))
_FULL_MIX_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_FULL_MIX_DIR)
for _p in (_REPO_ROOT, _FULL_MIX_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from full_mix.common.budget_prompt import (  # noqa: E402
    MODE_BUDGET,
    MODE_FREE,
    MODE_NOTHINK,
    NO_BUDGET,
)
from full_mix.rewards import lcpo  # noqa: E402

THINK_OPEN_ID = 151667   # <think>
THINK_CLOSE_ID = 151668  # </think>

CHAT_SOURCES = ("local/dolci-chat-32b",)

# Every key emitted on every row, always, in this order. Values must be int or
# float — the rollout dump path casts these columns with numpy and chokes on
# strings or None.
EXTRA_KEYS = (
    "think_budget", "n_think", "n_answer", "n_total",
    "abs_len_err", "rel_len_err", "len_mult", "len_penalty",
    "is_wellformed", "is_nothink_mode", "nothink_format_ok",
    "chat_exact_half", "truncated", "task_score",
    # Collapse instrumentation. `think_collapsed` is the canary: a fixed
    # threshold, independent of the reward gate, so it keeps reading true even
    # if the gate is retuned. In the stage-a run of 2026-08-19 this share went
    # 3% -> 38% -> 90% across steps 15-19 and nothing surfaced it.
    "min_think_ok", "think_collapsed",
)

# Absolute think-token count below which a budgeted response is "collapsed".
# Reporting only — the reward gate is lcpo.min_think_tokens.
THINK_COLLAPSE_TOKENS = 32


def _defaults() -> dict:
    out = {k: -1.0 if k in ("think_budget", "abs_len_err", "rel_len_err") else 0.0
           for k in EXTRA_KEYS}
    # Rows outside budget mode are never gated on think length.
    out["min_think_ok"] = 1.0
    return out


def split_think(valid_ids) -> tuple[int, int, bool, list]:
    """Locate the reasoning segment in a response by token id.

    Returns (n_think, close_pos, closed, think_span_ids) where
      n_think       tokens up to and including the first </think>
      close_pos     index of that </think>, or -1
      closed        whether a </think> was emitted at all
      think_span_ids the ids strictly between <think> and </think>, for the
                    whitespace check in zero mode

    Counting by token id rather than by parsing text means a literal "</think>"
    written inside prose cannot be mistaken for the real tag.
    """
    ids = valid_ids.tolist() if hasattr(valid_ids, "tolist") else list(valid_ids)
    try:
        close_pos = ids.index(THINK_CLOSE_ID)
    except ValueError:
        return len(ids), -1, False, ids
    start = 1 if ids and ids[0] == THINK_OPEN_ID else 0
    return close_pos + 1, close_pos, True, ids[start:close_pos]


class LCPORewardManager(RewardManagerBase):
    """Task score plus the LCPO length rules, per mode."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None,
                 reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        kw = config.reward.get("reward_kwargs", {}) or {}
        lcpo_cfg = kw.get("lcpo", {}) or {}
        self.stage = str(lcpo_cfg.get("stage", "a")).lower()
        self.delta = float(lcpo_cfg.get("delta", lcpo.DEFAULT_DELTA))
        self.alpha_default = float(lcpo_cfg.get("alpha", lcpo.DEFAULT_ALPHA))
        # Per-source alpha. alpha is a reward-per-token exchange rate, and the
        # right value depends on how much the task score varies within a group:
        # math spreads 0.30-0.50, ifeval only 0.10-0.20, chat 0.15-0.25. One
        # global alpha would let the length term dominate the task term on
        # ifeval and chat by 1.5-3x. Calibrated at gate G2.
        self.alpha_by_source = {str(k): float(v) for k, v in (lcpo_cfg.get("alpha_by_source", {}) or {}).items()}
        assert self.stage in ("a", "b"), f"lcpo.stage must be 'a' or 'b', got {self.stage!r}"

        # Which length the budget governs: "total" (think + answer) or "think".
        # "think" leaves the answer unconstrained, which makes relocating the
        # reasoning the cheapest way to comply — see lcpo.py's docstring.
        self.length_target = str(lcpo_cfg.get("length_target", lcpo.DEFAULT_LENGTH_TARGET)).lower()
        assert self.length_target in (lcpo.LENGTH_TARGET_TOTAL, lcpo.LENGTH_TARGET_THINK), (
            f"lcpo.length_target must be 'total' or 'think', got {self.length_target!r}"
        )
        self.min_think_floor = int(lcpo_cfg.get("min_think_floor", lcpo.DEFAULT_MIN_THINK_FLOOR))
        self.min_think_frac = float(lcpo_cfg.get("min_think_frac", lcpo.DEFAULT_MIN_THINK_FRAC))
        print(
            f"[LCPORewardManager] stage={self.stage} length_target={self.length_target} "
            f"min_think=max({self.min_think_floor}, {self.min_think_frac}*budget) "
            f"alpha={self.alpha_default} alpha_by_source={self.alpha_by_source}",
            flush=True,
        )

    def alpha_for(self, data_source: str) -> float:
        return self.alpha_by_source.get(str(data_source), self.alpha_default)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        item = data[0]

        response_ids = item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(item.batch["attention_mask"][-response_length:].sum())
        valid_response_ids = response_ids[:valid_response_length]

        data_source = item.non_tensor_batch["data_source"]
        ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = item.non_tensor_batch.get("extra_info", {})

        # Rows built before the budget columns existed (the current val files)
        # fall back to unconstrained, which keeps the key set identical.
        mode = str(item.non_tensor_batch.get("think_mode", MODE_FREE) or MODE_FREE)
        budget = int(item.non_tensor_batch.get("think_budget", NO_BUDGET))

        n_think, close_pos, closed, think_span = split_think(valid_response_ids)
        n_answer = max(0, valid_response_length - n_think) if closed else 0

        answer_ids = valid_response_ids[n_think:] if closed else []
        answer_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(answer_ids, skip_special_tokens=True)
        )
        wellformed = closed and bool(answer_text.strip())

        # Zero mode: whitespace inside the block is CORRECT (SFT taught
        # "<think>\n\n</think>"), so this checks for non-whitespace content
        # rather than counting tokens.
        nothink_ok = True
        if mode == MODE_NOTHINK:
            span_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(think_span, skip_special_tokens=True)
            )
            nothink_ok = lcpo.nothink_format_ok(span_text)

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source, solution_str=response_str,
                ground_truth=ground_truth, extra_info=extra_info,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source, solution_str=response_str,
                    ground_truth=ground_truth, extra_info=extra_info,
                ),
            )

        extra = _defaults()
        if isinstance(result, dict):
            task_score = float(result["score"])
            for k, v in result.items():
                if isinstance(v, (int, float, bool)):
                    extra[k] = float(v)
        else:
            task_score = float(result)

        out = lcpo.compute_reward(
            mode=mode, stage=self.stage, task_score=task_score, wellformed=wellformed,
            n_think=n_think, n_total=int(valid_response_length), budget=budget,
            nothink_ok=nothink_ok, alpha=self.alpha_for(data_source), delta=self.delta,
            length_target=self.length_target,
            min_think_floor=self.min_think_floor, min_think_frac=self.min_think_frac,
        )

        extra["score"] = float(out["reward"])
        extra["task_score"] = task_score
        extra["think_budget"] = float(budget)
        extra["n_think"] = float(n_think)
        extra["n_answer"] = float(n_answer)
        extra["n_total"] = float(valid_response_length)
        extra["len_mult"] = float(out["len_mult"])
        extra["len_penalty"] = float(out["len_penalty"])
        extra["is_wellformed"] = 1.0 if wellformed else 0.0
        extra["is_nothink_mode"] = 1.0 if mode == MODE_NOTHINK else 0.0
        extra["nothink_format_ok"] = 1.0 if (mode != MODE_NOTHINK or nothink_ok) else 0.0
        extra["truncated"] = 1.0 if valid_response_length >= response_length else 0.0
        # Judge-outage signal. The chat scorer returns exactly 0.5 when the judge
        # is unreachable, and under stage B's multiplication that is invisible:
        # every chat sample still gets a varying length multiplier, so the slice
        # produces confident-looking gradient about length and none about quality.
        # We are not changing the judge, so watch this share instead.
        extra["chat_exact_half"] = 1.0 if (data_source in CHAT_SOURCES and task_score == 0.5) else 0.0

        extra["min_think_ok"] = float(out["min_think_ok"])

        if mode == MODE_BUDGET and budget > 0:
            # Measured against the length the reward actually controls, so the
            # error curve and the penalty can never disagree.
            n_len = lcpo.controlled_length(n_think, int(valid_response_length), self.length_target)
            extra["abs_len_err"] = float(abs(budget - n_len))
            extra["rel_len_err"] = float(abs(budget - n_len) / budget)
            extra["think_collapsed"] = 1.0 if n_think < THINK_COLLAPSE_TOKENS else 0.0

        return {"reward_score": float(out["reward"]), "reward_extra_info": extra}
