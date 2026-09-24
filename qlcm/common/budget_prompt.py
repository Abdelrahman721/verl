"""Marker-less budget prompt for the qlcm models.

The qlcm chat template has no `/think` / `/no_think` machinery: the model
always reasons inside `<think>...</think>` and then answers. So a budget is
stated as a plain trailing sentence on the last user turn and nothing else,
and "free" mode is the prompt exactly as the model was trained on it:

    budget  {task}\\n\\nThink for a maximum of {n} tokens.
    free    {task}

The sentence itself comes from `full_mix.common.budget_prompt.budget_line`
(the "max" wording, which is what the stage-b reward enforces), so the two
model families cannot drift apart on the one string that matters. Nothing in
`full_mix/common/budget_prompt.py` is changed; that module keeps its markers
for the 4B runs.

There is no no-think mode here. `render_suffix_plain` raises on it rather than
rendering something the model has never seen.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from full_mix.common.budget_prompt import (  # noqa: E402
    MODE_BUDGET,
    MODE_FREE,
    NO_BUDGET,
    SEP,
    STYLE_MAX,
    _PREFIX,
    _SUFFIX,
    budget_line,
)

MODES = (MODE_BUDGET, MODE_FREE)

__all__ = [
    "MODE_BUDGET", "MODE_FREE", "MODES", "NO_BUDGET",
    "render_suffix_plain", "render_prompt_plain", "apply_to_messages_plain",
    "parse_suffix_plain",
]


def render_suffix_plain(mode: str, budget: int = NO_BUDGET) -> str:
    """Text appended after the task: the budget sentence, or nothing."""
    if mode == MODE_FREE:
        return ""
    if mode == MODE_BUDGET:
        if budget is None or int(budget) <= 0:
            raise ValueError(f"budget mode needs a positive budget, got {budget!r}")
        return SEP + budget_line(budget, STYLE_MAX)
    raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")


def render_prompt_plain(task_text: str, mode: str, budget: int = NO_BUDGET) -> str:
    """Full user-turn text for one row. Free mode returns the task text unchanged."""
    if mode == MODE_FREE:
        return task_text
    return task_text.rstrip() + render_suffix_plain(mode, budget)


def apply_to_messages_plain(messages: list[dict], mode: str, budget: int = NO_BUDGET) -> list[dict]:
    """Copy of `messages` with the suffix on the LAST non-assistant turn.

    Multi-turn medical conversations carry gold assistant turns; only the
    final user turn is the one the model answers, so only it gets the budget.
    """
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") != "assistant":
            m["content"] = render_prompt_plain(m["content"], mode, budget)
            return out
    raise ValueError("no non-assistant turn to attach a budget to")


def parse_suffix_plain(user_text: str) -> tuple[str, int]:
    """Recover (mode, budget) from a rendered user turn. For tests and checks."""
    last = user_text.rstrip().rsplit("\n", 1)[-1].strip()
    pre = _PREFIX[STYLE_MAX]
    if last.startswith(pre) and last.endswith(_SUFFIX):
        inner = last[len(pre):-len(_SUFFIX)]
        if inner.isdigit():
            return MODE_BUDGET, int(inner)
    return MODE_FREE, NO_BUDGET
