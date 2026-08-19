"""Strip chain-of-thought <think>...</think> spans from model outputs before grading.

Fails CLOSED on an unterminated trace. A response that opens <think> and never
closes it (the model ran out of room) has no answer in it, so there is nothing
to grade and the right result is the empty string.

Returning the raw trace instead — which this function used to do, because the
regex needs both tags and the rsplit guard was false — silently handed the whole
chain of thought to the graders as if it were the answer. math_reward then does
``rfind("\\boxed")`` over that text, so a truncated math rollout scored a full
1.0 whenever a boxed value appeared anywhere in its reasoning. Roughly 18% of
this model's rollouts hit the cap without closing the tag, and short reasoning
budgets make that the common case rather than the tail.
"""

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def strip_think(text: str) -> str:
    if not text:
        return text
    # Unterminated trace: opened but never closed -> no answer exists.
    if _THINK_OPEN in text and _THINK_CLOSE not in text:
        return ""
    text = _THINK_RE.sub("", text)
    if _THINK_CLOSE in text:
        text = text.rsplit(_THINK_CLOSE, 1)[1]
    return text.lstrip()
