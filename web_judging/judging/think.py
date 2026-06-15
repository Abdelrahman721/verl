"""Strip chain-of-thought <think>...</think> spans from model outputs before grading."""

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.lstrip()
