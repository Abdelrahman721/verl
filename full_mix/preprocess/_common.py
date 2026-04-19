"""Shared helpers for the full_mix preprocessors."""

import os


MATH_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
GSM8K_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def strip_user_prefix(text: str) -> str:
    """Strip a leading 'user:' prefix some Dolci prompts ship with."""
    if not text:
        return text
    lowered = text.lstrip()
    if lowered[:6].lower() == "user: ":
        return lowered[6:].lstrip()
    if lowered[:5].lower() == "user:":
        return lowered[5:].lstrip()
    return text


def ensure_dir(path: str) -> str:
    path = os.path.expanduser(path)
    os.makedirs(path, exist_ok=True)
    return path
