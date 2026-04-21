"""DeepSeek helpers: same pattern as mcs-instruct-dataset/src/llm.py call_deepseek."""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

# Set to "1", "true", "yes" to enable DeepSeek API calls (requires DEEPSEEK_API_KEY).
_USE_LLM_ENV = "IF_USE_LLM_JUDGES"

_DEEPSEEK_MODEL = "deepseek-reasoner"
_DEEPSEEK_BASE = "https://api.deepseek.com"


def judges_enabled() -> bool:
    v = os.getenv(_USE_LLM_ENV, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def call_deepseek(
    prompt: Optional[str] = None,
    *,
    system_message: Optional[str] = None,
    user_message: Optional[str] = None,
    max_retries: int = 3,
):
    """Call DeepSeek (deepseek-reasoner only); same contract as mcs-instruct-dataset/src/llm.call_deepseek.

    Returns the full chat completion response, or ``None`` if no API key or all attempts fail.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError:
        return None

    client = OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE)

    if user_message is not None:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
    elif prompt is not None:
        messages = [{"role": "user", "content": prompt}]
    else:
        raise ValueError("call_deepseek requires either prompt= or user_message=")

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=_DEEPSEEK_MODEL,
                messages=messages,
            )
            return response
        except Exception as e:
            print(f"DeepSeek API error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2**attempt)

    return None


def _assistant_message_text(message: object) -> str:
    content = (getattr(message, "content", None) or "").strip()
    reasoning = (getattr(message, "reasoning_content", None) or "").strip()
    if content:
        return content
    return reasoning


def deepseek_completion_text(system_message: str, user_message: str) -> str | None:
    """Run DeepSeek with system + user messages; return assistant text or None."""
    resp = call_deepseek(system_message=system_message, user_message=user_message)
    if resp is None or not resp.choices:
        return None
    out = _assistant_message_text(resp.choices[0].message)
    return out if out else None


def call_judge(system_message: str, user_message: str) -> str | None:
    if _judge_fn is not None:
        return _judge_fn(system_message, user_message)
    if not judges_enabled():
        return None
    return deepseek_completion_text(system_message, user_message)


# --- v09_multiple_choice: LLM extracts final chosen IDs from free-form answer ---

V09_EXTRACT_SYSTEM = (
    "An assistant was asked a multiple-choice question whose options are SNOMED CT concept IDs "
    "(possibly with descriptions).\n"
    "From the assistant response below, output ONLY the SNOMED CT concept IDs, exactly as they appear "
    "in the response, for every option the assistant chose as their FINAL answer.\n"
    "Ignore IDs that appear only for justification, comparison, or discussion.\n"
    "Ignore IDs for options the assistant did not select as part of the final chosen answer set.\n"
    "Output only the IDs (one line, comma- or space-separated is fine). No other text."
)


def call_v09_pred_extraction(assistant_response: str) -> str | None:
    if _v09_extract_fn is not None:
        return _v09_extract_fn(assistant_response)
    if not judges_enabled():
        return None
    return deepseek_completion_text(
        V09_EXTRACT_SYSTEM,
        f"ASSISTANT_RESPONSE:\n{assistant_response}\n",
    )


# --- v08_code_validation ---

JUDGE_V08_SYSTEM = (
    "You are a strict grader. You are given the true SNOMED CT concept ID (correct answer), "
    "a proposed SNOMED CT concept ID, and an assistant response.\n"
    "Decide whether the response agrees with the correct code (whether the assistant appropriately "
    "reflects that the proposed code matches or does not match the true code, as required).\n"
    "Return ONLY 'yes' or 'no'."
)


def judge_v08_user(true_code: str, proposed_code: str, assistant_response: str) -> str:
    return (
        f"TRUE_CODE (correct):\n{true_code}\n\n"
        f"PROPOSED_CODE:\n{proposed_code}\n\n"
        f"ASSISTANT_RESPONSE:\n{assistant_response}\n"
    )


# --- v11_code_desc_consistency (rules.py 955–963) ---

JUDGE_V11_SYSTEM = (
    "You are a strict grader. You are given:\n"
    "- The SNOMED CT ID-description pair that was shown to the assistant in the user prompt,\n"
    "- The ground truth ID-description pair for this example,\n"
    "- The assistant's response.\n\n"
    "Decide whether the assistant correctly concluded whether the shown pair is consistent with SNOMED CT, "
    "and when the shown pair was wrong, whether they fixed the description (or pair) appropriately "
    "so it aligns with the ground truth. Return ONLY 'yes' or 'no'."
)


def judge_v11_user(
    gold_id: str,
    shown_desc: str,
    gold_desc: str,
    assistant_response: str,
) -> str:
    return (
        f"PAIR_SHOWN_TO_ASSISTANT:\nID {gold_id}\nDescription: {shown_desc}\n\n"
        f"GROUND_TRUTH_PAIR:\nID {gold_id}\nDescription: {gold_desc}\n\n"
        f"ASSISTANT_RESPONSE:\n{assistant_response}\n"
    )


# --- v12_set_correction (rules.py 1066–1073) ---

JUDGE_V12_SYSTEM = (
    "You are a strict grader. You are given:\n"
    "- The ground truth SNOMED CT concept IDs (with descriptions) for this clinical note,\n"
    "- The proposed code set that appeared in the user prompt (may be incomplete or contain extras),\n"
    "- The assistant's response about checking and correcting that set.\n\n"
    "Decide whether the assistant correctly evaluated the proposed set against the note and ground truth. "
    "Return ONLY 'yes' or 'no'."
)


def judge_v12_user(
    ground_truth_codes: str,
    proposed_for_judge: str,
    assistant_response: str,
) -> str:
    return (
        f"GROUND_TRUTH_LABELS:\n{ground_truth_codes}\n\n"
        f"PROPOSED_SET_FROM_USER_PROMPT:\n{proposed_for_judge}\n\n"
        f"ASSISTANT_RESPONSE:\n{assistant_response}\n"
    )


def verdict_yes(raw: str | None) -> bool:
    if not raw:
        return False
    return raw.strip().lower() == "yes"


def set_judge_fn(fn: Callable[[str, str], str | None] | None) -> None:
    global _judge_fn
    _judge_fn = fn


def set_v09_extract_fn(fn: Callable[[str], str | None] | None) -> None:
    global _v09_extract_fn
    _v09_extract_fn = fn


_judge_fn: Callable[[str, str], str | None] | None = None
_v09_extract_fn: Callable[[str], str | None] | None = None
