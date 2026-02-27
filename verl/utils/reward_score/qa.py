"""
Reward scoring for medical QA using LLM-as-a-judge (OpenAI).

The dataset stores ground_truth as a dict:
    {"gold_answer": str, "criteria": list[str]}

The compute_score function is the single entry point for verl's reward manager.
It returns a dict with individual score components and a combined "score" key.

Expected model format: <think>reasoning</think> then the answer directly after.

Reward structure:
    - judge_score: 0.0 to 1.0 (single LLM call, holistic 5-category rating)
    - format_score: -0.5 to 0.0 (penalty only, no bonus for correct format)
    - score: judge_score + format_score

LLM judge categories (mapped to rewards):
    A (Excellent)  → 1.0  — correct, complete, concise, may exceed gold
    B (Good)       → 0.5  — correct on main points, minor gaps or verbosity
    C (Partial)    → 0.2  — partially correct, significant gaps or errors
    D (Fail)       → 0.0  — wrong, irrelevant, or harmful
"""

import asyncio
import json
import os
import re

from openai import AsyncOpenAI

# ============================================================================
# FORMAT TAGS
# ============================================================================
THINKING_START = "<think>"
THINKING_END = "</think>"

# ============================================================================
# LLM JUDGE CONFIG
# ============================================================================
_GPT_CLIENT = None


def _get_judge_client():
    global _GPT_CLIENT
    if _GPT_CLIENT is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        _GPT_CLIENT = AsyncOpenAI(api_key=api_key)
    return _GPT_CLIENT


JUDGE_MODEL = os.environ.get("QA_GRADING_MODEL", "gpt-5.1")

JUDGE_SYSTEM_PROMPT = (
    "You are an expert medical examiner grading a student's answer.\n"
    "You will receive a question, the gold standard answer, and the student's answer.\n"
    "Grade the student's answer into exactly one category:\n\n"
    "A — Excellent: The answer is correct, complete, well-written, and concise. "
    "It covers all key points from the gold answer (or even adds correct detail "
    "beyond it). No significant errors.\n\n"
    "B — Good: The answer is correct on the main points and mostly complete, "
    "but has minor gaps, slight verbosity, or small imprecisions that don't "
    "change clinical meaning.\n\n"
    "C — Partial: The answer is partially correct but has significant gaps, "
    "missing key information, excessive verbosity, or notable inaccuracies.\n\n"
    "D — Fail: The answer is wrong, irrelevant, severely incomplete, contradicts "
    "the gold standard, or could be harmful.\n\n"
    "Respond with ONLY a single letter: A, B, C, or D. Nothing else."
)

JUDGE_USER_TEMPLATE = """QUESTION:
{question}

GOLD STANDARD ANSWER:
{gold}

STUDENT'S ANSWER:
{answer}

Grade (A/B/C/D):"""

# Category → reward mapping
GRADE_TO_SCORE = {
    "A": 1.0,
    "B": 0.5,
    "C": 0.2,
    "D": 0.0,
}

# ============================================================================
# FORMAT SCORING — penalty only, no bonus
# ============================================================================
_match_format_perfect = re.compile(
    rf"^\s*"
    rf"{re.escape(THINKING_START)}.+?{re.escape(THINKING_END)}\s*"
    rf".+\s*\Z",
    flags=re.DOTALL,
)

_extract_after_think = re.compile(
    rf"{re.escape(THINKING_END)}\s*(.+)",
    flags=re.DOTALL,
)


def _extract_answer(response: str) -> str | None:
    """Extract everything after </think> as the answer."""
    m = _extract_after_think.search(response)
    return m.group(1).strip() if m else None


def _compute_format_score(response: str) -> float:
    """
    Format score — penalty only, no bonus.

    Perfect format gets 0.0 (no free reward).
    Broken format gets penalized.
    """
    ts_count = response.count(THINKING_START)
    te_count = response.count(THINKING_END)

    if ts_count == 1 and te_count == 1 and _match_format_perfect.search(response):
        return 0.0  # correct format — no penalty, no bonus

    score = 0.0
    if ts_count == 0:
        score -= 0.25
    elif ts_count > 1:
        score -= 0.5 * (ts_count - 1)
    if te_count == 0:
        score -= 0.25
    elif te_count > 1:
        score -= 0.5 * (te_count - 1)

    return max(score, -0.5)


# ============================================================================
# LLM JUDGE — single holistic call
# ============================================================================
async def _call_judge(question: str, gold: str, answer: str, max_retries: int = 3) -> str:
    """Ask the LLM judge to grade the answer. Returns a single letter grade."""
    if not answer:
        return "D"

    prompt = JUDGE_USER_TEMPLATE.format(question=question, gold=gold, answer=answer)
    client = _get_judge_client()

    for attempt in range(max_retries):
        try:
            resp = await client.responses.create(
                model=JUDGE_MODEL,
                instructions=JUDGE_SYSTEM_PROMPT,
                reasoning={"effort": "low"},
                input=prompt,
            )
            letter = resp.output_text.strip().upper()
            # Extract first valid grade letter
            for ch in letter:
                if ch in GRADE_TO_SCORE:
                    return ch
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[QA REWARD] LLM judge failed: {e}")
                return "D"
            await asyncio.sleep(0.5 * (attempt + 1))
    return "D"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute reward score for medical QA.

    Args:
        data_source: dataset identifier (unused but required by verl API)
        solution_str: the model's full response text
        ground_truth: dict with keys "gold_answer" (str) and "criteria" (list[str])
        extra_info: optional dict with additional metadata

    Returns:
        dict with keys:
            "score": combined reward (float)
            "reward/judge_score": holistic LLM grade (0.0-1.0)
            "reward/judge_grade": letter grade (A/B/C/D/F)
            "reward/format_score": format penalty (-0.5 to 0.0)
    """
    # Parse ground_truth - handle both dict and JSON string
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    gold_answer = ground_truth.get("gold_answer", "")

    # Extract question from extra_info if available
    question = ""
    if extra_info and isinstance(extra_info, dict):
        question = extra_info.get("question", "")

    # Extract answer: everything after </think>
    # If format is broken (no </think>), grade empty string — don't reward
    # the model for leaking thinking into the answer.
    extracted = _extract_answer(solution_str)
    answer_to_grade = extracted if extracted else ""

    # Format score (penalty only)
    format_score = _compute_format_score(solution_str)

    # LLM judge — single holistic call
    if gold_answer:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                grade = pool.submit(asyncio.run, _call_judge(question, gold_answer, answer_to_grade)).result()
        else:
            grade = asyncio.run(_call_judge(question, gold_answer, answer_to_grade))

        judge_score = GRADE_TO_SCORE[grade]
    else:
        grade = "D"
        judge_score = 0.0

    # Length stats for monitoring
    think_match = re.search(
        rf"{re.escape(THINKING_START)}(.+?){re.escape(THINKING_END)}",
        solution_str,
        flags=re.DOTALL,
    )
    reasoning_len = len(think_match.group(1).split()) if think_match else 0
    answer_len = len(answer_to_grade.split()) if answer_to_grade else 0

    scores = {
        "reward/judge_score": judge_score,
        "reward/judge_grade": grade,
        "reward/format_score": format_score,
        "reward/reasoning_length": reasoning_len,
        "reward/answer_length": answer_len,
    }
    scores["score"] = judge_score + format_score
    return scores
