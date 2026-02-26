"""
Reward scoring for medical QA using LLM-as-a-judge (OpenAI).

The dataset stores ground_truth as a dict:
    {"gold_answer": str, "criteria": list[str]}

The compute_score function is the single entry point for verl's reward manager.
It returns a dict with individual score components and a combined "score" key.

Expected model format: <think>reasoning</think> then the answer directly after.

Reward structure (multiplicative — penalties scale with accuracy):
    - accuracy_score: 0.0 to 1.0 (binary per-criterion recall, fraction met)
    - hallucination_penalty: -0.5 to 0.0 (precision, penalizes incorrect claims)
    - format_score: -0.5 to 0.0 (penalty only, no bonus for correct format)
    - brevity_bonus: 0.0 to 0.1 (small bonus for concise answers near gold length)
    - score: accuracy * max(0.1, 1 + hallucination + brevity) + format
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
# LLM GRADER CONFIG
# ============================================================================
_GPT_CLIENT = None


def _get_grader_client():
    global _GPT_CLIENT
    if _GPT_CLIENT is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        _GPT_CLIENT = AsyncOpenAI(api_key=api_key)
    return _GPT_CLIENT


GRADING_MODEL = os.environ.get("QA_GRADING_MODEL", "gpt-5.1")

GRADING_SYSTEM_PROMPT = (
    "You are an expert medical evaluator. For each criterion below, "
    "determine whether the candidate answer contains the specific "
    "information described, compared to the gold standard.\n"
    "For each criterion, respond with 0 or 1:\n"
    "1 = The answer explicitly and correctly states this specific fact or concept\n"
    "0 = The answer does not state this, states it incorrectly, or only vaguely alludes to it\n"
    "Be strict. Apply these rules:\n"
    "- Vaguely related content without the specific fact is 0.\n"
    "- Hedging (e.g. 'it may be X or Y', 'possibly', 'could be') without "
    "committing to the correct specific answer is 0.\n"
    "- Burying the correct fact inside excessive irrelevant text does not count — "
    "the fact must be clearly stated, not hidden in a wall of text.\n"
    "- Only clear, correct, committed statements count as 1.\n"
    "Respond with ONLY comma-separated 0 or 1 values, one per criterion. "
    "Nothing else. Example for 3 criteria: 1,0,1"
)

GRADING_USER_TEMPLATE = """
QUESTION:
{question}

GOLD ANSWER:
{gold}

CANDIDATE ANSWER:
{answer}

CRITERIA:
{criteria}

Scores (comma-separated, one per criterion):"""

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
# BREVITY SCORING — reward conciseness, penalize bloat
# ============================================================================
def _compute_brevity_score(answer_text: str, gold_answer: str) -> float:
    """
    Brevity score that rewards concise answers and penalizes verbose ones.

    This is used multiplicatively with accuracy, so verbose answers
    get their accuracy scaled down — you can't brute-force criteria hits
    by dumping text.

    Returns:
        +0.1  if answer is within 0.5x-1.5x gold length (concise bonus)
         0.0  at 2x gold length
        -0.2  per 1x excess beyond 2x, uncapped
    """
    if not answer_text or not gold_answer:
        return 0.0

    answer_words = len(answer_text.split())
    gold_words = max(len(gold_answer.split()), 1)

    ratio = answer_words / gold_words

    if 0.5 <= ratio <= 1.5:
        return 0.1  # concise bonus
    elif ratio <= 2.0:
        return 0.0  # neutral zone
    else:
        excess = ratio - 2.0
        return -0.2 * excess  # no cap — 10x answer gets -1.6


# ============================================================================
# LLM GRADING — binary per-criterion
# ============================================================================
def _parse_scores(text: str, expected: int) -> list[int]:
    """Parse comma-separated binary scores (0 or 1) from grader response."""
    if not text:
        return []
    numbers = re.findall(r"\b([01])\b", text.strip())
    if len(numbers) >= expected:
        return [int(n) for n in numbers[:expected]]
    return []


async def _call_grader(
    question: str, gold: str, answer: str, criteria: list[str], max_retries: int = 3
) -> list[int]:
    if not answer or not criteria:
        return []

    criteria_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))
    prompt = GRADING_USER_TEMPLATE.format(
        question=question, gold=gold, answer=answer, criteria=criteria_text
    )
    client = _get_grader_client()

    for attempt in range(max_retries):
        try:
            resp = await client.responses.create(
                model=GRADING_MODEL,
                instructions=GRADING_SYSTEM_PROMPT,
                input=prompt,
                max_output_tokens=2048,
            )
            scores = _parse_scores(resp.output_text, len(criteria))
            if scores:
                return scores
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[QA REWARD] GPT grading failed: {e}")
                return []
            await asyncio.sleep(0.5 * (attempt + 1))
    return []


def _accuracy_from_scores(scores: list[int], num_criteria: int) -> float:
    """Fraction of criteria met: 0.0 to 1.0."""
    if not scores or num_criteria == 0:
        return 0.0
    return sum(scores) / num_criteria


# ============================================================================
# HALLUCINATION CHECK — precision penalty
# ============================================================================
HALLUCINATION_SYSTEM_PROMPT = (
    "You are an expert medical fact-checker. Your job is to count factually "
    "incorrect medical claims in the candidate answer.\n"
    "Compare the candidate answer against the gold standard answer and "
    "established medical knowledge.\n"
    "Count ONLY clear factual errors — wrong drug names, wrong mechanisms, "
    "wrong anatomy, incorrect numbers/thresholds, reversed relationships, "
    "or fabricated claims. Do NOT count:\n"
    "- Omissions (missing information is not a hallucination)\n"
    "- Stylistic issues or vague language\n"
    "- Minor imprecisions that don't change clinical meaning\n"
    "Respond with ONLY a single integer: the number of factual errors found. "
    "If there are no errors, respond with 0."
)

HALLUCINATION_USER_TEMPLATE = """
QUESTION:
{question}

GOLD ANSWER:
{gold}

CANDIDATE ANSWER:
{answer}

Number of factual errors:"""


async def _call_hallucination_check(
    question: str, gold: str, answer: str, max_retries: int = 3
) -> int:
    """Count factual errors in the answer. Returns an integer >= 0."""
    if not answer:
        return 0

    prompt = HALLUCINATION_USER_TEMPLATE.format(
        question=question, gold=gold, answer=answer
    )
    client = _get_grader_client()

    for attempt in range(max_retries):
        try:
            resp = await client.responses.create(
                model=GRADING_MODEL,
                instructions=HALLUCINATION_SYSTEM_PROMPT,
                input=prompt,
                max_output_tokens=2048,
            )
            numbers = re.findall(r"\b(\d+)\b", resp.output_text.strip())
            if numbers:
                return int(numbers[0])
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[QA REWARD] Hallucination check failed: {e}")
                return 0
            await asyncio.sleep(0.5 * (attempt + 1))
    return 0


def _compute_hallucination_penalty(num_errors: int) -> float:
    """
    Convert hallucination count to a penalty.

    -0.1 per factual error, capped at -0.5.
    0 errors = 0.0 (no penalty).
    """
    if num_errors <= 0:
        return 0.0
    return max(-0.1 * num_errors, -0.5)


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
            "reward/accuracy_score": binary recall (0.0-1.0)
            "reward/hallucination_penalty": precision penalty (-0.5 to 0.0)
            "reward/format_score": format penalty (-0.5 to 0.0)
            "reward/brevity_score": brevity bonus/penalty (+0.1 to uncapped negative)
    """
    # Parse ground_truth - handle both dict and JSON string
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    gold_answer = ground_truth.get("gold_answer", "")
    criteria = ground_truth.get("criteria", [])

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

    # Brevity score (on answer only, relative to gold)
    brevity_score = _compute_brevity_score(answer_to_grade, gold_answer)

    # Run both LLM calls concurrently: accuracy (recall) + hallucination (precision)
    async def _run_grading():
        acc_task = _call_grader(question, gold_answer, answer_to_grade, criteria)
        hal_task = _call_hallucination_check(question, gold_answer, answer_to_grade)
        return await asyncio.gather(acc_task, hal_task)

    if gold_answer and criteria:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                llm_scores, num_hallucinations = pool.submit(
                    asyncio.run, _run_grading()
                ).result()
        else:
            llm_scores, num_hallucinations = asyncio.run(_run_grading())

        accuracy_score = _accuracy_from_scores(llm_scores, len(criteria))
        hallucination_penalty = _compute_hallucination_penalty(num_hallucinations)
    else:
        accuracy_score = 0.0
        hallucination_penalty = 0.0

    scores = {
        "reward/accuracy_score": accuracy_score,
        "reward/hallucination_penalty": hallucination_penalty,
        "reward/format_score": format_score,
        "reward/brevity_score": brevity_score,
    }
    # Multiplicative scoring: penalties scale WITH accuracy.
    # A verbose/hallucinating answer can't score high just by hitting criteria.
    # score = accuracy * max(0.1, 1 + hallucination + brevity) + format
    quality_multiplier = max(0.1, 1.0 + hallucination_penalty + brevity_score)
    scores["score"] = accuracy_score * quality_multiplier + format_score
    return scores