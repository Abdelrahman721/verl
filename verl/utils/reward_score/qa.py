"""
Reward scoring for medical QA using LLM-as-a-judge (OpenAI).

The dataset stores ground_truth as a dict:
    {"gold_answer": str, "criteria": list[str]}

The compute_score function is the single entry point for verl's reward manager.
It returns a dict with individual score components and a combined "score" key.

Expected model format: <think>reasoning</think> then the answer directly after.

Reward structure:
    - accuracy_score: 0.0 to 1.0 (LLM judge average over criteria)
    - format_score: -0.5 to 0.5 (<think>...</think> tag compliance)
    - score: combined total (the value verl uses for RL)
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
    "You are an expert medical evaluator. Score how well an answer meets "
    "each criterion compared to a gold standard.\n"
    "For each criterion, score 1-5:\n"
    "1 = Missing/wrong\n"
    "2 = Partially addressed with errors\n"
    "3 = Addressed but incomplete\n"
    "4 = Well addressed\n"
    "5 = Fully and accurately addressed\n"
    "Respond with ONLY comma-separated integers, one per criterion. "
    "Nothing else. Example for 3 criteria: 4,5,3"
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
# FORMAT SCORING — only <think>...</think> tags
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
    Format score based on <think>...</think> usage:
      +0.5 for perfect format: <think>...</think> followed by answer text
      -0.25 per missing <think> or </think> tag
      -0.5 per duplicate tag
    Capped at -0.5.
    """
    ts_count = response.count(THINKING_START)
    te_count = response.count(THINKING_END)

    if ts_count == 1 and te_count == 1 and _match_format_perfect.search(response):
        return 0.5

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
# LLM GRADING
# ============================================================================
def _parse_scores(text: str, expected: int) -> list[int]:
    if not text:
        return []
    numbers = re.findall(r"\b([1-5])\b", text.strip())
    if len(numbers) >= expected:
        return [int(n) for n in numbers[:expected]]
    return []


async def _call_grader(
    question: str, gold: str, answer: str, criteria: list[str], max_retries: int = 3
) -> list[int]:
    if not answer or not criteria:
        return []
    
    criteria += ["The answer does not contain hallucinations, is easy, concise, and well written"]
    criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))
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
                reasoning={"effort": "low"},
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
    """Map average 1-5 score to 0.0-1.0."""
    if not scores or num_criteria == 0:
        return 0.0
    avg = sum(scores) / num_criteria  # 1.0 to 5.0
    return (avg - 1) / 4  # 0.0 to 1.0



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
            "reward/accuracy_score": LLM judge accuracy (0.0-1.0)
            "reward/format_score": format compliance (-0.5 to 0.5)
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
    extracted = _extract_answer(solution_str)
    answer_to_grade = extracted if extracted else solution_str

    # Format score
    format_score = _compute_format_score(solution_str)

    # LLM accuracy score (run async in sync context)
    if gold_answer and criteria:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an existing event loop (verl's async reward manager)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                llm_scores = pool.submit(
                    asyncio.run,
                    _call_grader(question, gold_answer, answer_to_grade, criteria),
                ).result()
        else:
            llm_scores = asyncio.run(
                _call_grader(question, gold_answer, answer_to_grade, criteria)
            )
        accuracy_score = _accuracy_from_scores(llm_scores, len(criteria))
    else:
        accuracy_score = 0.0

    scores = {
        "reward/accuracy_score": accuracy_score,
        "reward/format_score": format_score,
    }
    scores["score"] = sum(scores.values())
    return scores
