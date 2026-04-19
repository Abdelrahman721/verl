"""Chat reward — LLM-as-judge for general chat responses.

The local chat dataset ships a short `ground_truth` which is a FACTUAL
reference, not a style/format reference. The judge is instructed to use it
only as a factual grounding signal; style/helpfulness is graded on the
response itself. Output score 1-10 is linearly mapped to [0, 1].
"""

import logging

from full_mix.common.judge_client import JudgeUnavailable, call_judge, parse_json_object

logger = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = (
    "You are ChatJudge, a careful evaluator of AI assistant responses.\n\n"
    "You receive a user prompt, a reference answer (which contains factual information "
    "but may be terse or lack helpfulness), and a candidate response to score.\n\n"
    "STEP 1 — Factual correctness: does the candidate's response agree with the facts "
    "expressed in the reference answer? Minor rephrasing is fine; contradictions are not. "
    "Use the reference ONLY for factual grounding, not for style or format.\n\n"
    "STEP 2 — Helpfulness / quality: is the candidate response clear, well-structured, "
    "appropriately detailed, and genuinely useful to the user?\n\n"
    "STEP 3 — Score on a 1-10 scale:\n"
    "  10 = Fully correct, clear, genuinely helpful, well-formatted.\n"
    "   8 = Correct + helpful, minor issues.\n"
    "   6 = Mostly correct but unclear, under-detailed, or slight factual drift.\n"
    "   4 = Significant factual issues or largely unhelpful.\n"
    "   2 = Mostly wrong or incoherent.\n"
    "   1 = Completely wrong, irrelevant, or harmful.\n\n"
    "Return exactly one JSON object:\n"
    '{\"factual_correct\": <boolean>, \"score\": <integer 1-10>, \"reason\": \"<under 25 words>\"}\n\n'
    "Output valid JSON only. No markdown. No extra commentary."
)

JUDGE_USER_TEMPLATE = (
    "[USER_PROMPT]\n{user_prompt}\n[/USER_PROMPT]\n\n"
    "[REFERENCE_ANSWER]\n{reference_answer}\n[/REFERENCE_ANSWER]\n\n"
    "[CANDIDATE_RESPONSE]\n{candidate_response}\n[/CANDIDATE_RESPONSE]"
)


def _reward_from_score(s) -> float:
    try:
        s = int(s)
    except (TypeError, ValueError):
        return 0.0
    s = max(1, min(10, s))
    return (s - 1) / 9.0


def compute_score(solution_str: str, ground_truth, extra_info=None) -> float:
    """Grade a chat response via LLM judge. Returns reward in [0, 1]."""
    if not solution_str or not solution_str.strip():
        return 0.0

    user_prompt = ""
    if extra_info and isinstance(extra_info, dict):
        user_prompt = extra_info.get("user_prompt", "") or ""

    ref = ground_truth if isinstance(ground_truth, str) else str(ground_truth)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                user_prompt=user_prompt,
                reference_answer=ref,
                candidate_response=solution_str,
            ),
        },
    ]

    try:
        raw = call_judge(messages)
    except JudgeUnavailable as e:
        logger.warning("chat judge unavailable; returning 0.0 (%s)", e)
        return 0.0
    except Exception:
        logger.warning("chat judge call raised", exc_info=True)
        return 0.0

    try:
        obj = parse_json_object(raw)
    except ValueError:
        logger.warning("chat judge returned unparseable JSON: %r", raw[:200])
        return 0.0

    return _reward_from_score(obj.get("score"))
