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

# """Chat reward — pairwise LLM-as-judge against a fixed SFT baseline.

# For each prompt, we have pre-generated a baseline response from the SFT base
# model (see generate_baseline.py). At training time the judge compares the
# policy's rollout against that baseline; reward is the policy's preference
# score.

# Why pairwise vs. the previous absolute 1-10 reward:
#   - Reference-free (judge uses its own knowledge for factual correctness).
#   - Anchored to the SFT model the policy is trying not to regress against:
#     "no improvement" -> 0.5, no gradient pull. The policy is only rewarded
#     for actually beating its starting point.
#   - Higher resolution per call: 5 verdict levels -> {0.0, 0.25, 0.5, 0.75, 1.0}
#     rather than the old 1-10 scale that clustered at 7-8 in practice.
#   - Mirrors AlpacaEval 2.0 LC: pairwise wins, no reference, length-aware judge.

# Framework wiring:
#   The training framework's data adapter must populate
#       extra_info["baseline_response"] = row["baseline_response"]
#   alongside the existing extra_info["user_prompt"] mapping. The new column is
#   added to the dataset by generate_baseline.py.
# """

# import logging
# import random
# import re

# from full_mix.common.judge_client import JudgeUnavailable, call_judge, parse_json_object

# logger = logging.getLogger(__name__)


# JUDGE_SYSTEM_PROMPT = (
#     "You are PairwiseChatJudge, a careful evaluator comparing two AI assistant "
#     "responses to the same user prompt.\n\n"
#     "You will see a USER_PROMPT and two candidate responses, RESPONSE_A and "
#     "RESPONSE_B. Decide which response is better, or whether they are tied.\n\n"
#     "EVALUATION CRITERIA (in priority order):\n"
#     "  1. Factual correctness — penalize hallucinations, wrong facts, or "
#     "incorrect reasoning. Use your own knowledge; there is no reference.\n"
#     "  2. Helpfulness — does the response actually address what the user asked, "
#     "with appropriate depth and structure?\n"
#     "  3. Clarity — is it well-organized, easy to follow, free of confusion?\n"
#     "  4. Completeness — does it cover the important parts of the question "
#     "without being padded or repetitive?\n\n"
#     "BIAS CONTROLS (read carefully):\n"
#     "  - Position: A and B are randomized. Do not favor either position.\n"
#     "  - Length: longer is not better. Prefer the response that is appropriately "
#     "detailed for the question. Penalize padding, repetition, or filler.\n"
#     "  - Style: do not prefer markdown formatting, bullet lists, or headers for "
#     "their own sake — only when they aid clarity for this specific question.\n"
#     "  - Confidence: confident-sounding writing is not better if it is wrong.\n\n"
#     "VERDICTS:\n"
#     "  A_much_better      — A is clearly and substantially better.\n"
#     "  A_slightly_better  — A is somewhat better; the gap is small.\n"
#     "  tie                — Roughly equivalent in overall quality.\n"
#     "  B_slightly_better  — B is somewhat better; the gap is small.\n"
#     "  B_much_better      — B is clearly and substantially better.\n\n"
#     "Return exactly one JSON object:\n"
#     '{"verdict": "<one of the five labels above>", "reason": "<under 25 words>"}\n\n'
#     "Output valid JSON only. No markdown. No extra commentary."
# )

# JUDGE_USER_TEMPLATE = (
#     "[USER_PROMPT]\n{user_prompt}\n[/USER_PROMPT]\n\n"
#     "[RESPONSE_A]\n{response_a}\n[/RESPONSE_A]\n\n"
#     "[RESPONSE_B]\n{response_b}\n[/RESPONSE_B]"
# )


# # Verdict -> reward from A's perspective. We flip if policy is in B.
# _A_PERSPECTIVE_REWARD = {
#     "A_much_better": 1.0,
#     "A_slightly_better": 0.75,
#     "tie": 0.5,
#     "B_slightly_better": 0.25,
#     "B_much_better": 0.0,
# }

# _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# def _strip_thinking(text: str) -> str:
#     """Remove <think>...</think> blocks, then strip surrounding whitespace.

#     Handles the common case where the model emits <think>...</think>FINAL and
#     also the case where the closing </think> appears without an opening tag
#     (some chat templates inject the opening tag in the prompt).
#     """
#     if "</think>" in text and "<think>" not in text:
#         return text.split("</think>", 1)[1].strip()
#     return _THINK_RE.sub("", text).strip()


# def _reward_from_verdict(verdict: str, policy_is_a: bool) -> float:
#     if verdict not in _A_PERSPECTIVE_REWARD:
#         return 0.5
#     a_reward = _A_PERSPECTIVE_REWARD[verdict]
#     return a_reward if policy_is_a else 1.0 - a_reward


# def compute_score(solution_str: str, ground_truth, extra_info=None) -> float:
#     """Pairwise judge reward in [0, 1]. ground_truth is unused (reference-free)."""
#     del ground_truth  # reference-free judging

#     if not solution_str or not solution_str.strip():
#         return 0.0

#     if not extra_info or not isinstance(extra_info, dict):
#         logger.warning("pairwise chat reward: missing extra_info; returning 0.5")
#         return 0.5

#     user_prompt = (extra_info.get("user_prompt") or "").strip()
#     baseline = (extra_info.get("baseline_response") or "").strip()

#     if not user_prompt or not baseline:
#         logger.warning(
#             "pairwise chat reward: missing user_prompt or baseline_response; "
#             "returning 0.5 (user_prompt=%s, baseline=%s)",
#             bool(user_prompt),
#             bool(baseline),
#         )
#         return 0.5

#     policy_response = _strip_thinking(solution_str)
#     if not policy_response:
#         return 0.0

#     # Random A/B swap to mitigate position bias. Over a GRPO group, the policy
#     # appears in each position in expectation; bias cancels in the advantage.
#     policy_is_a = random.random() < 0.5
#     if policy_is_a:
#         response_a, response_b = policy_response, baseline
#     else:
#         response_a, response_b = baseline, policy_response

#     messages = [
#         {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
#         {
#             "role": "user",
#             "content": JUDGE_USER_TEMPLATE.format(
#                 user_prompt=user_prompt,
#                 response_a=response_a,
#                 response_b=response_b,
#             ),
#         },
#     ]

#     try:
#         raw = call_judge(messages)
#     except JudgeUnavailable as e:
#         logger.warning("pairwise chat judge unavailable; returning 0.5 (%s)", e)
#         return 0.5
#     except Exception:
#         logger.warning("pairwise chat judge call raised", exc_info=True)
#         return 0.5

#     try:
#         obj = parse_json_object(raw)
#     except ValueError:
#         logger.warning(
#             "pairwise chat judge returned unparseable JSON: %r", raw[:200]
#         )
#         return 0.5

#     verdict = obj.get("verdict")
#     if not isinstance(verdict, str) or verdict not in _A_PERSPECTIVE_REWARD:
#         logger.warning("pairwise chat judge returned bad verdict: %r", verdict)
#         return 0.5

#     return _reward_from_verdict(verdict, policy_is_a)
