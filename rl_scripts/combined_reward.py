"""Combined reward function for multi-domain GRPO training.

Dispatches to domain-specific scorers based on `data_source`:
  - Math domains  → boxed-format extraction via verl's math_reward
  - GSM8K         → #### extraction via verl's gsm8k reward
  - IF domains    → constraint checking via ifeval_reward
  - General chat  → LM-as-judge comparing response to reference answer
"""

import ast
import logging
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if os.path.dirname(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

logger = logging.getLogger(__name__)

MATH_BOXED_SOURCES = {
    "allenai/Dolci-RL-Zero-Math-7B",
    "HuggingFaceH4/MATH-500",
    "DigitalLearningGmbH/MATH-lighteval",
    "lighteval/MATH",
}

GSM8K_SOURCES = {
    "openai/gsm8k",
}

IF_SOURCES = {
    "allenai/Dolci-RL-Zero-IF-7B",
    "google/IFEval",
    "allenai/IFBench_test",
    "allenai/IF_multi_constraints_upto5",
}

GENERAL_SOURCES = {
    "allenai/Dolci-RL-Zero-General-7B",
}


def _math_boxed_score(solution_str, ground_truth):
    from verl.utils.reward_score.math_reward import compute_score as math_compute_score
    return float(math_compute_score(solution_str, ground_truth))


def _gsm8k_score(solution_str, ground_truth):
    from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score
    return float(gsm8k_compute_score(solution_str, ground_truth))


def _if_score(data_source, solution_str, ground_truth, extra_info=None):
    from rl_scripts.ifeval.instructions_registry import FUNCTION_DICT

    if not solution_str or not solution_str.strip():
        return 0.0

    try:
        if isinstance(ground_truth, str):
            constraints = ast.literal_eval(ground_truth)
        else:
            constraints = ground_truth
    except (ValueError, SyntaxError):
        logger.warning("Failed to parse IF ground_truth: %s", str(ground_truth)[:200])
        return 0.0

    if not constraints:
        return 0.0

    total = 0
    passed = 0

    for constraint_group in constraints:
        instruction_ids = constraint_group.get("instruction_id", [])
        kwargs_list = constraint_group.get("kwargs", [])

        if not isinstance(instruction_ids, list):
            instruction_ids = [instruction_ids]
        if not isinstance(kwargs_list, list):
            kwargs_list = [kwargs_list]

        while len(kwargs_list) < len(instruction_ids):
            kwargs_list.append(None)

        for instr_id, kw in zip(instruction_ids, kwargs_list):
            total += 1
            try:
                checker_cls = FUNCTION_DICT.get(instr_id)
                if checker_cls is None:
                    logger.warning("Unknown instruction_id: %s", instr_id)
                    continue
                checker = checker_cls(instr_id)
                if kw is None:
                    kw = {}
                checker.build_description(**kw)
                if checker.check_following(solution_str):
                    passed += 1
            except Exception:
                logger.debug("Error evaluating constraint %s", instr_id, exc_info=True)

    return passed / total if total > 0 else 0.0


def _lm_judge_score(solution_str, ground_truth, extra_info=None):
    """Score a general-chat response using an LM judge.

    Requires environment variables:
      LM_JUDGE_API_BASE  – OpenAI-compatible base URL (e.g. http://localhost:8000/v1)
      LM_JUDGE_MODEL     – model name served at that endpoint
      LM_JUDGE_API_KEY   – API key (default: "EMPTY")

    Falls back to 0.0 if the judge is unreachable.
    """
    api_base = os.environ.get("LM_JUDGE_API_BASE")
    model = os.environ.get("LM_JUDGE_MODEL")
    api_key = os.environ.get("LM_JUDGE_API_KEY", "EMPTY")

    if not api_base or not model:
        logger.warning(
            "LM_JUDGE_API_BASE / LM_JUDGE_MODEL not set; returning 0.0 for general-chat reward"
        )
        return 0.0

    if not solution_str or not solution_str.strip():
        return 0.0

    ref_answer = ground_truth if isinstance(ground_truth, str) else str(ground_truth)

    judge_prompt = (
        "You are an impartial judge evaluating the quality of an AI assistant's response.\n\n"
        "## Reference Answer\n"
        f"{ref_answer}\n\n"
        "## Assistant's Response\n"
        f"{solution_str}\n\n"
        "## Instructions\n"
        "Compare the assistant's response to the reference answer. "
        "Rate the response on a scale from 0 to 10, where:\n"
        "  0 = completely wrong, irrelevant, or harmful\n"
        "  5 = partially correct but missing key information\n"
        "  10 = fully correct, comprehensive, and well-written\n\n"
        "Output ONLY a single integer score between 0 and 10, nothing else."
    )

    try:
        import openai
        client = openai.OpenAI(base_url=api_base, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=8,
            temperature=0.0,
        )
        score_text = response.choices[0].message.content.strip()
        score = int(score_text)
        return max(0.0, min(1.0, score / 10.0))
    except Exception:
        logger.warning("LM judge call failed; returning 0.0", exc_info=True)
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Unified reward function dispatching by data_source."""
    if data_source in MATH_BOXED_SOURCES:
        return _math_boxed_score(solution_str, ground_truth)
    elif data_source in GSM8K_SOURCES:
        return _gsm8k_score(solution_str, ground_truth)
    elif data_source in IF_SOURCES:
        return _if_score(data_source, solution_str, ground_truth, extra_info)
    elif data_source in GENERAL_SOURCES:
        return _lm_judge_score(solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError(
            f"Combined reward: no handler for data_source={data_source!r}"
        )
