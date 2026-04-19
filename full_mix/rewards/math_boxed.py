"""Math reward — wraps verl's boxed-answer scorer."""

import logging

from verl.utils.reward_score.math_reward import compute_score as _verl_math_score

logger = logging.getLogger(__name__)


def compute_score(solution_str: str, ground_truth) -> float:
    if not solution_str or not solution_str.strip():
        return 0.0
    try:
        return float(_verl_math_score(solution_str, ground_truth))
    except Exception:
        logger.debug("math_boxed scorer error", exc_info=True)
        return 0.0
