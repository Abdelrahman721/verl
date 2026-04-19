"""GSM8K reward — wraps verl's #### extractor."""

import logging

from verl.utils.reward_score.gsm8k import compute_score as _verl_gsm8k_score

logger = logging.getLogger(__name__)


def compute_score(solution_str: str, ground_truth) -> float:
    if not solution_str or not solution_str.strip():
        return 0.0
    try:
        return float(_verl_gsm8k_score(solution_str, ground_truth, method="strict"))
    except Exception:
        logger.debug("gsm8k scorer error", exc_info=True)
        return 0.0
