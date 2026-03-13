"""Custom reward function for instruction-following (IFEval) RLVR training.

Evaluates model responses against instruction-following constraints defined
in the allenai/IF_multi_constraints_upto5 dataset, using verification logic
ported from allenai/open-instruct IFEvalG.
"""

import ast
import logging
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

from rl_scripts.ifeval.instructions_registry import FUNCTION_DICT

logger = logging.getLogger(__name__)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Compute a fractional reward based on how many IF constraints are satisfied.

    Args:
        data_source: Dataset identifier string (unused, for API compatibility).
        solution_str: The model's response string to evaluate.
        ground_truth: A string encoding a list of constraint dicts, each with
            'instruction_id' (list of str) and 'kwargs' (list of dict/None).
            Parsed via ast.literal_eval.
        extra_info: Optional dict with additional metadata (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        float: Fraction of constraints satisfied, in [0.0, 1.0].
    """
    if not solution_str or not solution_str.strip():
        return 0.0

    try:
        if isinstance(ground_truth, str):
            constraints = ast.literal_eval(ground_truth)
        else:
            constraints = ground_truth
    except (ValueError, SyntaxError):
        logger.warning("Failed to parse ground_truth: %s", ground_truth[:200])
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

    if total == 0:
        return 0.0

    return passed / total
