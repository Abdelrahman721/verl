from __future__ import annotations

import json
import random


def compute_score(
    solution_str: str,
    ground_truth: str,
    format_score: float = 0.5,
    score: float = 1.0,
) -> float:
    """
    SNOMED reward:
    - If exact sct_id string match found anywhere in solution_str -> 1.0
    - Else if exact label string match found (case insensitive) -> 0.5
    - Else -> 0.0
    """
    # Parse ground truth JSON produced by snomed_preprocess.py.
    # This must always be valid; if not, crash loudly instead of hiding the issue.
    gt = json.loads(ground_truth)

    target_sctid = str(gt["sct_id"]).strip()
    target_label = str(gt["label"]).strip()

    # Check for exact SCTID match anywhere in solution_str
    if target_sctid in solution_str:
        return 1.0

    # Check for exact label match (case insensitive) anywhere in solution_str
    if target_label.lower() in solution_str.lower():
        return 0.5

    # No match found
    return 0.0


# def compute_score(
#     solution_str: str,
#     ground_truth: str,
#     format_score: float = 0.5,
#     score: float = 1.0,
# ) -> float:
#     """
#     Dummy reward function that randomly returns either 0 or 1.
#     """
#     print("Dumdum")
#     return float(random.choice([0, 1]))
