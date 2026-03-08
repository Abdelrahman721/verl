from __future__ import annotations

import json
import re
import random

ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def compute_score(
    solution_str: str,
    ground_truth: str,
    format_score: float = 0.5,
    score: float = 1.0,
) -> float:
    """
    SNOMED reward (evaluates only text inside <answer>...</answer>):
    - If exact sct_id string match found in answer tag content -> 1.0
    - Else if exact label string match found in answer tag (case insensitive) -> 0.5
    - Else or if no <answer> tag present -> 0.0
    """
    # Parse ground truth JSON produced by snomed_preprocess.py.
    # This must always be valid; if not, crash loudly instead of hiding the issue.
    gt = json.loads(ground_truth)

    target_sctid = str(gt["sct_id"]).strip()
    target_label = str(gt["label"]).strip()

    # Restrict scoring to content inside <answer>...</answer> only
    answer_match = ANSWER_TAG_PATTERN.search(solution_str or "")
    if not answer_match:
        return 0.0
    answer_text = (answer_match.group(1) or "").strip()

    # Check for exact SCTID match in answer tag only
    if target_sctid in answer_text:
        return 1.0

    # Check for exact label match (case insensitive) in answer tag only
    if target_label.lower() in answer_text.lower():
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
