from __future__ import annotations

import json
import re
import random

THINK_CLOSE_PATTERN = re.compile(r"</think>", re.IGNORECASE)
SCT_ID_PATTERN = re.compile(r"\d+")


def _extract_sct_ids(answer_text: str) -> list[str]:
    """Extract candidate SCT concept IDs (contiguous digit spans) from the answer."""
    return [
        tok
        for tok in SCT_ID_PATTERN.findall(answer_text or "")
        if 6 <= len(tok) <= 18
    ]


def compute_score(
    solution_str: str,
    ground_truth: str,
    format_score: float = 0.5,
    score: float = 1.0,
) -> float:
    """
    SNOMED reward (evaluates only the answer after </think>):
    - model output format is "<think> reasoning </think> answer"; the answer is
      everything after the closing </think> tag
    - extract all SNOMED codes from the answer; if more than one or none are
      found -> 0.0
    - otherwise, if the single extracted id matches the ground truth -> 1.0
    - else if the ground truth label is found in the answer (case insensitive)
      -> 0.5, else 0.0
    """
    # Parse ground truth JSON produced by snomed_preprocess.py.
    # This must always be valid; if not, crash loudly instead of hiding the issue.
    gt = json.loads(ground_truth)

    target_sctid = str(gt["sct_id"]).strip()
    target_label = str(gt["label"]).strip()

    # Restrict scoring to the answer that follows the closing </think> tag.
    think_match = THINK_CLOSE_PATTERN.search(solution_str or "")
    if not think_match:
        return 0.0
    answer_text = (solution_str or "")[think_match.end():].strip()

    # Extract all SNOMED codes; only score when exactly one is present.
    extracted_ids = _extract_sct_ids(answer_text)
    if len(extracted_ids) != 1:
        return 0.0

    # Exact id match.
    if extracted_ids[0] == target_sctid:
        return 1.0

    # Wrong id, but the correct label appears in the answer (case insensitive).
    if target_label.lower() in answer_text.lower():
        return 0.5

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
