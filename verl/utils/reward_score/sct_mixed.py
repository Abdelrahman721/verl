from __future__ import annotations

import json
from typing import Any


def compute_score(solution_str: str, ground_truth: str, **kwargs: Any) -> float:
    """
    Mixed single-label / multi-label reward dispatcher.

    Reads the "source" field from ground_truth JSON and delegates to the
    appropriate reward function:
      - "single_label" -> snomed.compute_score  (exact ID / label match)
      - "multi_label"  -> sct_multilabel.compute_score  (set-F1 over IDs)
    """
    try:
        gt = json.loads(ground_truth)
    except Exception:
        return 0.0

    source = gt.get("source", "")

    if source == "multi_label":
        from . import sct_multilabel
        return sct_multilabel.compute_score(solution_str, ground_truth, **kwargs)

    from . import snomed
    return snomed.compute_score(solution_str, ground_truth, **kwargs)
