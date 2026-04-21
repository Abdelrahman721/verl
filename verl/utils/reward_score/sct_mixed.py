from __future__ import annotations

import json
from typing import Any


def compute_score(solution_str: str, ground_truth: str, **kwargs: Any) -> float:
    """
    Mixed single-label / multi-label / instruction-following reward dispatcher.

    Reads the "source" field from ground_truth JSON and delegates to the
    appropriate reward function:
      - "single" / "single_label" -> snomed.compute_score  (exact ID / label match)
      - "multilabel" / "multi_label" -> sct_multilabel.compute_score  (set-F1 over IDs)
      - "if" -> sct_if.compute_score  (instruction-following, rule-based)
    """
    try:
        gt = json.loads(ground_truth)
    except Exception:
        return 0.0

    source = gt.get("source", "")
    # `default_compute_score` may forward `extra_info`, but not all reward functions accept it.
    extra_info = kwargs.pop("extra_info", None)

    if source in ("multilabel", "multi_label"):
        from . import sct_multilabel

        return sct_multilabel.compute_score(solution_str, ground_truth, **kwargs)

    if source == "if":
        from . import sct_if

        return sct_if.compute_score(solution_str, ground_truth, extra_info=extra_info, **kwargs)

    from . import snomed
    return snomed.compute_score(solution_str, ground_truth, **kwargs)
