from __future__ import annotations

import json
from typing import Any


def _safe_json_loads(s: str) -> dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}


def compute_score(solution_str: str, ground_truth: str, **kwargs: Any) -> float:
    """
    Mixed SNOMED reward: ground_truth JSON must include "source":
      - "multilabel" -> sct_multilabel (set-F1 on gt_sct_ids)
      - "single"     -> snomed (exact sct_id / label in <answer>)
    """
    gt = _safe_json_loads(ground_truth or "")
    src = gt.get("source")

    if src == "multilabel":
        from . import sct_multilabel

        return sct_multilabel.compute_score(solution_str, ground_truth)
    if src == "single":
        from . import snomed

        return snomed.compute_score(solution_str, ground_truth)

    raise ValueError(
        'snomed_mixed reward: ground_truth must include "source": "single" or "multilabel"; '
        f"got {src!r}"
    )
