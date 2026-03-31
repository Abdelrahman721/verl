from __future__ import annotations

import json
import re
from typing import Any, Iterable, Set


ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
SCT_ID_PATTERN = re.compile(r"\d+")


def _extract_sct_ids_from_answer(answer_text: str) -> Set[str]:
    """
    Extract candidate SCT concept IDs from inside <answer>...</answer>.

    Heuristic:
    - tokens are contiguous digit spans
    - keep only lengths in a reasonable range for your SCT IDs
    """
    out: Set[str] = set()
    for tok in SCT_ID_PATTERN.findall(answer_text or ""):
        tok = tok.strip()
        if 5 <= len(tok) <= 20:
            out.add(tok)
    return out


def _safe_json_loads(s: str) -> dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}


def _f1_from_sets(pred: Set[str], gold: Set[str]) -> float:
    if not gold:
        # If there's no ground truth, we only reward correct empty prediction.
        return 1.0 if not pred else 0.0

    tp = len(pred & gold)
    if tp == 0:
        return 0.0

    precision = tp / max(1, len(pred))
    recall = tp / max(1, len(gold))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_score(solution_str: str, ground_truth: str, **kwargs: Any) -> float:
    """
    Multilabel reward for SCT tasks (GRPO-friendly):
    - parse ground_truth JSON string: expects gt_sct_ids: List[str]
    - extract predicted SCT IDs from the model text inside <answer>...</answer>
    - return set-F1 between predicted and gold SCT IDs
    """
    gt = _safe_json_loads(ground_truth or "")
    gold_ids: Iterable[str] = gt.get("gt_sct_ids", []) or []
    gold_set: Set[str] = {str(x).strip() for x in gold_ids if str(x).strip()}

    answer_match = ANSWER_TAG_PATTERN.search(solution_str or "")
    if not answer_match:
        return 0.0

    pred_set = _extract_sct_ids_from_answer(answer_match.group(1) or "")
    return float(_f1_from_sets(pred_set, gold_set))

