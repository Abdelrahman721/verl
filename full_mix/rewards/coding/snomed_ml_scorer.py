"""SNOMED CT multilabel scorer.

Verbatim port of the parsing / scoring LOGIC from hazem's
`snomed_ml_reward.py` (no behaviour change), reshaped to the verl-compatible
scorer signature used by the rest of the coding suite.

The original takes `(solution_str, ground_truth)` and returns a sparse dict
with `score`, `reward/accuracy_score`, `reward/precision`, `reward/recall`.
This shim:
  - accepts the full verl scorer signature
  - extracts gold from `ground_truth.gt_sct_ids` (already parsed dict from
    the dispatcher's JSON pre-pass)
  - extracts predictions from the post-`</think>` answer with the same
    6-18 digit heuristic
  - applies format_penalty and pads to the coding union via `_empty_score_dict`
"""

from __future__ import annotations

import re
from typing import Iterable

from .common import _empty_score_dict

# Verbatim from snomed_ml_reward.py
_SCT_ID_PATTERN = re.compile(r"\d+")


def _extract_sct_ids_from_answer(answer_text: str) -> set[str]:
    """6-18 digit contiguous spans, same as snomed_ml_reward.py."""
    out: set[str] = set()
    for tok in _SCT_ID_PATTERN.findall(answer_text or ""):
        tok = tok.strip()
        if 6 <= len(tok) <= 18:
            out.add(tok)
    return out


def _prf_from_sets(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    """Verbatim from snomed_ml_reward.py — set P/R/F1 with empty-gold edge case."""
    if not gold:
        val = 1.0 if not pred else 0.0
        return val, val, val
    tp = len(pred & gold)
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / max(1, len(pred))
    recall = tp / max(1, len(gold))
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def score(eval_mode: str,
          solution_str: str,
          ground_truth,
          format_penalty: float,
          reasoning_len: int,
          answer_len: int,
          answer_to_grade: str) -> dict:
    """Verl-shape SNOMED multilabel scorer.

    `ground_truth` is a dict (parsed upstream) with `gt_sct_ids: list[str]`.
    """
    out = _empty_score_dict(eval_mode, format_penalty, reasoning_len, answer_len)

    gold_ids: Iterable = (ground_truth or {}).get("gt_sct_ids", []) or []
    gold_set: set[str] = {str(x).strip() for x in gold_ids if str(x).strip()}

    # The original logic scored 0 when there was no </think> tag. The verl
    # dispatcher computes `answer_to_grade` for us — it'll be "" if there's
    # no </think>, which yields an empty pred set → score 0 (same effect).
    pred_set = _extract_sct_ids_from_answer(answer_to_grade or "")
    precision, recall, f1 = _prf_from_sets(pred_set, gold_set)

    out["reward/raw_score"]      = f1
    out["reward/judge_score"]    = f1   # telemetry alias
    out["reward/precision"]      = precision
    out["reward/recall"]         = recall
    out["reward/f1"]             = f1
    out["reward/accuracy"]       = f1 * 10.0
    # Keep the snomed_ml_reward.py-original `reward/accuracy_score` alias
    # populated so any downstream dashboards keyed on that name still work.
    out["reward/accuracy_score"] = f1
    out["score"] = max(0.0, f1 + format_penalty)
    return out
