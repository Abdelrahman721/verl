"""ICD-10 multilabel scorer (used by both `icd10_multilabel` and
`icd10_instruction_follow` eval_modes — per the user's spec, both ICD
variants score identically).

Pure deterministic: regex-extract ICD-style codes from the candidate, regex-
extract from the ground-truth string (which is a JSON-string of the gold code
list after `medical_qa_compat.transform_table`), then compute set-based
precision / recall / F1.

This is a self-contained port of the icd_multi_label logic that lives in
`qa_openrouter_bench.py` — copied here (not imported) so the coding package
has no cross-domain dependencies.
"""

from __future__ import annotations

import re

from .common import _empty_score_dict

# `\b[A-Z][A-Z0-9]{2}(?:\.[A-Z0-9]{1,4})?\b` matches ICD-10 codes like
# `A01.00`, `Z79.84`, `E11.9`. The wider character class than `\d` is
# intentional — ICD-10-PCS allows letters in subsequent positions.
_ICD_CODE_REGEX = re.compile(r"\b[A-Z][A-Z0-9]{2}(?:\.[A-Z0-9]{1,4})?\b")


def _extract_codes_from_answer(model_answer: str) -> set[str]:
    """Pull all ICD-10-shaped tokens from the model's text."""
    if not model_answer:
        return set()
    # Look inside <answer>…</answer> if present, otherwise everything after
    # </think>, otherwise the raw string. Matches the pattern from icd.py.
    if "<answer>" in model_answer:
        scope = model_answer.split("<answer>", 1)[1]
    elif "</think>" in model_answer:
        scope = model_answer.split("</think>", 1)[1]
    else:
        scope = model_answer
    return set(_ICD_CODE_REGEX.findall(scope))


def _extract_codes_from_gold(ground_truth) -> set[str]:
    """Pull all ICD-10-shaped tokens from the ground_truth string.

    The build pipeline JSON-serialises `list<string>` and `list<list<string>>`
    ICD ground_truths, so `str(ground_truth)` yields something like
    `'["E11.9","Z79.84"]'` or `'[["E11.9"]]'` — the regex finds the codes
    inside regardless of the bracket nesting.
    """
    return set(_ICD_CODE_REGEX.findall(str(ground_truth or "")))


def score(eval_mode: str,
          solution_str: str,
          ground_truth,
          format_penalty: float,
          reasoning_len: int,
          answer_len: int,
          answer_to_grade: str) -> dict:
    """Verl-shape ICD multilabel scorer.

    `eval_mode` is one of `icd10_multilabel` / `icd10_instruction_follow`
    (the dispatcher passes whatever was on the row so it shows up in the
    reward dict / dump). Both score identically per user spec.
    """
    out = _empty_score_dict(eval_mode, format_penalty, reasoning_len, answer_len)

    pred = _extract_codes_from_answer(answer_to_grade or solution_str or "")
    gold = _extract_codes_from_gold(ground_truth)

    if not pred:
        # No codes predicted → P=R=F1=0 (matches the icd.py contract).
        precision = recall = f1 = 0.0
    else:
        tp = len(pred & gold)
        precision = tp / len(pred) if pred else 0.0
        recall    = tp / len(gold) if gold else 0.0
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    out["reward/raw_score"]   = f1
    out["reward/judge_score"] = f1   # telemetry alias
    out["reward/precision"]   = precision
    out["reward/recall"]      = recall
    out["reward/f1"]          = f1
    out["reward/accuracy"]    = f1 * 10.0
    out["score"] = max(0.0, f1 + format_penalty)
    return out
