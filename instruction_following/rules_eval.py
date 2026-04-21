"""Per-rule evaluation: instruction (format) vs accuracy (semantic)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .judges import (
    JUDGE_V08_SYSTEM,
    JUDGE_V11_SYSTEM,
    JUDGE_V12_SYSTEM,
    call_judge,
    call_v09_pred_extraction,
    judge_v08_user,
    judge_v11_user,
    judge_v12_user,
    verdict_yes,
)
from .types import EvalResult
from .utils import (
    extract_sct_ids,
    f1_set,
    get_answer_text,
    get_labels,
    normalize_for_desc_match,
    prompt_ids_from_user_prompt,
)

# v17: descendants of 373572006 in B1 index (from rules.py)
NEGATION_CODES = {
    "440565004",
    "846756007",
    "860719001",
    "1141702004",
    "1344906007",
}


class CodesNotInLabelsError(ValueError):
    """A SNOMED-shaped ID (6–18 digits) in original_codes or processed_codes is missing from the labels index."""


class V17InputError(ValueError):
    """original_codes must include a negation code; processed_codes must not include any."""


def _snomed_id_strings(codes: list[str]) -> set[str]:
    """Only validate IDs that look like SNOMED concept IDs (6–18 digits)."""
    out: set[str] = set()
    for x in codes:
        s = str(x).strip()
        if re.fullmatch(r"\d{6,18}", s):
            out.add(s)
    return out


def _validate_all_codes_in_labels(
    original_codes: list[str],
    processed_codes: list[str],
    labels: dict[str, str],
) -> None:
    all_ids = _snomed_id_strings(original_codes) | _snomed_id_strings(processed_codes)
    if not all_ids:
        return
    missing = sorted(x for x in all_ids if x not in labels)
    if missing:
        raise CodesNotInLabelsError(
            "Every SNOMED ID in original_codes and processed_codes must exist in the labels table; "
            f"missing: {missing}"
        )


def _validate_v17_inputs(original_codes: list[str], processed_codes: list[str]) -> None:
    orig = {str(x) for x in original_codes}
    proc = {str(x) for x in processed_codes}
    if not (orig & NEGATION_CODES):
        raise V17InputError(
            "v17_exclude_negation expects original_codes to contain at least one negation "
            f"concept ID (from the v17 negation set); got original_codes={sorted(orig)}"
        )
    bad = proc & NEGATION_CODES
    if bad:
        raise V17InputError(
            "v17_exclude_negation expects processed_codes (gold target) to contain no "
            f"negation codes; got {sorted(bad)} in processed_codes"
        )


def _v10_answer_ids_valid(answer: str, gold_id: str, prompt_ids: set[str]) -> bool:
    extracted = extract_sct_ids(answer)
    if gold_id not in extracted:
        return False
    allowed = {gold_id} | prompt_ids
    return extracted.issubset(allowed)


def _v08_heuristic_accuracy(response: str, agree_expected: bool) -> bool:
    """When LLM judge is off: coarse keyword stance."""
    t = response.lower()
    positive = any(
        x in t
        for x in (
            "yes",
            "correct",
            "true",
            "agree",
            "valid",
            "appropriate",
            "matches",
            "is correct",
            "is appropriate",
        )
    )
    negative = bool(
        re.search(
            r"\b(no|incorrect|wrong|invalid|disagree|not\s+correct|not\s+appropriate|is\s+incorrect)\b",
            t,
        )
    )
    if agree_expected:
        return positive and not negative
    return negative and not positive


def eval_v01_direct_coding(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    **kwargs: Any,
) -> EvalResult:
    """v01: no instruction score; accuracy = F1(pred IDs, original_codes)."""
    text = get_answer_text(response)
    gold = {str(x) for x in original_codes}
    pred = extract_sct_ids(text)
    acc = f1_set(pred, gold)
    return EvalResult(None, acc, None)


def eval_v13_surface_perturbation(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    **kwargs: Any,
) -> EvalResult:
    return eval_v01_direct_coding(response, original_codes, processed_codes, **kwargs)


def eval_v02_id_only(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    **kwargs: Any,
) -> EvalResult:
    text = get_answer_text(response)
    gold = {str(x) for x in original_codes}
    pred = extract_sct_ids(text)
    inst = 0.0 if re.search(r"[A-Za-z]", text) else 1.0
    acc = f1_set(pred, gold)
    return EvalResult(inst, acc, None)


def _pred_codes_v03_descriptions(
    text: str, gold: set[str], labels: dict[str, str]
) -> set[str]:
    """Predicted codes: any extracted IDs plus gold codes whose description appears in text."""
    pred = extract_sct_ids(text)
    answer_cf = normalize_for_desc_match(text).casefold()
    for c in gold:
        if c not in labels:
            continue
        g = normalize_for_desc_match(labels[c]).casefold()
        if g and g in answer_cf:
            pred.add(c)
    return pred


def eval_v03_descriptions_only(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    labels_path: Path | None = None,
    **kwargs: Any,
) -> EvalResult:
    text = get_answer_text(response)
    labels = get_labels(labels_path)
    inst = 0.0 if re.search(r"\b\d{6,18}\b", text) else 1.0
    gold = {str(x) for x in original_codes}
    pred = _pred_codes_v03_descriptions(text, gold, labels)
    acc = f1_set(pred, gold)
    return EvalResult(inst, acc, None)


def eval_v04_id_plus_description(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    labels_path: Path | None = None,
    **kwargs: Any,
) -> EvalResult:
    """
    Instruction: at least one SNOMED ID present and at least one letter (ID + text).
    Accuracy: F1 between extracted IDs and original_codes.
    """
    text = get_answer_text(response)
    gold = {str(x) for x in original_codes}
    pred = extract_sct_ids(text)
    has_ids = bool(pred)
    has_letters = bool(re.search(r"[A-Za-z]", text))
    inst = 1.0 if (has_ids and has_letters) else 0.0
    acc = f1_set(pred, gold)
    return EvalResult(inst, acc, None)


def eval_v05_json_output(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    **kwargs: Any,
) -> EvalResult:
    text = get_answer_text(response).strip()
    gold = {str(x) for x in original_codes}
    try:
        json.loads(text)
        inst = 1.0
    except json.JSONDecodeError:
        return EvalResult(0.0, 0.0, None)
    pred = extract_sct_ids(text)
    acc = f1_set(pred, gold)
    return EvalResult(inst, acc, None)


def _v06_instruction_numeric_order_only(text: str) -> float:
    """1 iff every extracted ID appears in ascending numeric order; 0 if none extracted."""
    ids = re.findall(r"\b(\d{6,18})\b", text)
    if not ids:
        return 0.0
    nums = [int(x) for x in ids]
    return 1.0 if nums == sorted(nums) else 0.0


def eval_v06_sorted_multilabel(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    **kwargs: Any,
) -> EvalResult:
    text = get_answer_text(response)
    gold = {str(x) for x in original_codes}
    pred = extract_sct_ids(text)
    inst = _v06_instruction_numeric_order_only(text)
    acc = f1_set(pred, gold)
    return EvalResult(inst, acc, None)


def eval_v07_count_prediction(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    **kwargs: Any,
) -> EvalResult:
    text = get_answer_text(response)
    target = len({str(x) for x in original_codes})
    m = re.fullmatch(r"\s*(\d+)\s*", text)
    inst = 1.0 if m else 0.0
    acc = 1.0 if m and int(m.group(1)) == target else 0.0
    return EvalResult(inst, acc, None)


def eval_v08_code_validation(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    clinical_note: str | None = None,
    **kwargs: Any,
) -> EvalResult:
    """
    Single-label: true code = original_codes[0], proposed = processed_codes[0].
    Instruction: non-empty response. Accuracy: LLM judge (true code, proposed code, response
    only) returns yes/no; if judge disabled, keyword heuristic vs agree/disagree expectation.
    """
    text = get_answer_text(response)
    inst = 1.0 if text.strip() else 0.0
    if len(original_codes) != 1 or len(processed_codes) != 1:
        return EvalResult(inst, 0.0, None)

    gold_id = str(original_codes[0])
    candidate_id = str(processed_codes[0])
    agree_expected = gold_id == candidate_id

    judge_raw = call_judge(
        JUDGE_V08_SYSTEM,
        judge_v08_user(gold_id, candidate_id, text),
    )
    if judge_raw is not None:
        acc = 1.0 if verdict_yes(judge_raw) else 0.0
        return EvalResult(inst, acc, judge_raw)

    acc = 1.0 if _v08_heuristic_accuracy(text, agree_expected) else 0.0
    return EvalResult(inst, acc, None)


def eval_v09_multiple_choice(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    **kwargs: Any,
) -> EvalResult:
    """
    pred set: DeepSeek lists only final chosen MCQ IDs; then extract_sct_ids on that text.
    If extraction is unavailable (no API), falls back to extract_sct_ids on the full answer.
    model_response: raw extraction text when the LLM path ran.
    """
    text = get_answer_text(response)
    gold = {str(x) for x in processed_codes}
    extracted_raw = call_v09_pred_extraction(text)
    if extracted_raw is not None:
        pred = extract_sct_ids(extracted_raw)
        model_resp: str | None = extracted_raw.strip() or None
    else:
        pred = extract_sct_ids(text)
        model_resp = None
    inst = 1.0 if pred.issubset(gold) else 0.0
    acc = f1_set(pred, gold)
    return EvalResult(inst, acc, model_resp)


def eval_v10_parent_child_specificity(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    prompt: str | None = None,
    **kwargs: Any,
) -> EvalResult:
    """No instruction score; accuracy uses prompt IDs + gold-in-answer (see rules)."""
    if len(original_codes) != 1:
        return EvalResult(None, 0.0, None)
    if not prompt or not prompt.strip():
        return EvalResult(None, 0.0, None)

    text = get_answer_text(response)
    labels = get_labels(kwargs.get("labels_path"))
    gold_id = str(original_codes[0])

    prompt_ids = prompt_ids_from_user_prompt(prompt)
    if gold_id in prompt_ids:
        return EvalResult(None, 0.0, None)

    acc = 1.0 if _v10_answer_ids_valid(text, gold_id, prompt_ids) else 0.0
    return EvalResult(None, acc, None)


def eval_v11_code_desc_consistency(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    **kwargs: Any,
) -> EvalResult:
    """No instruction score; judge for accuracy."""
    text = get_answer_text(response)
    if len(original_codes) != 1 or len(processed_codes) != 1:
        return EvalResult(None, 0.0, None)

    labels = get_labels(kwargs.get("labels_path"))
    gold_id = str(original_codes[0])
    src_id = str(processed_codes[0])

    shown_desc = labels[src_id]
    gold_desc = labels[gold_id]
    judge_raw = call_judge(
        JUDGE_V11_SYSTEM,
        judge_v11_user(gold_id, shown_desc, gold_desc, text),
    )
    if judge_raw is not None:
        return EvalResult(None, 1.0 if verdict_yes(judge_raw) else 0.0, judge_raw)
    return EvalResult(None, 0.0, None)


def eval_v12_set_correction(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    **kwargs: Any,
) -> EvalResult:
    """No instruction score; judge for accuracy."""
    text = get_answer_text(response)
    labels = get_labels(kwargs.get("labels_path"))
    gold_set = {str(x) for x in original_codes}
    proposed_set = {str(x) for x in processed_codes}

    gt_sorted = sorted(gold_set, key=int)
    ground_truth_codes = ", ".join(f"{g} ({labels[g]})" for g in gt_sorted)
    prop_sorted = sorted(proposed_set, key=int)
    proposed_for_judge = ", ".join(f"{p} ({labels[p]})" for p in prop_sorted)

    judge_raw = call_judge(
        JUDGE_V12_SYSTEM,
        judge_v12_user(ground_truth_codes, proposed_for_judge, text),
    )
    if judge_raw is not None:
        return EvalResult(None, 1.0 if verdict_yes(judge_raw) else 0.0, judge_raw)
    return EvalResult(None, 0.0, None)


def eval_v14_policy_conditioned_one_code_only(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    **kwargs: Any,
) -> EvalResult:
    text = get_answer_text(response)
    ext = extract_sct_ids(text)
    gold = {str(x) for x in original_codes}
    inst = 1.0 if len(ext) == 1 else 0.0
    acc = 1.0 if len(ext) == 1 and next(iter(ext)) in gold else 0.0
    return EvalResult(inst, acc, None)


def eval_v15_description_from_code(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    labels_path: Path | None = None,
    **kwargs: Any,
) -> EvalResult:
    """No instruction score; accuracy = gold description substring in answer."""
    text = get_answer_text(response)
    labels = get_labels(labels_path)
    if len(original_codes) != 1:
        raise ValueError("eval_v15_description_from_code requires exactly one original code")
   
    code_id = str(original_codes[0])
    gold_desc = labels[code_id]
    answer_cf = normalize_for_desc_match(text).casefold()
    acc = 1.0 if normalize_for_desc_match(gold_desc).casefold() in answer_cf else 0.0
    return EvalResult(None, acc, None)


def eval_v16_code_from_description(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    **kwargs: Any,
) -> EvalResult:
    """No instruction score; accuracy = single extracted ID equals gold."""
    text = get_answer_text(response)
    ext_list = re.findall(r"\b(\d{6,18})\b", text)
    gold_id = str(original_codes[0])
    acc = 1.0 if len(ext_list) == 1 and ext_list[0] == gold_id else 0.0
    return EvalResult(None, acc, None)


def eval_v17_exclude_negation(
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    **kwargs: Any,
) -> EvalResult:
    """
    Callers must only pass rows where original_codes includes a negation label and
    processed_codes is the gold target with negations removed; see _validate_v17_inputs.
    Accuracy: F1(extracted IDs, processed_codes). No instruction score.
    """
    _validate_v17_inputs(original_codes, processed_codes)
    text = get_answer_text(response)
    gold = {str(x) for x in processed_codes}
    ext = extract_sct_ids(text)
    acc = f1_set(ext, gold)
    return EvalResult(None, acc, None)


RULE_EVALUATORS: dict[str, Any] = {
    "v01_direct_coding": eval_v01_direct_coding,
    "v13_surface_perturbation": eval_v13_surface_perturbation,
    "v02_id_only": eval_v02_id_only,
    "v03_descriptions_only": eval_v03_descriptions_only,
    "v04_id_plus_description": eval_v04_id_plus_description,
    "v05_json_output": eval_v05_json_output,
    "v06_sorted_multilabel": eval_v06_sorted_multilabel,
    "v07_count_prediction": eval_v07_count_prediction,
    "v08_code_validation": eval_v08_code_validation,
    "v09_multiple_choice": eval_v09_multiple_choice,
    "v10_parent_child_specificity": eval_v10_parent_child_specificity,
    "v11_code_desc_consistency": eval_v11_code_desc_consistency,
    "v12_set_correction": eval_v12_set_correction,
    "v14_policy_conditioned_one_code_only": eval_v14_policy_conditioned_one_code_only,
    "v15_description_from_code": eval_v15_description_from_code,
    "v16_code_from_description": eval_v16_code_from_description,
    "v17_exclude_negation": eval_v17_exclude_negation,
}


def evaluate_row(
    rule: str,
    response: str,
    original_codes: list[str],
    processed_codes: list[str],
    **kwargs: Any,
) -> EvalResult:
    fn = RULE_EVALUATORS.get(rule)
    if fn is None:
        raise KeyError(f"Unknown rule: {rule!r}. Known: {sorted(RULE_EVALUATORS)}")
    labels = get_labels(kwargs.get("labels_path"))
    _validate_all_codes_in_labels(original_codes, processed_codes, labels)
    return fn(response, original_codes, processed_codes, **kwargs)
