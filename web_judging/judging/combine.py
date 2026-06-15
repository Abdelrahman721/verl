"""Combine a qa-judge breakdown with the shared penalty verdict.

Verbatim port of ``qa_bedrock_with_penalties._normalise_penalty_into`` (and its
``_PENALTY_OUTPUT_KEYS`` map). Kept self-contained per the no-cross-import rule.

    final score = max(0.0, qa_score - penalty_total)
"""

# Keys produced by score_penalties() -> names exposed on the reward dict.
_PENALTY_OUTPUT_KEYS = {
    "penalty_self_id":              "reward/penalty_self_id",
    "penalty_disclaimer_intrusion": "reward/penalty_disclaimer_intrusion",
    "penalty_over_conservative":    "reward/penalty_over_conservative",
    "penalty_total":                "reward/penalty_total",
}


def normalise_penalty_into(qa_result: dict, penalty: dict) -> dict:
    """Merge penalty fields onto ``qa_result``. Pure (returns a new dict).

    Sets ``reward/score_pre_penalty`` to the original qa score, replaces
    ``score`` with the post-penalty value (floored at 0), and sets
    ``reward/penalty_judge_ok`` to 1/0. Emits the full penalty key set with
    sentinel 0 values when the penalty judge failed / was disabled.
    """
    out = dict(qa_result) if isinstance(qa_result, dict) else {"score": 0.0}
    pre_penalty = float(out.get("score", 0.0))
    total_penalty = float(penalty.get("penalty_total", 0.0)) if penalty else 0.0
    out["reward/score_pre_penalty"] = pre_penalty
    out["score"] = max(0.0, pre_penalty - total_penalty)
    out["reward/penalty_judge_ok"] = int(bool(penalty.get("judge_ok", False))) if penalty else 0
    for src, dst in _PENALTY_OUTPUT_KEYS.items():
        out[dst] = float(penalty.get(src, 0.0)) if penalty else 0.0
    return out
