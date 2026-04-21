"""Types for instruction-following evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalResult:
    """Scores for one evaluated example."""

    instruction_score: float | None
    """None when this rule does not define an instruction-following score."""
    accuracy_score: float
    model_response: str | None = None
    """Raw LLM judge output, or v09 extraction output when that path ran."""
