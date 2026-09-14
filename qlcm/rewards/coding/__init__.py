"""Self-contained medical-coding reward suite.

Public API: import `compute_score` from this package and call it with verl's
standard signature. Routes ICD-10 (ml + if) and SNOMED CT (ml + if) rows to
the right deterministic / framework-backed scorer based on
`extra_info.task_type`. Returns a dict normalised to the 15-key coding union
(see `common._REWARD_UNION_DEFAULTS`).

Layout:
  common.py                            — shared helpers + union defaults
  icd_scorer.py                        — ICD-10 multilabel (P/R/F1)
  snomed_ml_scorer.py                  — SNOMED CT multilabel (P/R/F1)
  snomed_if_scorer.py                  — SNOMED CT instruction-following
  dispatcher.py                        — per-row router + compute_score entry point
  instruction_following/               — embedded IF framework (rules_eval, etc.)
  instruction_following/labels/        — embedded sct_id → description JSON
"""

from .dispatcher import compute_score
from .common import _REWARD_UNION_DEFAULTS, _serialize_gold, _normalize_reward_dict

__all__ = [
    "compute_score",
    "_REWARD_UNION_DEFAULTS",
    "_serialize_gold",
    "_normalize_reward_dict",
]
