"""Identity reward — reference-aligned LLM judge.

For identity-dataset samples, scores the candidate against the row's
reference response across every identity dimension the prompt probes
(name, company, capability scope, manipulation resistance, labeling
resistance, lineage non-disclosure, mission, capabilities). See
``identity_judges.reference_aligned_identity_score`` for the full rubric
and rationale.

The dispatcher (full_mix/rewards/compute_score.py) ALSO applies the
GLOBAL identity gate (``identity_judges.check_global_identity_violation``)
to every sample in every domain — so an identity sample that makes a
positive false-identity claim will already score 1 from this rubric. We
don't double-apply the gate here.

Notes:
  - ``ground_truth`` carries the first ``reference_responses`` entry from
    the source dataset (see ``full_mix.preprocess.identity_train``).
  - When the reference is missing, ``identity_judges`` uses a built-in
    stand-in so the judge still has a stance anchor.
"""

import logging

from full_mix.rewards.identity_judges import reference_aligned_identity_score


logger = logging.getLogger(__name__)


def compute_score(solution_str: str, ground_truth, extra_info=None) -> float:
    """Grade an identity-domain response. Returns reward in [0, 1]."""
    if not solution_str or not solution_str.strip():
        return 0.0

    user_prompt = ""
    if extra_info and isinstance(extra_info, dict):
        user_prompt = (extra_info.get("user_prompt") or "").strip()

    reference = ground_truth if isinstance(ground_truth, str) else str(ground_truth or "")

    return reference_aligned_identity_score(
        user_prompt=user_prompt,
        reference=reference,
        candidate=solution_str,
    )
