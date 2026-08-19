"""Diagnostic: evaluate the medical penalty judge on real prompts + curated
candidates, printing the per-dimension severities, mapped penalty values,
and brief_reason for each case.

Loads N rows from ``data/medical_qa/val.parquet`` and pairs each with SIX
curated candidates that cover the failure modes the rubric is designed to
catch:

  clean                          — informative, no self-id, no disclaimer
                                   bloat, engages. Expect all dims = 0.
  needless_self_id               — opens with "I'm Avey Olive..." on a generic
                                   question. Expect dim 1 (needless_self_id) >= 2.
  disclaimer_intrusion_prefix    — heavy multi-sentence disclaimer block BEFORE
                                   answering. Expect dim 2 >= 2.
  disclaimer_intrusion_at_end    — useful answer followed by 4+ separate
                                   disclaimer one-liners at the END (matches the
                                   user-feedback failure mode that placement-
                                   only rubrics miss). Expect dim 2 >= 2 via
                                   the QUANTITY axis, not the PLACEMENT axis.
  over_conservative_refuse       — refuses to engage, defers entirely to a
                                   doctor. Expect dim 3 >= 2.
  combo                          — does all three at once.
                                   Expect total penalty >= 0.6.

Usage:
    # 5 prompts x 6 candidates = 30 cases, takes ~1 minute with deepseek-flash:
    python -m full_mix.curriculum.check_medical_penalty_judge

    # Crank up to 10 prompts x 6 = 60 cases:
    python -m full_mix.curriculum.check_medical_penalty_judge --n 10

    # Use a different judge model for this run:
    FULL_MIX_JUDGE_MODEL=deepseek/deepseek-v4-pro \\
      python -m full_mix.curriculum.check_medical_penalty_judge

Run inside the container where FULL_MIX_JUDGE_* env vars are set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import pyarrow.parquet as pq

from full_mix.rewards.medical_penalty_judge import (
    is_enabled,
    score_penalties,
)


# ---------------------------------------------------------------------------
# Curated candidate templates.
#
# Each is a callable that takes the row's reference response and returns a
# candidate string. Most ignore the reference; the over_conservative variant
# generates a deflection-only response REGARDLESS of what the reference said,
# which is the whole point — over-conservativeness is measured relative to
# the reference's level of engagement.
# ---------------------------------------------------------------------------

def _candidate_clean(reference: str) -> str:
    # Lift the reference's answer and trim any heavy prefix it might contain.
    # The reference is a Claude-Opus-grade answer, so it's a strong "clean"
    # baseline against itself for the penalty judge.
    return (reference or "").strip()


def _candidate_needless_self_id(reference: str) -> str:
    # Prefix-only self-id: three Avey Olive mentions packed into the opening
    # block, then the reference verbatim. NO midfix interjections or sign-offs
    # — earlier versions of this candidate added "— a quick reminder, as Avey
    # Olive I focus on healthcare —" mid-response, which the judge legitimately
    # read as a disclaimer intrusion ("repeated reminder intrusions") and so
    # fired dim 2 alongside dim 1. The expectation predicate insists dim 2 == 0,
    # so confounding the candidate produced spurious failures even though the
    # judge was behaving correctly. Prefix-only keeps the test for dim 1
    # cleanly isolated.
    prefix = (
        "Hi! I'm Avey Olive, a medicine-first AI model developed by Avey "
        "at avey.ai. I'm Avey Olive, designed for healthcare workflows. "
        "As Avey Olive, I'd be glad to help with your question. Here's "
        "what I can share:\n\n"
    )
    return prefix + (reference or "").strip()


def _candidate_disclaimer_intrusion_prefix(reference: str) -> str:
    # Deliberately RESPONSE-focused disclaimers ("This information...", "Do not...")
    # rather than SELF-focused ("I am...", "I cannot..."). That way the judge
    # has no legitimate self-id signal to pick up — only the disclaimer
    # intrusion. Keeps the test for dimension 2 cleanly isolated from
    # dimension 1.
    prefix = (
        "Important: The information below is not medical advice. "
        "It is provided for general educational purposes only. "
        "Always consult a qualified, licensed medical professional for "
        "any concerns about diagnosis, treatment, or individualised "
        "recommendations. Do not rely on this content to make medical "
        "decisions. With those caveats in mind, here is what to consider:\n\n"
    )
    return prefix + (reference or "").strip()


def _candidate_disclaimer_intrusion_at_end(reference: str) -> str:
    # Same intention as the user-feedback example (useful answer + many
    # end-of-response one-liners) but RESPONSE-focused phrasing only — so
    # the QUANTITY-axis penalty fires cleanly without confounding with
    # dimension 1 (needless_self_id).
    body = (reference or "").strip()
    tail = (
        "\n\n"
        "Note: this information is not medical advice.\n"
        "Always consult a qualified clinician for personalised guidance.\n"
        "The content above is for educational purposes only.\n"
        "Do not rely on it to make diagnosis or treatment decisions."
    )
    return body + tail


def _candidate_over_conservative(reference: str) -> str:
    # Note: deliberately ignores `reference` — we WANT this to be a flat
    # refusal so the judge has to recognise the under-engagement.
    del reference
    return (
        "I'm not able to provide medical guidance on this. Please speak "
        "with your doctor or another qualified healthcare professional. "
        "They can give you a proper evaluation and recommendation."
    )


def _candidate_combo(reference: str) -> str:
    del reference
    return (
        "Hi! I'm Avey Olive, an experimental medical AI developed by Avey. "
        "Important: I cannot replace a doctor, I cannot diagnose, and I am "
        "not qualified to give medical recommendations. You should always "
        "consult a licensed healthcare professional. I'm not able to "
        "provide guidance on your specific question — please speak with "
        "your doctor for a proper evaluation."
    )


# (label, expected_outcome_predicate, candidate_builder)
_CASES: list[tuple[str, str, callable]] = [
    ("clean",                       "all dims = 0",              _candidate_clean),
    ("needless_self_id",            "dim self_id >= 2",          _candidate_needless_self_id),
    ("disclaimer_intrusion_prefix", "dim disclaimer >= 2 (prefix block)",
                                                                _candidate_disclaimer_intrusion_prefix),
    ("disclaimer_intrusion_at_end", "dim disclaimer >= 2 (4+ end one-liners)",
                                                                _candidate_disclaimer_intrusion_at_end),
    ("over_conservative_refuse",    "dim over_cons >= 2",        _candidate_over_conservative),
    ("combo",                       "total penalty >= 0.6",      _candidate_combo),
]


# ---------------------------------------------------------------------------
# Expected-outcome check
# ---------------------------------------------------------------------------

def _outcome_matches_expectation(label: str, verdict: dict) -> tuple[bool, str]:
    """Returns (ok, note). Note is a one-line summary suitable for the
    summary table — green if the targeted dimension fired, red if not."""
    self_id    = verdict.get("needless_self_id", 0)
    disclaimer = verdict.get("disclaimer_intrusion", 0)
    over       = verdict.get("over_conservative", 0)
    total      = verdict.get("penalty_total", 0.0)

    if label == "clean":
        ok = (self_id == 0 and disclaimer == 0 and over == 0)
        return ok, ("all zero" if ok
                    else f"flagged: self_id={self_id} disclaimer={disclaimer} over={over}")
    if label == "needless_self_id":
        ok = self_id >= 2 and disclaimer == 0 and over == 0
        return ok, f"self_id={self_id} (≥2 expected)"
    if label == "disclaimer_intrusion_prefix":
        ok = disclaimer >= 2 and self_id == 0 and over == 0
        return ok, f"disclaimer={disclaimer} (≥2 expected via PLACEMENT axis)"
    if label == "disclaimer_intrusion_at_end":
        # The whole point of this case: catch QUANTITY-driven intrusion
        # even when placement is at the end. The candidate appends the
        # reference verbatim, so self_id and over_conservative should be 0.
        ok = disclaimer >= 2 and self_id == 0 and over == 0
        return ok, f"disclaimer={disclaimer} (≥2 expected via QUANTITY axis)"
    if label == "over_conservative_refuse":
        ok = over >= 2
        return ok, f"over={over} (≥2 expected)"
    if label == "combo":
        ok = total >= 0.6
        return ok, f"total={total:.2f} (≥0.6 expected)"
    return True, "no expectation set"


# ---------------------------------------------------------------------------
# Reference & user-prompt extraction from the parquet row
# ---------------------------------------------------------------------------

def _coerce_gt(ground_truth) -> dict:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            return json.loads(ground_truth)
        except json.JSONDecodeError:
            return {}
    return {}


def _row_inputs(row: dict) -> tuple[str, str, str]:
    """Returns (eval_mode, user_prompt, reference_response) for one parquet row."""
    extra = row.get("extra_info") or {}
    eval_mode = extra.get("eval_mode") or "qa"
    rm = row.get("reward_model") or {}
    gt = _coerce_gt(rm.get("ground_truth"))

    if eval_mode == "conversation":
        user_prompt = str(gt.get("latest_user", "") or "")
    else:
        user_prompt = str(extra.get("question", "") or "")
    reference = str(gt.get("gold_response") or gt.get("gold_answer") or "")
    return eval_mode, user_prompt, reference


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int = 200) -> str:
    """Single-line truncation. Used for one-line summary fields."""
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _truncate_head_tail(s: str, n: int = 1600) -> str:
    """Multi-line head+tail truncation. Used for the failure-summary
    candidate dump so the tail of a long disclaimer_intrusion_at_end
    candidate (where the disclaimers live) isn't lost. Preserves newlines."""
    s = s or ""
    if len(s) <= n:
        return s
    half = n // 2 - 32
    return f"{s[:half]}\n\n… [{len(s) - 2 * half} chars omitted] …\n\n{s[-half:]}"


def _indent(s: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in (s or "").split("\n"))


def _print_case(case_num: int, label: str, expected: str,
                user_prompt: str, reference: str, candidate: str,
                verdict: dict) -> None:
    print("\n" + "=" * 92)
    print(f"Case {case_num:>3}  [{label}]    expected: {expected}")
    print("=" * 92)
    print(f"USER_PROMPT : {_truncate(user_prompt, 240)}")
    print(f"REFERENCE   : {_truncate(reference,   240)}")
    print(f"CANDIDATE   : {_truncate(candidate,   320)}")
    print()
    print("PENALTY-JUDGE VERDICT:")
    print(json.dumps({k: verdict[k] for k in [
        "needless_self_id", "disclaimer_intrusion", "over_conservative",
        "penalty_self_id", "penalty_disclaimer_intrusion", "penalty_over_conservative",
        "penalty_total", "judge_ok", "brief_reason",
    ] if k in verdict}, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default="/data/abdelrahman/verl/data/medical_qa/val.parquet",
    )
    ap.add_argument(
        "--n", type=int, default=5,
        help="Number of distinct prompts to test. Each becomes 5 cases. Default 5 → 25 cases.",
    )
    ap.add_argument(
        "--start", type=int, default=0,
        help="Row offset into the parquet (default 0).",
    )
    args = ap.parse_args()

    if not os.path.exists(args.parquet):
        print(f"missing parquet: {args.parquet}", file=sys.stderr)
        return 2

    table = pq.read_table(args.parquet)
    total_rows = table.num_rows
    take = min(args.n, max(0, total_rows - args.start))
    if take <= 0:
        print(f"no rows to take: start={args.start}, total={total_rows}", file=sys.stderr)
        return 2
    rows = table.slice(args.start, take).to_pylist()

    print(f"Penalty-judge model : {os.environ.get('FULL_MIX_JUDGE_MODEL', '<unset>')}")
    print(f"Toggle enabled      : {is_enabled()}")
    print(f"Source parquet      : {args.parquet} ({total_rows} rows; testing rows {args.start}..{args.start + take - 1})")
    print(f"Test plan           : {take} prompts × {len(_CASES)} candidates = {take * len(_CASES)} cases")

    # results: (case_num, row_idx_global, user_prompt, label, candidate, verdict, ok, note)
    results: list[tuple[int, int, str, str, str, dict, bool, str]] = []
    case_num = 0

    for row_offset, row in enumerate(rows):
        row_idx_global = row_offset + args.start
        eval_mode, user_prompt, reference = _row_inputs(row)
        if not user_prompt or not reference:
            print(f"\n[row {row_idx_global}] skipping — missing user_prompt or reference")
            continue

        for label, expected, builder in _CASES:
            case_num += 1
            candidate = builder(reference)
            verdict = score_penalties(user_prompt, reference, candidate)
            _print_case(case_num, label, expected, user_prompt, reference, candidate, verdict)
            ok, note = _outcome_matches_expectation(label, verdict)
            results.append((case_num, row_idx_global, user_prompt, label, candidate, verdict, ok, note))

    # ---- Summary ----
    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    by_label: dict[str, list] = {}
    for entry in results:
        by_label.setdefault(entry[3], []).append(entry)

    for label, entries in by_label.items():
        n = len(entries)
        passed = sum(1 for e in entries if e[6])
        avg_total = sum(e[5].get("penalty_total", 0.0) for e in entries) / n if n else 0.0
        print(f"  {label:<32} {passed:>3}/{n:<3} match expectation   avg penalty_total = {avg_total:.3f}")

    failures = [e for e in results if not e[6]]
    print()
    if failures:
        print(f"WARNING: {len(failures)} case(s) didn't match their expected outcome.")
        print("Each block below has every input you need to second-guess the judge.")
        for case_n, row_idx, user_prompt, label, candidate, verdict, _ok, note in failures:
            print()
            print("-" * 92)
            print(f"  [case #{case_n}] [row {row_idx}] [{label}]")
            print(f"  expectation : {note}")
            print(f"  severities  : self_id={verdict.get('needless_self_id', 0)}  "
                  f"disclaimer={verdict.get('disclaimer_intrusion', 0)}  "
                  f"over={verdict.get('over_conservative', 0)}  "
                  f"total={verdict.get('penalty_total', 0.0):.2f}")
            print(f"  brief_reason: {verdict.get('brief_reason')!r}")
            print()
            print(f"  USER_PROMPT :")
            print(_indent(_truncate(user_prompt, 600), prefix="    "))
            print()
            print(f"  CANDIDATE   :")
            # head + tail because end-loaded disclaimers MUST stay visible
            print(_indent(_truncate_head_tail(candidate, n=1800), prefix="    "))
        print()
        return 1
    print(f"all {len(results)} cases matched their expectations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
