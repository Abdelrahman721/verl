"""Paired leak report for a validation dump.

Every validation prompt is asked at each budget and free, so for each prompt the
answer body at budget b can be compared with the SAME prompt's free-mode answer.
This prints, per source and budget, the median of (answer_b - answer_free) and
the share of prompts whose budgeted answer body exceeds 1.5x its free-mode
answer — the stop-rule quantity for the think-target runs.

    python3 -m full_mix.lcpo.leak_report dumps/qlcm/medical_qa_stage9b_val/40.jsonl
    python3 -m full_mix.lcpo.leak_report <dump> --threshold 2.0 --by-src-id

Prompts are paired on the md5 of the dumped `input` with the budget sentence
removed, NOT on `prompt_uid`: the coding/medical scorers stamp a constant
retention sentinel there (every mlb, ml, if, snomed_ml, snomed_if, medical_qa
and medical_conv row shares one id), so pairing on it compared every budgeted
row of those sources against one arbitrary free row. When a prompt has several
free rows (rollout dumps: n samples per prompt) the median free answer is used.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import statistics as st
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from full_mix.common import budget_prompt as bp  # noqa: E402

# The budget sentence in either wording, built from budget_line so it cannot
# drift from what the data builders write.
_N = "123456789"
_SENTENCE = "|".join(re.escape(bp.budget_line(int(_N), s)).replace(_N, r"\d+")
                     for s in (bp.STYLE_MAX, bp.STYLE_EXACT))
# Everything after the task: the budget sentence (budgeted rows only), the
# full_mix `/think` marker and the chat template's assistant header, with the
# whitespace around them. Anchored at the end, so a task that merely contains
# the words is never touched. Covers both prompt formats:
#   qlcm      {task}\n\nThink for a maximum of N tokens.      free: {task}
#   full_mix  {task}\n\nThink for N tokens.\n\n/think         free: {task}\n\n/think
# The task's own trailing whitespace goes too: budgeted rows rstrip the task
# before appending the sentence (budget_prompt.render_prompt) while qlcm free
# rows keep it, so a note ending in "\n" would otherwise never pair.
_PROMPT_TAIL = re.compile(
    r"(?:\s*(?:" + _SENTENCE + r"))?"
    r"(?:\s*" + re.escape(bp.MARKER_THINK) + r")?"
    r"\s*(?:<\|im_end\|>)?\s*(?:<\|im_start\|>)?(?:assistant)?\s*\Z"
)
# The tail is under 100 characters; scanning only the end keeps the regex from
# trying every position of a multi-thousand-character clinical note.
_TAIL_WINDOW = 512


def prompt_key(text: str) -> str:
    """Identity of a prompt, equal for its budgeted and free renderings."""
    cut = max(0, len(text) - _TAIL_WINDOW)
    body = (text[:cut] + _PROMPT_TAIL.sub("", text[cut:], count=1)).rstrip()
    return hashlib.md5(body.encode("utf-8")).hexdigest()

SRC_ID = {1: "mlb", 2: "ml", 3: "if", 4: "snomed_ml", 5: "snomed_if", 6: "sl", 7: "snomed_sl", 8: "slb",
          9: "medical_qa", 10: "medical_conv", 11: "ifeval", 12: "chat", 13: "safety", 14: "identity",
          21: "IFEval", 22: "GSM8K", 23: "MATH-500"}


def source_of(r: dict, by_src_id: bool) -> str:
    if by_src_id or "data_source" not in r or r.get("data_source") is None:
        return SRC_ID.get(int(r.get("src_id", 0)), str(r.get("data_source_code", "?")))
    return str(r["data_source"]).split("@", 1)[0]


def pair_rows(rows: list[dict], threshold: float = 1.5, by_src_id: bool = False):
    """Pair each budgeted row with its own prompt's free answer.

    Returns (per, free, free_rows, budgeted): `per[(source, budget)]` holds one
    (answer_b - answer_free, leaked, answer_b, think_b) tuple per paired row, and
    `free[(source, key)]` the median free answer of that prompt.
    """
    free_rows: dict[tuple, list] = collections.defaultdict(list)
    budgeted: dict[tuple, list] = collections.defaultdict(list)
    for r in rows:
        src = source_of(r, by_src_id)
        key = (src, prompt_key(r["input"]))
        if r["think_budget"] > 0:
            budgeted[key].append((int(r["think_budget"]), float(r["n_answer"]), float(r["n_think"])))
        elif r.get("is_nothink_mode", 0) != 1:
            free_rows[key].append(float(r["n_answer"]))
    free = {k: st.median(v) for k, v in free_rows.items()}

    per: dict[tuple, list] = collections.defaultdict(list)
    for key, items in budgeted.items():
        if key not in free:
            continue
        f = free[key]
        for b, a, t in items:
            per[(key[0], b)].append((a - f, a > threshold * max(f, 1.0), a, t))
    return per, free, free_rows, budgeted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--threshold", type=float, default=1.5, help="leak = answer_b > threshold * answer_free")
    ap.add_argument("--by-src-id", action="store_true", help="name sources by the numeric src_id stamp")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.dump)]
    if any("input" not in r for r in rows):
        print(f"{args.dump}: rows without `input`; prompts cannot be paired", file=sys.stderr)
        return 2
    per, free, free_rows, budgeted = pair_rows(rows, args.threshold, args.by_src_id)
    budgets = sorted({b for _, b in per})
    srcs = sorted({s for s, _ in per})
    per_prompt = collections.Counter(len(v) for v in free_rows.values())
    print(f"paired prompts: {len({k for k in budgeted if k in free})} of {len(budgeted)} budgeted; "
          f"free rows per prompt: {dict(sorted(per_prompt.items()))}; threshold {args.threshold}x")
    print(f"{'source':14s} {'budget':>6s} {'n':>4s} {'answer_free':>11s} {'answer_b':>9s} {'median diff':>11s} {'leak share':>10s} {'think_b':>8s}")
    for s in srcs:
        fr = [free[k] for k in free if k[0] == s]
        for b in budgets:
            v = per.get((s, b))
            if not v:
                continue
            print(f"{s:14s} {b:6d} {len(v):4d} {st.median(fr):11.0f} {st.median([x[2] for x in v]):9.0f} "
                  f"{st.median([x[0] for x in v]):+11.0f} {sum(x[1] for x in v) / len(v):10.2f} {st.median([x[3] for x in v]):8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
