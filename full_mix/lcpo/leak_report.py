"""Paired leak report for a validation dump.

Every validation prompt is asked at each budget and free, so for each prompt the
answer body at budget b can be compared with the SAME prompt's free-mode answer.
This prints, per source and budget, the median of (answer_b - answer_free) and
the share of prompts whose budgeted answer body exceeds 1.5x its free-mode
answer — the stop-rule quantity for the think-target runs.

    python3 -m full_mix.lcpo.leak_report dumps/qlcm/medical_qa_stage9b_val/40.jsonl
    python3 -m full_mix.lcpo.leak_report <dump> --threshold 2.0 --by-src-id
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st

SRC_ID = {1: "mlb", 2: "ml", 3: "if", 4: "snomed_ml", 5: "snomed_if", 6: "sl", 7: "snomed_sl", 8: "slb",
          9: "medical_qa", 10: "medical_conv", 11: "ifeval", 12: "chat", 13: "safety", 14: "identity",
          21: "IFEval", 22: "GSM8K", 23: "MATH-500"}


def source_of(r: dict, by_src_id: bool) -> str:
    if by_src_id or "data_source" not in r or r.get("data_source") is None:
        return SRC_ID.get(int(r.get("src_id", 0)), str(r.get("data_source_code", "?")))
    return str(r["data_source"]).split("@", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--threshold", type=float, default=1.5, help="leak = answer_b > threshold * answer_free")
    ap.add_argument("--by-src-id", action="store_true", help="name sources by the numeric src_id stamp")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.dump)]
    free: dict[tuple, float] = {}
    budgeted: dict[tuple, list] = collections.defaultdict(list)
    for r in rows:
        src = source_of(r, args.by_src_id)
        key = (src, r["prompt_uid"])
        if r["think_budget"] > 0:
            budgeted[key].append((int(r["think_budget"]), float(r["n_answer"]), float(r["n_think"])))
        elif r.get("is_nothink_mode", 0) != 1:
            free[key] = float(r["n_answer"])

    per: dict[tuple, list] = collections.defaultdict(list)
    for key, items in budgeted.items():
        if key not in free:
            continue
        f = free[key]
        for b, a, t in items:
            per[(key[0], b)].append((a - f, a > args.threshold * max(f, 1.0), a, t))
    budgets = sorted({b for _, b in per})
    srcs = sorted({s for s, _ in per})
    print(f"paired prompts: {len({k for k in budgeted if k in free})} of {len(budgeted)} budgeted; threshold {args.threshold}x")
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
