"""Build the validation ladder: each eval set rendered at every rung.

Training draws budgets at random so the model has to learn the actual mapping
from the number to a length, rather than memorising a handful of lengths.
Validation does the opposite — fixed rungs, so score-vs-budget is a clean,
comparable curve across checkpoints. That split is the whole reason both exist.

One parquet per variant, with the eval sets concatenated inside it:

    val_budget_00256.parquet ... val_budget_06000.parquet
    val_free.parquet          (bare /think, no budget)
    val_nothink.parquet       (/no_think)

Each row's data_source is tagged `@b00256` / `@free` / `@nothink` so verl emits
one validation series PER RUNG instead of pooling all eight into a single
number. The tag is a reporting dimension only — encoding.split_data_source
strips it before any routing or scoring decision.

The rungs bracket the training range [200, 6000] rather than extending past it.
6000 is trained, so the top of the ladder is interpolation; that is deliberate,
because a monotonicity claim that only holds inside the trained range is the one
we can actually defend.

extra_info is homogenized to the same union schema the training builder uses, so
train and val rows agree and verl can concatenate either set.

Usage:
    python -m full_mix.lcpo.build_val_ladder
"""

import argparse
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from full_mix.common.budget_prompt import (  # noqa: E402
    MODE_BUDGET,
    MODE_FREE,
    MODE_NOTHINK,
    NO_BUDGET,
    apply_to_messages,
)
from full_mix.common.lcpo_schema import homogenize_extra_info  # noqa: E402

RUNGS = [256, 512, 1024, 2048, 4000, 6000]

# NOTE: ifbench_eval dropped — its constraint IDs are not in full_mix/ifeval/
# FUNCTION_DICT, so 100% of constraints score 0 (unknown IDs count toward the
# denominator in ifeval_reward.compute_score, so every IFBench row returns a
# flat 0.0 no matter what the model writes). Re-add once IFBench checkers are
# ported from allenai/open-instruct into full_mix/ifeval/instructions.py.
EVAL_FILES = [
    "gsm8k_eval.parquet",
    "ifeval_eval.parquet",
    "math500_eval.parquet",
]

def variant_tag(mode: str, budget: int) -> str:
    """Reporting tag appended to data_source as ``@tag``.

    verl groups validation metrics by the data_source STRING, so without this
    all eight variants pool into one number per eval set and the score-vs-budget
    curve — the entire point of the ladder — cannot be plotted. The 2026-08-19
    run showed the symptom clearly: `think_budget/mean@1` sat at a constant
    1729.75, which is the mean of the six rungs plus the two -1 sentinels, a
    budget no prompt actually carries.

    The tag is stripped by encoding.split_data_source, so reward routing, int
    encoding and uid construction never see it.
    """
    if mode == MODE_BUDGET:
        return f"b{int(budget):05d}"
    return str(mode)


def render(rows: list[dict], mode: str, budget: int) -> list[dict]:
    tag = variant_tag(mode, budget)
    return [
        {
            "prompt": apply_to_messages(r["prompt"], mode, budget),
            "data_source": f"{r['data_source']}@{tag}",
            "ability": r["ability"],
            "reward_model": r["reward_model"],
            "extra_info": homogenize_extra_info(r["extra_info"]),
            "think_budget": int(budget),
            "think_mode": mode,
        }
        for r in rows
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="/data/abdelrahman/verl/data/full_mix/eval")
    ap.add_argument("--out_dir", default="/data/abdelrahman/verl/data/lcpo_val")
    ap.add_argument("--rungs", type=int, nargs="+", default=RUNGS)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows: list[dict] = []
    per_set = {}
    for fname in EVAL_FILES:
        path = os.path.join(args.eval_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        got = pq.read_table(path).to_pylist()
        per_set[fname] = len(got)
        rows.extend(got)
    if not rows:
        raise SystemExit(f"no eval parquets found under {args.eval_dir}")

    variants = [(f"val_budget_{b:05d}", MODE_BUDGET, b) for b in args.rungs]
    variants.append(("val_free", MODE_FREE, NO_BUDGET))
    variants.append(("val_nothink", MODE_NOTHINK, NO_BUDGET))

    written = {}
    for name, mode, budget in variants:
        built = render(rows, mode, budget)
        dst = os.path.join(args.out_dir, f"{name}.parquet")
        pq.write_table(pa.Table.from_pylist(built), dst)
        written[name] = {"path": dst, "rows": len(built), "mode": mode, "budget": budget}
        print(f"  {name:20s} {len(built):5d} rows  mode={mode:8s} budget={budget}")

    with open(os.path.join(args.out_dir, "_build_report.json"), "w") as f:
        json.dump({"rungs": args.rungs, "eval_sets": per_set,
                   "rows_per_variant": len(rows), "variants": written}, f, indent=2)

    print(f"\n  {len(variants)} variants x {len(rows)} rows = {len(variants) * len(rows)} val rows")
    print(f"  saved -> {args.out_dir}")
    # The exact list to paste into VAL_FILES.
    print("\n  VAL_FILES=[" + ",".join(written[n]["path"] for n, _, _ in variants) + "]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
