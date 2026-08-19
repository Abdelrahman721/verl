"""Build the LCPO training mix: every prompt materialized in three modes/budgets.

Sources (disjoint by construction — mix2 was partitioned against math_mix):

    chat_ifeval_mix2/chat_with_baseline_train.parquet   2000
    chat_ifeval_mix2/ifeval_train.parquet               4000
    chat_ifeval_math_mix/math_train.parquet             2000

Each of the 8000 prompts becomes K=3 rows, so 24000 rows total, split
60% budgeted / 20% unconstrained / 20% zero.

Hitting that split exactly while guaranteeing every prompt sees at least one
budget is a small combinatorial fact: over 3 slots, give 40% of prompts
(budget, budget, free), 40% (budget, budget, nothink), and 20%
(budget, free, nothink). That averages 1.8 / 0.6 / 0.6 slots per prompt, i.e.
exactly 60/20/20, and no prompt is left without a budgeted row. Patterns are
assigned by `index % 5`, so the layout is reproducible and inspectable.

Budgets are drawn uniformly from [200, 6000] and FIXED here, not resampled per
epoch. verl reads rows verbatim from the parquet with no per-epoch hook, and
re-rolling inside the dataset's __getitem__ would run in dataloader workers —
unreproducible and awkward to resume. Replicating K times at build time gets the
same variety, reproducibly, with zero framework changes.

Why 200-6000 rather than the paper's 100-4000: this model's own reasoning sits
at p05 ~330 / p50 ~1200-1500 / p95 ~7000-8000 tokens. Below ~200 it cannot get
anything right, so those groups would only ever teach "be short" with no link to
being correct. And a ceiling of 4000 sits near this model's median, so it would
only ever be asked to compress — leaving anything above 4000 at eval time as
pure extrapolation, which is exactly the range where we claim scores keep rising.

`think_budget` and `think_mode` are TOP-LEVEL columns, not nested in extra_info,
because verl concatenates the three files and their extra_info structs differ.
This builder also homogenizes extra_info to the union of all three schemas
(missing string fields become ""), so concatenation cannot silently coerce.

Usage:
    python -m full_mix.curriculum.build_lcpo_mix
    python -m full_mix.curriculum.build_lcpo_mix --k 3 --seed 43 --out_dir /tmp/lcpo
"""

import argparse
import json
import os
import random
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

SOURCES = [
    ("chat_ifeval_mix2", "chat_with_baseline_train.parquet", "local/dolci-chat-32b"),
    ("chat_ifeval_mix2", "ifeval_train.parquet", "local/dolci-ifeval-32b"),
    ("chat_ifeval_math_mix", "math_train.parquet", "local/dolci-math-7b"),
]

# index % 5 -> the three slot modes for that prompt. 2/5 + 2/5 + 1/5 of prompts
# gives exactly 60/20/20 over all slots (see module docstring).
SLOT_PATTERNS = [
    (MODE_BUDGET, MODE_BUDGET, MODE_FREE),
    (MODE_BUDGET, MODE_BUDGET, MODE_FREE),
    (MODE_BUDGET, MODE_BUDGET, MODE_NOTHINK),
    (MODE_BUDGET, MODE_BUDGET, MODE_NOTHINK),
    (MODE_BUDGET, MODE_FREE, MODE_NOTHINK),
]

def build_rows(rows: list[dict], k: int, budget_min: int, budget_max: int, seed: int) -> list[dict]:
    out = []
    for row in rows:
        idx = int(row["extra_info"].get("index", 0))
        pattern = SLOT_PATTERNS[idx % len(SLOT_PATTERNS)]
        # Seed per (source, row index) so budgets are reproducible and a rebuild
        # with the same seed is byte-identical.
        rng = random.Random(f"{seed}:{row['data_source']}:{idx}")
        for slot in range(k):
            mode = pattern[slot % len(pattern)]
            budget = rng.randint(budget_min, budget_max) if mode == MODE_BUDGET else NO_BUDGET
            out.append(
                {
                    "prompt": apply_to_messages(row["prompt"], mode, budget),
                    "data_source": row["data_source"],
                    "ability": row["ability"],
                    "reward_model": row["reward_model"],
                    "extra_info": homogenize_extra_info(row["extra_info"]),
                    "think_budget": int(budget),
                    "think_mode": mode,
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/data/abdelrahman/verl/data")
    ap.add_argument("--out_dir", default="/data/abdelrahman/verl/data/lcpo_mix")
    ap.add_argument("--k", type=int, default=3, help="rows materialized per prompt")
    ap.add_argument("--budget_min", type=int, default=200)
    ap.add_argument("--budget_max", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report = {
        "k": args.k, "seed": args.seed,
        "budget_range": [args.budget_min, args.budget_max],
        "mode_split_target": {"budget": 0.6, "free": 0.2, "nothink": 0.2},
        "by_source": {},
    }
    total = 0
    mode_counts: dict[str, int] = {MODE_BUDGET: 0, MODE_FREE: 0, MODE_NOTHINK: 0}

    for subdir, fname, base_ds in SOURCES:
        src = os.path.join(args.data_root, subdir, fname)
        table = pq.read_table(src)
        rows = table.to_pylist()
        built = build_rows(rows, args.k, args.budget_min, args.budget_max, args.seed)

        # Row identity: extra_info.index must survive so prompt_uid stays
        # comparable with every earlier mix built from these parquets.
        src_idx = [int(r["extra_info"].get("index", 0)) for r in rows]
        got_idx = sorted({int(r["extra_info"]["index"]) for r in built})
        assert got_idx == sorted(set(src_idx)), f"{fname}: extra_info.index not preserved"
        assert len(built) == len(rows) * args.k, f"{fname}: expected {len(rows)*args.k} rows"

        for r in built:
            mode_counts[r["think_mode"]] += 1
        dst = os.path.join(args.out_dir, fname)
        pq.write_table(pa.Table.from_pylist(built), dst)
        total += len(built)

        budgets = [r["think_budget"] for r in built if r["think_mode"] == MODE_BUDGET]
        report["by_source"][base_ds] = {
            "filename": fname, "source": src, "source_rows": len(rows),
            "output": dst, "output_rows": len(built),
            "budget_rows": len(budgets),
            "budget_min_seen": min(budgets) if budgets else None,
            "budget_max_seen": max(budgets) if budgets else None,
            "budget_mean": round(sum(budgets) / len(budgets), 1) if budgets else None,
        }
        print(f"  {fname}: {len(rows)} prompts -> {len(built)} rows  ({len(budgets)} budgeted)")

    report["total_rows"] = total
    report["mode_counts"] = mode_counts
    report["mode_fractions"] = {m: round(c / total, 4) for m, c in mode_counts.items()}

    with open(os.path.join(args.out_dir, "_build_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  total rows : {total}")
    print(f"  mode split : " + "  ".join(f"{m}={c} ({c/total:.1%})" for m, c in mode_counts.items()))
    print(f"  saved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
