"""Build a SECOND chat + ifeval mix, disjoint from the first, plus the leftovers.

Partitions each source parquet into three non-overlapping pieces:

    used      rows already sampled into data/chat_ifeval_math_mix (read from
              that build's _build_report.json — not re-derived from a seed)
    mix2      newly sampled rows, drawn only from what `used` left behind
    remaining everything else, written out as its own dataset for later use

    used + mix2 + remaining == every row in the source, with no overlap.

Both invariants are asserted before anything is written, so a stale or wrong
exclusion report fails loudly instead of silently producing an overlapping mix.

Defaults: 2000 chat + 4000 ifeval into the new mix; the rest (16708 chat /
21847 ifeval) into the remaining dir.

Row identity and schema are preserved exactly as in
``build_chat_ifeval_math_mix`` — rows are sliced with ``.take()`` and
``extra_info.index`` keeps pointing at the source row, so ``prompt_uid`` stays
comparable across every mix built from these parquets.

Usage:
    python -m full_mix.curriculum.build_chat_ifeval_mix2
    python -m full_mix.curriculum.build_chat_ifeval_mix2 \\
        --n_chat 2000 --n_ifeval 4000 --seed 43 \\
        --exclude_report /data/abdelrahman/verl/data/chat_ifeval_math_mix/_build_report.json
"""

import argparse
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

from full_mix.curriculum.build_chat_ifeval_math_mix import check_row_identity
from full_mix.curriculum.encoding import DATASET_ID, UID_DATASET_STRIDE

# base data_source -> (source parquet filename, CLI arg holding the count)
SOURCES: dict[str, tuple[str, str]] = {
    "local/dolci-chat-32b": ("chat_with_baseline_train.parquet", "n_chat"),
    "local/dolci-ifeval-32b": ("ifeval_train.parquet", "n_ifeval"),
}


def load_used_indices(report_paths: list[str]) -> dict[str, set[int]]:
    """Collect already-sampled row indices per base data_source.

    Accepts several reports so further mixes can exclude every earlier one by
    passing --exclude_report repeatedly.
    """
    used: dict[str, set[int]] = {}
    for path in report_paths:
        if not os.path.exists(path):
            raise SystemExit(f"exclusion report not found: {path}")
        with open(path) as f:
            report = json.load(f)
        for base_ds, entry in report.get("by_source", {}).items():
            idx = entry.get("row_indices")
            if idx is None:
                raise SystemExit(
                    f"{path} has no row_indices for {base_ds}; cannot prove disjointness"
                )
            used.setdefault(base_ds, set()).update(int(i) for i in idx)
        print(f"  loaded exclusions from {path}")
    return used


def partition(
    n_rows: int, used: set[int], n_wanted: int, seed: int
) -> tuple[list[int], list[int]]:
    """Split [0, n_rows) minus `used` into (newly sampled, remaining)."""
    stale = {i for i in used if i >= n_rows}
    if stale:
        raise SystemExit(
            f"exclusion report references {len(stale)} row indices >= n_rows={n_rows}; "
            f"the source parquet has changed since that mix was built"
        )
    available = sorted(set(range(n_rows)) - used)
    if n_wanted > len(available):
        raise SystemExit(
            f"requested {n_wanted} prompts but only {len(available)} unused rows remain "
            f"({n_rows} total - {len(used)} already sampled)"
        )
    rng = random.Random(seed)
    picked = sorted(rng.sample(available, n_wanted))
    remaining = sorted(set(available) - set(picked))
    return picked, remaining


def write_subset(table: pa.Table, indices: list[int], dst: str, base_ds: str) -> int:
    sub = table.take(pa.array(indices, type=pa.int64()))
    check_row_identity(sub, indices, base_ds)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(sub, dst)
    return sub.num_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_train_dir", default="/data/abdelrahman/verl/data/full_mix/train")
    ap.add_argument("--mix_dir", default="/data/abdelrahman/verl/data/chat_ifeval_mix2")
    ap.add_argument("--remaining_dir", default="/data/abdelrahman/verl/data/chat_ifeval_remaining")
    ap.add_argument(
        "--exclude_report",
        action="append",
        default=None,
        help="Build report whose row_indices must be excluded. Repeatable. "
        "Defaults to the chat_ifeval_math_mix report.",
    )
    ap.add_argument("--n_chat", type=int, default=2000)
    ap.add_argument("--n_ifeval", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument(
        "--chat_variant",
        choices=["chat", "chat_with_baseline"],
        default="chat_with_baseline",
    )
    args = ap.parse_args()

    reports = args.exclude_report or [
        "/data/abdelrahman/verl/data/chat_ifeval_math_mix/_build_report.json"
    ]

    sources = dict(SOURCES)
    if args.chat_variant == "chat":
        sources["local/dolci-chat-32b"] = ("chat_train.parquet", "n_chat")

    print("Loading exclusions …")
    used_by_source = load_used_indices(reports)

    summary: dict[str, dict] = {}
    for base_ds, (filename, count_arg) in sources.items():
        n_wanted = getattr(args, count_arg)
        dataset_id = DATASET_ID[base_ds]
        src = os.path.join(args.source_train_dir, filename)
        table = pq.read_table(src)
        used = used_by_source.get(base_ds, set())

        picked, remaining = partition(table.num_rows, used, n_wanted, seed=args.seed + dataset_id)

        # Invariants: the three pieces must partition the source exactly.
        assert not (set(picked) & used), "new mix overlaps the excluded rows"
        assert not (set(picked) & set(remaining)), "new mix overlaps the remaining rows"
        assert not (used & set(remaining)), "remaining overlaps the excluded rows"
        assert len(used) + len(picked) + len(remaining) == table.num_rows, "partition is not exhaustive"

        mix_dst = os.path.join(args.mix_dir, filename)
        rem_dst = os.path.join(args.remaining_dir, filename)
        n_mix = write_subset(table, picked, mix_dst, base_ds)
        n_rem = write_subset(table, remaining, rem_dst, base_ds)

        print(f"\n{base_ds}  (dataset_id={dataset_id})")
        print(f"  source            : {src}  ({table.num_rows} rows)")
        print(f"  already sampled   : {len(used)}  (excluded)")
        print(f"  new mix           : {n_mix}  -> {mix_dst}")
        print(f"  remaining         : {n_rem}  -> {rem_dst}")
        print(f"  partition check   : {len(used)} + {n_mix} + {n_rem} = {table.num_rows}  OK")

        summary[base_ds] = {
            "filename": filename,
            "source": src,
            "source_rows": table.num_rows,
            "excluded_rows": len(used),
            "mix_output": mix_dst,
            "mix_rows": n_mix,
            "remaining_output": rem_dst,
            "remaining_rows": n_rem,
            "seed": args.seed + dataset_id,
            "row_indices": picked,
            "remaining_row_indices": remaining,
            "uid_stride": UID_DATASET_STRIDE,
        }

    print(f"\nnew mix total     : {sum(s['mix_rows'] for s in summary.values())} prompts")
    print(f"remaining total   : {sum(s['remaining_rows'] for s in summary.values())} prompts")

    # The mix report keeps the same shape as build_chat_ifeval_math_mix's, so a
    # third mix can exclude both by passing --exclude_report twice.
    os.makedirs(args.mix_dir, exist_ok=True)
    mix_report = os.path.join(args.mix_dir, "_build_report.json")
    with open(mix_report, "w") as f:
        json.dump(
            {
                "source_train_dir": args.source_train_dir,
                "chat_variant": args.chat_variant,
                "seed": args.seed,
                "excluded_reports": reports,
                "total_prompts": sum(s["mix_rows"] for s in summary.values()),
                "by_source": {
                    k: {kk: vv for kk, vv in v.items() if kk != "remaining_row_indices"}
                    for k, v in summary.items()
                },
            },
            f,
            indent=2,
        )
    print(f"wrote report -> {mix_report}")

    os.makedirs(args.remaining_dir, exist_ok=True)
    rem_report = os.path.join(args.remaining_dir, "_build_report.json")
    with open(rem_report, "w") as f:
        json.dump(
            {
                "source_train_dir": args.source_train_dir,
                "chat_variant": args.chat_variant,
                "note": "Every row not used by chat_ifeval_math_mix or chat_ifeval_mix2.",
                "excluded_reports": reports + [mix_report],
                "total_prompts": sum(s["remaining_rows"] for s in summary.values()),
                "by_source": {
                    k: {
                        "filename": v["filename"],
                        "source": v["source"],
                        "source_rows": v["source_rows"],
                        "output": v["remaining_output"],
                        "rows": v["remaining_rows"],
                        "row_indices": v["remaining_row_indices"],
                    }
                    for k, v in summary.items()
                },
            },
            f,
            indent=2,
        )
    print(f"wrote report -> {rem_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
