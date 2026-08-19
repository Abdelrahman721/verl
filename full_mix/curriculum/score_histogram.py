"""Per-prompt score histogram for a curriculum stage's rollouts.

Reads every JSONL under ``rollout_dir``, groups records by ``prompt_uid``,
takes each prompt's mean rollout score, then prints:
  1. A 5%-bucket histogram across ALL prompts.
  2. The same histogram broken out per base data source (ifeval / chat / safety).

Usage:
    python -m full_mix.curriculum.score_histogram --stage 1
    python -m full_mix.curriculum.score_histogram --rollout_dir ~/some/path
    python -m full_mix.curriculum.score_histogram --stage 2 --buckets 10
"""

import argparse
import json
import os
import statistics
from collections import defaultdict
from glob import glob

from full_mix.curriculum.encoding import decode_data_source


DEFAULT_BUCKETS = 20  # 5% each


def _iter_records(rollout_dir: str):
    paths = sorted(glob(os.path.join(rollout_dir, "*.jsonl")))
    if not paths:
        raise SystemExit(f"no *.jsonl files under {rollout_dir}")
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def load_per_prompt(rollout_dir: str):
    """Returns {prompt_uid: (mean_score, base_data_source, n_rollouts)}."""
    scores_by_uid: dict[int, list[float]] = defaultdict(list)
    base_by_uid: dict[int, str] = {}
    for rec in _iter_records(rollout_dir):
        if "prompt_uid" not in rec or "data_source_code" not in rec:
            raise SystemExit(
                f"rollout record missing prompt_uid/data_source_code "
                f"(keys={sorted(rec.keys())}). This dump is from a pre-curriculum "
                f"run; only dumps produced after the prompt_uid/data_source_code "
                f"plumbing was added are supported."
            )
        puid = int(rec["prompt_uid"])
        scores_by_uid[puid].append(float(rec["score"]))
        if puid not in base_by_uid:
            ds_code = int(rec["data_source_code"])
            try:
                base, _shard = decode_data_source(ds_code)
            except ValueError:
                base = f"unknown(code={ds_code})"
            base_by_uid[puid] = base
    return {
        uid: (statistics.mean(scores), base_by_uid[uid], len(scores))
        for uid, scores in scores_by_uid.items()
    }


def histogram_counts(values, n_buckets: int) -> list[int]:
    """Bucket i covers ``[i/n, (i+1)/n)``; the last bucket includes 1.0."""
    buckets = [0] * n_buckets
    for v in values:
        v = max(0.0, min(1.0, v))
        idx = min(int(v * n_buckets), n_buckets - 1)
        buckets[idx] += 1
    return buckets


def print_histogram(title: str, values: list[float], n_buckets: int, bar_width: int = 40):
    print(f"\n=== {title} ===")
    if not values:
        print("  (no prompts)")
        return
    print(f"  prompts={len(values)}  mean={statistics.mean(values):.3f}  "
          f"median={statistics.median(values):.3f}  min={min(values):.3f}  max={max(values):.3f}")
    counts = histogram_counts(values, n_buckets)
    total = len(values)
    max_count = max(counts) or 1
    width = len(str(max_count))
    for i, c in enumerate(counts):
        lo = i / n_buckets
        hi = (i + 1) / n_buckets
        bracket = ")" if i < n_buckets - 1 else "]"
        bar = "#" * round(c / max_count * bar_width)
        pct = 100.0 * c / total if total else 0.0
        print(f"  [{lo:0.2f}, {hi:0.2f}{bracket}  {bar:<{bar_width}}  {c:>{width}}  ({pct:5.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1,
                    help="Curriculum stage; used to derive default rollout_dir.")
    ap.add_argument("--rollout_dir", default=None,
                    help="Explicit rollout dir. Defaults to ~/verl_dumps/curriculum_stage{stage}_rollouts.")
    ap.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS,
                    help=f"Number of histogram buckets (default {DEFAULT_BUCKETS} = 5%% each).")
    ap.add_argument("--bar_width", type=int, default=40,
                    help="Max width of the ASCII bar (default 40 chars).")
    args = ap.parse_args()

    rollout_dir = args.rollout_dir or os.path.expanduser(
        f"~/verl_dumps/curriculum_stage{args.stage}_rollouts"
    )
    if not os.path.isdir(rollout_dir):
        raise SystemExit(f"rollout dir not found: {rollout_dir}")

    per_prompt = load_per_prompt(rollout_dir)
    total_rollouts = sum(n for _, _, n in per_prompt.values())
    print(f"loaded {len(per_prompt)} unique prompts ({total_rollouts} rollouts) "
          f"from {rollout_dir}")

    all_means = [m for m, _, _ in per_prompt.values()]
    print_histogram("ALL prompts", all_means, n_buckets=args.buckets, bar_width=args.bar_width)

    by_source: dict[str, list[float]] = defaultdict(list)
    for mean, base, _n in per_prompt.values():
        by_source[base].append(mean)
    for base in sorted(by_source):
        print_histogram(base, by_source[base], n_buckets=args.buckets, bar_width=args.bar_width)


if __name__ == "__main__":
    main()
