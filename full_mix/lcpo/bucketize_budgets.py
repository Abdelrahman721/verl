"""Snap the budgets of a built LCPO parquet set onto a fixed grid, without a rebuild.

`build_lcpo_mix.py` draws every training budget uniformly from the integers in
[200, 6000], so the model sees ~14k distinct numbers, each once. This tool
re-draws every budgeted row from a small, evenly spaced grid instead —
{256, 512, ..., 6144} by default, 24 values — so each budget recurs hundreds of
times across the mix and the validation ladder sits on the same grid.

WHY REWRITE RATHER THAN REBUILD
-------------------------------
Same reason as `rephrase_budget_prompts.py`: rewriting the built set leaves every
prompt, mode, row order and `extra_info.index` bit-identical, so `prompt_uid`
still lines up with every earlier rollout dump and the runs stay comparable.
Only the number inside the budget sentence and the `think_budget` column move.
The sentence's wording ("Think for N tokens." vs "Think for a maximum of N
tokens.") is detected per row and preserved, so this runs on either style.

TRAIN  (--kind train)
    Every budgeted row gets a fresh draw from the grid. The RNG is keyed on
    (seed, data_source, extra_info.index) exactly as the builder keys its own,
    so a rerun with the same seed is byte-identical and two sets that share rows
    and indices draw the same budgets. A prompt's budgeted slots (at most two in
    the K=3 layout) are drawn WITHOUT replacement so no prompt sees the same
    bucket twice — with 24 buckets that would otherwise hit ~4% of prompts and
    hand GRPO two identical groups.

VAL  (--kind val)
    Each `val_budget_NNNNN.parquet` is one fixed rung, so nothing is sampled: the
    rung is rounded UP to the grid (4000 -> 4096, 6000 -> 6144; the power-of-two
    rungs are already on it), the file is renamed, and the `@bNNNNN` reporting
    tag on data_source is retagged to match. `val_free` and `val_nothink` are
    copied unchanged. The top rung lands on the new training maximum, which keeps
    the ladder's rule that it brackets the trained range instead of leaving it.

Non-destructive: writes a new directory and refuses to write into one that
already holds parquets (pass --force to allow it). The source is never touched.

    python -m full_mix.lcpo.bucketize_budgets --kind train \\
        --src data/lcpo_mix --dst data/lcpo_mix_bucket
    python -m full_mix.lcpo.bucketize_budgets --kind val \\
        --src data/lcpo_val --dst data/lcpo_val_bucket

then `rephrase_budget_prompts.py` on each output for the `_max` wording, exactly
as `lcpo_mix_max` / `lcpo_val_max` were made from `lcpo_mix` / `lcpo_val`.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from full_mix.common.budget_prompt import (  # noqa: E402
    MODE_BUDGET,
    budget_line,
    parse_budget_style,
    parse_suffix,
)
from full_mix.curriculum.encoding import split_variant  # noqa: E402
from full_mix.lcpo.build_val_ladder import variant_tag  # noqa: E402

_VAL_RUNG_RE = re.compile(r"^val_budget_(\d+)\.parquet$")

# Columns that this tool must never move, on any row.
_FROZEN_COLS = ("ability", "reward_model", "extra_info", "think_mode")


def make_grid(step: int, lo: int, hi: int) -> list[int]:
    if step <= 0 or lo <= 0 or hi < lo or lo % step or hi % step:
        raise SystemExit(
            f"grid needs 0 < min <= max with both multiples of step; got step={step} min={lo} max={hi}"
        )
    return list(range(lo, hi + 1, step))


def snap_up(budget: int, grid: list[int]) -> int:
    """Smallest grid value >= budget, clamped to the top of the grid."""
    for g in grid:
        if g >= budget:
            return g
    return grid[-1]


def _last_user_idx(messages: list[dict]) -> int:
    for j in range(len(messages) - 1, -1, -1):
        if messages[j].get("role") != "assistant":
            return j
    raise ValueError("no non-assistant turn to rewrite")


def rewrite_budget(messages: list[dict], old_budget: int, new_budget: int) -> list[dict]:
    """Replace the number in the budget sentence on the LAST non-assistant turn.

    Same approach as rephrase_budget_prompts: build the exact rendered sentence in
    the wording the row already uses and replace its LAST occurrence, so task text
    that happens to mention a token count is never touched.
    """
    out = [dict(m) for m in messages]
    m = out[_last_user_idx(out)]
    content = m["content"]
    mode, parsed = parse_suffix(content)
    if mode != MODE_BUDGET or parsed != int(old_budget):
        raise ValueError(
            f"prompt says ({mode}, {parsed}) but the row says (budget, {old_budget}); refusing to rewrite"
        )
    style = parse_budget_style(content)
    old = budget_line(old_budget, style)
    new = budget_line(new_budget, style)
    head, sep, tail = content.rpartition(old)
    if not sep:
        raise ValueError(f"budget sentence {old!r} not found in the last user turn")
    m["content"] = head + new + tail
    return out


def bucketize_train_file(src: str, dst: str, grid: list[int], seed: int) -> dict:
    table = pq.read_table(src)
    rows = table.to_pylist()

    # Budgeted rows grouped by prompt identity, in row order. The builder keyed
    # its RNG on (seed, data_source, index); reuse that key so a rerun with the
    # same seed is byte-identical and sibling sets with the same rows agree.
    slots: dict[tuple[str, int], list[int]] = {}
    for i, r in enumerate(rows):
        if str(r.get("think_mode") or "") != MODE_BUDGET:
            continue
        if int(r.get("think_budget", -1)) <= 0:
            raise ValueError(f"{src} row {i}: budget mode with think_budget={r.get('think_budget')!r}")
        slots.setdefault((str(r["data_source"]), int(r["extra_info"]["index"])), []).append(i)

    rewritten = 0
    for (ds, idx), positions in slots.items():
        if len(positions) > len(grid):
            raise ValueError(
                f"{src}: prompt {ds}:{idx} has {len(positions)} budgeted rows, more than the grid's {len(grid)}"
            )
        rng = random.Random(f"{seed}:{ds}:{idx}")
        for i, b in zip(positions, rng.sample(grid, k=len(positions))):
            r = rows[i]
            r["prompt"] = rewrite_budget(r["prompt"], int(r["think_budget"]), b)
            r["think_budget"] = int(b)
            rewritten += 1

    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), dst)
    return {"rows": len(rows), "rewritten": rewritten, "untouched": len(rows) - rewritten,
            "prompts_with_budget": len(slots)}


def bucketize_val_file(src: str, dst_dir: str, grid: list[int], written: set[str]) -> tuple[str, dict]:
    name = os.path.basename(src)
    m = _VAL_RUNG_RE.match(name)
    if m is None:
        dst = os.path.join(dst_dir, name)
        shutil.copyfile(src, dst)
        written.add(dst)
        n = pq.read_metadata(src).num_rows
        return dst, {"rows": n, "rewritten": 0, "untouched": n, "copied": True}

    old = int(m.group(1))
    new = snap_up(old, grid)
    dst = os.path.join(dst_dir, f"val_budget_{new:05d}.parquet")
    if dst in written:
        raise ValueError(f"{name}: rung {old} snaps to {new}, which another rung already produced")

    table = pq.read_table(src)
    rows = table.to_pylist()
    old_tag, new_tag = variant_tag(MODE_BUDGET, old), variant_tag(MODE_BUDGET, new)
    for i, r in enumerate(rows):
        if str(r.get("think_mode") or "") != MODE_BUDGET or int(r["think_budget"]) != old:
            raise ValueError(
                f"{name} row {i}: expected budget mode at {old}, got {r.get('think_mode')!r}/{r.get('think_budget')!r}"
            )
        base, tag = split_variant(str(r["data_source"]))
        if tag != old_tag:
            raise ValueError(f"{name} row {i}: data_source tag {tag!r} != {old_tag!r}")
        r["prompt"] = rewrite_budget(r["prompt"], old, new)
        r["think_budget"] = int(new)
        r["data_source"] = f"{base}@{new_tag}"

    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), dst)
    written.add(dst)
    return dst, {"rows": len(rows), "rewritten": len(rows), "untouched": 0, "rung_from": old, "rung_to": new}


def verify(src: str, dst: str, grid: list[int], kind: str) -> dict:
    """Every budgeted row in dst sits on the grid, parses back to its column,
    keeps its wording, and nothing but the number (and, for val, the reporting
    tag) moved. Non-budget rows must be identical to the source."""
    a = pq.read_table(src).to_pylist()
    b = pq.read_table(dst).to_pylist()
    if len(a) != len(b):
        raise SystemExit(f"VERIFY FAILED {dst}: {len(b)} rows, source has {len(a)}")

    def fail(i: int, why: str):
        raise SystemExit(f"VERIFY FAILED {dst} row {i}: {why}")

    hist: collections.Counter = collections.Counter()
    styles: collections.Counter = collections.Counter()
    for i, (ra, rb) in enumerate(zip(a, b)):
        mode = str(rb.get("think_mode") or "")
        for col in _FROZEN_COLS:
            if ra[col] != rb[col]:
                fail(i, f"{col} changed")
        if kind == "train" and ra["data_source"] != rb["data_source"]:
            fail(i, "data_source changed")
        if kind == "val" and split_variant(ra["data_source"])[0] != split_variant(rb["data_source"])[0]:
            fail(i, "data_source base changed")
        if mode != MODE_BUDGET:
            if ra != rb:
                fail(i, f"{mode} row changed")
            styles[f"{mode}/None"] += 1
            continue

        pa_, pb_ = ra["prompt"], rb["prompt"]
        if len(pa_) != len(pb_):
            fail(i, "message count changed")
        j = _last_user_idx(pb_)
        for jj in range(len(pa_)):
            if jj != j and pa_[jj] != pb_[jj]:
                fail(i, f"message {jj} changed")
        if pa_[j].get("role") != pb_[j].get("role"):
            fail(i, "rewritten turn changed role")

        ta, tb = pa_[j]["content"], pb_[j]["content"]
        pm, pbud = parse_suffix(tb)
        if pm != MODE_BUDGET or pbud != int(rb["think_budget"]):
            fail(i, f"prompt parses to ({pm}, {pbud}) but think_budget={rb['think_budget']}")
        if pbud not in grid:
            fail(i, f"budget {pbud} is off the grid")
        sa, sb = parse_budget_style(ta), parse_budget_style(tb)
        if sa != sb:
            fail(i, f"wording changed {sa!r} -> {sb!r}")
        ha, _, la = ta.rpartition(budget_line(int(ra["think_budget"]), sa))
        hb, _, lb = tb.rpartition(budget_line(pbud, sb))
        if ha != hb or la != lb:
            fail(i, "text around the budget sentence changed")
        if kind == "val" and split_variant(rb["data_source"])[1] != variant_tag(MODE_BUDGET, pbud):
            fail(i, f"data_source tag {rb['data_source']!r} does not match budget {pbud}")

        hist[pbud] += 1
        styles[f"{mode}/{sb}"] += 1

    return {"styles_after": dict(styles),
            "budget_hist": {int(k): v for k, v in sorted(hist.items())}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["train", "val"])
    ap.add_argument("--src", required=True, help="directory of built parquets to read")
    ap.add_argument("--dst", required=True, help="directory to write the bucketized copies")
    ap.add_argument("--step", type=int, default=256)
    ap.add_argument("--budget_min", type=int, default=256)
    ap.add_argument("--budget_max", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=42,
                    help="train only; keyed per (seed, data_source, index) like build_lcpo_mix")
    ap.add_argument("--force", action="store_true", help="write into a --dst that already holds parquets")
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("--src and --dst are the same directory; this tool never rewrites in place")
    grid = make_grid(args.step, args.budget_min, args.budget_max)

    files = sorted(glob.glob(os.path.join(args.src, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files in {args.src}")
    if glob.glob(os.path.join(args.dst, "*.parquet")) and not args.force:
        raise SystemExit(f"{args.dst} already holds parquets; pass --force to overwrite them")
    os.makedirs(args.dst, exist_ok=True)

    report = {"kind": args.kind, "src": args.src, "dst": args.dst,
              "grid": {"step": args.step, "min": args.budget_min, "max": args.budget_max,
                       "n_values": len(grid), "values": grid},
              "seed": args.seed if args.kind == "train" else None,
              "files": {}}
    written: set[str] = set()
    for src in files:
        name = os.path.basename(src)
        if args.kind == "train":
            dst = os.path.join(args.dst, name)
            stats = bucketize_train_file(src, dst, grid, args.seed)
        else:
            dst, stats = bucketize_val_file(src, args.dst, grid, written)
        stats.update(verify(src, dst, grid, args.kind))
        report["files"][name] = {"output": dst, **stats}
        extra = f"  rung {stats['rung_from']} -> {stats['rung_to']}" if "rung_from" in stats else ""
        print(f"  {name}: {stats['rewritten']} rewritten, {stats.get('untouched', 0)} untouched{extra}"
              f" -> {dst}")

    hist: collections.Counter = collections.Counter()
    for v in report["files"].values():
        hist.update(v["budget_hist"])
    total = sum(hist.values())
    report["budget_rows"] = total
    report["budget_hist"] = {int(k): v for k, v in sorted(hist.items())}
    report["budget_min_seen"] = min(hist) if hist else None
    report["budget_max_seen"] = max(hist) if hist else None
    report["budget_mean"] = round(sum(k * v for k, v in hist.items()) / total, 1) if total else None

    with open(os.path.join(args.dst, "_bucketize_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{total} budgeted rows on a {len(grid)}-value grid "
          f"[{args.budget_min}..{args.budget_max} step {args.step}]; mean {report['budget_mean']}")
    if args.kind == "train":
        print("  per-bucket counts: " + "  ".join(f"{k}:{v}" for k, v in report["budget_hist"].items()))
    else:
        outs = [v["output"] for v in report["files"].values()]
        rungs = sorted(p for p in outs if _VAL_RUNG_RE.match(os.path.basename(p)))
        rest = [p for p in outs if p not in rungs]
        print("\n  VAL_FILES=[" + ",".join(rungs + rest) + "]")
    print(f"saved -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
