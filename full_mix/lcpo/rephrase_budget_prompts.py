"""Rewrite the budget sentence in an existing LCPO parquet set, in place of a rebuild.

Stage b enforces a ceiling, not a target, so the prompt has to say so — otherwise
the model cannot tell the two rules apart and stage b just overwrites stage a's
behaviour on byte-identical text. See `budget_prompt.STYLE_EXACT` / `STYLE_MAX`.

    Think for 1234 tokens.               ->  Think for a maximum of 1234 tokens.

WHY REWRITE RATHER THAN REBUILD
-------------------------------
`build_lcpo_mix.py` needs the upstream parquets under `chat_ifeval_mix2/` and
`chat_ifeval_math_mix/`, and those are no longer on disk — only their
`_build_report.json` stubs remain. Rewriting the built mix also has a property a
rebuild does not: every prompt, budget, mode, row order and `extra_info.index`
stays bit-identical, so `prompt_uid` still lines up with every stage-a rollout
dump and the two runs remain directly comparable. The only thing that changes is
the wording of one sentence.

Non-destructive: writes a new directory and leaves the source untouched, so the
stage-a data survives for comparison and a bad rewrite costs nothing.

    python -m full_mix.lcpo.rephrase_budget_prompts \\
        --src data/lcpo_mix --dst data/lcpo_mix_max
    python -m full_mix.lcpo.rephrase_budget_prompts \\
        --src data/lcpo_val --dst data/lcpo_val_max

Needs pyarrow, so run it inside the container (`bash dev/dev.sh`), where the repo
is mounted at /workspace/verl.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from full_mix.common.budget_prompt import (  # noqa: E402
    MODE_BUDGET,
    STYLE_EXACT,
    STYLE_MAX,
    budget_line,
    parse_budget_style,
)


def rewrite_messages(messages, budget: int, src_style: str, dst_style: str):
    """Swap the budget sentence on the LAST non-assistant turn.

    Targets the exact rendered sentence rather than a regex, so a task that
    happens to contain the words "Think for 5 tokens." is never touched — only
    the suffix this module put there.
    """
    old = budget_line(budget, src_style)
    new = budget_line(budget, dst_style)
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "assistant":
            continue
        content = m["content"]
        if not content.rstrip().endswith("/think"):
            raise ValueError("budget row does not end with the /think marker")
        # rsplit: replace the LAST occurrence, which is the suffix we appended.
        head, sep, tail = content.rpartition(old)
        if not sep:
            raise ValueError(f"budget sentence {old!r} not found in the last user turn")
        m["content"] = head + new + tail
        return out
    raise ValueError("no non-assistant turn to rewrite")


def convert_file(src: str, dst: str, src_style: str, dst_style: str) -> dict:
    table = pq.read_table(src)
    rows = table.to_pylist()
    touched = skipped = 0
    for r in rows:
        mode = str(r.get("think_mode", "") or "")
        budget = int(r.get("think_budget", -1))
        if mode != MODE_BUDGET or budget <= 0:
            skipped += 1
            continue
        r["prompt"] = rewrite_messages(r["prompt"], budget, src_style, dst_style)
        touched += 1
    out = table.__class__.from_pylist(rows, schema=table.schema)
    pq.write_table(out, dst)
    return {"rows": len(rows), "rewritten": touched, "untouched": skipped}


def verify(path: str, dst_style: str) -> dict:
    """Every budgeted row must now carry the new wording, and nothing else may."""
    rows = pq.read_table(path).to_pylist()
    seen: dict[str, int] = {}
    for r in rows:
        mode = str(r.get("think_mode", "") or "")
        last = next(m for m in reversed(r["prompt"]) if m.get("role") != "assistant")
        style = parse_budget_style(last["content"])
        key = f"{mode}/{style}"
        seen[key] = seen.get(key, 0) + 1
    bad = {k: v for k, v in seen.items()
           if k.startswith(MODE_BUDGET) and not k.endswith(dst_style)}
    if bad:
        raise SystemExit(f"VERIFY FAILED in {path}: budgeted rows with wrong wording: {bad}")
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of built parquets to read")
    ap.add_argument("--dst", required=True, help="directory to write the rephrased copies")
    ap.add_argument("--from_style", default=STYLE_EXACT, choices=[STYLE_EXACT, STYLE_MAX])
    ap.add_argument("--to_style", default=STYLE_MAX, choices=[STYLE_EXACT, STYLE_MAX])
    args = ap.parse_args()

    if args.from_style == args.to_style:
        raise SystemExit("--from_style and --to_style are the same; nothing to do")
    os.makedirs(args.dst, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.src, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files in {args.src}")

    report = {"src": args.src, "dst": args.dst,
              "from_style": args.from_style, "to_style": args.to_style, "files": {}}
    for src in files:
        name = os.path.basename(src)
        dst = os.path.join(args.dst, name)
        stats = convert_file(src, dst, args.from_style, args.to_style)
        stats["styles_after"] = verify(dst, args.to_style)
        report["files"][name] = stats
        print(f"  {name}: {stats['rewritten']} rewritten, {stats['untouched']} untouched "
              f"-> {dst}")

    with open(os.path.join(args.dst, "_rephrase_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    total = sum(v["rewritten"] for v in report["files"].values())
    print(f"\nrewrote {total} budgeted prompts across {len(files)} files")
    print(f"saved -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
