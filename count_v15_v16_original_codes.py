#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter


TARGETS = ("160352002", "161659007")
RULES = ("v15_description_from_code", "v16_code_from_description")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Count rows whose gts.original_codes contains specific SNOMED IDs. "
            "Reports both v15/v16-only counts and counts across all rules."
        )
    )
    p.add_argument(
        "--jsonl",
        default="/data/muhsen/repos/verl/verl_dumps/val/470.jsonl",
        help="Path to a VERL jsonl dump containing a 'gts' field.",
    )
    return p.parse_args()


def _iter_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield line_no, row


def _get_gts(row: dict) -> dict | None:
    gts = row.get("gts")
    if isinstance(gts, dict):
        return gts
    if isinstance(gts, str):
        try:
            return json.loads(gts)
        except json.JSONDecodeError:
            return None
    return None


def main() -> int:
    args = _parse_args()

    v15v16_total = Counter()
    v15v16_by_rule: dict[str, Counter] = {r: Counter() for r in RULES}
    v15v16_seen = 0

    all_total = Counter()
    all_by_rule: dict[str, Counter] = {}
    all_seen = 0

    for _line_no, row in _iter_rows(args.jsonl):
        gts = _get_gts(row)
        if not gts:
            continue
        rule = gts.get("rule")
        if not isinstance(rule, str) or not rule:
            continue

        orig = gts.get("original_codes") or []
        orig_set = {str(x) for x in orig}

        # Across all rules
        all_seen += 1
        if rule not in all_by_rule:
            all_by_rule[rule] = Counter()
        for t in TARGETS:
            if t in orig_set:
                all_total[t] += 1
                all_by_rule[rule][t] += 1

        # v15/v16 subset
        if rule in RULES:
            v15v16_seen += 1
            for t in TARGETS:
                if t in orig_set:
                    v15v16_total[t] += 1
                    v15v16_by_rule[rule][t] += 1

    print(f"rows_considered_all_rules: {all_seen}")
    for t in TARGETS:
        print(f"{t}: {all_total[t]}")

    print("--- by rule (all rules) ---")
    for r in sorted(all_by_rule):
        print(r)
        for t in TARGETS:
            print(f"  {t}: {all_by_rule[r][t]}")

    print("--- v15/v16 subset ---")
    print(f"rows_considered_v15_v16: {v15v16_seen}")
    for t in TARGETS:
        print(f"{t}: {v15v16_total[t]}")
    print("--- by rule (v15/v16) ---")
    for r in RULES:
        print(r)
        for t in TARGETS:
            print(f"  {t}: {v15v16_by_rule[r][t]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

