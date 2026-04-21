#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatchRef:
    line_no: int
    file_pos: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Interactively browse v12_set_correction rows in a VERL jsonl dump and print the exact "
            "DeepSeek judge prompt (system + user) used by the instruction-following framework."
        )
    )
    p.add_argument(
        "--jsonl",
        default="/data/muhsen/repos/verl/verl_dumps/val/470.jsonl",
        help="Path to the VERL jsonl file.",
    )
    p.add_argument(
        "--labels-path",
        default=None,
        help=(
            "Optional path to labels CSV (set_B1). If omitted, uses the same default resolution as "
            "verl.instruction_following.utils.get_labels()."
        ),
    )
    return p.parse_args()


def _load_row_at(f, ref: MatchRef) -> dict:
    f.seek(ref.file_pos)
    line = f.readline()
    if not line:
        raise RuntimeError(f"Failed to read line at pos={ref.file_pos} (line_no={ref.line_no})")
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON at line {ref.line_no}: {e}") from e


def _extract_gts(row: dict) -> dict:
    gts = row.get("gts")
    if isinstance(gts, str):
        try:
            return json.loads(gts)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Row has non-JSON gts string: {e}") from e
    if isinstance(gts, dict):
        return gts
    raise RuntimeError(f"Row missing gts or wrong type: {type(gts)}")


def _score_to_yes_no(score: object) -> str:
    try:
        v = float(score)
    except Exception:
        return "UNKNOWN"
    if v == 1.0:
        return "yes"
    if v == 0.0:
        return "no"
    return f"UNKNOWN({v})"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.getenv("NO_COLOR", "").strip() == ""


def _color(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _highlight_ids_in_text(
    text: str,
    *,
    green_ids: set[str],
    red_ids: set[str],
) -> str:
    """
    Highlight whole-word occurrences:
    - first occurrence: green_ids in green, red_ids in red
    - subsequent occurrences of the same ID: yellow
    """
    if not text:
        return text

    ids = set(green_ids) | set(red_ids)
    if not ids:
        return text

    # Sort by length desc so we don't partially wrap longer numeric strings (defensive).
    ids_sorted = sorted(ids, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, ids_sorted)) + r")\b")

    seen_counts: dict[str, int] = {}

    def repl(m: re.Match) -> str:
        sid = m.group(1)
        seen_counts[sid] = seen_counts.get(sid, 0) + 1
        if seen_counts[sid] > 1:
            return _color(sid, "33")  # yellow for repeats
        if sid in green_ids:
            return _color(sid, "32")  # green
        if sid in red_ids:
            return _color(sid, "31")  # red
        return sid

    return pattern.sub(repl, text)


def _build_v12_prompt(*, gts: dict, assistant_response: str, labels_path: str | None) -> tuple[str, str]:
    if gts.get("rule") != "v12_set_correction":
        raise RuntimeError(f"Expected rule v12_set_correction, got: {gts.get('rule')!r}")

    def load_module(mod_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed creating module spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # Avoid importing the top-level `verl` package (can pull heavy deps like numpy).
    repo_root = Path(__file__).resolve().parent
    judges_mod = load_module(
        "_if_judges",
        repo_root / "instruction_following" / "judges.py",
    )
    utils_mod = load_module(
        "_if_utils",
        repo_root / "instruction_following" / "utils.py",
    )

    lp = Path(labels_path) if labels_path else None
    labels = utils_mod.get_labels(lp)
    assistant_text_for_judge = utils_mod.get_answer_text(assistant_response)

    gold_set = {str(x) for x in (gts.get("original_codes") or [])}
    proposed_set = {str(x) for x in (gts.get("processed_codes") or [])}

    gt_sorted = sorted(gold_set, key=int)
    ground_truth_codes = ", ".join(f"{g} ({labels[g]})" for g in gt_sorted)
    prop_sorted = sorted(proposed_set, key=int)
    proposed_for_judge = ", ".join(f"{p} ({labels[p]})" for p in prop_sorted)

    system_message = judges_mod.JUDGE_V12_SYSTEM
    user_message = judges_mod.judge_v12_user(
        ground_truth_codes,
        proposed_for_judge,
        assistant_text_for_judge,
    )
    return system_message, user_message


def _build_refs(path: str) -> list[MatchRef]:
    refs: list[MatchRef] = []
    # In this dump, the per-example metadata is stored under "gts" as a JSON string, so the
    # literal text in the jsonl line will look like: \"rule\": \"v12_set_correction\".
    needle = "v12_set_correction"
    with open(path, "r", encoding="utf-8") as f:
        line_no = 0
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            line_no += 1
            if needle in line:
                refs.append(MatchRef(line_no=line_no, file_pos=pos))
    return refs


def _visualize_answer_against_ground_truth(*, gts: dict, assistant_raw_output: str) -> None:
    """
    Visual view of answer text:
    - highlight ground-truth IDs that appear in the answer in green
    - show ground-truth IDs that are missing from the answer
    """
    repo_root = Path(__file__).resolve().parent

    def load_module(mod_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed creating module spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    utils_mod = load_module(
        "_if_utils_for_visual",
        repo_root / "instruction_following" / "utils.py",
    )

    answer_text = utils_mod.get_answer_text(str(assistant_raw_output or ""))
    gt_ids = {str(x) for x in (gts.get("original_codes") or [])}

    present = {sid for sid in gt_ids if re.search(rf"\b{re.escape(sid)}\b", answer_text)}
    missing = sorted(gt_ids - present, key=int) if gt_ids else []

    # Any other SNOMED-shaped IDs (6–18 digits) in the answer that are not in ground truth.
    all_ids_in_answer = set(re.findall(r"\b(\d{6,18})\b", answer_text))
    other_ids = all_ids_in_answer - gt_ids

    print("-" * 110)
    print("ANSWER_VIEW:")
    print(
        _highlight_ids_in_text(
            answer_text.rstrip(),
            green_ids=present,
            red_ids=other_ids,
        )
    )

    print("-" * 110)
    print("GROUND_TRUTH_ID_COVERAGE:")
    if gt_ids:
        present_sorted = sorted(present, key=int)
        if present_sorted:
            print("present_in_answer:")
            for sid in present_sorted:
                print("  " + _color(sid, "32"))
        else:
            print("present_in_answer: (none)")

        if missing:
            # Use yellow for missing so green remains reserved for "present".
            print("missing_from_answer:")
            for sid in missing:
                print("  " + _color(sid, "33"))
        else:
            print("missing_from_answer: (none)")
    else:
        print("(no ground truth IDs found)")


def _print_record(*, idx: int, total: int, ref: MatchRef, row: dict, system_message: str, user_message: str) -> None:
    score = row.get("score")
    yn = _score_to_yes_no(score)
    step = row.get("step")
    acc = row.get("acc")
    reward = row.get("reward")
    print("=" * 110)
    print(f"[{idx + 1}/{total}] jsonl_line={ref.line_no}  step={step}  score={score} -> {yn}  acc={acc}  reward={reward}")
    print("-" * 110)
    print("SYSTEM_MESSAGE:")
    print(system_message.rstrip())
    print("USER_MESSAGE:")
    print(user_message.rstrip())
    print("=" * 110)


def main() -> int:
    args = _parse_args()
    jsonl_path = args.jsonl
    labels_path = args.labels_path

    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}", file=sys.stderr)
        return 2

    refs = _build_refs(jsonl_path)
    if not refs:
        print("No rows found with rule v12_set_correction.")
        return 0

    i = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        while True:
            ref = refs[i]
            row = _load_row_at(f, ref)
            gts = _extract_gts(row)
            assistant_response = str(row.get("output") or "")
            system_message, user_message = _build_v12_prompt(
                gts=gts,
                assistant_response=assistant_response,
                labels_path=labels_path,
            )
            _print_record(
                idx=i,
                total=len(refs),
                ref=ref,
                row=row,
                system_message=system_message,
                user_message=user_message,
            )
            _visualize_answer_against_ground_truth(gts=gts, assistant_raw_output=assistant_response)

            cmd = input("Command [n=next, p=prev, j <k>=jump, q=quit] > ").strip()
            if cmd in ("q", "quit", "exit"):
                return 0
            if cmd == "" or cmd == "n":
                i = (i + 1) % len(refs)
                continue
            if cmd == "p":
                i = (i - 1) % len(refs)
                continue
            if cmd.startswith("j"):
                parts = cmd.split()
                if len(parts) == 2 and parts[1].isdigit():
                    k = int(parts[1])
                    if 1 <= k <= len(refs):
                        i = k - 1
                        continue
                print(f"Invalid jump. Use: j <1..{len(refs)}>")
                continue
            print("Unknown command.")


if __name__ == "__main__":
    raise SystemExit(main())

