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
            "Interactively browse v11_code_desc_consistency rows in a VERL jsonl dump and print the "
            "exact judge prompt (system + user) used by the instruction-following framework, plus a "
            "compact visualization tailored to v11."
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
            "instruction_following/utils.py get_labels()."
        ),
    )
    return p.parse_args()


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.getenv("NO_COLOR", "").strip() == ""


def _color(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


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


def _load_module(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed creating module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def _load_row_at(f, ref: MatchRef) -> dict:
    f.seek(ref.file_pos)
    line = f.readline()
    if not line:
        raise RuntimeError(f"Failed to read line at pos={ref.file_pos} (line_no={ref.line_no})")
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON at line {ref.line_no}: {e}") from e


def _build_refs(path: str) -> list[MatchRef]:
    refs: list[MatchRef] = []
    needle = "v11_code_desc_consistency"
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


def _build_v11_prompt(*, gts: dict, assistant_raw_output: str, labels_path: str | None) -> tuple[str, str, dict]:
    """
    Reconstruct v11 judge prompt exactly as in rules_eval.py:
      - text = get_answer_text(response)
      - gold_id = original_codes[0]
      - src_id = processed_codes[0]
      - shown_desc = labels[src_id]
      - gold_desc = labels[gold_id]
      - call_judge(JUDGE_V11_SYSTEM, judge_v11_user(gold_id, shown_desc, gold_desc, text))
    """
    if gts.get("rule") != "v11_code_desc_consistency":
        raise RuntimeError(f"Expected rule v11_code_desc_consistency, got: {gts.get('rule')!r}")

    repo_root = Path(__file__).resolve().parent
    judges_mod = _load_module("_if_judges_v11", repo_root / "instruction_following" / "judges.py")
    utils_mod = _load_module("_if_utils_v11", repo_root / "instruction_following" / "utils.py")

    lp = Path(labels_path) if labels_path else None
    labels = utils_mod.get_labels(lp)
    assistant_text_for_judge = utils_mod.get_answer_text(str(assistant_raw_output or ""))

    orig = gts.get("original_codes") or []
    proc = gts.get("processed_codes") or []
    if len(orig) != 1 or len(proc) != 1:
        raise RuntimeError(f"v11 expects single-label: original_codes={orig}, processed_codes={proc}")

    gold_id = str(orig[0])
    src_id = str(proc[0])
    shown_desc = labels[src_id]
    gold_desc = labels[gold_id]

    system_message = judges_mod.JUDGE_V11_SYSTEM
    user_message = judges_mod.judge_v11_user(gold_id, shown_desc, gold_desc, assistant_text_for_judge)
    meta = {
        "gold_id": gold_id,
        "src_id": src_id,
        "shown_desc": shown_desc,
        "gold_desc": gold_desc,
        "assistant_text": assistant_text_for_judge,
    }
    return system_message, user_message, meta


def _highlight_answer_v11(answer_text: str, *, gold_id: str, src_id: str) -> str:
    """
    In-place answer highlighting:
    - first occurrence of gold_id: green
    - first occurrence of src_id (if different): cyan
    - any other SNOMED-shaped IDs: red
    - repeats of any ID after first: yellow
    """
    if not answer_text:
        return answer_text

    ids_in_answer = set(re.findall(r"\b(\d{6,18})\b", answer_text))
    other_ids = ids_in_answer - {gold_id, src_id}

    candidates = {gold_id, src_id} | other_ids
    if not candidates:
        return answer_text

    ids_sorted = sorted(candidates, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, ids_sorted)) + r")\b")

    seen: dict[str, int] = {}

    def repl(m: re.Match) -> str:
        sid = m.group(1)
        seen[sid] = seen.get(sid, 0) + 1
        if seen[sid] > 1:
            return _color(sid, "33")  # yellow repeats
        if sid == gold_id:
            return _color(sid, "32")  # green
        if sid == src_id and src_id != gold_id:
            return _color(sid, "36")  # cyan
        return _color(sid, "31")  # red

    return pattern.sub(repl, answer_text)


def _print_prompt(*, idx: int, total: int, ref: MatchRef, row: dict, system_message: str, user_message: str) -> None:
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


def _visualize_v11(meta: dict) -> None:
    gold_id = meta["gold_id"]
    src_id = meta["src_id"]
    shown_desc = meta["shown_desc"]
    gold_desc = meta["gold_desc"]
    answer_text = meta["assistant_text"]

    same_id = gold_id == src_id
    print("-" * 110)
    print("PAIR_VIEW:")
    print(f"shown_to_assistant:  ID {src_id}")
    print(f"                 desc {shown_desc}")
    print(f"ground_truth:       ID {gold_id}")
    print(f"                 desc {gold_desc}")
    if not same_id:
        print(f"relationship: shown_id != gold_id  ({_color('mismatch', '31')})")
    else:
        print("relationship: shown_id == gold_id")

    print("-" * 110)
    print("ANSWER_VIEW:")
    print(_highlight_answer_v11(answer_text.rstrip(), gold_id=gold_id, src_id=src_id))

    print("-" * 110)
    print("COVERAGE:")
    gold_present = bool(re.search(rf"\b{re.escape(gold_id)}\b", answer_text))
    src_present = bool(re.search(rf"\b{re.escape(src_id)}\b", answer_text))
    print(f"mentions_gold_id: {gold_present}")
    print(f"mentions_shown_id: {src_present}")


def main() -> int:
    args = _parse_args()
    jsonl_path = args.jsonl
    labels_path = args.labels_path

    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}", file=sys.stderr)
        return 2

    refs = _build_refs(jsonl_path)
    if not refs:
        print("No rows found with rule v11_code_desc_consistency.")
        return 0

    i = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        while True:
            ref = refs[i]
            row = _load_row_at(f, ref)
            gts = _extract_gts(row)
            assistant_raw_output = str(row.get("output") or "")

            system_message, user_message, meta = _build_v11_prompt(
                gts=gts,
                assistant_raw_output=assistant_raw_output,
                labels_path=labels_path,
            )
            _print_prompt(
                idx=i,
                total=len(refs),
                ref=ref,
                row=row,
                system_message=system_message,
                user_message=user_message,
            )
            _visualize_v11(meta)

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

