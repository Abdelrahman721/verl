#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Iterator


TARGET_ID = "160352002"
TARGET_WORD_RE = re.compile(r"kidney", re.IGNORECASE)


@dataclass(frozen=True)
class Match:
    source: str  # "sft_messages_ifsnomed" | "instruction_following_train.parquet"
    row_id: str  # stable-ish identifier for navigation/logging


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Interactively browse assistant responses across two instruction-following datasets, "
            "filtering for responses that contain 160352002 and the word 'kidney'."
        )
    )
    p.add_argument(
        "--sft-dataset-dir",
        default="/data/muhsen/repos/llm-pretrainer/sft_messages_ifsnomed",
        help="Path to the HF datasets directory for sft_messages_ifsnomed.",
    )
    p.add_argument(
        "--if-train-parquet",
        default="/data/muhsen/repos/verl/rl-data/instruction_following_train.parquet",
        help="Path to instruction_following_train.parquet.",
    )
    return p.parse_args()


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.getenv("NO_COLOR", "").strip() == ""


def _green(text: str) -> str:
    if not _supports_color():
        return text
    return f"\x1b[32m{text}\x1b[0m"


def _highlight_targets(text: str) -> str:
    """
    Highlight all occurrences of:
    - the exact substring TARGET_ID
    - the substring 'kidney' (case-insensitive)
    in green, no matter where/how many times they occur.
    """
    if not text:
        return text

    # We do a single pass over a combined regex to preserve original casing.
    combined = re.compile(rf"({re.escape(TARGET_ID)}|kidney)", re.IGNORECASE)

    def repl(m: re.Match) -> str:
        return _green(m.group(0))

    return combined.sub(repl, text)


def _join_if_split_think_answer(chunks: list[str]) -> str:
    """
    If thinking/answer are split across multiple assistant chunks, join them before processing.
    """
    chunks = [c for c in (x.strip() for x in chunks) if c]
    if not chunks:
        return ""
    return "\n\n".join(chunks)


def _assistant_response_from_row(row: dict[str, Any]) -> str:
    """
    Best-effort extraction of the assistant-only model response, excluding the prompt.
    Handles common shapes:
    - {"messages":[{"role":"user"...},{"role":"assistant","content":...}, ...]}
    - {"output": "..."} / {"response": "..."} / {"completion": "..."}
    - {"thinking": "...", "answer": "..."} (join)
    """
    # messages (OpenAI-style)
    msgs = row.get("messages")
    if isinstance(msgs, list):
        assistant_chunks: list[str] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if str(m.get("role", "")).strip().lower() != "assistant":
                continue
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                assistant_chunks.append(content)
        if assistant_chunks:
            return _join_if_split_think_answer(assistant_chunks)

    # explicit fields
    if isinstance(row.get("thinking"), str) or isinstance(row.get("answer"), str):
        chunks = []
        if isinstance(row.get("thinking"), str):
            chunks.append(row["thinking"])
        if isinstance(row.get("answer"), str):
            chunks.append(row["answer"])
        out = _join_if_split_think_answer([c for c in chunks if isinstance(c, str)])
        if out:
            return out

    for k in ("output", "response", "completion", "assistant_response", "text"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def _is_match(text: str) -> bool:
    return (TARGET_ID in text) and (TARGET_WORD_RE.search(text) is not None)


def _iter_sft_rows(dataset_dir: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """
    Yields (row_id, row_dict) for the HF dataset at dataset_dir.
    """
    try:
        from datasets import load_from_disk  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: 'datasets'. Install it or run in an environment that has it."
        ) from e

    ds = load_from_disk(dataset_dir)
    # ds can be DatasetDict or Dataset
    if hasattr(ds, "keys"):
        # pick a split if present
        for split in ("train", "validation", "test"):
            if split in ds:
                d = ds[split]
                break
        else:
            # first split
            d = ds[next(iter(ds.keys()))]
    else:
        d = ds

    for i in range(len(d)):
        row = d[i]
        # datasets returns dict-like mapping; make it a plain dict
        row_dict = dict(row)
        yield f"idx={i}", row_dict


def _iter_parquet_rows(parquet_path: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """
    Yields (row_id, row_dict) for rows in a parquet file without loading everything into memory.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: 'pyarrow'. Install it or run in an environment that has it."
        ) from e

    pf = pq.ParquetFile(parquet_path)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        cols = table.column_names
        # Convert row-by-row (small overhead but ok for search + interactive usage)
        for j in range(table.num_rows):
            row = {c: table[c][j].as_py() for c in cols}
            yield f"rg={rg},row={j}", row


def _build_matches(sft_dir: str, parquet_path: str) -> tuple[list[Match], dict[tuple[str, str], str]]:
    """
    Returns:
      - list of matches
      - mapping (source,row_id) -> assistant_text
    """
    matches: list[Match] = []
    cache: dict[tuple[str, str], str] = {}

    # SFT dataset
    for row_id, row in _iter_sft_rows(sft_dir):
        text = _assistant_response_from_row(row)
        if text and _is_match(text):
            m = Match(source="sft_messages_ifsnomed", row_id=row_id)
            matches.append(m)
            cache[(m.source, m.row_id)] = text

    # IF train parquet
    for row_id, row in _iter_parquet_rows(parquet_path):
        text = _assistant_response_from_row(row)
        if text and _is_match(text):
            m = Match(source="instruction_following_train.parquet", row_id=row_id)
            matches.append(m)
            cache[(m.source, m.row_id)] = text

    return matches, cache


def _print_match(idx: int, total: int, m: Match, text: str) -> None:
    print("=" * 110)
    print(f"[{idx + 1}/{total}] source={m.source}  row={m.row_id}")
    print("-" * 110)
    print("MODEL_RESPONSE:")
    print(_highlight_targets(text.rstrip()))
    print("=" * 110)


def main() -> int:
    args = _parse_args()
    sft_dir = args.sft_dataset_dir
    parquet_path = args.if_train_parquet

    if not os.path.exists(sft_dir):
        print(f"Not found: {sft_dir}", file=sys.stderr)
        return 2
    if not os.path.exists(parquet_path):
        print(f"Not found: {parquet_path}", file=sys.stderr)
        return 2

    print("Scanning datasets for matches... (this can take a bit)")
    matches, cache = _build_matches(sft_dir, parquet_path)

    if not matches:
        print("No matching responses found.")
        return 0

    i = 0
    while True:
        m = matches[i]
        text = cache.get((m.source, m.row_id), "")
        _print_match(i, len(matches), m, text)

        cmd = input("Command [n=next, p=prev, j <k>=jump, q=quit] > ").strip()
        if cmd in ("q", "quit", "exit"):
            return 0
        if cmd == "" or cmd == "n":
            i = (i + 1) % len(matches)
            continue
        if cmd == "p":
            i = (i - 1) % len(matches)
            continue
        if cmd.startswith("j"):
            parts = cmd.split()
            if len(parts) == 2 and parts[1].isdigit():
                k = int(parts[1])
                if 1 <= k <= len(matches):
                    i = k - 1
                    continue
            print(f"Invalid jump. Use: j <1..{len(matches)}>")
            continue
        if cmd == "dump":
            # Optional: emit current record as JSON for debugging schema
            print(json.dumps({"source": m.source, "row": m.row_id}, indent=2))
            continue
        print("Unknown command.")


if __name__ == "__main__":
    raise SystemExit(main())

