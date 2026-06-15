"""Parquet loading + per-row field extraction for the medical_qa datasets.

Mirrors how qa_bedrock / qa_bedrock_with_penalties read a row:
  - extra_info (dict) -> eval_mode (default "qa"), question
  - reward_model.ground_truth is a JSON string -> json.loads -> key_points,
    gold_answer / gold_response, conv fields (latest_user, ...)
  - prompt = list of chat-message dicts (the policy input)

qa user_prompt   = extra_info["question"]
conv user_prompt = ground_truth["latest_user"]
reference        = ground_truth["gold_response"] or ground_truth["gold_answer"]
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

import pyarrow.parquet as pq

log = logging.getLogger("web_judging.data")

_SPLIT_ENV = {
    "val": "VAL_PARQUET",
    "train": "TRAIN_PARQUET",
}
_DEFAULT_PATHS = {
    "val": "/data/abdelrahman/verl/data/medical_qa/val.parquet",
    "train": "/data/abdelrahman/verl/data/medical_qa/train.parquet",
}


def _path_for(split: str) -> str:
    if split not in _SPLIT_ENV:
        raise ValueError(f"unknown split {split!r}; expected one of {list(_SPLIT_ENV)}")
    return os.environ.get(_SPLIT_ENV[split], _DEFAULT_PATHS[split])


def _train_limit() -> int:
    try:
        return int(os.environ.get("WEB_JUDGING_TRAIN_LIMIT", "500"))
    except ValueError:
        return 500


@lru_cache(maxsize=4)
def _load_rows(split: str) -> list[dict]:
    """Load (a capped slice of) a split into a list of plain dicts. Cached."""
    path = _path_for(split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"parquet for split {split!r} not found: {path}")
    table = pq.read_table(path)
    total = table.num_rows
    if split == "train":
        limit = _train_limit()
        if total > limit:
            log.warning("train split capped to first %d of %d rows for the dropdown "
                        "(set WEB_JUDGING_TRAIN_LIMIT to change)", limit, total)
            table = table.slice(0, limit)
    rows = table.to_pylist()
    log.info("loaded %d rows from %s (%s, total=%d)", len(rows), path, split, total)
    return rows


def _coerce_gt(ground_truth) -> dict:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            return json.loads(ground_truth)
        except json.JSONDecodeError:
            return {}
    return {}


def _extra_info(row: dict) -> dict:
    ei = row.get("extra_info") or {}
    return ei if isinstance(ei, dict) else {}


def _question(row: dict) -> str:
    """The medical question. For qa rows it's extra_info['question']; fall back
    to the first user prompt message (matches qa_bedrock's preference order)."""
    ei = _extra_info(row)
    q = ei.get("question")
    if q:
        return str(q)
    prompt = row.get("prompt") or []
    if isinstance(prompt, (list, tuple)) and prompt:
        first = prompt[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return ""


def list_entries(split: str) -> list[dict]:
    """Lightweight listing for the dropdown."""
    rows = _load_rows(split)
    out = []
    for i, row in enumerate(rows):
        ei = _extra_info(row)
        q = _question(row)
        preview = q.replace("\n", " ").strip()
        if len(preview) > 90:
            preview = preview[:89] + "…"
        out.append({
            "split": split,
            "index": i,
            "eval_mode": ei.get("eval_mode", "qa"),
            "preview": preview or "(no question text)",
        })
    return out


def get_entry(split: str, index: int) -> dict:
    """Full entry payload: everything the UI and the judges need."""
    rows = _load_rows(split)
    if index < 0 or index >= len(rows):
        raise IndexError(f"index {index} out of range for split {split!r} ({len(rows)} rows)")
    row = rows[index]
    ei = _extra_info(row)
    gt = _coerce_gt((row.get("reward_model") or {}).get("ground_truth"))
    eval_mode = ei.get("eval_mode", "qa")

    if eval_mode == "conversation":
        user_prompt = str(gt.get("latest_user", "") or "")
    else:
        user_prompt = _question(row)

    reference = str(gt.get("gold_response") or gt.get("gold_answer") or "")

    prompt = row.get("prompt") or []
    # Normalise chat messages to plain dicts (parquet may give tuples/np types).
    messages = []
    if isinstance(prompt, (list, tuple)):
        for m in prompt:
            if isinstance(m, dict) and "role" in m and "content" in m:
                messages.append({"role": str(m["role"]), "content": str(m["content"])})

    return {
        "split": split,
        "index": index,
        "eval_mode": eval_mode,
        "data_source": row.get("data_source"),
        "question": _question(row),
        "user_prompt": user_prompt,        # what the judge / penalty judge sees as the prompt
        "reference": reference,            # gold answer / response
        "key_points": gt.get("key_points", []),
        "messages": messages,              # policy chat input
        "ground_truth": gt,                # full parsed ground_truth (for the judges)
        "extra_info": {                    # minimal extra_info the judges read
            "eval_mode": eval_mode,
            "question": _question(row),
            "prompt": messages,
        },
    }
