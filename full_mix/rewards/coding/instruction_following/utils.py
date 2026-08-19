"""Shared helpers: SNOMED ID extraction, description normalization, labels."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

# Embedded labels live next to this file under labels/ — relocated from the
# original repos-relative path so the coding package is fully self-contained.
# The INSTRUCTION_FOLLOWING_LABELS_CSV env var still overrides this default.
_HERE = Path(__file__).resolve().parent
_DEFAULT_LABELS = _HERE / "labels" / "all_codes_to_descriptions_reconciled.json"


def default_labels_path() -> Path:
    env = os.getenv("INSTRUCTION_FOLLOWING_LABELS_CSV", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_LABELS

_SCT_ID_PATTERN = re.compile(r"\b(\d{6,18})\b")


def get_answer_text(response: str) -> str:
    """Take everything after the last </think> tag.

    Case-sensitive on </think> — matches the dispatcher's
    `_compute_format_penalty` and `_extract_answer` in `coding/common.py`,
    so a response that puts the boundary in a non-canonical case (e.g.
    `</THINK>`) gets a format penalty AND no answer extracted on this side.
    Symmetric treatment avoids the asymmetric case where the IF framework
    rewards a "correct" answer while the dispatcher penalises its format
    (was -0.5 cap on max attainable score)."""
    text = response or ""
    matches = list(re.finditer(r"</think>", text))
    if not matches:
        return text.strip()
    return text[matches[-1].end() :].strip()


def extract_sct_ids(text: str) -> set[str]:
    """Extract SNOMED CT concept IDs (6–18 digits) from text."""
    return set(_SCT_ID_PATTERN.findall(text))


def f1_set(pred: set[str], gold: set[str]) -> float:
    """
    Micro F1 for two sets of IDs: precision/recall on membership overlap.
    pred = predicted / extracted labels; gold = reference labels.
    """
    if not gold and not pred:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def normalize_for_desc_match(text: str) -> str:
    """
    Remove only an outer quote pair (single or double) when present on both sides.
    Mirrors domain/snomed/rules.py.
    """
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1].strip()
    return text


def load_labels(labels_path: Path | None = None) -> dict[str, str]:
    """Load sct_id -> description from CSV or JSON mapping file."""
    path = labels_path or default_labels_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {path}. The default labels JSON ships under "
            "full_mix/rewards/coding/instruction_following/labels/; set "
            "INSTRUCTION_FOLLOWING_LABELS_CSV to override."
        )
    out: dict[str, str] = {}

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Labels JSON must be an object mapping sct_id -> label. Got: {type(data).__name__}")
        for sid, label in data.items():
            sid_s = str(sid).strip()
            label_s = str(label).strip()
            if sid_s and label_s:
                out[sid_s] = label_s
        return out

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row["sct_id"]).strip()
            out[sid] = str(row["label"]).strip()
    return out


_LABELS_CACHE: dict[str, str] | None = None


def get_labels(labels_path: Path | None = None) -> dict[str, str]:
    global _LABELS_CACHE
    if labels_path is not None:
        return load_labels(labels_path)
    if _LABELS_CACHE is None:
        _LABELS_CACHE = load_labels()
    return _LABELS_CACHE


def prompt_ids_from_user_prompt(prompt: str) -> set[str]:
    """SNOMED IDs embedded in the user prompt (for v10)."""
    return set(_SCT_ID_PATTERN.findall(prompt))


def parse_json_codes_list(s: str) -> list[str]:
    """Parse original_codes / processed_codes column from CSV JSON array."""
    data = json.loads(s)
    return [str(x) for x in data]


def v06_sorted_positions_ok(answer: str, sorted_ids: list[str]) -> bool:
    """First occurrences of sorted_ids appear in strictly increasing positions."""
    positions: list[int] = []
    for sct_id in sorted_ids:
        m = re.search(rf"\b{re.escape(sct_id)}\b", answer)
        if not m:
            return False
        positions.append(m.start())
    return positions == sorted(positions)
