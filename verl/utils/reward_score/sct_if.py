# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Instruction-following reward for SNOMED (data_source ``sct_if``).

Uses ``instruction_following.rules_eval.evaluate_row`` per ``rule`` in ground truth JSON.

``ground_truth`` must include ``rule``, ``original_codes``, and ``processed_codes``
(see ``examples/data_preprocess/sct_if_preprocess.py``).

For ``extra_info``, include ``user_prompt`` (full user message text) for rules that need it
(e.g. v10); the preprocess script stores this automatically.

Score aggregation:
- If both instruction_score and accuracy_score are numeric: their sum normalized to [0, 1]
  (i.e. average).
- If exactly one is ``None``: the other score, unchanged.
- If both are ``None``: ``0.0``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from instruction_following.rules_eval import (
    CodesNotInLabelsError,
    V17InputError,
    evaluate_row,
)


def _combine_instruction_accuracy(
    instruction_score: float | None,
    accuracy_score: float | None,
) -> float:
    if instruction_score is not None and accuracy_score is not None:
        return (float(instruction_score) + float(accuracy_score)) / 2.0
    if instruction_score is not None:
        return float(instruction_score)
    if accuracy_score is not None:
        return float(accuracy_score)
    return 0.0


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> float:
    del kwargs
    try:
        gt = json.loads(ground_truth or "{}")
    except json.JSONDecodeError:
        return 0.0

    rule = gt.get("rule")
    original_codes = gt.get("original_codes")
    processed_codes = gt.get("processed_codes")
    if rule is None or original_codes is None or processed_codes is None:
        return 0.0

    oc = [str(x) for x in original_codes]
    pc = [str(x) for x in processed_codes]

    labels_path: Path | None = None
    env_labels = os.getenv("INSTRUCTION_FOLLOWING_LABELS_CSV", "").strip()
    if env_labels:
        labels_path = Path(env_labels).expanduser()
    if extra_info and extra_info.get("labels_path"):
        labels_path = Path(str(extra_info["labels_path"])).expanduser()

    user_prompt: str | None = None
    if extra_info:
        up = extra_info.get("user_prompt")
        if isinstance(up, str) and up.strip():
            user_prompt = up

    try:
        result = evaluate_row(
            str(rule),
            solution_str or "",
            oc,
            pc,
            prompt=user_prompt,
            labels_path=labels_path,
        )
    except (CodesNotInLabelsError, V17InputError, KeyError, ValueError):
        return 0.0

    return _combine_instruction_accuracy(result.instruction_score, result.accuracy_score)
