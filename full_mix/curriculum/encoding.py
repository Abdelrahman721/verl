"""Encoding tables shared by sharding, scoring, and stage-transition scripts.

The reward dispatcher in ``full_mix/rewards/compute_score.py`` may only return
int/float values (the rollout dump path serializes via numpy column-cast and
chokes on dtype=object string arrays). So the dict-form ``data_source`` is a
two-digit int code; the parquet's string column keeps the human-readable
``"<base>#shard{N}"`` form.

Encoding:
    tens digit = dataset id (1=ifeval, 2=chat, 3=safety)
    ones digit = shard id (1..NUM_SHARDS, currently 1..4)

Examples:
    "local/dolci-ifeval-32b#shard2"     -> 12
    "local/dolci-chat-32b#shard4"       -> 24
    "local/safety-dpo-reference#shard1" -> 31

Per-row uid:
    uid = dataset_id * 10_000_000 + original_row_index
where ``original_row_index`` is the row's position in the full original
parquet for that dataset (NOT the per-shard index), so a carried prompt keeps
the same uid across all stages.
"""

from __future__ import annotations

import re

NUM_SHARDS = 4

DATASET_ID: dict[str, int] = {
    # ---- training sources (the curriculum shards these) ----
    "local/dolci-ifeval-32b":     1,
    "local/dolci-chat-32b":       2,
    "local/safety-dpo-reference": 3,
    # ---- training source not used by the no-math curriculum ----
    "local/dolci-math-7b":        4,
    # ---- validation sources (never sharded; appear during val_before_train
    # and periodic eval). compute_score must return the same dict shape for
    # train and val rollouts, so each val source needs an id too. Their
    # encoded data_source will be id * 10 (shard digit 0). ----
    "HuggingFaceH4/MATH-500":     5,
    "openai/gsm8k":               6,
    "google/IFEval":              7,
    "allenai/IFBench_test":       8,
    # ---- additional training sources (curriculum-sharded) ----
    "local/avey-identity":        9,
}

ID_TO_DATASET: dict[int, str] = {v: k for k, v in DATASET_ID.items()}

UID_DATASET_STRIDE = 10_000_000

_SHARD_RE = re.compile(r"^(?P<base>.+)#shard(?P<shard>\d+)$")

# Validation-variant suffix, e.g. "openai/gsm8k@b00256" or "google/IFEval@free".
# Purely a reporting dimension: verl groups val metrics by the data_source
# STRING, so this is what turns one pooled `val-core/openai/gsm8k/reward` into
# one series per budget rung. Stripped everywhere else, so scoring, routing and
# int encoding never see it. `@` is safe as a delimiter — no base contains one.
_VARIANT_RE = re.compile(r"^(?P<base>[^@]+)@(?P<variant>[A-Za-z0-9_]+)$")


def split_variant(value: str) -> tuple[str, str | None]:
    """Split off a trailing ``@variant`` tag. Returns (rest, variant_or_None)."""
    m = _VARIANT_RE.match(value)
    if m is None:
        return value, None
    return m.group("base"), m.group("variant")


def variant_of(value: str) -> str | None:
    """The ``@variant`` tag on a data_source, or None."""
    return split_variant(value)[1]


def split_data_source(value: str) -> tuple[str, int | None]:
    """Split a parquet ``data_source`` string into (base, shard_id_or_None).

    ``shard_id_or_None`` is ``None`` for un-suffixed strings (the originals).
    Any ``@variant`` tag is stripped first, so every existing caller — reward
    routing, int encoding, uid construction — is unaffected by it.
    """
    value, _variant = split_variant(value)
    m = _SHARD_RE.match(value)
    if m is None:
        return value, None
    return m.group("base"), int(m.group("shard"))


def base_data_source(value: str) -> str:
    return split_data_source(value)[0]


def encode_data_source(value: str) -> int:
    """Encode a parquet data_source string into the int code carried in the
    rollout JSONL dump.

    Suffixed (curriculum) input -> dataset_id * 10 + shard_id (1..NUM_SHARDS).
    Un-suffixed (pre-curriculum) input -> dataset_id * 10 + 0, so dumps from
    non-curriculum runs still carry recognizable codes.
    """
    base, shard = split_data_source(value)
    if base not in DATASET_ID:
        raise ValueError(f"unknown base data_source: {base!r}")
    if shard is None:
        return DATASET_ID[base] * 10
    if not 1 <= shard <= NUM_SHARDS:
        raise ValueError(f"shard id out of range 1..{NUM_SHARDS}: {shard}")
    return DATASET_ID[base] * 10 + shard


def decode_data_source(code: int) -> tuple[str, int]:
    dataset_id, shard_id = divmod(code, 10)
    if dataset_id not in ID_TO_DATASET:
        raise ValueError(f"unknown dataset id in code {code}: tens={dataset_id}")
    if not 1 <= shard_id <= NUM_SHARDS:
        raise ValueError(f"unknown shard id in code {code}: ones={shard_id}")
    return ID_TO_DATASET[dataset_id], shard_id


def make_uid(base: str, original_row_index: int) -> int:
    if base not in DATASET_ID:
        raise ValueError(f"unknown base data_source: {base!r}")
    if original_row_index < 0 or original_row_index >= UID_DATASET_STRIDE:
        raise ValueError(
            f"row index {original_row_index} out of range [0, {UID_DATASET_STRIDE})"
        )
    return DATASET_ID[base] * UID_DATASET_STRIDE + original_row_index


def shard_suffix(shard_id: int) -> str:
    if not 1 <= shard_id <= NUM_SHARDS:
        raise ValueError(f"shard id out of range 1..{NUM_SHARDS}: {shard_id}")
    return f"#shard{shard_id}"


def with_shard(base: str, shard_id: int) -> str:
    return f"{base}{shard_suffix(shard_id)}"


def retag_shard(value: str, new_shard_id: int) -> str:
    base, _ = split_data_source(value)
    return with_shard(base, new_shard_id)
