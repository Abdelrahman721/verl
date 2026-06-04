"""Per-data-source training reward metrics for wandb.

Verl emits aggregate `critic/score/mean`, `critic/rewards/mean` and
`critic/advantages/*` for the whole training batch. For a mixed-source run
(chat / safety / ifeval / identity / ...) the aggregate is an average over
heterogeneous reward functions and tells you almost nothing about whether any
individual source is learning. This module wraps
`verl.trainer.ppo.metric_utils.compute_data_metrics` so each source also gets
its own curves:

    critic/per_source/<source>/score/mean            # raw reward (pre-KL)
    critic/per_source/<source>/reward/mean           # token-level reward (post-KL)
    critic/per_source/<source>/response_length/mean
    critic/per_source/<source>/adv/mean              # mean per-seq advantage
    critic/per_source/<source>/adv/std               # spread of per-seq advantage
    critic/per_source/<source>/adv/zero_var_frac     # frac of seqs with ~0 |adv|
    critic/per_source/<source>/count

`adv/std` and `zero_var_frac` are the GRPO health signal: if within-source
advantage variance is ~0, every response in a group got the same reward and
there is no gradient for that source.

IMPORTANT — why this is not the old metric_utils-only patch:
Every trainer does `from verl.trainer.ppo.metric_utils import compute_data_metrics`,
which binds the *original* function object into that trainer module's namespace
at import time. Reassigning `metric_utils.compute_data_metrics` alone does NOT
affect those already-bound references, so the metric never changes. We instead
rebind every live reference across `sys.modules`, so the patch works regardless
of import order.

Usage: ensure this module is imported once before training starts. It is
imported from `full_mix/rewards/compute_score.py` (the custom reward module,
loaded during trainer setup, after the trainer modules are already imported).
The patch is idempotent.
"""

from __future__ import annotations

import sys

import numpy as np

from verl.trainer.ppo import metric_utils

_PATCH_ATTR = "_full_mix_per_source_wrapped"
_orig_compute_data_metrics = metric_utils.compute_data_metrics


def _per_seq_advantages(batch) -> np.ndarray | None:
    """Mean advantage over each sequence's valid response tokens -> shape [B]."""
    if "advantages" not in batch.batch.keys():
        return None
    adv = batch.batch["advantages"].detach().float()
    if "response_mask" in batch.batch.keys():
        mask = batch.batch["response_mask"].detach().float()
    else:
        mask = batch.batch["responses"].ne(0).detach().float()
    tok = mask.sum(-1).clamp(min=1.0)
    return ((adv * mask).sum(-1) / tok).cpu().numpy()


def _compute_per_source_metrics(batch) -> dict:
    """Group sequence-level score / reward / length / advantage by data_source."""
    if "data_source" not in batch.non_tensor_batch:
        return {}

    sequence_score = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()
    sequence_reward = batch.batch["token_level_rewards"].sum(-1).detach().cpu().numpy()
    sources = np.asarray(batch.non_tensor_batch["data_source"])
    response_length = batch.batch["responses"].ne(0).sum(-1).detach().cpu().numpy()
    per_seq_adv = _per_seq_advantages(batch)

    out: dict = {}
    for src in np.unique(sources):
        mask = sources == src
        out[f"critic/per_source/{src}/count"] = int(mask.sum())
        non_aborted = mask & (response_length > 0)
        if not non_aborted.any():
            continue
        out[f"critic/per_source/{src}/score/mean"] = float(sequence_score[non_aborted].mean())
        out[f"critic/per_source/{src}/reward/mean"] = float(sequence_reward[non_aborted].mean())
        out[f"critic/per_source/{src}/response_length/mean"] = float(response_length[non_aborted].mean())
        if per_seq_adv is not None:
            a = per_seq_adv[non_aborted]
            out[f"critic/per_source/{src}/adv/mean"] = float(a.mean())
            out[f"critic/per_source/{src}/adv/std"] = float(a.std())
            out[f"critic/per_source/{src}/adv/zero_var_frac"] = float((np.abs(a) < 1e-6).mean())
    return out


def compute_data_metrics_with_sources(batch, use_critic: bool = True) -> dict:
    metrics = _orig_compute_data_metrics(batch, use_critic=use_critic)
    try:
        metrics.update(_compute_per_source_metrics(batch))
    except Exception as e:  # never let metrics break the training step
        metrics["critic/per_source/_error"] = 1.0
        print(f"[per_source_metrics] skipped, error: {e!r}", flush=True)
    return metrics


def install() -> int:
    """Rebind every live `compute_data_metrics` reference. Returns #sites patched."""
    wrapped = compute_data_metrics_with_sources
    setattr(wrapped, _PATCH_ATTR, True)

    # 1) the canonical attribute (covers modules imported AFTER us)
    metric_utils.compute_data_metrics = wrapped

    # 2) every module that already did `from ... import compute_data_metrics`
    patched = 0
    for mod in list(sys.modules.values()):
        if mod is None or mod is metric_utils:
            continue
        try:
            ref = getattr(mod, "compute_data_metrics", None)
        except Exception:
            continue
        if ref is _orig_compute_data_metrics:
            setattr(mod, "compute_data_metrics", wrapped)
            patched += 1
    print(f"[per_source_metrics] installed, rebound {patched} call site(s)", flush=True)
    return patched


# Idempotent install at import time.
if not getattr(metric_utils.compute_data_metrics, _PATCH_ATTR, False):
    install()
