"""Per-data-source training reward metrics for wandb.

Verl emits aggregate `critic/score/mean` and `critic/rewards/mean` for the whole
training batch. This module monkey-patches `verl.trainer.ppo.metric_utils.compute_data_metrics`
to also emit `critic/per_source/<source>/{score,reward}/mean` so each dataset
(chat / safety / ifeval / etc.) gets its own curve in wandb.

Usage: import this module from the training entry point before the trainer
starts. Easiest path is to add `import full_mix.per_source_metrics  # noqa`
to the top of the verl trainer launch, OR set PYTHONSTARTUP, OR import it
from the custom reward function file (which is loaded before training starts).
"""

from __future__ import annotations

import numpy as np

from verl.trainer.ppo import metric_utils

_orig_compute_data_metrics = metric_utils.compute_data_metrics


def _compute_per_source_metrics(batch) -> dict:
    """Group sequence_score and sequence_reward by data_source."""
    if "data_source" not in batch.non_tensor_batch:
        return {}

    sequence_score = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()
    sequence_reward = batch.batch["token_level_rewards"].sum(-1).detach().cpu().numpy()
    sources = np.asarray(batch.non_tensor_batch["data_source"])
    response_length = batch.batch["responses"].ne(0).sum(-1).detach().cpu().numpy()

    out: dict = {}
    for src in np.unique(sources):
        mask = sources == src
        if not mask.any():
            continue
        non_aborted = mask & (response_length > 0)
        if non_aborted.any():
            out[f"critic/per_source/{src}/score/mean"] = float(sequence_score[non_aborted].mean())
            out[f"critic/per_source/{src}/reward/mean"] = float(sequence_reward[non_aborted].mean())
            out[f"critic/per_source/{src}/response_length/mean"] = float(response_length[non_aborted].mean())
        out[f"critic/per_source/{src}/count"] = int(mask.sum())
    return out


def compute_data_metrics_with_sources(batch, use_critic: bool = True) -> dict:
    metrics = _orig_compute_data_metrics(batch, use_critic=use_critic)
    metrics.update(_compute_per_source_metrics(batch))
    return metrics


# Patch in place. Idempotent.
if metric_utils.compute_data_metrics is not compute_data_metrics_with_sources:
    metric_utils.compute_data_metrics = compute_data_metrics_with_sources
