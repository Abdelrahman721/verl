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

Usage: this module must be imported IN THE PROCESS THAT COMPUTES METRICS.

That is not the process it looks like. `compute_data_metrics` runs inside the
`TaskRunner` Ray actor (main_ppo.run_ppo creates it with `ray.remote(TaskRunner)`),
whereas the import from `full_mix/rewards/compute_score.py` happens in the
RewardLoopWorker processes that score rollouts. Importing there patches those
workers and nothing else, so every curve below silently goes missing while the
run looks healthy. That is exactly what happened to the stage-a run of
2026-08-19: `rebound 0 call site(s)` in the reward workers, no per-source or
LCPO curves in wandb, and a math-slice collapse that took 170 steps to notice
instead of 20.

The training script therefore also passes this module's `install` as a Ray
worker setup hook, which runs in every Ray worker including the TaskRunner:

    +ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=full_mix.per_source_metrics.install

That requires the repo root on PYTHONPATH in the Ray workers; the training
script exports it. The patch is idempotent, so importing it several ways is
harmless.
"""

from __future__ import annotations

import os
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


def _response_length(batch) -> np.ndarray:
    """Per-sequence response length in real (non-padding) tokens.

    Uses the attention mask, the way verl's own compute_data_metrics does. The
    previous implementation counted ``responses.ne(0)``, but this tokenizer pads
    with <|endoftext|> = 151643, not 0 — so padding counted as content and every
    source reported a length pinned at max_response_length.
    """
    responses = batch.batch["responses"]
    attention_mask = batch.batch.get("attention_mask", None)
    if attention_mask is not None:
        # The response occupies the trailing `response_length` positions.
        return attention_mask[:, -responses.shape[-1]:].sum(-1).detach().cpu().numpy()
    return responses.ne(_pad_token_id(batch)).sum(-1).detach().cpu().numpy()


def _pad_token_id(batch) -> int:
    meta = getattr(batch, "meta_info", None) or {}
    return int(meta.get("pad_token_id", 151643))


def _compute_per_source_metrics(batch) -> dict:
    """Group sequence-level score / reward / length / advantage by data_source."""
    if "data_source" not in batch.non_tensor_batch:
        return {}

    sequence_score = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()
    sequence_reward = batch.batch["token_level_rewards"].sum(-1).detach().cpu().numpy()
    sources = np.asarray(batch.non_tensor_batch["data_source"])
    response_length = _response_length(batch)
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


# Budget buckets for the length curves. Edges match the validation rungs so the
# in-training view and the eval ladder line up.
_BUDGET_EDGES = [0, 384, 768, 1536, 3024, 5000, 10**9]
_BUDGET_LABELS = ["le384", "le768", "le1536", "le3024", "le5000", "gt5000"]

# Emitted per sample by LCPORewardManager. Mean of each, per group.
_LCPO_MEAN_KEYS = (
    "task_score", "n_think", "n_answer", "abs_len_err", "rel_len_err",
    "len_mult", "is_wellformed", "nothink_format_ok", "truncated",
    "chat_exact_half",
)


def _budget_bucket(b: float) -> str:
    for lo, label in zip(_BUDGET_EDGES[1:], _BUDGET_LABELS):
        if b <= lo:
            return label
    return _BUDGET_LABELS[-1]


def _compute_lcpo_metrics(batch) -> dict:
    """Length / mode curves for budget training.

    Grouped by think_mode, and for budgeted rows also by budget bucket, because
    the three modes are scored by different rules — a combined average mixes
    populations that are not comparable, which is the easiest way to miss
    unconstrained mode quietly drifting.

    Reads only from non_tensor_batch, which is where the reward manager's
    numeric extra keys land. Returns {} when this is not an LCPO run.
    """
    ntb = batch.non_tensor_batch
    if "n_think" not in ntb:
        return {}

    n = len(batch)
    vals = {k: np.asarray(ntb[k], dtype=float) for k in _LCPO_MEAN_KEYS if k in ntb}
    if not vals:
        return {}
    budgets = np.asarray(ntb["think_budget"], dtype=float) if "think_budget" in ntb else np.full(n, -1.0)

    # Derive the mode from the NUMERIC reward keys rather than the dataset's
    # `think_mode` string column. Dataset columns do not reach the trainer batch
    # (see the note in compute_data_metrics_with_sources), and reward_extra_info
    # can only carry numbers anyway.
    if "think_mode" in ntb:
        modes = np.asarray(ntb["think_mode"])
    else:
        is_nothink = np.asarray(ntb["is_nothink_mode"], dtype=float) if "is_nothink_mode" in ntb else np.zeros(n)
        modes = np.where(is_nothink > 0.5, "nothink", np.where(budgets > 0, "budget", "free"))

    out: dict = {}

    def emit(prefix: str, mask: np.ndarray) -> None:
        if not mask.any():
            return
        out[f"{prefix}/count"] = int(mask.sum())
        for key, arr in vals.items():
            out[f"{prefix}/{key}/mean"] = float(arr[mask].mean())

    for mode in np.unique(modes):
        mask = modes == mode
        emit(f"critic/lcpo/mode_{mode}", mask)
        # Per-budget curves only make sense where a budget exists. This is the
        # monotonicity view: task_score by budget bucket.
        if mode != "budget":
            continue
        buckets = np.array([_budget_bucket(b) for b in budgets])
        for label in _BUDGET_LABELS:
            emit(f"critic/lcpo/budget_{label}", mask & (buckets == label))

    return out


def compute_data_metrics_with_sources(batch, use_critic: bool = True) -> dict:
    metrics = _orig_compute_data_metrics(batch, use_critic=use_critic)
    try:
        metrics.update(_compute_per_source_metrics(batch))
    except Exception as e:  # never let metrics break the training step
        metrics["critic/per_source/_error"] = 1.0
        print(f"[per_source_metrics] skipped, error: {e!r}", flush=True)
    # Called separately, NOT from inside _compute_per_source_metrics, because
    # that function returns early when `data_source` is absent — and it is
    # absent here. With rule-based rewards `enable_agent_reward_loop` is true,
    # so ray_trainer passes reward_loop_worker_handles and agent_loop.py skips
    # `non_tensor_batch.update(input_non_tensor_batch)`, dropping every dataset
    # column from the trainer's batch. The LCPO curves ride the reward_extra_info
    # channel instead, which IS propagated.
    try:
        metrics.update(_compute_lcpo_metrics(batch))
    except Exception as e:
        metrics["critic/lcpo/_error"] = 1.0
        print(f"[per_source_metrics] lcpo metrics skipped, error: {e!r}", flush=True)
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

    # Be explicit about which of the two mechanisms actually took, because
    # "rebound 0 call site(s)" is NOT a failure on its own — step 1 above still
    # covers every module that imports metric_utils later. The failure mode we
    # actually had was subtler: the right process never imported this module.
    # `trainer_seen` answers that: it is true only where a trainer is loaded.
    trainer_seen = any(
        "trainer" in name and getattr(mod, "compute_data_metrics", None) is wrapped
        for name, mod in list(sys.modules.items()) if mod is not None
    )
    print(
        f"[per_source_metrics] installed in pid={os.getpid()}: "
        f"canonical=patched, late-bound sites={patched}, trainer_in_process={trainer_seen}",
        flush=True,
    )
    if not trainer_seen:
        print(
            "[per_source_metrics] NOTE: no trainer module in this process — if this "
            "is the only place the module is imported, per-source and LCPO curves "
            "will NOT appear in wandb. See this module's docstring.",
            flush=True,
        )
    return patched


# Idempotent install at import time.
if not getattr(metric_utils.compute_data_metrics, _PATCH_ATTR, False):
    install()
