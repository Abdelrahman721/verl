"""Training entry point: verl's `main_ppo`, with the per-source/LCPO metric patch
installed inside the TaskRunner actor.

Use this instead of `python3 -m verl.trainer.main_ppo`:

    python3 -m full_mix.main_ppo <same hydra overrides as before>

WHY NOT A RAY WORKER SETUP HOOK
-------------------------------
The obvious way to get the patch into the TaskRunner is
`ray_kwargs.ray_init.runtime_env.worker_process_setup_hook`, and that is what the
training script did until 2026-08-23. It works, and it also breaks the run: Ray
executes that hook in EVERY worker process at process startup, i.e. BEFORE it
assigns the process its GPUs. The hook imports this package, which imports verl,
and `verl/utils/device.py` runs `torch.cuda.is_available()` at import time. That
first CUDA call freezes the process's device list to all 8 GPUs, and the CUDA
runtime then ignores the per-actor CUDA_VISIBLE_DEVICES that Ray sets afterwards.

Two failures came out of that, both from the same cause:

  * TaskRunner (a CPU-only actor, so Ray gives it CUDA_VISIBLE_DEVICES=""):
    torch's cached count said 8 while a fresh count said 0, so transformer_engine's
    import-time device probe raised AssertionError("Invalid device id") and the
    actor died in its constructor.
  * Every GPU worker: `set_device(0)` resolved against the frozen all-8 list, so
    all ranks landed on physical GPU 0 and NCCL aborted with
    "Duplicate GPU detected : rank 3 and rank 0 both on CUDA device 6000".

Verified: with CUDA touched before CUDA_VISIBLE_DEVICES=3 is applied, device 0
resolves to physical GPU 0's UUID; with the assignment first, it correctly
resolves to GPU 3's.

`run_ppo(config, task_runner_class=...)` is verl's own extension point for this
("For recipe to change TaskRunner"), so the patch runs in exactly the one process
that computes metrics, at a point where every process already holds the GPUs Ray
gave it. Nothing runs at worker startup any more.

The correctness argument from `per_source_metrics` is unchanged: the patch must
land in the process that runs `compute_data_metrics`, which is the TaskRunner
actor, not the RewardLoopWorkers. Confirm on startup by looking for
"[per_source_metrics] installed in pid=..." with trainer_in_process=True.
"""

from __future__ import annotations

import ray

from verl.trainer import main_ppo as _verl_main


class TaskRunnerWithPerSourceMetrics(_verl_main.TaskRunner):
    """verl's TaskRunner, with the metric patch installed in this actor's process.

    `run` is the right seam rather than `__init__`: by the time it executes, this
    process has imported `ray_trainer`, so `install()` rebinds both the canonical
    `metric_utils.compute_data_metrics` and the copy `ray_trainer` bound at import.
    """

    def run(self, config):
        from full_mix import per_source_metrics

        per_source_metrics.install()
        return super().run(config)


_verl_run_ppo = _verl_main.run_ppo


def run_ppo(config, task_runner_class=None) -> None:
    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunnerWithPerSourceMetrics)
    return _verl_run_ppo(config, task_runner_class=task_runner_class)


# verl's `main` is hydra-decorated and calls `run_ppo` through a module-global
# lookup, so rebinding the name in that module is what routes it to the TaskRunner
# above. Hydra still resolves config_path against verl/trainer/, which keeps every
# CLI override identical to `python3 -m verl.trainer.main_ppo`.
_verl_main.run_ppo = run_ppo


if __name__ == "__main__":
    _verl_main.main()
