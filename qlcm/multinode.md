# Multi-node training for QLCM

The pipeline defaults to a **single 8-GPU node** (2 train + 6 rollout), which is
enough for a 1.7B actor. This doc covers spreading it across more machines —
the flow the 32B production run used on five.

verl's `fully_async_policy` puts rollout and training on **disjoint** GPU pools,
so the total the cluster must provide is:

```
NNODES_TRAIN * NGPUS_TRAIN  +  NNODES_ROLLOUT * NGPUS_ROLLOUT
```

The pools must not overlap. Ray orchestrates both, so the flow is:

1. Start a Ray cluster spanning all nodes
2. Launch `run_pipeline.sh` on the **head node only**

---

## Prerequisites

- **Identical docker image** on every node (the tag `dev/dev.sh` pins).
- **Identical filesystem layout** — repo, datasets and checkpoints at the same
  absolute paths everywhere. Either shared storage mounted at
  `/data/abdelrahman` on each node, or rsync to identical paths.
- **Inter-node network**: every node's IP reachable from every other.
  `dev/dev.sh` uses `--net=host`, so container ports equal host ports.
- **Same judge configuration** on every node — see below.

---

## Step 1 — dev container on every node

```bash
bash dev/dev.sh          # drops you into /workspace/verl inside the container
bash qlcm/setup.sh       # extra deps, once per node
```

Leave a terminal open on each node.

## Step 2 — start the Ray cluster

Find the head node's cluster-facing IP **on the host**, not in the container:

```bash
hostname -I | awk '{print $1}'
```

On the head node, inside the container:

```bash
ray start --head --port=6379 \
  --dashboard-host=0.0.0.0 --dashboard-port=8265 \
  --num-gpus=8 --num-cpus=<physical cores>
```

On every worker node:

```bash
ray start --address=<head-ip>:6379 --num-gpus=8 --num-cpus=<physical cores>
```

Verify on the head with `ray status` — you should see N nodes and 8·N GPUs.

> **Always pass `--num-cpus` explicitly.** On the 200+ core hosts, leaving Ray
> to autodetect makes a second `ray job submit` hang and kills the head raylet.

> **Troubleshooting**: if `ray start --address=...` reports "connection
> refused", check port 6379 is reachable worker → head. Strict firewalls also
> need 10001-10999 (Ray workers) and 8265 (dashboard).

## Step 3 — submit the job

From the head node, inside the container:

```bash
bash qlcm/submit.sh
```

That wraps `ray job submit` with the right topology, entrypoint quoting and
preflight checks. Defaults to **1 trainer pool (2 GPUs) + 2 rollout pools
(6 GPUs each)** = 14 of 16 GPUs and 12 vLLM engines at `GEN_TP=1`.

It refuses to launch if `runtime_env.yaml` is missing, if
`QLCM_JUDGE_API_KEY` is still a placeholder, or if the two pools do not fit the
cluster. Stage-scoped keys that are still placeholders produce a warning, not a
failure — the general stage does not need them.

Useful variants:

```bash
START_STAGE=general END_STAGE=general bash qlcm/submit.sh   # one stage only
FOLLOW=1 bash qlcm/submit.sh                                # stream logs
NNODES_ROLLOUT=1 NGPUS_ROLLOUT=6 bash qlcm/submit.sh        # back to single node
bash qlcm/submit.sh trainer.test_freq=5                     # extra Hydra overrides
```

Detached by default. Track it with:

```bash
ray job list  --address http://127.0.0.1:8265
ray job logs  --address http://127.0.0.1:8265 -f <submission-id>
ray job stop  --address http://127.0.0.1:8265 <submission-id>
```

### The raw command

`submit.sh` exists because this is easy to get subtly wrong — a mis-quoted
`bash -c` silently launches with the single-node defaults. For reference:

```bash
ray job submit \
  --address http://127.0.0.1:8265 \
  --runtime-env qlcm/runtime_env.yaml \
  --no-wait \
  -- bash -c "cd /workspace/verl && \
      NNODES_TRAIN=1 NGPUS_TRAIN=2 \
      NNODES_ROLLOUT=2 NGPUS_ROLLOUT=6 \
      bash qlcm/run_pipeline.sh"
```

Two things about this command:

- **No `--working-dir`.** `/data` is shared storage and `dev/dev.sh` mounts the
  repo at `/workspace/verl` on every node, so there is nothing to upload. Adding
  `--working-dir` would ship the whole tree — including the datasets — through
  the object store on every submit.
- **Topology travels as entrypoint env, not in `runtime_env.yaml`.** It is a
  per-launch decision; the YAML holds configuration that should not change
  between launches.

### Sizing

Rollout is the bottleneck for a 1.7B actor — the general stage measured
`trainer/idle_ratio: 0.588`, i.e. the trainer spent 59% of each step waiting for
samples. So the split is deliberately lopsided rather than one node each:

| split | engines | note |
|---|---|---|
| 1×2 train + 2×6 rollout | 12 | default; 14/16 GPUs used |
| 1×8 train + 1×6 rollout | 6 | one node each — wastes 8 GPUs on the half that isn't the bottleneck |
| 1×2 train + 1×6 rollout | 6 | single node |

Raise `GEN_TP` only if the actor stops fitting on one GPU. A 1.7B model never
will.

## Environment across nodes

`qlcm/runtime_env.yaml` is the single source of configuration, and it only has
to exist on the node you launch from. `load_runtime_env.sh` loads it into the
training script *and* forwards the resolved values into
`ray_kwargs.ray_init.runtime_env.env_vars`, so they reach the Ray actors on
every node — including the reward workers that make the judge calls.

This is the part that used to bite: `export` in a shell script configures only
the driver process. Actors take their environment from the job runtime_env or
from the raylet, so before this helper existed you had to either export the
judge vars on every node *before* `ray start`, or hand-write
`RAY_ENV_FLAGS`. Neither is needed now.

Both launch paths work and produce the same worker environment:

```bash
# direct, from the head node
bash qlcm/run_pipeline.sh

# submitted — Ray applies the YAML itself; the loader detects the job and
# suppresses its own overrides so the two runtime_envs cannot conflict
ray job submit --runtime-env=qlcm/runtime_env.yaml \
  -- bash -c "cd /workspace/verl && bash qlcm/run_pipeline.sh"
```

`RAY_ENV_FLAGS` is still honoured if you set it, and still takes effect
alongside the generated flags — useful for a one-off override without touching
the YAML.

Verify after launch:

```bash
grep 'judge_client boot' <log>
```

It prints the resolved API base once per worker process. You want lines from
**multiple PIDs on multiple hostnames**; if you only see the driver, the
propagation did not happen.

Set `QLCM_ENV_VERBOSE=1` to have the loader print which variables it is
forwarding (names only, values elided) before training starts.

---

## Shutdown

```bash
ray stop        # on each worker, inside the container
ray stop        # then on the head
```

`ray stop --force` if anything misbehaves.

---

## Sanity checks before a long run

1. `ray status` on the head shows N nodes and 8·N GPUs.
2. `DRY_RUN=1 bash qlcm/run_pipeline.sh` prints the full plan without running.
3. On each node: `ls /data/abdelrahman/verl/data/qlcm/train/*.parquet` lists the
   four general parquets.
4. `test -f $BASE_MODEL/config.json` on every node.
5. Launch with a high `trainer.test_freq` for a few steps and watch for
   `judge_client boot` lines from multiple hostnames, and GPU utilisation on
   every node in `ray status`.
6. If weight sync or collectives hang, `export NCCL_DEBUG=INFO` on every node
   before `ray start` and check for interface-selection errors.
