# Multi-node training for full_mix

This doc covers running [train.sh](train.sh), [train_no_math.sh](train_no_math.sh),
[train_one_step_off.sh](train_one_step_off.sh), and [train_one_step_off_no_math.sh](train_one_step_off_no_math.sh)
across multiple nodes. VERL uses a Ray cluster for multi-node orchestration, so the flow is:

1. Start a Ray cluster that spans all your nodes
2. Launch the usual training script on the **head node only** — Ray distributes the work

The existing scripts already read `NNODES` (and for OSOP, `N_GPUS_ROLLOUT` / `N_GPUS_TRAINING`) from the environment, so no code changes are needed once the Ray cluster is up.

---

## Prerequisites

- **Identical docker image** on every node (same tag of `verlai/verl:vllm012.latest` you use today).
- **Identical filesystem layout** — the repo and dataset parquets must live at the same absolute paths on every node. Options:
  - **Shared storage** (NFS / Lustre / shared NVMe): mount the same `/data/abdelrahman` on every node.
  - **Rsync** the repo and data to each node before launching, at identical paths.
- **Inter-node network**: each node's LAN/IB IP must be reachable from every other node. [dev/dev.sh](../dev/dev.sh) uses `--net=host`, so container ports equal host ports — no docker networking surgery needed.
- **Same number of GPUs per node** (recommended). If heterogeneous, set `trainer.n_gpus_per_node` to the minimum and accept idle GPUs on the bigger nodes.
- **Same judge configuration**: the `FULL_MIX_JUDGE_*` env vars need to reach every node's worker actors. The scripts auto-propagate these via Ray `runtime_env` when `RAY_ENV_FLAGS` is set; see [Judge env across nodes](#judge-env-across-nodes) below.

---

## Step 1 — Start the dev container on every node

On each node (from the host):

```bash
bash dev/dev.sh
```

This drops you into a bash shell inside the container at `/workspace/verl`. Leave a terminal open on each node.

---

## Step 2 — Start the Ray cluster

Find the head node's cluster-facing IP (on the host, **not** inside the container):

```bash
hostname -I | awk '{print $1}'   # or: ip route get 1.1.1.1 | awk '{print $7}'
```

Pick one node as the head (e.g. `10.x.x.120`).

**On the head node**, inside the container:

```bash
ray start --head \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-gpus=8
```

Verify:
```bash
ray status
```
It should list 1 node with 8 GPUs.

**On every worker node**, inside the container:

```bash
ray start --address=<head-ip>:6379 --num-gpus=8
```

Back on the head node, re-check:
```bash
ray status
```
You should now see N nodes with 8·N GPUs total.

> **Troubleshooting**: if `ray start --address=...` hangs or complains about "connection refused", check that port 6379 is reachable from worker → head. For strict firewall environments, also open 10001-10999 (Ray worker ports) and 8265 (dashboard).

---

## Step 3 — Launch training on the head node

The scripts only need to be invoked on the **head node**. Ray takes care of scheduling actors across the cluster.

### Sync GRPO — train.sh / train_no_math.sh

```bash
# from inside the container on the head node
NNODES=2 N_GPUS_PER_NODE=8 bash full_mix/train_no_math.sh
```

That gives you 2·8 = 16 training GPUs. Internally VERL colocates actor + rollout on the same GPUs (standard PPO pattern), so rollout also gets 16 vLLM instances (TP=1).

### One-step-off-policy — train_one_step_off.sh / train_one_step_off_no_math.sh

OSOP splits each node's GPUs between a **rollout pool** and a **training pool**. The split is **per-node**, then Ray replicates it across all nodes. The defaults in [train_one_step_off_no_math.sh](train_one_step_off_no_math.sh) are `N_GPUS_ROLLOUT=6` and `N_GPUS_TRAINING=2`, which on a 2-node / 16-GPU cluster gives:
- 12 rollout GPUs (6 per node × 2 nodes) → 12 vLLM instances
- 4 training GPUs (2 per node × 2 nodes) → FSDP-4 training

```bash
NNODES=2 \
N_GPUS_ROLLOUT=6 \
N_GPUS_TRAINING=2 \
bash full_mix/train_one_step_off_no_math.sh
```

Alternate splits worth trying on 2 nodes:
| Split / node | Total rollout | Total training | Best for |
|---|---|---|---|
| 6R + 2T (default) | 12 | 4 | balanced sweet spot |
| 4R + 4T | 8 | 8 | training-heavy (if updates still bottleneck) |
| 7R + 1T | 14 | 2 | gen-max (only if training fits on 1 GPU) |

All of these work on 2 nodes; just set `N_GPUS_ROLLOUT` + `N_GPUS_TRAINING` so they sum to 8.

---

## Judge env across nodes

The chat + safety reward path calls an LLM judge via OpenRouter or a self-hosted vLLM. On a multi-node Ray cluster, worker actors on non-head nodes may not inherit the shell env where you launched training. Two defenses:

1. **Export on every node before `ray start`** (easiest):
   ```bash
   # on each node, in the container, BEFORE ray start
   export FULL_MIX_JUDGE_API_BASE=https://openrouter.ai/api/v1
   export FULL_MIX_JUDGE_API_KEY=sk-or-v1-...
   export FULL_MIX_JUDGE_MODEL=x-ai/grok-4.1-fast
   export FULL_MIX_JUDGE_DISABLE_THINKING=0

   ray start ...
   ```
   Ray worker processes inherit the raylet's env, so this propagates correctly.

2. **Use Ray runtime_env** (robust against cluster restarts): set `RAY_ENV_FLAGS` before launching to pass the env vars through Ray's runtime_env (the OSOP scripts already interpolate `${RAY_ENV_FLAGS}`):
   ```bash
   RAY_ENV_FLAGS=$(for v in FULL_MIX_JUDGE_API_BASE FULL_MIX_JUDGE_API_KEY \
                          FULL_MIX_JUDGE_MODEL FULL_MIX_JUDGE_DISABLE_THINKING; do
     echo -n "+ray_kwargs.ray_init.runtime_env.env_vars.${v}=${!v} "
   done) \
   NNODES=2 bash full_mix/train_one_step_off_no_math.sh
   ```

Whichever path you choose, verify after launch by grepping logs for the `[judge_client boot ...]` line — it prints the resolved API base on every worker process.

---

## Shutdown

Clean shutdown order:
```bash
# on each worker node, inside the container
ray stop

# on the head node, inside the container
ray stop
```

Or kill the whole cluster with `ray stop --force` if anything misbehaves.

If you're fully done, `exit` the container shell and re-run `dev/dev.sh` later (it auto-recreates with `RECREATE=1`).

---

## Sanity checks before a long run

1. **Cluster visibility**: `ray status` on the head node shows N nodes, 8·N GPUs.
2. **Data availability**: on each node, inside the container, `ls /data/abdelrahman/verl/data/full_mix/train/*.parquet` should list the four parquets.
3. **Model path**: `test -f $MODEL_PATH/config.json` on each node, or `load_from_disk` succeeds on a Python smoke test.
4. **Dry run**: launch with `trainer.total_epochs=1` and a high `trainer.test_freq` for a few steps, watch for:
   - `[judge_client boot pid=...]` lines from **multiple PIDs on multiple hostnames** (confirms judge config reached all workers).
   - `ray status` output during training — all nodes should show GPU utilization.
5. **NCCL**: if weight sync or collective ops hang, `export NCCL_DEBUG=INFO` on every node before `ray start` and re-launch; check logs for interface-selection errors.

---

## Performance expectations

With math dropped and a 2-node OSOP setup (12R / 4T):
- Gen floor: ~350-450 s/step (12 vLLM instances instead of 6; straggler tail stays similar)
- Train: ~200 s/step (4 GPUs, no offload likely fits)
- Step time: **~400-450 s** — ~40% better than the single-node 720 s

If you see step times much worse than this, the bottleneck is probably either:
- **Stale Ray cluster env** (see [Judge env across nodes](#judge-env-across-nodes))
- **Inter-node network bandwidth** during weight sync — check with `timing_s/sync_rollout_weights` on wandb; should be <5 s/sync on 25+ Gbps links.
