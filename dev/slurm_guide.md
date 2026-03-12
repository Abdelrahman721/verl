# SLURM Training Guide — MedQA GRPO

## Cluster info

- **Cluster**: pi-heavy (4 nodes, 8×H100 80GB each)
- **SLURM binaries**: `/data/slurm/bin/` (add to PATH)
- **Container runtime**: pyxis (NVIDIA enroot) — use `--container-image` with srun
- **Docker image**: `verlai/verl:vllm012.latest`

```bash
# Always do this first on pi-heavy
export PATH=/data/slurm/bin:$PATH
```

---

## Single-node (1×8 H100)

### Files

| File | Purpose |
|---|---|
| `dev/slurm_train.sbatch` | SLURM submission script |
| `rl_scripts/grpo_qa_8h100.sh` | Training hyperparameters |

### Submit

```bash
export QA_JUDGE_API_KEY="your-gemini-key"
export WANDB_API_KEY="your-wandb-key"
sbatch dev/slurm_train.sbatch
```

### Config summary

| Parameter | Value |
|---|---|
| Nodes | 1 |
| GPUs | 8 |
| TP (vLLM rollout) | 2 |
| Samples per prompt | 16 |
| Train batch size | 4 prompts × 16 samples = 64 samples/step |
| Gen batch size | 12 |
| Steps per epoch | ~1234 |
| Judge calls per step | 64 |

---

## Multi-node (2×8 H100)

### Files

| File | Purpose |
|---|---|
| `dev/slurm_train_multinode.sbatch` | SLURM submission script |
| `rl_scripts/grpo_qa_multinode.sh` | Training hyperparameters |

### Submit

```bash
export QA_JUDGE_API_KEY="your-gemini-key"
export WANDB_API_KEY="your-wandb-key"
sbatch dev/slurm_train_multinode.sbatch
```

### Config summary

| Parameter | Value |
|---|---|
| Nodes | 2 |
| GPUs | 16 total (8 per node) |
| TP (vLLM rollout) | 2 |
| Samples per prompt | 16 |
| Train batch size | 8 prompts × 16 samples = 128 samples/step |
| Gen batch size | 16 |
| Steps per epoch | ~617 |
| Judge calls per step | 128 |

### Multi-node notes

- verl uses Ray under the hood for multi-node coordination
- NCCL env vars are set in the rl script for cross-node GPU comms
- The model must be accessible from all nodes at the same path (`/home/hazem/serve/no-paraphrase`)
- If the model is on local disk, either use shared storage (NFS) or copy to each node before training

---

## Monitoring

### Logs

```bash
# Follow the log of a running job
tail -f /home/hazem/verl-rl/train-<JOBID>.log      # single-node
tail -f /home/hazem/verl-rl/train-mn-<JOBID>.log   # multi-node
```

### Job status

```bash
squeue -u hazem                    # list your jobs
squeue -u hazem -o "%.8i %.9P %.20j %.2t %.10M %.6D %.20R"  # detailed
scancel <JOBID>                    # kill a job
sacct -j <JOBID> --format=JobID,Elapsed,MaxRSS,State  # post-mortem
```

### Getting a shell on the running job's node

```bash
# Bare-metal shell on the node (outside container)
# Good for: nvidia-smi, htop, checking disk, network
srun --jobid=<JOBID> --overlap --pty bash

# Shell inside the container
# Good for: inspecting Python env, checking verl internals, debugging
srun --jobid=<JOBID> --overlap \
  --container-image=verlai/verl:vllm012.latest \
  --container-mounts="/home/hazem/verl:/workspace/verl" \
  --container-workdir=/workspace/verl \
  --pty bash
```

For multi-node jobs, the above lands on the **first node**. To pick a specific node:

```bash
srun --jobid=<JOBID> --overlap --nodelist=denvrbm-2023 --pty bash
```

### GPU monitoring

```bash
# From bare-metal shell on the node
nvidia-smi                         # snapshot
watch -n 2 nvidia-smi              # live refresh every 2s
nvidia-smi dmon -s u -d 5          # utilization sampled every 5s
```

### WandB

Training logs to WandB project `MedQA-RL`. Check:
- `reward/judge_score` — the 0-1 multi-dimensional reward
- `reward/accuracy`, `reward/completeness`, `reward/clarity` — dimension breakdown
- `reward/format_penalty` — format compliance
- `reward/reasoning_length`, `reward/answer_length` — generation stats

---

## Troubleshooting

### Container fails to start

```
pyxis: container start failed
enroot-mount: failed to mount: /home/hazem/.cache/huggingface ... No such file or directory
```

**Fix**: The cache dir doesn't exist on the allocated node. SSH in and create it:
```bash
ssh denvrbm-XXXX
mkdir -p ~/.cache/huggingface ~/.cache/torch
```
The sbatch scripts already do this via `mkdir -p` before `srun`, but if a new mount is added, update accordingly.

### OOM on GPU

Reduce micro batch sizes in the rl script:
```bash
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2   # down from 4
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
```

Or reduce `gpu_memory_utilization` for vLLM (trades generation speed for less memory):
```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.6   # down from 0.8
```

### Gemini rate limits

If you see `429 Too Many Requests` in logs, reduce concurrency or batch size:
- In `verl/utils/reward_score/qa.py`: the calls are sequential per sample, so rate is bounded by `train_batch_size × n_samples / step_duration`
- Reduce `TRAIN_BATCH_SIZE` to lower judge calls per step
- Gemini Flash paid tier: 2000 RPM, 4M TPM — shouldn't hit this with conservative batches

### NCCL timeout (multi-node)

```
NCCL WARN Timeout on barrier
```

Check that nodes can reach each other on high-speed network. Try:
```bash
export NCCL_SOCKET_IFNAME=eth0   # or the correct interface
export NCCL_DEBUG=INFO           # for verbose logs
```

### Model not found

The model path must be identical on all nodes. If using local disk:
```bash
# Copy model to each node (run from login node)
for node in denvrbm-2022 denvrbm-2023; do
  ssh $node "mkdir -p /home/hazem/serve"
  rsync -avP /home/hazem/serve/no-paraphrase/ $node:/home/hazem/serve/no-paraphrase/
done
```
