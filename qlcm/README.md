# QLCM — sequential GRPO pipeline for a Qwen3-1.7B medical model

This directory runs the same multi-stage GRPO pipeline that produced the 32B
production model in [`full_mix/`](../full_mix), pruned to only the stages that
are on the path to the best checkpoint, and re-targeted at a CPT'ed + SFT'ed
**Qwen3-1.7B-Base** actor.

```
general → 1 → 2 → 3 → 4 → 5 → 6 → 8
```

One command runs the whole thing:

```bash
cp qlcm/runtime_env.yaml.example qlcm/runtime_env.yaml   # fill in the 4 API keys
bash qlcm/preprocess_all.sh                    # general train/eval parquets
bash qlcm/run_pipeline.sh                      # all eight stages, in order
```

---

## What was removed, and why

The 32B run had eleven training runs. Three of them are not on the path to the
checkpoint that scored best, so they are not in this directory:

| Removed | Reason |
|---|---|
| `medical-qa-stage7-coding` | Dead end. It reached step 34 and was never merged forward — stage 8 re-branched from **stage 6 / step 42** instead. Nothing downstream depends on it. |
| `medical-qa-stage9-mix` | Ran *after* the best checkpoint. Stage 8's output was the best-performing model, so stage 9 is off the path. |
| `medical-qa-stage10-ddx` | Same — a DDx-ranking finisher applied after stage 9. |

Because stage 8 already branched off stage 6 in the original run, dropping
stage 7 changes nothing about the chain: **stage 8 still starts from stage 6.**
The stage numbering is deliberately left with a gap at 7 so the mapping back to
the original experiments stays 1:1.

Also dropped, as they were only ever used by the removed stages or by R&D
branches that never touched the production chain:

- `rewards/ddx_rank.py` (stage 10 only)
- `rewards/qa_bedrock.py`, `rewards/qa_openai.py` — superseded judge transports.
  Medical stage 1 ran one day on Bedrock before switching to OpenRouter; this
  pipeline starts on OpenRouter.
- `rewards/lcpo*.py`, `lcpo/`, `sft_modes/`, `common/budget_prompt.py`,
  `common/lcpo_schema.py` — length-controlled-policy-optimisation and dual-mode
  SFT workstreams. (`per_source_metrics.py` still carries its LCPO wandb-curve
  branch; it is guarded and returns nothing when the batch has no LCPO keys,
  which is always here. Left in place rather than surgically removed.)
- `rewards/compute_score_ifeval.py`, `rewards/ifeval_quality_gated.py` —
  IFEval-only R&D.
- `preprocess/shard_train_data.py`, `curriculum/build_next_stage*.py`,
  `run_curriculum.sh` — the 4-shard curriculum driver, written for the 4B
  experiments and never used at scale.
- `preprocess/math_train.py` — math was dropped before the production run.
  GSM8K and MATH-500 remain as **eval** sets in the general phase.

---

## The chain

Every stage starts from the previous stage's merged HF checkpoint. `run_pipeline.sh`
resolves that automatically, merging FSDP shards when needed.

| Stage | Script | Experiment | Trains on | Reward |
|---|---|---|---|---|
| `general` | `train_general.sh` | `general-phase` | IFEval + chat + safety + identity | `compute_score.py` |
| `1` | `train_medical_qa_stage1.sh` | `medical-qa-fresh` | medical QA corpus + curated general retention | `compute_score_medical_mix.py` |
| `2` | `train_medical_qa_stage2.sh` | `medical-qa-stage2` | stage-2 curriculum (carry-forward + unseen conversations) | `qa_openrouter.py` |
| `3` | `train_medical_qa_stage3.sh` | `medical-qa-stage3` | replay pools + medical benchmarks | `qa_openrouter_bench.py` |
| `4` | `train_medical_qa_stage4.sh` | `medical-qa-stage4-coding` | SNOMED single-label + ICD multi-label, retention folded in | `compute_score_medical_mix.py` |
| `5` | `train_medical_qa_stage5.sh` | `medical-qa-stage5-coding` | ICD + SNOMED × multilabel + instruction-following, + retention | `compute_score_coding_mix.py` |
| `6` | `train_medical_qa_stage6.sh` | `medical-qa-stage6-coding` | **real** clinical-note coding, + retention | `compute_score_coding_mix.py` |
| `8` | `train_medical_qa_stage8.sh` | `medical-qa-stage8-coding` | coding only, **no retention** — final stage | `compute_score_coding_mix.py` |

Between stages, `run_pipeline.sh` scores the previous stage's rollout dumps and
builds the next stage's parquets — that is what makes this a curriculum rather
than a fixed schedule. A prompt the model has mastered (mean rollout score ≥
`THRESHOLD`, default 0.6) is dropped or demoted to retention; unmastered
prompts are carried forward.

---

## What changed for 1.7B

Only the actor and the things that scale with it. Algorithm, curriculum, reward
routing, judges and data are unchanged.

| Knob | 32B run | Here | Why |
|---|---|---|---|
| `MODEL_PATH` | `out-qwen-32b/coding-mixed-sft/final` | `out-qwen3-1.7b/coding-mixed-sft/final` | the target model |
| Cluster | 2 train + 3 rollout nodes (5 machines, 40 GPUs) | 1 node: 2 train + 6 rollout GPUs | a 1.7B actor does not need 40 GPUs |
| `GEN_TP` | 4 | **1** | 1.7B fits on one GPU; TP=1 gives one vLLM engine per rollout GPU |
| `SP_SIZE` | 2 → 8 | **1** | no Ulysses sequence parallelism needed |
| `FSDP_SIZE` | 16 | `NNODES_TRAIN * NGPUS_TRAIN` | follows the train pool |
| `ACTOR_OFFLOAD` / `REF_OFFLOAD` | True | **False** | actor + optimizer fit in HBM; offload is pure overhead |
| `MAX_NUM_SEQS` | 64 | **256** | far more free KV cache per engine |
| `MAX_RESPONSE_LEN` | 16,384 | **8,192** | measured p50 is 996 tokens; ~10% ran to the 16k cap and starved the trainer |
| `PROJECT_NAME` | `RL-Exps` | `QLCM` | keeps wandb and checkpoints separate |

Node counts stay env-overridable. To reproduce the original topology:

```bash
NNODES_TRAIN=2 NGPUS_TRAIN=8 NNODES_ROLLOUT=3 NGPUS_ROLLOUT=8 \
  bash qlcm/run_pipeline.sh
```

For a real multi-node run use `qlcm/submit.sh`, which wraps `ray job submit`
with the topology, entrypoint quoting and preflight checks — see
[multinode.md](multinode.md).

Everything else — GRPO with `norm_adv_by_std=False`, `rollout.n=16`, clip
0.2/0.28 with `clip_ratio_c=10`, lr 1e-6 with 20 warmup steps, token-mean loss,
`staleness_threshold=0.25`, token-level rollout importance sampling at
threshold 2.0, and the 1,024-token overlong buffer, KL-in-reward off for the general phase and on at 1e-4 for every medical
stage — is carried over verbatim.

> The 1.7B checkpoint has `max_position_embeddings=32768`, so the 8,192 prompt +
> 8,192 response budget used from stage 6 onward leaves ample headroom.

## Judges

Three judges score different slices. All read credentials from
`runtime_env.yaml`, and every training script preflight-checks
`QLCM_JUDGE_API_KEY` and exits 2 if it is empty or still the template
placeholder — an unset key silently turns judged rows into a constant reward,
which is a very expensive way to find out.

| Env prefix | Model | Scores |
|---|---|---|
| `QLCM_JUDGE_*` | `deepseek/deepseek-v4-flash-0731`, **provider-pinned** | chat, safety, identity |
| `QA_JUDGE_*` | `openai/gpt-5.4-mini` | medical QA, conversations, bench text/medec |
| `OPENROUTER_API_KEY` + `IF_USE_LLM_JUDGES=1` | `deepseek/deepseek-v3.2`, hardcoded | SNOMED instruction-following (stages 5-6) |

### When the judge fails

A judge failure is not a verdict. `chat.py` returns 0.5 and `safety.py` returns
`QLCM_SAFETY_JUDGE_FAILURE_REWARD` (0.5). Safety previously returned 0.0, which
turned a transient upstream 429 into a confident zero for a response that had
correctly refused — and under GRPO that shifts advantages for every sample in
the group, not just the affected one.

Genuine zeros are untouched: a wrong safety stance, a score at or below 4, and
an empty response all still score 0.0.

Both safety and chat now send a strict `json_schema` plus a `salvage` parser, so
the failure path is reached far less often. Adding the schema took deepinfra —
which produced 2/10 malformed verdicts on the schema-less identity gate — to 6/6
correct on safety.

### Global identity gate — OFF

`QLCM_IDENTITY_GATE` defaults to `0`. The gate issued one extra judge call per
non-identity sample and was the dominant source of judge failures: it uses
`json_object` mode with no schema, and providers in the pool returned degenerate
objects with the key names stripped — `{": false, ": false, "brief_reason": …}`
instead of `{"false_model_identity": false, …}`. Those fail validation, burn all
four attempts, then fail open anyway.

Turning it off is close to behaviourally neutral, because the gate already
failed open on every error. What you lose is catching an identity claim made on
a *non*-identity row. `local/avey-identity` rows are unaffected — they are
scored by `reference_aligned_identity_score`, which never used this gate.

Set `QLCM_IDENTITY_GATE: "1"` in `runtime_env.yaml` to re-enable.

**The general judge is pinned to a named provider pool**, following the LCPO
runs' configuration:

```
QLCM_JUDGE_PROVIDER_ONLY  = together,novita,makora,baseten,baidu,deepinfra
QLCM_JUDGE_PROVIDER_ORDER = together,novita,makora,baseten,baidu,deepinfra
QLCM_JUDGE_REQUIRE_PARAMETERS = 0
```

The model is served by many independently-operated OpenRouter deployments at
quantizations from fp4 to bf16, and they are not interchangeable — unpinned,
sampled calls spread across ~8 providers and one of them (Morph, bf16) produced
4k-13k-token repetition loops that failed JSON parsing. `ONLY` is a permission
set; `ORDER` is the priority list, and it matters: without it OpenRouter
load-balances within the pool and the slowest member sets the median latency.

`REQUIRE_PARAMETERS` must stay **0**. The judge client defaults it to 1, which
routes only to providers implementing every parameter sent — that excludes
baseten (no `response_format`), and a request pinned to it 404s. With it off, OpenRouter strips the parameter for those two and the judge
follows the prompt alone; `_salvage_verdict` in `rewards/chat.py` is the net
underneath. `TEMPERATURE=0.6` is also deliberate: at 0.0 a pinned provider is
deterministic, so a malformed verdict regenerates byte-identically on every
retry.

The full rationale, including why fireworks and morph are excluded, is in the
judge block of any `train_*.sh`.

Pool measured 2026-09-02 with the real identity-gate call, one attempt each:

| provider | n | ok | malformed | other |
|---|---:|---:|---:|---:|
| together | 20 | 20 | 0 | 0 |
| novita | 20 | 20 | 0 | 0 |
| makora | 20 | 19 | 0 | 1 (429) |
| baseten | 10 | 9 | 0 | 1 (no choices) |
| baidu | 10 | 2 | 0 | 8 (429) |
| deepinfra | 10 | 8 | **2** | 0 |
| ~~digitalocean~~ | 10 | 4 | **6** | 0 |

digitalocean was dropped — it invented key names (`false_foundation_lineage`)
the validator rejects. makora replaced it. deepinfra is kept last as
last-resort overflow. baidu is fine on retries but rate-limits heavily, so it
sits low in the order rather than first.


> The SNOMED-IF judge in
> `rewards/coding/instruction_following/judges.py` is a separate code path with
> its own rubric. It is still on `deepseek/deepseek-v3.2` with only a 4-provider
> *denylist*, and the model is hardcoded rather than env-configurable. Left as
> it was, since changing it changes the reward for coding-IF rows.

---

## Layout

```
qlcm/
├── run_pipeline.sh          sequential driver: data → train → merge → next
├── preprocess_all.sh        general train/eval parquets from the raw sources
├── setup.sh                 per-node dependency install
├── train_general.sh         general phase
├── train_medical_qa_stage{1,2,3,4,5,6,8}.sh
├── rewards/                 reward dispatchers + judges + coding scorers
├── curriculum/              per-stage data builders, rollout scorers, harmoniser
├── preprocess/              raw dataset → verl parquet
├── common/                  judge client, strip_think
├── ifeval/                  IFEval constraint checker
├── load_runtime_env.sh      loads runtime_env.yaml + forwards it to Ray actors
└── runtime_env.yaml.example copy to runtime_env.yaml (gitignored)
```

## Data

Derived, per-run data lives under `data/qlcm/` so it never collides with the
32B run's. The raw medical corpus is a shared read-only input.

```
data/qlcm/train/            general train parquets      (preprocess_all.sh)
data/qlcm/eval/             gsm8k / math500 / ifeval    (preprocess_all.sh)
data/qlcm/curated/          curated general retention   (built from general rollouts)
data/qlcm/medical_qa_stage{2,3,4,5,6,8}/   per-stage curriculum
data/medical_qa/            raw medical corpus          (shared, read-only)
```

Set `BENCH_PARQUET` before stage 3 — it folds in the medical benchmark pool and
has no default. The SNOMED and ICD coding sources are module constants in
`curriculum/build_medical_qa_stage{4,5,6,8}.py`.

---

## Two things the original did by hand

**Schema harmonisation.** Stages 5 and 6 hand verl *two* train parquets, which
HF `datasets` concat-loads into one dataset — so both files must carry an
identical Arrow schema. Nothing in `full_mix/` did that reconciliation; the only
evidence it happened is the `*.preharmonise.bak` files left beside the outputs.
`curriculum/harmonise_train_files.py` is that pass, made reproducible, and
`run_pipeline.sh` calls it automatically after building stages 5 and 6.

Run against the original's own pre-harmonisation backups it reproduces the
39-field union exactly, minus `extra_info.code`, which it drops as a genuine
type conflict (`list<list<string>>` vs `list<string>`) — the same field
`build_medical_qa_stage6.py` drops for the same reason.

**Checkpoint merging.** `run_pipeline.sh` merges each stage's final FSDP
checkpoint to `merged_hf_model/` before the next stage loads it. Pass
`SKIP_MERGE=1` to do it yourself.

## Stage-specific validation

Stages 2, 3 and 4 have no val-set builder — the 32B run carved those by hand —
so they default to the shared 48-row medical QA val split. Stages 5, 6 and 8
build their own. Override `VAL_FILES` if you carve stage-specific sets.

---

## Environment configuration

One file: **`qlcm/runtime_env.yaml`** (gitignored; copy the `.example`). It is a
Ray `runtime_env` document, so it works unchanged with `ray job submit`.

`load_runtime_env.sh`, sourced by every script, does two things with it:

1. Loads the `env_vars:` block into the script's own environment, filling only
   variables that are not already set.
2. Builds `_QLCM_RAY_FLAGS[]` — Hydra overrides that push the resolved values
   into `ray_kwargs.ray_init.runtime_env.env_vars`, so they reach the Ray
   **actors**. This matters: `export` in a shell script configures only the
   driver, while reward and rollout workers take their environment from the job
   runtime_env or the raylet. Without step 2, the judge settings in the training
   scripts would never reach the workers that actually call the judge.

Precedence:

```
shell env  >  runtime_env.yaml  >  train-script defaults
```

so `QLCM_JUDGE_MODEL=x bash qlcm/train_general.sh` still wins for a one-off.

Both launch paths work and agree:

```bash
bash qlcm/run_pipeline.sh                                        # direct
ray job submit --runtime-env=qlcm/runtime_env.yaml -- \
  bash -c "cd /workspace/verl && bash qlcm/run_pipeline.sh"      # submitted
```

Under `ray job submit --runtime-env=...` Ray applies the YAML itself — to the
entrypoint **and** to every actor — so both steps stand down: step 1 finds each
variable already set, and step 2 emits nothing.

Step 2 standing down is not an optimisation. Ray rejects a driver-supplied
runtime_env that repeats any key the job runtime_env already declares, **even
with an identical value** (`Failed to merge the Job's runtime env … because of a
conflict`), and the job dies at `ray.init()`. The loader detects the job via
`RAY_JOB_CONFIG_JSON_ENV_VAR` and suppresses its own overrides. A job submitted
*without* `--runtime-env` sets that variable but carries no `env_vars`, and
there the loader does propagate — otherwise the actors would get nothing.

Values containing commas (the provider lists) are single-quoted in the Hydra
override so they parse as strings rather than lists. A value containing a
literal single quote cannot be expressed this way; the loader skips it with a
warning rather than emitting something malformed.

## Operating notes

- Run everything inside the container from `dev/dev.sh`; `setup.sh` installs
  the extra deps on each node.
- All environment configuration lives in **one** file, `qlcm/runtime_env.yaml`,
  and it only has to exist on the node you launch from. `load_runtime_env.sh`
  loads it into the script *and* forwards the resolved values to the Ray actors,
  so `bash qlcm/run_pipeline.sh` and
  `ray job submit --runtime-env=qlcm/runtime_env.yaml -- ...` behave identically.
  Confirm with `grep 'judge_client boot' <log>` — it prints the resolved API base
  once per worker process.
- On the 200+ CPU hosts, pass `--num-cpus` explicitly to `ray start`, or a
  second `ray job submit` hangs and takes the head raylet down with it.
- `DRY_RUN=1 bash qlcm/run_pipeline.sh` prints every command without running
  anything — worth doing once before a long run.
- `START_STAGE=4 bash qlcm/run_pipeline.sh` resumes mid-chain.
