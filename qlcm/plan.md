# QLCM — pipeline design notes

Reference for how the pieces fit together. For how to *run* it, see
[README.md](README.md); for spreading it across machines, [multinode.md](multinode.md).

## Context

Sequential GRPO on a CPT'ed + SFT'ed **Qwen3-1.7B-Base** actor, via verl's
`fully_async_policy` entrypoint. Eight stages, each starting from the previous
stage's merged checkpoint, taking the model from general alignment through
medical QA and benchmarks to real clinical-note coding.

This is the [`full_mix/`](../full_mix) 32B pipeline with the three off-path
stages removed (7, 9, 10) and the actor swapped. Algorithm, curriculum logic,
reward routing and judges are unchanged.

## Fully-async resource model

Rollout and training run on **disjoint** GPU pools inside one Ray cluster.
Generation streams continuously; the trainer pulls batches as they complete and
pushes weights back every `trigger_parameter_sync_step` optimizer steps.

- `staleness_threshold=0.25` — a training batch may be up to 25% stale samples
- `partial_rollout=True` — long in-flight generations are preempted at weight
  sync rather than stalling the step on a straggler
- `rollout_correction.rollout_is=token` at threshold 2.0 — token-level
  importance-sampling correction, required once training is decoupled from
  rollout
- `use_rollout_log_probs=True` — pinned explicitly rather than inherited

## `data_source` dispatch keys

verl routes rewards by each row's `data_source` column. The reward function is a
batch-signature dispatcher that partitions the batch by key, scores each family
with its native scorer (so the LLM-judge families and the deterministic ones run
concurrently), then reassembles in original order.

**General** — `local/dolci-ifeval-32b`, `local/dolci-chat-32b`,
`local/safety-dpo-reference`, `local/avey-identity`

**Medical** — `medical_qa`, `medical_conv`

**Benchmarks** — `medical_benchmark_mcq`, `medical_benchmark_numeric`,
`medical_benchmark_text`, `medical_benchmark_medec`

**Coding** — `snomed_sl`, `snomed_ml`, `snomed_if`, `sl`, `ml`, `if`, `mlb`, `slb`

**Eval** — `openai/gsm8k`, `HuggingFaceH4/MATH-500`, `google/IFEval`

Coding rows carry `extra_info.task_type` (`icd10_multilabel`,
`snomed_multilabel`, `snomed_instruction_follow`, …), which the coding
dispatcher sub-routes on.

## Row format

Every parquet row, general or medical:

```python
{
  "data_source": <key above>,
  "prompt": [{"role": "user", "content": <user turn>}],
  "ability": "instruction_following" | "chat" | "safety" | "medical" | ...,
  "reward_model": {"style": "rule", "ground_truth": <string>},
  "extra_info": {"split": ..., "index": ..., "task_type": ..., ...},
}
```

`extra_info` is a struct whose field set widens as the pipeline progresses —
by stage 5 it is a ~38-field union. When verl loads two train parquets as one
dataset, both must agree on that struct exactly; see *Schema harmonisation*.

## Reward dispatchers

| Module | Branches |
|---|---|
| `compute_score.py` | IFEval constraint checker (rule) · chat judge · safety judge · identity judge, plus a global identity gate applied to non-identity rows |
| `qa_openrouter.py` | OpenRouter judge over QA key points and the conversation rubric |
| `qa_openrouter_bench.py` | the above, plus the four benchmark scorers — mcq and numeric are deterministic (no judge call), text and medec are judged |
| `compute_score_medical_mix.py` | medical + bench → `qa_openrouter_bench` · everything else → `compute_score` |
| `compute_score_coding_mix.py` | coding → `rewards.coding` (regex P/R/F1 on ICD codes and SCT IDs, plus the IF framework) · medical + bench → `qa_openrouter_bench` · general retention → `compute_score` |

Every scorer receives `strip_think(solution_str)`, so `<think>…</think>` traces
never reach a grader.

Per-family return dicts have different key sets, so the mix dispatchers pad
every output to the union of all branches — verl's `DataProto.concat` requires
the first row's dict to be a superset of the rest.

### Normalisation

- IFEval / mcq / numeric / coding F1 → already in [0, 1]
- Chat (judge 1–10) → `(score - 1) / 9`
- Safety → 0 if `stance_correct` is False, else `max(0, (score - 4) / 6)`
- Judge refusal → `QA_JUDGE_REFUSAL_REWARD` (0.5), dumped to
  `QA_JUDGE_REFUSAL_DIR` if set
- Judge failure after retries → a NEUTRAL reward, not 0.0: chat 0.5,
  safety `QLCM_SAFETY_JUDGE_FAILURE_REWARD` (0.5). A failed judge produced no
  verdict, so scoring it 0.0 would assert "wrong stance" / "worst response" on
  the strength of an upstream 429. A genuine wrong-stance verdict still scores
  0.0.

### Length cost

A linear cost on TOTAL words (think + answer) is folded in by
`rewards/length_penalty.py` at each dispatcher's outer entry point:

    penalty = -0.07 * total_words / 1000

No threshold, no cap, no dead zone. Because this pipeline runs
`norm_adv_by_std_in_grpo=False`, the advantage is `r_i - mean(r_group)`, so an
additive term contributes exactly `-0.07 * (L_i - mean_group(L)) / 1000`: the
constant part cancels against GRPO's own baseline and only the deviation from
the group survives. The group mean IS the reference length, which means there
is no threshold to choose and nothing for the policy to pile up beneath, and
the pressure is per-prompt adaptive — a hard question where every sample runs
long incurs no net cost, because the mean moves with them. Only being longer
than your peers on the same question is charged.

Scores may go negative. That matches how verl's DAPO manager applies its own
overlong penalty (`reward += overlong_reward`, unfloored); clamping at zero
would put a flat region back at the long tail.

Why it exists: stage 1 of the first 1.7B run drifted 95% longer in the think
block over 373 steps (680 -> 1,328 words) while the answer stayed flat
(481 -> 499). Both nominal brakes were inert — the DAPO overlong buffer only
bites above 7,168 tokens and the longest response ever emitted was 6,619, while
`_compute_length_penalty` watches the answer at a 14x gold threshold and fired
on 0.0-0.1% of rows. The answer was the only watched quantity, so it was the
only bounded one: the mirror image of the failure in `lcpo.py`, and the reason
this cost is on the total rather than on the think span.

The extra length buys nothing. Within a single prompt's own 16 samples the
longer half scored higher 50.4% of the time versus lower 49.4% (paired
t = -0.08) across 2,481 prompts. The population-level "longer is worse" trend
is confounded by prompt difficulty and vanishes under that comparison. The
drift is free to cut — and its lack of correlation with reward is also why it
grew, since GRPO had no gradient either way.

Calibration, on 39,678 rollouts from the last quarter of that run: within-group
score std 0.084, within-group length std 297 words. At 0.07/1k the length
signal is 0.021, or 0.25x the task spread — enough to break ties between
equally-good samples, too little to overturn a real quality difference. It cuts
149 words per group for 0.0042 of judge score.

A rejected alternative worth recording: a multiplicative form
(`score * exp(-L/L0)`) is quality-gated for free, so a zero-scoring answer
gains nothing by being short, guarding against a "give up quickly" attractor.
That attractor does not exist here — 0 of 2,481 groups score all-zero, 0 have a
score spread under 0.01, and only 0.5% have a best sample below 0.20, because
the judge returns graded rather than binary scores. Meanwhile, with
std-normalisation off, multiplicative's ~20% shrink of the reward scale is a
straight ~20% cut to advantage magnitude — a stealth learning-rate drop.

An earlier design switched on at 2,000 words and saturated at 2,800. It left
67.2% of real rollouts with an exactly-zero length gradient (60.9% below the
knee, 6.2% at the cap) and steered only inside an 800-word band — a constraint
rather than an incentive. Replaced for that reason.

Disable with `QLCM_LEN_PENALTY_ENABLE=0`; tune with
`QLCM_LEN_COST_PER_1K_WORDS`. `qlcm/rewards/test_length_penalty.py` replays it
against the stage-1 dumps and asserts the calibration and the zero dead zone
still hold.

## Judges

| Env prefix | Default | Used by |
|---|---|---|
| `QLCM_JUDGE_*` | OpenRouter `deepseek/deepseek-v4-flash-0731`, pinned to a 6-provider pool | chat, safety, identity |
| `QA_JUDGE_*` | OpenRouter `openai/gpt-5.4-mini` | medical QA, conversations, bench text/medec |
| `OPENROUTER_API_KEY` + `IF_USE_LLM_JUDGES=1` | DeepSeek via OpenRouter | SNOMED instruction-following |

All three read from `runtime_env.yaml`, loaded by `load_runtime_env.sh`
(which also forwards them to the Ray actors — see the README). The judge client is a module-level singleton
with retries on timeout/429/5xx and tolerant JSON extraction (handles fenced
blocks and stray prose). It logs one `[judge_client boot pid=…]` line per worker
process with the resolved API base — the fastest way to confirm credentials
reached every node.

## Curriculum mechanics

Each stage dumps its rollouts to `$DUMPS_DIR/<stage>_rollouts/*.jsonl`. The
`stats_*` scripts read those dumps, group by prompt, and compute a mean reward
per prompt against the stage's source parquet. The `build_*` scripts consume
that cache:

- **mastered** (mean ≥ `THRESHOLD`, default 0.6) → dropped, or sampled thinly
  into the retention pool
- **unmastered** → carried forward into the next stage's train set
- **unseen** → sampled to introduce new material

`build_curated_subset.py` does the same for the general sets before medical
stage 1, keeping every hard prompt (< 0.6) and sampling easy ones (> 0.8) to a
target 20% of the output.

Retention pools are what stop the model regressing on earlier tasks while it
specialises. Stage 8 deliberately has none — it is the pure-coding finisher.

## Schema harmonisation

Stages 5 and 6 pass verl two train parquets. HF `datasets` concat-loads them
into a single dataset, so both files need an identical Arrow schema.
`curriculum/harmonise_train_files.py` reconciles them: drops non-canonical
columns, normalises `prompt` to `list<struct<role, content>>` and
`reward_model` to `struct<style, ground_truth>`, widens `extra_info` to the
union of both files' fields, and promotes `string → large_string`.

Genuine type conflicts are fatal unless `--drop-conflicts` is passed.
`extra_info.code` is one: `list<list<string>>` on one side,
`list<string>` on the other. It is pure telemetry, redundant with
`reward_model.ground_truth`, and `build_medical_qa_stage6.py` already drops it
for exactly this reason — so `run_pipeline.sh` passes `--drop-conflicts`.

## Sequence budget

`max_prompt_length` steps 2,048 → 4,096 (stage 2) → 8,192 (stage 6, when real
clinical notes arrive). `max_response_length` is 8,192 throughout, with a
DAPO-style overlong buffer: the last 1,024 tokens of that cap are a soft
penalty zone, so the model is pushed to finish before vLLM hard-truncates.

16,384 total fits inside the 1.7B checkpoint's 32,768 `max_position_embeddings`.

The response cap was 16,384 (inherited from the 32B run) until it was measured
on the general stage: p50 996 tokens, but ~10% of rollouts ran flat into the cap
without emitting EOS, holding a rollout slot for ~6 minutes each and leaving the
trainer idle 59% of the step. 8,192 halves the worst case and puts ~13% of
responses in the overlong penalty zone instead of ~2%.

## Files reused from verl unchanged

- `verl/experimental/fully_async_policy/fully_async_main.py` — entrypoint
- `verl/workers/reward_manager/` — `dapo` manager, selected by config
- `verl/model_merger` — FSDP → HF checkpoint merge between stages
- `verl/utils/reward_score/{math_reward,gsm8k}.py` — eval scorers

`per_source_metrics.py` monkey-patches `verl.trainer.ppo.metric_utils` to emit
per-`data_source` reward curves in wandb, which is what makes a mixed batch
readable. It is imported for its side effect by `compute_score.py`.

## Verification before a long run

1. `bash qlcm/preprocess_all.sh` → four train parquets + three eval parquets
2. `DRY_RUN=1 bash qlcm/run_pipeline.sh` → full plan, nothing executed
3. Launch stage `general` with a high `trainer.test_freq`, confirm
   `judge_client boot` lines from multiple hostnames and per-`data_source`
   panels appearing in wandb
4. Let it reach the first `save_freq` boundary and confirm the checkpoint merges
