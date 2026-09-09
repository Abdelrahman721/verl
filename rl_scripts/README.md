# GRPO RL on Qwen3-4B tooling checkpoints

Launch scripts and reward code for the RL runs on top of the Mega SFT
checkpoint (`models/Qwen3-4B-Base-sft-3850`). Evaluation lives in the sibling
gorilla repo; results are in its `analysis/all_bfcl_results.tsv` under corpus
`rl-*`.

## The task

Each row is one decision step. The model sees a system prompt with tool
definitions plus a conversation, and must either emit `<tool_call>` blocks or
reply in prose. The ground truth is the expert's action at that step:
`function_call`, `function_call_batch`, or `message`.

## The runs

| Script | Experiment | Reward |
|---|---|---|
| `nemotron_pivot_grpo.sh` | `grpo_nemotron_pivot_all` | rule only (`nemotron_pivot.py`) |
| `nemotron_pivot_grpo_judge.sh` | `grpo_nemotron_pivot_all_judge` (v1), `grpo_nemotron_pivot_v2` (v2) | judge + rule |
| `nemotron_unified_grpo.sh` | `grpo_nemotron_unified_v3` | judge + graded matching |

All require `OPENROUTER_API_KEY` in the environment except the first. No key is
stored in this repo. `dev/dev.sh` forwards no env vars, so export it inside the
container.

## Why the reward changed three times

**Rule only.** `nemotron_pivot.py` returns 1.0 for *any* prose reply when the
expert replied in prose — content is never compared — so prose is free points
and the policy collapses onto it. Val tool accuracy fell 0.211 to 0.133.

**v1: judge on prose.** `nemotron_pivot_judge.py` sends prose rows to an LLM
judge instead. But the tool side still followed Gym's canonical config, where a
response scores 1.0 if every expected call is matched by *some* emitted call
and surplus calls cost nothing. The policy sprayed: mean calls per response 0.53
to 4.24. BFCL Overall 34.41 to 17.28 to 12.36, with MultiTurn at 0.88 then 0.12
because multi-turn is graded on final API state and a sprayed call corrupts it.

**v2: hard count gate.** Wrong call count scores -1 outright. Spray died and
MultiTurn recovered to 21.62, but "4 of 5 calls correct" now scored exactly what
prose scored, leaving no gradient on multi-call rows. `parallel_multiple` fell
86.00 to 57.00 — nearly the whole NonLive AST regression.

**v3: graded name-multiset matching.** `smooth_call_score` matches emitted calls
to ground-truth calls by name respecting multiplicity; each matched call
contributes its per-argument score, and missing and extra calls are charged -1
each. Normalised by the ground-truth call count, so prose (every call missing)
lands at exactly -1 and shares a range with the prose reward, whose judge
verdict maps to +1/-1.

## Reward internals worth knowing

`verl/utils/reward_score/nemotron_pivot_judge.py`, wired in through
`verl/workers/reward_manager/nemotron_judge.py` with
`reward.reward_manager.source=importlib` so verl itself stays unpatched.

- **The manager binds the judge scorer unconditionally.** verl always injects
  `default_compute_score`, so `compute_score or judge` would silently train on
  the old reward without raising anything.
- **Judge provider is pinned to `minimax/fp8`.** DeepInfra, the earlier pin,
  returned HTTP 429 (`engine_overloaded`, `upstream_provider_shared_pool`) on
  ~90% of requests at concurrency 4 and cost 7-22% of judge calls per step;
  those rows fell back to a neutral 0.0, injecting a fake mid-value into GRPO
  groups where everything else is +/-1. `minimax/fp8` has since run ~11k calls
  across 233 steps at a 0.0% error rate. `NEMOTRON_JUDGE_PROVIDER=""` lets
  OpenRouter route freely at some cost in verdict stability.
- **`reward_extra_info` must be float.** np.int64 is not JSON-serializable and
  crashes the rollout dumper; np.float64 subclasses float and is fine.
- The judge never sees the gold reply. It asks only whether the candidate is
  coherent prose addressing the last user message, making it a leniency gate
  rather than a correctness check.

## Reading the rollout dumps

`trainer.rollout_data_dir` writes one JSONL per step under `verl_dumps/`, one
line per rollout, carrying `input`, `output`, `gts`, `score` and the reward
components (`is_call`, `type_match`, `name_match`, `format_error`, `judged`,
`judge_score`, `judge_error`). Group by `input` to reconstruct the GRPO group.

Two things to watch, both learned the hard way on v3:

- **`filter_groups` silently starves whichever category is failing.** A group
  whose 8 rollouts all score -1 has zero variance, so its advantage is zero and
  the group is discarded. On v3 single-call rows this reached 72%, and the
  category never recovered: it improved on the 28% that survived (-0.28 to
  -0.14) while the aggregate fell, because the 72% it never trained on drifted
  to a mean of exactly -1.000. Check surviving-group rate per category before
  concluding a reward is too weak — the v3 single signal was the *strongest* of
  the three per surviving group (mean |adv| 0.55 vs parallel's 0.32), just the
  rarest.
- **Zero variance is invariant to reward scale.** No rescaling, normaliser
  change or learning-rate change revives a unanimous group; only breaking the
  tie inside it does.

## Config notes

`data.filter_overlong_prompts_workers=16` cuts the startup scan substantially.
`algorithm.norm_adv_by_std_in_grpo=False` means advantage is `r - mean(r)`, so
reward *scale* affects gradient magnitude directly — keep components in a shared
range. Training shuffles, which matters because the unified parquet is written
in category-order blocks on disk.
