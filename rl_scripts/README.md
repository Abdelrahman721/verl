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
| `nemotron_unified_grpo_sft737.sh` | `grpo_nemotron_unified_v3_sft737` | v3, from `Qwen3-4B-Base-sft-737` |
| `nemotron_unified_grpo_v4_sft737.sh` | `grpo_nemotron_unified_v4_sft737` | v3 behind a strict format gate |

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

**v4: strict format gate.** v3 inherited the hermes parser's leniency: an unclosed
trailing `<tool_call>` was still parsed, text around and between blocks was ignored,
and a response that never closed `</think>` was scanned whole. Late v3 rollouts
exploited all of it (junk tokens between calls, stray `<think>` tags, 8% never
closing the reasoning) and still collected full tool reward.
`nemotron_unified_v4.py` accepts only the layout the SFT data renders, with no
recovery:

- Response must start with `<think>` and contain exactly one `<think>` and one
  `</think>`, else the row's floor (-1.5 tool rows, -1 prose rows), no judge call
  (`think_error`).
- If the post-think text has any `<tool_call>`/`</tool_call>` tag it must be only
  whitespace-separated blocks, each a JSON object with exactly `name` (non-empty
  string) and `arguments` (object), no duplicate keys, no NaN. Anything else is
  prose (`format_error`): -1.5 on tool rows, sent to the judge on prose rows.
- **Prose on a tool row scores -1.5**, below the worst possible call (-1). On the
  v3 sft737 run, calls on single-call rows fell from 43% to 26% of rollouts in 19
  steps: prose and a wrong call both scored -1, so all-prose groups had no signal
  while prose rows kept paying for prose. Replaying steps 18-21 under v4, single
  groups with signal rise from 30% to 42% and the mean advantage of call rollouts
  on those rows from +0.34 to +0.55.
- With valid format a call scores exactly what v3 gave it. Replaying v3 steps 1-5 and 229-233
  changed no valid-format tool-row score; v4 only removed credit v3 gave to
  malformed output (think errors 7-18% of rollouts, markup errors 1-6%).

Manager: `verl/workers/reward_manager/nemotron_judge_v4.py`
(`NemotronJudgeV4RewardManager`). Tests:
`tests/utils/reward_score/test_nemotron_unified_v4_on_cpu.py`.

## olive_rl300k run

`rl_scripts/olive_rl300k_grpo.sh` -> `nemotron_unified_grpo_sft737.sh` ->
`nemotron_unified_grpo.sh`. Data `rl-data/olive_rl300k` (296,814 train / 2,970 val,
`data_source="olive_rl300k"`, built by `examples/data_preprocess/olive_rl300k_preprocess.py`).
Reward `verl/utils/reward_score/olive_rl300k.py`, routed by `data_source` inside the
existing `NemotronJudgeRewardManager` - no new manager.

Policy: `models/Qwen3-4B-Base-olive-sft-1074`, epoch 3 of 3 (the final step) of
`models-sft/abdelrahman-qwen-olive-32k-bs8-3ep-30gpu`, copied in and weight-for-weight
identical to the source. **Its `tokenizer_config.json` shipped a 723-char template that
renders neither tool-role messages nor `tool_calls`** - it silently changed 220 of 300
sampled prompts, dropping every `<tool_response>` and every structured call from the
history. `chat_template_tooling.jinja` was installed over it (original preserved at
`tokenizer_config.json.pre-tooling-template.bak`); prompts now render byte-identically to
what `Qwen3-4B-Base-sft-737` produces. This is the same repair that checkpoint needed.

What the wrapper changes and why:

- **`data.max_prompt_length=28672`** (the other runs use 12288). Measured on 4,957 rows
  sampled across every row group: p50 4,273, p90 22,283, p99 34,904, max 51,064. At 12288,
  `filter_overlong_prompts` drops 23.5% of the corpus, and not evenly - 93.8% of
  openresearcher, 87.2% of openseeker and 60.9% of swe-zero-openhands, leaving roughly the
  old nemotron_unified corpus. 28672 drops 3.89% (23.3% / 27.7% / 7.6%) and keeps every
  source. The ceiling is the checkpoint: it was SFT'd at 32k and its
  `max_position_embeddings` is **32768**, not the 40960 of sft-737, so prompt + response
  must fit 32768 and 28672 + 4096 lands exactly on it.
- **`use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=32768`** on the actor and the rollout
  log-prob pass. The inherited fixed `ppo_micro_batch_size_per_gpu=2` would be ~66k tokens
  per backward at these lengths; the logits alone (vocab 151,936) are ~20 GB. Token-budgeted
  batching caps it instead. The budget must be >= one full sequence. With `use_dynamic_bsz`
  on, the inherited micro-batch settings are ignored, not conflicting.
- **`trainer.total_training_steps`**: 296,814 rows at `train_batch_size=32` is 9,275 steps
  for one epoch, so the run ends on a step count rather than on the corpus.
- **`trainer.max_actor_ckpt_to_keep=null`** overrides the inherited `=2`. `null` is verl's own
  default and disables pruning entirely (`checkpoint_manager.py` returns early on a falsy
  value). The v4 run was limited to 2 and left seven 12 KB stubs where the weights had been,
  so only its last two steps were ever convertible. ~47 GB per checkpoint against 451 T free
  on `/mnt/data01`: retention is not the constraint, losing a checkpoint you wanted is.

Corpus mix is the sources' natural one, *not* the 33/33/33 of the unified corpus:
54% single call, 31% prose, 15% parallel batch.

Two changes were made to the reward before this run, both measured on the val split:

- **v4's format gate replaced `nemotron_pivot.extract_action`.** The tiered scoring logic is
  untouched; only the parser changed. A response must be one `<think>` block then prose or
  well-formed `<tool_call>` blocks, nothing salvaged from broken markup. A broken think block
  scores `THINK_ERROR_SCORE` (-1.0) with no judge call. On the v3 run 3.90% of rollouts
  collected tool reward the gate rejects (7.20% markup errors); under v4 those are 0.42% and
  0.76%. The generation prompt ends at `<|im_start|>assistant\n` with no pre-filled `<think>`,
  exactly as in the v4 run, so the gate is asking for what the policy already does.
- **The key-set cliff was removed.** `set(expected_args) != set(emitted_args)` used to score
  0.0, the same as calling a completely different tool. Only 19% of call rows have no optional
  params. Now optional params left at their declared default are stripped from both sides
  before comparison, and a key set the tool would accept (all required present, nothing
  undeclared) earns the 0.5 tier. Effect, val split:

  | perturbation of the expert action | before | after |
  |---|---|---|
  | add an optional param at its schema default (14% of rows exposed) | 62% score 0.0 | 100% score 1.0 |
  | omit one optional param the expert supplied (77% exposed) | 82% score 0.0 | 0.7% score 0.0 |

  The ordering inversion is gone: a correct call missing an optional param used to score below
  a call with entirely wrong arguments. The cost is that a wrong-argument call to the right
  tool now averages +0.50 rather than +0.43, because key-set misses that used to fall to 0.0
  land on 0.5 instead. Expert actions still score 1.0 on 2,029/2,029 val call rows with the
  prompt schemas in play, and expert prose reaches the judge on 941/941 prose rows.

### What the run showed (stopped at step 480)

From `Qwen3-4B-Base-olive-sft-1074`, 480 steps over ~2 days on hgx11, then **stopped
deliberately: the model was degrading on inspection.** Checkpoints 100-450 were all retained.

Mean score per 40-step bucket, from the rollout dumps:

| steps | prose | single | parallel | ALL | full% | half% | zero% | think err | fmt err |
|---|---|---|---|---|---|---|---|---|---|
| 1-40 | 0.628 | 0.438 | 0.469 | **0.503** | 34.4% | 41.7% | 23.9% | 1.32% | 1.98% |
| 121-160 | 0.770 | 0.502 | 0.577 | **0.599** | 36.3% | 47.3% | 16.4% | 0.32% | 0.67% |
| 241-280 | 0.819 | 0.494 | 0.596 | **0.615** | 38.5% | 45.2% | 16.3% | 0.21% | 0.40% |
| 361-400 | 0.831 | 0.503 | 0.613 | **0.627** | 38.1% | 46.3% | 15.6% | 0.13% | 0.32% |
| 441-479 | 0.821 | 0.504 | 0.603 | **0.625** | 39.8% | 44.9% | 15.3% | 0.12% | 0.34% |

Three things to carry into the next run:

- **The reward never showed the degradation.** It rose to ~0.62 by step 240 and was flat
  after; format and think errors hit run lows (0.12% / 0.34%) and median output length held
  at ~1.9k chars with no repetition blowup. The quality loss that stopped the run was visible
  on inspection and invisible to `olive_rl300k.py`. Treat the score plateau as the signal to
  look at completions, not as evidence the run is fine.
- **The policy learned call SHAPE, not argument VALUES.** Over 479 steps `full%` moved
  34.4% -> 39.8% (+5.4) while `zero%` fell 23.9% -> 15.3% (-8.6) and `half%` stayed ~45%.
  Nearly all the learning was wrong-shape -> right-shape. That is what the tiers pay for, and
  it is the most likely mechanism behind the proxy divergence above.
- **`run_terminal` dominates and is capped at 0.5.** It is 46% of all single-call rollouts
  (10,520 of ~23,000 in the late window), scoring mean 0.474 with **7.3% full and 88.5%
  half**. Decomposing those 9,205 half-tier rollouts: 73.5% have both `keystrokes` and
  `duration` wrong, **19.5% have the command verbs right and only the `duration` float wrong**,
  7.1% only `keystrokes`. So ~1 in 5 is blocked purely by an unguessable timeout number, and
  excluding `duration` would lift `run_terminal` full-tier from 7.3% to roughly 25%.
  `browser_search` fails on `query` 99% of the time, which is close to unwinnable by design -
  there is rarely one correct search query.

Group survival under `filter_groups` fell as the policy converged: prose 38.6% -> 23.5%,
single 61.8% -> 49.9%, parallel 85.8% -> 76.3%. Weighted that is ~45% of 96 generated groups,
still clearing the 32 needed in one gen batch. Surviving prose groups carried `mean|adv|`
0.668 against single's 0.279 - the reward-scale asymmetry from
`norm_adv_by_std_in_grpo=False` is real and measurable, though partly self-limiting because
prose survival collapses fastest.

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
