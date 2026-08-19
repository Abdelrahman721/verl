# Dual-mode (thinking / instant) SFT data generation

Builds an SFT set that teaches the model two inference modes from the held-out
prompts in `data/chat_ifeval_remaining`. Every assistant turn is framed
identically; the two modes differ only in what sits between the think tags:

```
thinking : <|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>\n
instant  : <|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>\n
```

## Pipeline

```bash
# 1. serve the RL checkpoint, DP=8 (one engine per GPU). Leave running.
bash full_mix/sft_modes/serve_dp8.sh

# 2. generate 8 samples per prompt (resumable — re-run to continue)
python -m full_mix.sft_modes.generate_responses --source ifeval
python -m full_mix.sft_modes.generate_responses --source chat

# 3. verify, select <=4 correct, split half thinking / half instant
python -m full_mix.sft_modes.build_sft_dataset --source ifeval
export FULL_MIX_JUDGE_API_KEY=...          # chat only — pairwise judge
python -m full_mix.sft_modes.build_sft_dataset --source chat
```

Outputs land in `data/sft_dual_mode/{chat,ifeval}_sft.parquet` with columns
`messages`, `thinking`, `text`, `data_source`, `prompt_uid`, `sample_idx`,
`verifier_score`. `text` is pre-rendered with `chat_template_dual_mode.jinja`;
`messages` is the same content in structured form (assistant turns carry
`reasoning_content` only in thinking mode).

## Correctness rules

| source | correct means |
|---|---|
| ifeval | `full_mix.ifeval_reward.compute_score == 1.0` — ALL constraints satisfied |
| chat | `full_mix.rewards.chat.compute_score > 0.5` — pairwise judge prefers it over `baseline_response` |

Both verify the **answer**, never the reasoning.

A sample is usable only if it opens `<think>`, closes `</think>`, and leaves a
non-empty answer. This matters: ~18% of this model's rollouts hit the length cap
without ever closing the tag, and `strip_think` returns the *whole* text when
the close tag is missing — so without the check, raw chain-of-thought would be
handed to the verifier as if it were the answer.

Reasoning is re-emitted trimmed rather than verbatim because the model's own
framing is inconsistent (`<think>Okay`, `<think>\nOkay`, `<think> Okay` all
occur). Normalizing is what makes the two modes byte-identical in framing.

## The seed requirement

`generate_responses.py` sends an explicit `seed` per prompt. **This is not
optional.** Without it this vLLM build returns `n` byte-identical samples
regardless of temperature — measured 12 distinct out of 96 at `temperature=1.0`,
and still 1/4 distinct at `temperature=1.5`. With `seed=base+prompt_uid` it is
8/8 distinct on every prompt, and the run stays reproducible.

## Samples per prompt

`ifeval` draws **16**, `chat` draws **8** (`DEFAULT_N` in
`generate_responses.py`; `--n` overrides). Measured effect of the extra ifeval
draws, same 120 prompts:

| ifeval | n=8 | n=16 |
|---|---|---|
| prompts with 0 correct | 76.7% | **69.2%** |
| SFT records | 68 | **108** |
| records per prompt | 0.57 | **0.90** |
| generation rate @ concurrency 60 | 0.72 prompt/s | 0.39 prompt/s |

Doubling the draws returned +59% records for ~1.9x the generation cost — close
to break-even per token, and it buys coverage of harder prompts rather than
more samples of easy ones, which is what a 4-per-prompt cap wants.

## Measured yield (pilot: 120 ifeval @ n=16, 40 chat @ n=8)

| | ifeval | chat |
|---|---|---|
| malformed samples dropped | 8.4% | 7.8% |
| prompts with 0 correct | 69.2% | 22.5% |
| SFT records per prompt | 0.90 | 2.25 |
| projected records (full set) | ~19,700 | ~37,600 |

ifeval yield is low because the threshold is *all* constraints satisfied and
these prompts carry 3–5 sub-constraints each. Loosening `--ifeval_threshold`
(e.g. 0.8) would multiply the record count several-fold, at the cost of
training on partially-compliant answers. That is a deliberate judgement call —
the default stays strict.

The pilot samples are small (120 / 40 prompts); treat the projections as
order-of-magnitude. A 500-prompt pilot per source costs ~15 min and tightens
them considerably.

## Cost of the full run

- **Generation**: 21.8k ifeval prompts x 16 samples + 16.7k chat prompts x 8.
  Pilot rates at `--concurrency 60`: ifeval 0.39 prompt/s, chat ~1 prompt/s.
  That projects to ~16 h for ifeval and ~5 h for chat on 8xH100; raising
  concurrency helps until the KV cache saturates. Budget ~20 h total.
- **Chat judging**: up to 133k pairwise judge calls against OpenRouter, minus
  early-exit savings once 4 correct are found. This is the dominant *money*
  cost — expect low hundreds of dollars. Run a `--limit` pilot first.

Both stages are resumable: generation skips `prompt_uid`s already in the output
JSONL, so an interrupted run continues by re-issuing the same command.

## Do not serve the dual-mode template

`serve_dp8.sh` intentionally serves the model's own minimal template. Serving
`chat_template_dual_mode.jinja` would prepend an empty think block to the
generation prompt and suppress reasoning — the opposite of what the thinking
half of the dataset needs.
