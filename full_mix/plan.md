# Multi-domain GRPO training via VERL — `full_mix/`

## Context

Train a single Qwen actor via GRPO on a mix of four task types (Math, Instruction-Following, Chat, Safety) using VERL, with a validation harness covering four eval sets (GSM8K, MATH500, IFEval, IFBench — 100 samples each). VERL dispatches rewards by the `data_source` column on each sample, so the plan is to (1) preprocess all 8 datasets into VERL-compatible parquet, (2) implement a single `compute_score` dispatcher that routes to the correct domain scorer, and (3) drive training with a script modelled on `rl_scripts/modified_grpo.sh`.

Chat and Safety use an LLM-as-judge; the client must transparently support either a local vLLM OpenAI-compatible endpoint or OpenRouter via env vars. All reward paths must strip `<think>...</think>` content before grading.

Verified dataset counts (`load_from_disk`, not the stale `dataset_info.json`): Math 13,314 · IFEval 29,847 · Chat 20,708 · Safety 11,881 → **~76k** total. Default: concatenate + shuffle, 1 epoch, no subsampling. `--max_rows` on each preprocessor lets the user cap any domain later.

## Folder layout (everything under `full_mix/`)

```
full_mix/
├── plan.md
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── think.py                # strip_think(text)
│   └── judge_client.py         # unified vLLM / OpenRouter judge w/ retries + JSON parsing
├── ifeval_reward.py            # COPY of rl_scripts/ifeval_reward.py (imports now local)
├── ifeval/                     # COPY of rl_scripts/ifeval/ (constraint checker)
├── rewards/
│   ├── __init__.py
│   ├── compute_score.py        # top-level dispatcher (batch signature)
│   ├── math_boxed.py           # wraps verl.utils.reward_score.math_reward
│   ├── gsm8k.py                # wraps verl.utils.reward_score.gsm8k
│   ├── chat.py                 # LLM-judge for chat
│   └── safety.py               # LLM-judge using the SafetyJudge prompt
├── preprocess/
│   ├── __init__.py
│   ├── _common.py              # shared strip-user-prefix, emit-verl-row helpers
│   ├── math_train.py
│   ├── ifeval_train.py
│   ├── chat_train.py
│   ├── safety_train.py
│   ├── gsm8k_eval.py
│   ├── math500_eval.py
│   ├── ifeval_eval.py
│   └── ifbench_eval.py
├── preprocess_all.sh
└── train.sh
```

Data output dir: `/data/abdelrahman/verl/data/full_mix/{train,eval}/*.parquet`.

## `data_source` identifiers (dispatch keys)

Training: `local/dolci-math-7b`, `local/dolci-ifeval-32b`, `local/dolci-chat-32b`, `local/safety-dpo-reference`.
Eval: `openai/gsm8k`, `HuggingFaceH4/MATH-500`, `google/IFEval`, `allenai/IFBench_test`.

## Row formats (post-preprocessing) — VERL-compatible

Every row:
```python
{
  "data_source": <key above>,
  "prompt": [{"role": "user", "content": <user turn>}],
  "ability": "math" | "instruction_following" | "chat" | "safety",
  "reward_model": {"style": "rule", "ground_truth": <string>},
  "extra_info": {"split": "train"|"test", "index": idx, ...},
}
```

Domain transforms (all `datasets.load_from_disk`):

| Dataset | Prompt | Ground truth |
|---|---|---|
| Math train | `prompt + " " + MATH_INSTRUCTION` | `str(ground_truth)` |
| IFEval train | strip `"user: "` prefix | `ground_truth[0]` (stringified constraint list) |
| Chat train | strip `"user: "` prefix | `ground_truth[0]` (reference text) |
| Safety train | `prompt` verbatim | `reference_answer` |
| GSM8K eval | `question + " " + GSM8K_INSTRUCTION` | `####`-extracted number |
| MATH500 eval | `problem + " " + MATH_INSTRUCTION` | `answer.strip()` |
| IFEval/IFBench eval | `example["prompt"]` | `str([{"instruction_id": ..., "kwargs": ...}])` |

Every eval: `ds.shuffle(seed=42).select(range(min(100, len(ds))))`.

## Reward dispatcher

Uses VERL's `BatchRewardManager` so LLM-judge calls can parallelize. Signature:
```python
compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, **kwargs) -> list[float]
```
Splits the batch by `data_source`, calls rule-based scorers inline, dispatches chat/safety through a `ThreadPoolExecutor` (`FULL_MIX_JUDGE_CONCURRENCY`, default 32). Every scorer is called on `strip_think(solution_str)`.

Routing:
- `{local/dolci-math-7b, HuggingFaceH4/MATH-500}` → `math_boxed.py` → `verl.utils.reward_score.math_reward`
- `{openai/gsm8k}` → `gsm8k.py` → `verl.utils.reward_score.gsm8k`
- `{local/dolci-ifeval-32b, google/IFEval, allenai/IFBench_test}` → `ifeval_reward.compute_score`
- `{local/dolci-chat-32b}` → `chat.py` (LLM judge)
- `{local/safety-dpo-reference}` → `safety.py` (SafetyJudge LLM prompt)

### Normalization

- Math / GSM8K / IFEval → already [0, 1]
- Chat (judge score 1–10) → `(score - 1) / 9`
- Safety → **0 if `stance_correct` is False**, else `max(0, (score - 4) / 6)` (rubric caps wrong-stance at 4; requiring ≥5 for non-zero reward suppresses partial-compliance leakage; 10 → 1.0)
- Judge failure → fallback 0.0 with warning log

## `strip_think`

```python
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

def strip_think(text):
    if not text: return text
    text = _THINK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.lstrip()
```

## Judge client env vars

| Var | Default | |
|---|---|---|
| `FULL_MIX_JUDGE_API_BASE` | (required) | `http://127.0.0.1:8000/v1` / `https://openrouter.ai/api/v1` |
| `FULL_MIX_JUDGE_API_KEY` | `EMPTY` | |
| `FULL_MIX_JUDGE_MODEL` | (required) | |
| `FULL_MIX_JUDGE_TIMEOUT` | `60` | seconds |
| `FULL_MIX_JUDGE_MAX_RETRIES` | `3` | exponential backoff |
| `FULL_MIX_JUDGE_MAX_TOKENS` | `512` | |
| `FULL_MIX_JUDGE_CONCURRENCY` | `32` | ThreadPool size |

Module-level singleton; retries on timeouts/429/5xx/`RateLimitError`; robust JSON extraction (handles ```json fences + stray prose).

## Train script

Driven by `full_mix/train.sh`, based on `rl_scripts/modified_grpo.sh`. Major differences:
- `data.train_files` / `data.val_files` as lists (4 parquets each)
- `reward.reward_manager=batch`
- `reward.custom_reward_function.path=full_mix/rewards/compute_score.py`
- `max_response_length=16384` (think models need headroom)
- Judge env vars default to local vLLM

## Docker / deps

All training runs inside `dev/dev.sh`. Install:
```bash
pip install langdetect immutabledict nltk peft openai
```
Ad-hoc Python testing uses the uv venv at `/data/abdelrahman/.venv_full_mix`.

## Files reused unchanged
- `verl/utils/reward_score/math_reward.py`, `verl/utils/reward_score/gsm8k.py`
- `verl/workers/reward_manager/batch.py` (selected by config)

## Verification

1. Preprocess: `bash full_mix/preprocess_all.sh` → 8 parquets; correct row counts (math 13314, ifeval 29847, chat 20708, safety 11881, each eval 100).
2. Reward smoke: import `compute_score`, call with fabricated inputs incl. `<think>...</think>` — math/ifeval/gsm8k return expected 0/1, chat/safety fall back to 0.0 when judge unset.
3. Dry-run training: tiny model (`Qwen3-0.6B-Base`), `batch=4`, `rollout.n=2`, 1 validation step. Check per-data_source wandb panels.
4. Full launch with target model.
