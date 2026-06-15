# web_judging

A small, **self-contained** convenience web app to:

1. Serve the medical-QA policy checkpoint with **vLLM** (`--tensor-parallel-size 2`).
2. Browse entries from `train.parquet` / `val.parquet` via a dropdown.
3. Send a chosen question to the policy and view the generated answer.
4. Grade that answer with **two qa-judge backends** and one shared penalty judge,
   then show both combined final scores side by side.

## The comparison

Each "system" = a qa judge **plus** the shared Deepseek penalty judge, combined as
`final = max(0, qa_score − penalty_total)`. The penalty judge runs **once**; only
the qa-judge backend differs:

| | qa judge (accuracy/completeness/clarity) | penalty judge |
|---|---|---|
| **System A** | **Sonnet** via Anthropic Bedrock (`qa_bedrock.py`) | Deepseek via OpenRouter |
| **System B** | **gpt-5.4-mini** via OpenRouter (`qa_openrouter.py`) | Deepseek via OpenRouter (same call) |

## Prompt parity (the hard requirement)

The System-B (gpt-5.4-mini) judge **does not redefine any prompt**. It imports the
prompt constants *and* every scoring/parsing/validation helper from the verbatim
copy of `qa_bedrock.py`, and overrides only the model call. The prompts therefore
live in exactly one place and cannot drift. `server.py` calls
`qa_openrouter.assert_prompt_parity()` at startup, which asserts the System-B prompt
objects are the **same Python objects** as `qa_bedrock`'s — the strongest possible
guarantee of zero textual difference.

## Self-containment

Everything lives under `web_judging/`. The four files copied from `full_mix/` are
byte-identical to their sources except for `medical_penalty_judge.py`, whose only
change is rewriting two `from full_mix.common…` imports to local relative imports:

- `judging/qa_bedrock.py`            ← `full_mix/rewards/qa_bedrock.py` (verbatim)
- `judging/judge_client.py`          ← `full_mix/common/judge_client.py` (verbatim)
- `judging/think.py`                 ← `full_mix/common/think.py` (verbatim)
- `judging/medical_penalty_judge.py` ← `full_mix/rewards/medical_penalty_judge.py` (imports only)

New glue (not imported across the boundary): `qa_openrouter.py`, `qa_eval.py`,
`combine.py` (ports `qa_bedrock_with_penalties._normalise_penalty_into`), `data.py`,
`server.py`, `static/index.html`.

## Run

```bash
cd web_judging
pip install -r requirements.txt          # into the env that has vLLM
# Edit run.sh and replace every PLACEHOLDER_* value with real credentials.
bash run.sh                              # launches vLLM (TP=2) + the web app
# or, against an already-running vLLM:
LAUNCH_VLLM=0 POLICY_API_BASE=http://host:8001/v1 bash run.sh
```

Then open `http://localhost:7860`.

## Environment variables

See `run.sh` — every value is a placeholder/default and overridable from the
environment. Key groups: policy/vLLM (`POLICY_*`, `GEN_*`), System A
(`QA_JUDGE_AWS_*`, `QA_JUDGE_MODEL`), System B (`QA_OPENROUTER_*`), shared penalty
(`FULL_MIX_JUDGE_*`, `FULL_MIX_MEDICAL_PENALTY_ENABLED`), data/web (`*_PARQUET`,
`WEB_JUDGING_TRAIN_LIMIT`, `WEB_PORT`).

**Note:** `FULL_MIX_JUDGE_DISABLE_THINKING=0` is required for OpenRouter (the
`chat_template_kwargs` hook used for self-hosted vLLM is rejected by OpenRouter).

## Failure-open behaviour

Mirrors the production stack: if the penalty judge errors or is disabled, it
fails open (`penalty_total=0`, `penalty_judge_ok=0`) and both final scores still
render. If a qa judge errors, that system's panel shows the error and its final
falls back to `0 − penalty`.
