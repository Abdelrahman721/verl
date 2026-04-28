pip install langdetect immutabledict nltk peft openai liger-kernel
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

export FULL_MIX_JUDGE_API_BASE="http://10.223.235.31:8000/v1"
export FULL_MIX_JUDGE_API_KEY="EMPTY"             # vLLM ignores the value but the OpenAI client needs something non-empty
export FULL_MIX_JUDGE_MODEL="Qwen/Qwen3.5-27B"    # exactly what you passed to `vllm serve` (the HF repo id, unless you set --served-model-name)
export FULL_MIX_JUDGE_DISABLE_THINKING=1          # turn ON for vLLM (uses chat_template_kwargs extra_body that OpenRouter rejected)
export FULL_MIX_JUDGE_FORCE_JSON=1                # vLLM honors response_format={"type":"json_object"} via guided decoding
export FULL_MIX_JUDGE_MAX_RETRIES=5               # local is reliable; no need for 3+
export FULL_MIX_JUDGE_CONCURRENCY=64              # raise from 16 — local has no rate limit
export FULL_MIX_JUDGE_TIMEOUT=120                 # safety net; usually completes in seconds
export FULL_MIX_JUDGE_MAX_TOKENS=4096             # plenty for the JSON-only output

# export FULL_MIX_JUDGE_API_BASE=https://openrouter.ai/api/v1
# export FULL_MIX_JUDGE_API_KEY=sk-or-v1-...
# export FULL_MIX_JUDGE_MODEL=x-ai/grok-4.1-fast
# export FULL_MIX_JUDGE_DISABLE_THINKING=0
# export FULL_MIX_JUDGE_MAX_RETRIES=3
# export FULL_MIX_JUDGE_CONCURRENCY=16

export WANDB_API_KEY="fc..."
export MODEL_PATH=...

# bash full_mix/preprocess_all.sh
# bash full_mix/train.sh   # after setting FULL_MIX_JUDGE_* to point at your vLLM judge