"""Unified LLM-as-judge client supporting both self-hosted vLLM and OpenRouter.

Both backends expose an OpenAI-compatible HTTP API, so a single
``openai.OpenAI`` client works for both. Configuration is driven by
environment variables so the reward function stays stateless.

Env vars:
    FULL_MIX_JUDGE_API_BASE    — required, e.g. http://127.0.0.1:8000/v1
    FULL_MIX_JUDGE_API_KEY     — default "EMPTY"
    FULL_MIX_JUDGE_MODEL       — required, e.g. Qwen/Qwen2.5-7B-Instruct
    FULL_MIX_JUDGE_TIMEOUT     — request timeout seconds (default 60)
    FULL_MIX_JUDGE_MAX_RETRIES — exponential backoff attempts (default 3)
    FULL_MIX_JUDGE_MAX_TOKENS  — judge response cap (default 512)
"""

import json
import logging
import os
import random
import re
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Announce the resolved judge env on module import so we can see from Ray
# worker logs whether the vars actually propagated.
_resolved = {
    "API_BASE": os.environ.get("FULL_MIX_JUDGE_API_BASE"),
    "MODEL": os.environ.get("FULL_MIX_JUDGE_MODEL"),
    "DISABLE_THINKING": os.environ.get("FULL_MIX_JUDGE_DISABLE_THINKING"),
    "FORCE_JSON": os.environ.get("FULL_MIX_JUDGE_FORCE_JSON"),
    "HAS_KEY": bool(os.environ.get("FULL_MIX_JUDGE_API_KEY")),
}
print(f"[judge_client boot pid={os.getpid()}] env resolved: {_resolved}", flush=True)


class JudgeUnavailable(RuntimeError):
    """Raised when the judge client cannot be configured / reached."""


_CLIENT_LOCK = threading.Lock()
_CLIENT: Optional[Any] = None
_CLIENT_SIG: Optional[tuple] = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r; using default %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %f", name, raw, default)
        return default


def _get_client():
    """Return a lazily-constructed, singleton OpenAI-compatible client.

    Rebuilds if env vars change between calls (useful for tests).
    """
    global _CLIENT, _CLIENT_SIG

    api_base = os.environ.get("FULL_MIX_JUDGE_API_BASE")
    api_key = os.environ.get("FULL_MIX_JUDGE_API_KEY", "EMPTY")
    model = os.environ.get("FULL_MIX_JUDGE_MODEL")
    timeout = _env_float("FULL_MIX_JUDGE_TIMEOUT", 60.0)

    if not api_base or not model:
        raise JudgeUnavailable(
            "FULL_MIX_JUDGE_API_BASE and FULL_MIX_JUDGE_MODEL must be set"
        )

    sig = (api_base, api_key, model, timeout)
    with _CLIENT_LOCK:
        if _CLIENT is None or _CLIENT_SIG != sig:
            try:
                import openai
            except ImportError as e:
                raise JudgeUnavailable(
                    "openai package not installed; run `pip install openai`"
                ) from e
            _CLIENT = openai.OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
            _CLIENT_SIG = sig
        return _CLIENT, model


def _is_transient(exc: BaseException) -> bool:
    import openai as _openai  # local import to avoid hard dep at module load
    transient = (
        _openai.APITimeoutError,
        _openai.APIConnectionError,
        _openai.RateLimitError,
        _openai.InternalServerError,
    )
    if isinstance(exc, transient):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    return False


def call_judge(
    messages: list[dict],
    *,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    response_format: Optional[dict] = None,
) -> str:
    """Send a chat completion to the configured judge. Returns message content.

    Retries transient failures with exponential backoff. Raises
    JudgeUnavailable on persistent failure or misconfiguration.

    By default we constrain the server to emit a JSON object via the OpenAI
    ``response_format`` field. vLLM implements this via guided decoding, which
    stops thinking-style models from spewing a reasoning preamble before the
    JSON (and blowing through ``max_tokens``). Disable by setting
    ``FULL_MIX_JUDGE_FORCE_JSON=0`` or by passing ``response_format={}``.
    """
    client, model = _get_client()
    max_tokens = max_tokens or _env_int("FULL_MIX_JUDGE_MAX_TOKENS", 512)
    max_retries = _env_int("FULL_MIX_JUDGE_MAX_RETRIES", 3)

    if response_format is None and os.environ.get("FULL_MIX_JUDGE_FORCE_JSON", "1") != "0":
        response_format = {"type": "json_object"}

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format:
        kwargs["response_format"] = response_format

    # For Qwen3-family thinking models, turn off CoT in the chat template
    # so the output is JSON-only (safe to pass; non-Qwen models ignore it).
    if os.environ.get("FULL_MIX_JUDGE_DISABLE_THINKING", "1") != "0":
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    last_err: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            content = choice.message.content or ""
            return content
        except Exception as e:
            last_err = e
            if attempt < max_retries and _is_transient(e):
                # Exponential backoff with small jitter.
                delay = (2 ** attempt) + random.uniform(0.0, 0.5)
                logger.warning(
                    "Judge call failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1, max_retries + 1, type(e).__name__, delay,
                )
                time.sleep(delay)
                continue
            break

    raise JudgeUnavailable(f"Judge call failed after {max_retries + 1} attempts: {last_err!r}") from last_err


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(raw: str) -> dict:
    """Extract the first JSON object from `raw`, tolerating prose/code fences.

    Raises ValueError if no valid JSON object can be recovered.
    """
    if raw is None:
        raise ValueError("judge returned None")
    text = raw.strip()
    if not text:
        raise ValueError("judge returned empty string")

    # Preferred: full text is valid JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to pull out a ```json ... ``` fenced block.
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Fall back to the widest braced region. This is greedy on purpose — small
    # models sometimes emit explanation text surrounding the JSON.
    m = _JSON_OBJECT.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"could not parse JSON from judge output: {text[:200]!r}")
