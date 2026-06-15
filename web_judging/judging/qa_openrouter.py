"""System B qa judge: the *exact* qa_bedrock scoring, but the underlying model
call goes to an OpenRouter chat-completions endpoint (default
``openai/gpt-5.4-mini``) instead of Anthropic Bedrock (Sonnet).

PROMPT PARITY — the single hard requirement of this comparison:
    This module does NOT redefine a single prompt. It imports the prompt
    constants AND every parsing / repair / validation / scoring helper from the
    copied ``qa_bedrock`` module. The only thing that differs from qa_bedrock is
    how the bytes of those prompts get sent to a model (OpenRouter vs Bedrock)
    and how the response text is pulled back out. Because the prompts live in
    exactly one place, they cannot drift. See the ``assert_prompt_parity`` check.

What is intentionally re-implemented here (and ONLY this):
    - ``_build_judge_client``      Bedrock client  -> OpenAI-compatible client.
    - ``_call_qa_judge`` /
      ``_call_conv_judge``         streaming Anthropic call -> chat.completions.
      The retry loop, the prompts, the parse/repair/validate path, and the
      empty-answer short-circuit are kept structurally identical to qa_bedrock.

What is NOT carried over (no Bedrock equivalent on OpenRouter):
    - The ``stop_reason == "refusal"`` safety-filter path. Bedrock surfaces a
      model-level refusal as a distinct stop reason; an OpenAI-compatible
      gateway does not. Empty content is treated as a retryable error, exactly
      like qa_bedrock treats an empty body.

Env vars (read at client build, then cached per event loop):
    QA_OPENROUTER_API_BASE   default "https://openrouter.ai/api/v1"
    QA_OPENROUTER_API_KEY    required
    QA_OPENROUTER_MODEL      default "openai/gpt-5.4-mini"
    QA_OPENROUTER_MAX_TOKENS default = qa_bedrock.JUDGE_MAX_TOKENS
"""

import asyncio
import json
import os

from openai import AsyncOpenAI

# Import EVERYTHING prompt- and scoring-related from the verbatim qa_bedrock
# copy. These names are re-exported so callers (qa_eval) can reach them through
# either backend module interchangeably.
from . import qa_bedrock as _qa
from .qa_bedrock import (  # noqa: F401  (re-exported on purpose)
    QA_JUDGE_SYSTEM_PROMPT,
    QA_JUDGE_USER_TEMPLATE,
    CONV_JUDGE_SYSTEM_PROMPT,
    CONV_JUDGE_USER_TEMPLATE,
    _parse_json_response,
    _repair_qa_parsed,
    _repair_conv_parsed,
    _validate_qa_response,
    _validate_conv_response,
    _build_conv_context,
    _score_single,
    _extract_answer,
    log,
)

# Same knobs as qa_bedrock, but namespaced so they don't collide with the
# Bedrock judge's env. Default model is the comparison target the user asked for.
JUDGE_MODEL = os.environ.get("QA_OPENROUTER_MODEL", "openai/gpt-5.4-mini")
JUDGE_MAX_TOKENS = int(os.environ.get("QA_OPENROUTER_MAX_TOKENS", str(_qa.JUDGE_MAX_TOKENS)))


def _resolve(*names: str) -> str:
    """First env var among ``names`` that is set and isn't one of our own
    PLACEHOLDER_* defaults. Lets the System-B judge fall back to the shared
    OpenRouter creds (FULL_MIX_JUDGE_*) — both hit OpenRouter, so setting one
    key is enough. Read lazily (at call time) so run.sh exports are seen."""
    for n in names:
        v = os.environ.get(n, "")
        if v and not v.startswith("PLACEHOLDER"):
            return v
    return ""


def _api_base() -> str:
    return _resolve("QA_OPENROUTER_API_BASE", "FULL_MIX_JUDGE_API_BASE") or "https://openrouter.ai/api/v1"


def assert_prompt_parity() -> None:
    """Fail loudly if the System-B prompts ever stop being the SAME object as
    qa_bedrock's. Called at server startup. Object identity (``is``) is the
    strongest possible guarantee that there is zero textual difference."""
    pairs = [
        ("QA_JUDGE_SYSTEM_PROMPT",   QA_JUDGE_SYSTEM_PROMPT,   _qa.QA_JUDGE_SYSTEM_PROMPT),
        ("QA_JUDGE_USER_TEMPLATE",   QA_JUDGE_USER_TEMPLATE,   _qa.QA_JUDGE_USER_TEMPLATE),
        ("CONV_JUDGE_SYSTEM_PROMPT", CONV_JUDGE_SYSTEM_PROMPT, _qa.CONV_JUDGE_SYSTEM_PROMPT),
        ("CONV_JUDGE_USER_TEMPLATE", CONV_JUDGE_USER_TEMPLATE, _qa.CONV_JUDGE_USER_TEMPLATE),
    ]
    for name, b_obj, a_obj in pairs:
        if b_obj is not a_obj:
            raise AssertionError(
                f"PROMPT PARITY VIOLATION: qa_openrouter.{name} is not the same "
                f"object as qa_bedrock.{name}. The OpenRouter judge must reuse "
                f"qa_bedrock's prompts verbatim."
            )


def _build_judge_client():
    """Build a fresh AsyncOpenAI client pointed at OpenRouter from env creds.

    Mirrors qa_bedrock._build_judge_client's contract: a new client per event
    loop, used under ``async with`` so it closes with its loop.
    """
    api_key = _resolve("QA_OPENROUTER_API_KEY", "FULL_MIX_JUDGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenRouter qa judge key not set: provide QA_OPENROUTER_API_KEY "
            "(or FULL_MIX_JUDGE_API_KEY — the System-B judge reuses the shared "
            "OpenRouter key when its own is unset). Required for the "
            "System-B (gpt-5.4-mini) judge."
        )
    return AsyncOpenAI(base_url=_api_base(), api_key=api_key, max_retries=4)


def _is_fatal(e: Exception) -> bool:
    """Deterministic failures that retrying cannot fix (bad key, bad model id,
    no permission). Fail fast instead of burning the full retry budget."""
    status = getattr(e, "status_code", None)
    if status in (400, 401, 403, 404):
        return True
    name = type(e).__name__
    return name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError")


async def _chat_json(client, system_prompt: str, user_prompt: str) -> str:
    """One OpenRouter chat completion that returns the raw assistant text.

    Constrains the gateway to emit a JSON object (json_object response_format),
    same intent as the FULL_MIX judge client. No thinking-disable extra_body —
    OpenRouter rejects the chat_template_kwargs hook qa_bedrock's Bedrock path
    never used either.
    """
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    choices = getattr(resp, "choices", None)
    if not choices:
        upstream_err = getattr(resp, "error", None)
        raise ValueError(f"judge returned no choices (error={upstream_err!r})")
    return (choices[0].message.content or "").strip()


async def _call_qa_judge(question, key_points, answer, raw_generation, client, max_retries=10):
    """OpenRouter twin of qa_bedrock._call_qa_judge. Same prompts, same parse/
    repair/validate, same empty-answer short-circuit, same return shape."""
    if not answer:
        return {"accuracy": 1, "clarity": 1, "key_point_coverage": []}

    kp_json = json.dumps(key_points, indent=2)
    prompt = QA_JUDGE_USER_TEMPLATE.format(
        question=question, key_points_json=kp_json, answer=answer
    )

    for attempt in range(max_retries):
        raw_text = ""
        try:
            raw_text = await _chat_json(client, QA_JUDGE_SYSTEM_PROMPT, prompt)
            if not raw_text:
                raise ValueError("Judge returned empty content")
            parsed = _parse_json_response(raw_text)
            parsed = _repair_qa_parsed(parsed)
            _validate_qa_response(parsed)
            return parsed
        except Exception as e:
            snippet = raw_text[:400].replace("\n", " ")
            if _is_fatal(e):
                log.error(f"[openrouter] QA judge FATAL (not retrying): "
                          f"{type(e).__name__}: {e}")
                return {"accuracy": 1, "clarity": 1, "key_point_coverage": [], "error": str(e)}
            if attempt < max_retries - 1:
                wait = min(5.0 * (2 ** attempt) if "429" in str(e) else 2.0 * (attempt + 1), 30.0)
                log.warning(f"[openrouter] QA judge attempt {attempt+1}/{max_retries} failed: "
                            f"{type(e).__name__}: {e} | raw[:400]={snippet!r}")
                await asyncio.sleep(wait)
            else:
                log.error(f"[openrouter] QA judge FAILED after {max_retries} attempts: {e} "
                          f"| raw[:400]={snippet!r}")
                return {"accuracy": 1, "clarity": 1, "key_point_coverage": [], "error": str(e)}


async def _call_conv_judge(ground_truth, answer, raw_generation, client, max_retries=10):
    """OpenRouter twin of qa_bedrock._call_conv_judge."""
    if not answer:
        return {"accuracy": 1, "behavior": 1, "compliance": 1}

    type_desc = ground_truth.get("type_descriptor", "")
    sub_index = ground_truth.get("sub_index", 1)
    total_subs = ground_truth.get("total_subs", 1)
    gold_response = ground_truth.get("gold_response", "")
    context = ground_truth.get("context",
              ground_truth.get("_context", "(No prior context)"))
    latest_user = ground_truth.get("latest_user",
                  ground_truth.get("_latest_user", ""))

    if context == "(No prior context)" and sub_index > 1:
        log.warning(f"[openrouter] Conv judge: sub_index={sub_index} but no prior context")

    prompt = CONV_JUDGE_USER_TEMPLATE.format(
        type_descriptor=type_desc if type_desc else "No type descriptor available.",
        sub_index=sub_index,
        total_subs=total_subs,
        context=context,
        latest_user_message=latest_user,
        gold_response=gold_response,
        candidate_response=answer,
    )

    for attempt in range(max_retries):
        raw_text = ""
        try:
            raw_text = await _chat_json(client, CONV_JUDGE_SYSTEM_PROMPT, prompt)
            if not raw_text:
                raise ValueError("Judge returned empty content")
            parsed = _parse_json_response(raw_text)
            parsed = _repair_conv_parsed(parsed)
            _validate_conv_response(parsed)
            return parsed
        except Exception as e:
            snippet = raw_text[:400].replace("\n", " ")
            if _is_fatal(e):
                log.error(f"[openrouter] Conv judge FATAL (not retrying): "
                          f"{type(e).__name__}: {e}")
                return {"accuracy": 1, "behavior": 1, "compliance": 1, "error": str(e)}
            if attempt < max_retries - 1:
                wait = min(5.0 * (2 ** attempt) if "429" in str(e) else 2.0 * (attempt + 1), 30.0)
                log.warning(f"[openrouter] Conv judge attempt {attempt+1}/{max_retries} failed: "
                            f"{type(e).__name__}: {e} | raw[:400]={snippet!r}")
                await asyncio.sleep(wait)
            else:
                log.error(f"[openrouter] Conv judge FAILED after {max_retries} attempts: {e} "
                          f"| raw[:400]={snippet!r}")
                return {"accuracy": 1, "behavior": 1, "compliance": 1, "error": str(e)}
