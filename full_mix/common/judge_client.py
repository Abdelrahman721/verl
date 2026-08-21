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
    FULL_MIX_JUDGE_TEMPERATURE — sampling temperature (default 0.0)

OpenRouter-only routing controls (ignored when API_BASE is not OpenRouter,
because a plain vLLM server rejects the unknown ``provider`` body field):
    FULL_MIX_JUDGE_PROVIDER_ONLY       — comma-separated provider slugs to pin
                                         routing to, e.g. "baidu". Empty = let
                                         OpenRouter pick from all of them.
    FULL_MIX_JUDGE_PROVIDER_IGNORE     — comma-separated slugs to exclude.
    FULL_MIX_JUDGE_REQUIRE_PARAMETERS  — 1 (default) routes only to providers
                                         that support every parameter we send.
                                         Without it OpenRouter silently DROPS
                                         unsupported params (response_format
                                         included) and routes anyway.
    FULL_MIX_JUDGE_ALLOW_FALLBACKS     — 0 to hard-fail instead of falling back
                                         to another provider. Unset = leave to
                                         OpenRouter's default (fallbacks on).
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
    "DISABLE_REASONING": os.environ.get("FULL_MIX_JUDGE_DISABLE_REASONING"),
    "FORCE_JSON": os.environ.get("FULL_MIX_JUDGE_FORCE_JSON"),
    "TEMPERATURE": os.environ.get("FULL_MIX_JUDGE_TEMPERATURE"),
    "PROVIDER_ONLY": os.environ.get("FULL_MIX_JUDGE_PROVIDER_ONLY"),
    "PROVIDER_IGNORE": os.environ.get("FULL_MIX_JUDGE_PROVIDER_IGNORE"),
    "HAS_KEY": bool(os.environ.get("FULL_MIX_JUDGE_API_KEY")),
}
print(f"[judge_client boot pid={os.getpid()}] env resolved: {_resolved}", flush=True)


class JudgeUnavailable(RuntimeError):
    """Raised when the judge client cannot be configured / reached."""


class JudgeReasoningOverrun(RuntimeError):
    """The model burned the whole token budget reasoning and emitted no answer.

    Signature is ``finish_reason == "length"`` with empty ``message.content``:
    reasoning tokens count against ``max_tokens``, so a model that loops in its
    chain of thought hits the cap before writing a single content token.

    Kept distinct from JudgeUpstreamError because the correct response differs.
    A plain upstream error is worth retrying as-is; this one is not — the same
    request will reason itself into the same wall. ``call_judge`` retries it
    with reasoning switched off instead.
    """


class JudgeUpstreamError(RuntimeError):
    """Raised when the judge endpoint returns a malformed/empty response.

    Common with OpenRouter and other proxy gateways: a 200 OK with
    ``choices=None`` (and the actual error tucked under ``resp.error``) when
    the upstream provider rate-limits, refuses, or errors. Treated as
    transient by ``_is_transient`` so call_judge retries.
    """


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


def _env_list(name: str) -> list[str]:
    """Parse a comma-separated env var into a list of non-empty stripped items."""
    raw = os.environ.get(name) or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _is_openrouter(api_base: Optional[str]) -> bool:
    return bool(api_base) and "openrouter.ai" in api_base


def _provider_block(api_base: Optional[str]) -> dict:
    """Build OpenRouter's ``provider`` routing object from env.

    Returns {} for non-OpenRouter backends: a self-hosted vLLM server has no
    such field and rejects the request rather than ignoring it.

    Why this exists: OpenRouter fans a single model out across dozens of
    independently-operated deployments at different quantizations. They are
    NOT interchangeable — one bad endpoint returns degenerate output that no
    parser can rescue, and for an RL reward the provider spread is variance
    injected straight into the reward function. Pinning removes both.
    """
    if not _is_openrouter(api_base):
        return {}

    provider: dict = {}

    only = _env_list("FULL_MIX_JUDGE_PROVIDER_ONLY")
    if only:
        provider["only"] = only

    # `only` is a permission set with no implied priority — OpenRouter still
    # load-balances within it, so the slowest member sets your median latency.
    # `order` is the priority list: try these in sequence, fall through on
    # failure. Use it when you care WHICH provider normally answers.
    order = _env_list("FULL_MIX_JUDGE_PROVIDER_ORDER")
    if order:
        provider["order"] = order

    ignore = _env_list("FULL_MIX_JUDGE_PROVIDER_IGNORE")
    if ignore:
        provider["ignore"] = ignore

    # Default ON. We always send response_format; without this OpenRouter
    # quietly strips it for providers that don't support structured output
    # and routes there anyway, so the judge silently stops being constrained.
    if os.environ.get("FULL_MIX_JUDGE_REQUIRE_PARAMETERS", "1") != "0":
        provider["require_parameters"] = True

    allow_fallbacks = os.environ.get("FULL_MIX_JUDGE_ALLOW_FALLBACKS")
    if allow_fallbacks is not None and allow_fallbacks != "":
        provider["allow_fallbacks"] = allow_fallbacks != "0"

    # "throughput" | "latency" | "price". Without this OpenRouter load-balances
    # across the pool, so the slowest member sets the median latency. Sorting
    # makes the pool a fast primary plus warm standbys instead of a round-robin.
    sort = os.environ.get("FULL_MIX_JUDGE_PROVIDER_SORT")
    if sort:
        provider["sort"] = sort

    return provider


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
    import json as _json
    import openai as _openai  # local import to avoid hard dep at module load
    transient: tuple = (
        _openai.APITimeoutError,
        _openai.APIConnectionError,
        _openai.RateLimitError,
        _openai.InternalServerError,
        JudgeUpstreamError,
        # Retryable, but only because call_judge drops reasoning first.
        JudgeReasoningOverrun,
        # OpenRouter/proxy gateways occasionally return a 200 with a malformed
        # / truncated body; the OpenAI SDK surfaces that as a raw
        # JSONDecodeError ("Expecting value: line N column 1") inside
        # client.chat.completions.create. Almost always transient.
        _json.JSONDecodeError,
    )
    # APIResponseValidationError exists in newer SDKs — schema mismatch on
    # the response, also very transient with proxies.
    api_response_validation = getattr(_openai, "APIResponseValidationError", None)
    if api_response_validation is not None:
        transient = transient + (api_response_validation,)
    if isinstance(exc, transient):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    return False


def _schema_rejected(exc: BaseException, kwargs: dict) -> bool:
    """True if this 400 looks like "I don't do json_schema" and we can downgrade.

    Provider support for structured output is finer-grained than OpenRouter's
    routing filter. ``response_format`` in a provider's supported_parameters
    only promises ``{"type": "json_object"}``; ``json_schema`` is gated behind
    the separate ``structured_outputs`` flag. Providers advertising the former
    without the latter (novita/fp8, for one) are therefore routed to even under
    require_parameters=1, and then hard-400 on the schema variant.

    A 400 is not transient, so without this the call dies and the caller falls
    back to a neutral reward. Downgrading to json_object costs the schema
    guarantee but keeps the verdict.
    """
    import openai as _openai

    if not isinstance(exc, _openai.BadRequestError):
        return False
    rf = kwargs.get("response_format")
    return isinstance(rf, dict) and rf.get("type") == "json_schema"


def call_judge(
    messages: list[dict],
    *,
    validate=None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    response_format: Optional[dict] = None,
    schema: Optional[dict] = None,
    schema_name: str = "verdict",
    salvage=None,
):
    """Send a chat completion to the configured judge.

    Two modes:
      validate=None        → returns the raw message-content string. Retries
                             only on transport errors (timeouts, 5xx, etc.).
      validate=callable    → parses the response with
                             ``parse_json_object(raw, validate=validate)`` and
                             returns the dict. Parse / schema failures are
                             treated as transient and retried with the same
                             exponential backoff used for transport errors —
                             so a flaky model that occasionally emits prose,
                             multiple JSON blocks, or a wrong-schema verdict
                             gets re-called until it complies (or we exhaust
                             ``FULL_MIX_JUDGE_MAX_RETRIES`` attempts and raise
                             JudgeUnavailable).

    ``model`` overrides the default ``FULL_MIX_JUDGE_MODEL`` for this call
    only — useful for letting a single judge endpoint serve a heavier model
    for the identity-domain reference judge while the global gate and the
    chat/safety judges keep using the cheap default. The underlying HTTP
    client is shared (same api_base/api_key/timeout); only the per-request
    ``model`` field on the payload changes.

    By default we constrain the server to emit a JSON object via the OpenAI
    ``response_format`` field. vLLM implements this via guided decoding, which
    stops thinking-style models from spewing a reasoning preamble before the
    JSON (and blowing through ``max_tokens``). Disable by setting
    ``FULL_MIX_JUDGE_FORCE_JSON=0`` or by passing ``response_format={}``.

    Pass ``schema`` (a JSON Schema dict) to upgrade that from "some JSON
    object" to strict structured output — ``response_format`` becomes
    ``{"type": "json_schema", ..., "strict": True}``, so the returned content
    is guaranteed to parse AND to match the schema. Strongly preferred over
    bare ``json_object``: it removes the whole class of failures where a model
    emits prose, a reasoning preamble, or a JSON block with the wrong fields.
    Strict mode requires ``additionalProperties: False`` and every property
    listed in ``required``. Only meaningful on backends that support it — pair
    with ``FULL_MIX_JUDGE_REQUIRE_PARAMETERS=1`` on OpenRouter so the request
    can't be routed to a provider that would ignore it.
    """
    client, default_model = _get_client()
    effective_model = model or default_model
    max_tokens = max_tokens or _env_int("FULL_MIX_JUDGE_MAX_TOKENS", 512)
    max_retries = _env_int("FULL_MIX_JUDGE_MAX_RETRIES", 3)
    if temperature is None:
        temperature = _env_float("FULL_MIX_JUDGE_TEMPERATURE", 0.0)

    if response_format is None and os.environ.get("FULL_MIX_JUDGE_FORCE_JSON", "1") != "0":
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            response_format = {"type": "json_object"}

    kwargs = {
        "model": effective_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format:
        kwargs["response_format"] = response_format

    extra_body: dict = {}

    # For Qwen3-family thinking models, turn off CoT in the chat template
    # so the output is JSON-only (safe to pass; non-Qwen models ignore it).
    if os.environ.get("FULL_MIX_JUDGE_DISABLE_THINKING", "1") != "0":
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    # OpenRouter's provider-agnostic reasoning switch. The chat_template_kwargs
    # flag above is a Qwen-ism that DeepSeek and friends ignore, so on
    # OpenRouter this is the one that actually stops the model reasoning.
    # Measured on deepseek-v4-flash-0731: 3.66s -> 0.59s median per verdict,
    # completion tokens 100 -> 33.
    #
    # Off by default: enabling it changes judge behaviour for the RL training
    # rewards too, so opt in explicitly (FULL_MIX_JUDGE_DISABLE_REASONING=1).
    if os.environ.get("FULL_MIX_JUDGE_DISABLE_REASONING", "0") != "0":
        extra_body["reasoning"] = {"enabled": False}

    provider = _provider_block(os.environ.get("FULL_MIX_JUDGE_API_BASE"))
    if provider:
        extra_body["provider"] = provider

    if extra_body:
        kwargs["extra_body"] = extra_body

    last_err: Optional[BaseException] = None
    attempts_made = 0
    for attempt in range(max_retries + 1):
        attempts_made = attempt + 1
        try:
            resp = client.chat.completions.create(**kwargs)
            # OpenRouter/proxy gateways sometimes return 200 OK with
            # choices=None (or []) when the upstream provider rate-limits or
            # rejects the request. The actual reason is in resp.error. Surface
            # that via a transient error class so we retry with backoff.
            choices = getattr(resp, "choices", None)
            if not choices:
                upstream_err = getattr(resp, "error", None)
                resp_id = getattr(resp, "id", None)
                raise JudgeUpstreamError(
                    f"judge returned no choices "
                    f"(resp.id={resp_id!r}, error={upstream_err!r})"
                )
            content = choices[0].message.content or ""
            # Empty content (model produced nothing or got truncated to 0
            # tokens by content moderation) — treat as transient so the call
            # is retried instead of falling into the unparseable-JSON path
            # downstream and silently scoring 0.5 / 0.0.
            if not content.strip():
                finish = getattr(choices[0], "finish_reason", None)
                resp_id = getattr(resp, "id", None)
                if finish == "length":
                    raise JudgeReasoningOverrun(
                        f"judge spent its entire {max_tokens}-token budget "
                        f"reasoning and emitted no content "
                        f"(resp.id={resp_id!r}); retrying without reasoning"
                    )
                raise JudgeUpstreamError(
                    f"judge returned empty content "
                    f"(resp.id={resp_id!r}, finish_reason={finish!r})"
                )

            if validate is None:
                return content

            # validate-mode: parse + schema-check this response. Any failure
            # is converted to JudgeUpstreamError so the existing retry loop
            # picks it up the same way it handles HTTP transients.
            try:
                return parse_json_object(content, validate=validate)
            except ValueError as parse_err:
                # Last resort before burning a retry: let the caller pull the
                # one field it actually needs out of syntactically-broken JSON.
                # Providers that advertise structured output do not all enforce
                # it, and the usual breakage is an unescaped quote inside a
                # free-text field — which leaves the enum field perfectly
                # readable. Salvaged objects still go through `validate`.
                if salvage is not None:
                    try:
                        rescued = salvage(content)
                    except Exception:
                        rescued = None
                    if isinstance(rescued, dict) and (validate is None or validate(rescued)):
                        logger.warning(
                            "judge JSON was malformed but the verdict was "
                            "salvageable: %r", content[:200],
                        )
                        return rescued
                raise JudgeUpstreamError(
                    f"judge JSON failed parse/validate: {parse_err}; "
                    f"raw={content[:300]!r}"
                ) from parse_err
        except Exception as e:
            last_err = e
            # Not transient, but recoverable: the provider rejected the strict
            # schema outright. Retry immediately with plain JSON mode.
            if attempt < max_retries and _schema_rejected(e, kwargs):
                kwargs["response_format"] = {"type": "json_object"}
                logger.warning(
                    "Judge provider rejected json_schema (attempt %d/%d): %s; "
                    "downgrading to json_object and retrying",
                    attempt + 1, max_retries + 1, str(e)[:200],
                )
                continue
            if attempt < max_retries and _is_transient(e):
                # A reasoning overrun is deterministic given the same request:
                # retrying unchanged just burns the budget again. Drop reasoning
                # for the remaining attempts so the model has to answer directly.
                if isinstance(e, JudgeReasoningOverrun):
                    extra_body["reasoning"] = {"enabled": False}
                    kwargs["extra_body"] = extra_body
                    # No backoff: nothing upstream is overloaded, and the next
                    # request is materially different. Sleeping here would just
                    # stall a reward worker for no reason.
                    delay = 0.0
                else:
                    # Exponential backoff with small jitter.
                    delay = (2 ** attempt) + random.uniform(0.0, 4)
                logger.warning(
                    "Judge call failed (attempt %d/%d): %s: %s; retrying in %.1fs",
                    attempt + 1, max_retries + 1, type(e).__name__, e, delay,
                )
                time.sleep(delay)
                continue
            break

    raise JudgeUnavailable(
        f"Judge call failed after {attempts_made} attempts: {last_err!r}"
    ) from last_err


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

# Thinking-mode delimiters emitted by Deepseek-V3/V4, Qwen, and similar models
# when reasoning is on. They appear around (or after) the JSON content and
# confuse downstream regex / json parsing. Stripped before parsing.
#
# Order matters — applied sequentially:
#   1. Full <begin>...<end> blocks (standard reasoning before final answer).
#   2. Open <begin> with no closing tag → strip from begin to end of string.
#   3. Stray <end> after the final answer → strip token + everything after.
#      (Models occasionally emit the final answer first, then continue
#      reasoning after a malformed end-of-thinking marker.)
#   4. Qwen-style <think>...</think>.
_THINK_TOKEN_PATTERNS = [
    re.compile(r"<｜begin▁of▁thinking｜>.*?<｜end▁of▁thinking｜>", re.DOTALL),
    re.compile(r"<｜begin▁of▁thinking｜>.*\Z",                    re.DOTALL),
    re.compile(r"<｜end▁of▁thinking｜>.*\Z",                      re.DOTALL),
    re.compile(r"<think>.*?</think>",                            re.DOTALL | re.IGNORECASE),
]


def _strip_thinking_tokens(text: str) -> str:
    for pat in _THINK_TOKEN_PATTERNS:
        text = pat.sub("", text)
    return text


def _iter_balanced_json_objects(text: str):
    """Yield every balanced ``{...}`` substring, string-aware.

    Walks the whole text and emits each top-level balanced object. Skips
    braces inside string literals (handles escaped quotes). Use to enumerate
    candidate JSON blocks when a model emits multiple — e.g. one inside its
    reasoning ("{Background: ...}") and another at the end (the actual final
    answer). Order is left-to-right.
    """
    in_str = False
    escape = False
    depth = 0
    start = -1
    for i, c in enumerate(text):
        if escape:
            escape = False
            continue
        if in_str:
            if c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]
                    start = -1


def parse_json_object(raw: str, *, validate=None) -> dict:
    """Extract a JSON object from `raw`, tolerating prose/code fences/multi-block output.

    Strategy:
      1. Strip thinking-mode delimiters (Deepseek, Qwen, etc.).
      2. Build candidate strings in this order:
           a) the whole text,
           b) the body of a fenced ```json``` block,
           c) every balanced ``{...}`` substring (left-to-right).
      3. For each candidate, try ``json.loads``. If it parses to a dict and
         ``validate`` is None or ``validate(obj)`` is True, return it.
      4. If nothing satisfied validation, raise ValueError. (Caller already
         logs and falls back to a default reward.)

    The ``validate`` callback exists because reasoning models (and especially
    judges that mirror structure from the candidate's response) sometimes
    emit multiple JSON-looking blocks. Without validation we'd return the
    first one — which is often an intermediate-reasoning fragment whose
    ``verdict`` is something like ``"Background"`` or ``"Uses"`` mirrored
    from a citation-classification task in the candidate. With validation
    we keep scanning until we find a block whose schema matches.
    """
    if raw is None:
        raise ValueError("judge returned None")
    text = _strip_thinking_tokens(raw).strip()
    if not text:
        raise ValueError("judge returned empty string")

    def _candidates():
        yield text
        m = _JSON_FENCE.search(text)
        if m:
            yield m.group(1)
        yield from _iter_balanced_json_objects(text)

    seen_any_dict = False
    for cand in _candidates():
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        seen_any_dict = True
        if validate is None or validate(obj):
            return obj

    if seen_any_dict and validate is not None:
        raise ValueError(
            f"no JSON object satisfied schema validation in: {text[:200]!r}"
        )
    raise ValueError(f"could not parse JSON from judge output: {text[:200]!r}")
