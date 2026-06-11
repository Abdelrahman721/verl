"""
Reward scoring for medical QA + conversations using LLM-as-a-judge (Bedrock).

This is the Anthropic-Bedrock variant of qa.py — same scoring, same prompts,
same entry point (compute_score), only the judge client changes:
AsyncOpenAI -> AsyncAnthropicBedrock.

Required env vars (read at first judge call, then cached):
  QA_JUDGE_AWS_ACCESS_KEY   AWS access key id
  QA_JUDGE_AWS_SECRET_KEY   AWS secret access key
  QA_JUDGE_AWS_REGION       AWS region (e.g. "eu-central-1")
Optional:
  QA_JUDGE_MODEL            Bedrock model / inference-profile id
                            (default: "eu.anthropic.claude-sonnet-4-6")
  QA_JUDGE_MAX_TOKENS       (default: 32768)

Handles two eval_modes:
- "qa": Single-turn medical Q&A with key points (accuracy + completeness + clarity)
- "conversation": Multi-turn conversation sub-convos (accuracy + behavior + compliance)

The eval_mode is determined from extra_info["eval_mode"] or data_source:
- data_source == "medical_qa" → qa mode
- data_source == "medical_conv" → conversation mode

Expected model format: <think>reasoning</think> then the answer directly after.

Reward structure:
    - score: 0.0 to 1.0 (weighted dimensions)
    - format_penalty: -0.5 to 0.0 (applied on top, floored at 0.0)

QA dimensions (40% accuracy + 50% completeness + 10% clarity)
Conversation dimensions (40% accuracy + 35% behavior + 25% compliance)
"""

import asyncio
import json
import logging
import os
import re
import time
import traceback
import uuid

from anthropic import AsyncAnthropicBedrock

# ============================================================================
# LOGGING
# ============================================================================
log = logging.getLogger("qa_v2_reward_bedrock")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(handler)
    log.setLevel(logging.WARNING)

# ============================================================================
# FORMAT TAGS
# ============================================================================
THINKING_START = "<think>"
THINKING_END = "</think>"

# ============================================================================
# LLM JUDGE CONFIG (Bedrock)
# ============================================================================
JUDGE_MODEL = os.environ.get("QA_JUDGE_MODEL", "eu.anthropic.claude-sonnet-4-6")
JUDGE_MAX_TOKENS = int(os.environ.get("QA_JUDGE_MAX_TOKENS", "16384"))
JUDGE_CONCURRENCY = int(os.environ.get("QA_JUDGE_CONCURRENCY", "4"))
# Fallback reward when the judge refuses (safety filter) — we can't grade the
# sample, so assign a neutral score rather than punishing the policy with 0.
REFUSAL_REWARD = float(os.environ.get("QA_JUDGE_REFUSAL_REWARD", "0.8"))


def _build_judge_client():
    """Build a fresh AsyncAnthropicBedrock client from env-supplied AWS creds.

    A new client must be created inside each event loop. The httpx connection
    pool / TCP transport binds to the loop it is first used on; reusing a cached
    client across the short-lived loops created by asyncio.run() (see the
    ThreadPoolExecutor bridge below) raises "TCPTransport closed ... handler is
    closed". Always use this under `async with` so the client closes with its loop.
    """
    access_key = os.environ.get("QA_JUDGE_AWS_ACCESS_KEY", "")
    secret_key = os.environ.get("QA_JUDGE_AWS_SECRET_KEY", "")
    region     = os.environ.get("QA_JUDGE_AWS_REGION", "")
    missing = [name for name, val in [
        ("QA_JUDGE_AWS_ACCESS_KEY", access_key),
        ("QA_JUDGE_AWS_SECRET_KEY", secret_key),
        ("QA_JUDGE_AWS_REGION",     region),
    ] if not val]
    if missing:
        raise RuntimeError(
            f"Bedrock judge env vars not set: {missing}. "
            "Required: QA_JUDGE_AWS_ACCESS_KEY, QA_JUDGE_AWS_SECRET_KEY, QA_JUDGE_AWS_REGION."
        )

    return AsyncAnthropicBedrock(
        aws_access_key=access_key,
        aws_secret_key=secret_key,
        aws_region=region,
        max_retries=4,
    )


def _extract_judge_text(resp) -> str:
    """Concatenate text blocks from an Anthropic messages.create response."""
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def _refusal_info(resp):
    """Return (category, explanation) for a stop_reason == 'refusal' response.

    Refusals are a model-level safety decision (not a client toggle); medical
    content commonly false-positives the 'bio' category. stop_details may be None.
    """
    details = getattr(resp, "stop_details", None)
    return getattr(details, "category", None), getattr(details, "explanation", None)


# Set QA_JUDGE_REFUSAL_DIR to a (shared) directory to persist every refusal for
# later manual inspection. Empty/unset → disabled (no-op).
REFUSAL_DUMP_DIR = os.environ.get("QA_JUDGE_REFUSAL_DIR", "")


def _dump_refusal(record: dict) -> None:
    """Persist one refused judge call to its own JSON file under REFUSAL_DUMP_DIR.

    One file per refusal (pid + uuid in the name) so concurrent async tasks and
    Ray worker processes never contend or interleave writes — no locking needed.
    Note: in a multi-node cluster, point REFUSAL_DUMP_DIR at a shared filesystem,
    otherwise each node writes to its own local copy of the path.
    """
    if not REFUSAL_DUMP_DIR:
        return
    try:
        os.makedirs(REFUSAL_DUMP_DIR, exist_ok=True)
        fname = f"refusal_{int(time.time() * 1000)}_{os.getpid()}_{uuid.uuid4().hex[:8]}.json"
        path = os.path.join(REFUSAL_DUMP_DIR, fname)
        with open(path, "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        log.warning(f"Failed to dump refusal record: {type(e).__name__}: {e}")


# ============================================================================
# SCORING CONFIGS
# ============================================================================
QA_WEIGHTS = {"accuracy": 0.40, "completeness": 0.50, "clarity": 0.10}
CONV_WEIGHTS = {"accuracy": 0.40, "behavior": 0.35, "compliance": 0.25}

# ============================================================================
# SAFE NUMERIC EXTRACTION
# ============================================================================
def _safe_float(val, default=1.0, field_name=""):
    """Safely convert a value to float, logging warnings on coercion."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            result = float(val)
            log.warning(f"Coerced string to float for '{field_name}': '{val}' -> {result}")
            return result
        except ValueError:
            pass
    log.warning(f"Could not convert '{field_name}' to float: {type(val).__name__}={val!r}, using default={default}")
    return default


# ============================================================================
# QA JUDGE PROMPT
# ============================================================================
QA_JUDGE_SYSTEM_PROMPT = """\
You are an expert medical examiner evaluating a student's answer against \
a structured checklist of key points derived from a gold-standard reference.

You will receive:
1. A medical QUESTION
2. A KEY POINTS CHECKLIST (with importance: CORE or SUPPLEMENTARY)
3. The STUDENT'S ANSWER

━━━ CRITICAL: STYLE BLINDNESS ━━━
Do NOT penalize or reward based on formatting, structural similarity, \
hedging language, or answer length. ONLY evaluate factual content.

━━━ SCORING DIMENSIONS ━━━

### Accuracy (0-10)
RULE 1 — Extra content is not automatically an error.
RULE 2 — But extra content CAN contain errors — verify specifics.
RULE 3 — Active error detection. Do NOT default to NO_ERRORS.

- NO_ERRORS: 10 | TRIVIAL_IMPRECISION: 8-9 | MINOR_ERROR: 7
- MODERATE_ERROR: 4-6 | MAJOR_ERROR: 2-3 | DANGEROUS: 0-1

### Clarity & Depth (0-10)
- EXCEPTIONAL: 9-10 | CLEAR: 7-8 | BASIC: 4-6
- DISORGANIZED: 2-3 | INCOMPREHENSIBLE: 1

━━━ KEY POINT COVERAGE ━━━
For each key point: COVERED / PARTIAL / MISSING

━━━ OUTPUT FORMAT ━━━
Respond in JSON only. No markdown fences.

CRITICAL — REQUIRED KEYS: Your JSON object MUST contain ALL of the following
keys, spelled EXACTLY as shown, every time:
  "accuracy_category", "accuracy", "clarity_category", "clarity",
  "key_point_coverage", "justification"

- "accuracy" and "clarity" are REQUIRED numeric scores from 0 to 10. NEVER omit them.
- The "_category" keys are SEPARATE, ADDITIONAL fields. Including
  "accuracy_category" does NOT replace "accuracy"; include BOTH. Same for clarity.
- Use these EXACT key names. Do NOT rename them, add suffixes, abbreviate, or
  substitute any variant — e.g. do NOT output "accuracy_score", "clarity_score",
  "accuracy_rating", "score", or anything other than the exact keys listed above.

{
  "accuracy_category": "...",
  "accuracy": <0-10>,
  "clarity_category": "...",
  "clarity": <0-10>,
  "key_point_coverage": [{"id": <int>, "status": "COVERED|PARTIAL|MISSING", "note": "..."}],
  "justification": "..."
}"""

QA_JUDGE_USER_TEMPLATE = """\
QUESTION:
{question}

KEY POINTS CHECKLIST:
{key_points_json}

STUDENT'S ANSWER:
{answer}

Evaluate the student's answer:"""

# ============================================================================
# CONVERSATION JUDGE PROMPT
# ============================================================================
CONV_JUDGE_SYSTEM_PROMPT = """\
You are an expert medical communication evaluator. You are evaluating a \
single response from a medical AI assistant within an ongoing conversation.

You will receive:
1. The EXPECTED BEHAVIOR for this conversation type
2. The CONVERSATION CONTEXT (prior turns)
3. The LATEST USER MESSAGE
4. A GOLD REFERENCE RESPONSE
5. The CANDIDATE RESPONSE
6. The TURN POSITION

━━━ CRITICAL ━━━
You are evaluating one turn in a conversation, not a standalone answer.
The GOLD REFERENCE shows what THIS turn needs to accomplish.

━━━ SCORING DIMENSIONS ━━━

### Accuracy (0-10) — medical facts correct?
- NO_ERRORS: 10 | TRIVIAL: 8-9 | MINOR: 7 | MODERATE: 4-6 | MAJOR: 2-3 | DANGEROUS: 0-1

### Behavioral Appropriateness (0-10)
Consider: tone, safety posture, turn awareness, scope, conversational progression.
- EXEMPLARY: 9-10 | APPROPRIATE: 7-8 | PARTIALLY_APPROPRIATE: 4-6
- INAPPROPRIATE: 2-3 | HARMFUL_BEHAVIOR: 0-1

### Instruction Compliance (0-10)
- FULLY_COMPLIANT: 9-10 | MOSTLY: 7-8 | PARTIALLY: 4-6
- NON_COMPLIANT: 2-3 | NOT_APPLICABLE: 8

━━━ OUTPUT FORMAT ━━━
Respond in JSON only. No markdown fences.

CRITICAL — REQUIRED KEYS: Your JSON object MUST contain ALL of the following
keys, spelled EXACTLY as shown, every time:
  "accuracy_category", "accuracy", "behavior_category", "behavior",
  "compliance_category", "compliance", "justification"

- "accuracy", "behavior", and "compliance" are REQUIRED numeric scores from 0 to
  10. NEVER omit them.
- The "_category" keys are SEPARATE, ADDITIONAL fields. Including
  "accuracy_category" does NOT replace "accuracy"; include BOTH. Same for
  behavior and compliance.
- Use these EXACT key names. Do NOT rename them, add suffixes, abbreviate, or
  substitute any variant — e.g. do NOT output "accuracy_score", "behavior_score",
  "compliance_score", "score", or anything other than the exact keys listed above.

{
  "accuracy_category": "...",
  "accuracy": <0-10>,
  "behavior_category": "...",
  "behavior": <0-10>,
  "compliance_category": "...",
  "compliance": <0-10>,
  "justification": "..."
}"""

CONV_JUDGE_USER_TEMPLATE = """\
EXPECTED BEHAVIOR FOR THIS CONVERSATION TYPE:
{type_descriptor}

TURN POSITION: Response {sub_index} of {total_subs} in this conversation.

CONVERSATION CONTEXT:
{context}

LATEST USER MESSAGE:
{latest_user_message}

GOLD REFERENCE RESPONSE:
{gold_response}

CANDIDATE RESPONSE:
{candidate_response}

Evaluate the candidate response:"""


# ============================================================================
# JSON PARSING
# ============================================================================
def _iter_balanced_json_objects(s: str):
    """Yield every top-level balanced {...} substring, respecting string escapes.

    Unlike a greedy regex, this does not merge multiple objects or grab a span
    that spills past the real object when the text has a preamble, trailing
    commentary, or braces inside string values.
    """
    depth = 0
    start = None
    in_str = False
    escape = False
    for i, ch in enumerate(s):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield s[start:i + 1]
                start = None


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Fast path: the whole response is one JSON object.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Robust path: among all balanced {...} spans, return the largest that
    # parses to a dict. Tolerates a prose preamble, trailing commentary, code
    # fences left behind, or an example object embedded in the text.
    best = None
    for cand in _iter_balanced_json_objects(cleaned):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (best is None or len(cand) > best[1]):
            best = (obj, len(cand))
    if best is not None:
        return best[0]

    raise ValueError(f"Could not parse JSON:\n{cleaned[:500]}")


# ============================================================================
# VALIDATION
# ============================================================================
def _validate_qa_response(parsed: dict) -> None:
    for field in ("accuracy", "clarity"):
        if field not in parsed:
            raise ValueError(f"Missing required field: {field}")
        val = parsed[field]
        if not isinstance(val, (int, float, str)):
            raise ValueError(f"Field '{field}' has unexpected type: {type(val).__name__}")

    kpc = parsed.get("key_point_coverage", [])
    if not isinstance(kpc, list):
        raise ValueError(f"key_point_coverage must be a list, got {type(kpc).__name__}")
    for i, entry in enumerate(kpc):
        if not isinstance(entry, dict):
            raise ValueError(f"key_point_coverage[{i}] must be a dict")
        if "id" not in entry or "status" not in entry:
            raise ValueError(f"key_point_coverage[{i}] missing id or status: {entry}")


def _validate_conv_response(parsed: dict) -> None:
    for field in ("accuracy", "behavior", "compliance"):
        if field not in parsed:
            raise ValueError(f"Missing required field: {field}")
        val = parsed[field]
        if not isinstance(val, (int, float, str)):
            raise ValueError(f"Field '{field}' has unexpected type: {type(val).__name__}")


# ============================================================================
# COMPLETENESS FROM KEY POINTS (QA mode only)
# ============================================================================
def _compute_completeness(judge_result: dict, key_points: list) -> float:
    KP_SCORE = {"COVERED": 1.0, "PARTIAL": 0.5, "MISSING": 0.0, "WRONG": 0.0}
    importance_by_id = {kp["id"]: kp.get("importance", "CORE") for kp in key_points}
    coverage = judge_result.get("key_point_coverage", [])

    core_scores, supp_scores = [], []
    for kp in coverage:
        if not isinstance(kp, dict) or "id" not in kp:
            continue
        score = KP_SCORE.get(kp.get("status", "MISSING"), 0.0)
        imp = importance_by_id.get(kp["id"], "CORE")
        (core_scores if imp == "CORE" else supp_scores).append(score)

    core_avg = sum(core_scores) / len(core_scores) if core_scores else 1.0
    supp_avg = sum(supp_scores) / len(supp_scores) if supp_scores else 1.0

    completeness = 0.5 + (core_avg * 4.1) + (core_avg * supp_avg * 0.4)
    return max(1.0, min(5.0, completeness))


# ============================================================================
# REWARD COMPUTATION
# ============================================================================
def _compute_qa_reward(judge_result: dict, key_points: list) -> dict:
    completeness_val = _compute_completeness(judge_result, key_points)
    dims = {}
    for d in QA_WEIGHTS:
        if d == "completeness":
            val = max(1.0, min(5.0, completeness_val))
            dims[d] = (val - 1) / 4
        else:
            val = _safe_float(judge_result.get(d, 1), default=1.0, field_name=f"qa_{d}")
            val = min(10, max(0, val))
            dims[d] = val / 10

    raw = sum(dims[k] * QA_WEIGHTS[k] for k in QA_WEIGHTS)
    return {"raw_score": round(raw, 4), "penalized_score": round(raw, 4), "dimension_scores": dims}


def _compute_conv_reward(judge_result: dict) -> dict:
    dims = {}
    for d in CONV_WEIGHTS:
        val = _safe_float(judge_result.get(d, 5), default=5.0, field_name=f"conv_{d}")
        val = min(10, max(0, val))
        dims[d] = val / 10

    raw = sum(dims[k] * CONV_WEIGHTS[k] for k in CONV_WEIGHTS)
    return {"raw_score": round(raw, 4), "penalized_score": round(raw, 4), "dimension_scores": dims}


# ============================================================================
# FORMAT SCORING
# ============================================================================
_match_format_perfect = re.compile(
    rf"^\s*{re.escape(THINKING_START)}.+?{re.escape(THINKING_END)}\s*.+\s*\Z",
    flags=re.DOTALL,
)
_extract_after_think = re.compile(
    rf"{re.escape(THINKING_END)}\s*(.+)", flags=re.DOTALL,
)


def _extract_answer(response: str) -> str | None:
    m = _extract_after_think.search(response)
    return m.group(1).strip() if m else None


def _compute_format_penalty(response: str) -> float:
    ts_count = response.count(THINKING_START)
    te_count = response.count(THINKING_END)
    if ts_count == 1 and te_count == 1 and _match_format_perfect.search(response):
        return 0.0
    penalty = 0.0
    if ts_count == 0:
        penalty -= 0.25
    elif ts_count > 1:
        penalty -= 0.5 * (ts_count - 1)
    if te_count == 0:
        penalty -= 0.25
    elif te_count > 1:
        penalty -= 0.5 * (te_count - 1)
    return max(penalty, -0.5)


# ============================================================================
# LLM JUDGE CALLS
# ============================================================================
async def _call_qa_judge(question, key_points, answer, raw_generation, client, max_retries=10):
    if not answer:

        return {"accuracy": 1, "clarity": 1, "key_point_coverage": []}

    kp_json = json.dumps(key_points, indent=2)
    prompt = QA_JUDGE_USER_TEMPLATE.format(
        question=question, key_points_json=kp_json, answer=answer
    )

    for attempt in range(max_retries):
        raw_text = ""
        stop_reason = None
        try:
            async with client.messages.stream(
                model=JUDGE_MODEL,
                system=QA_JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=JUDGE_MAX_TOKENS,
            ) as stream:
                resp = await stream.get_final_message()
            stop_reason = getattr(resp, "stop_reason", None)
            if stop_reason == "refusal":
                # Deterministic for identical input — do NOT retry.
                category, explanation = _refusal_info(resp)
                log.warning(f"QA judge REFUSED by safety filter "
                            f"(category={category}): {explanation}")
                _dump_refusal({
                    "eval_mode": "qa",
                    "refusal_category": category,
                    "refusal_explanation": explanation,
                    "judge_model": JUDGE_MODEL,
                    "judge_system_prompt": QA_JUDGE_SYSTEM_PROMPT,
                    "judge_user_prompt": prompt,          # full LLM prompt sent to the judge
                    "question": question,                  # original prompt to the policy model
                    "key_points": key_points,
                    "candidate_answer": answer,            # the extracted answer that was judged
                    "raw_generation": raw_generation,      # full policy output incl. <think>
                })
                return {"accuracy": 1, "clarity": 1, "key_point_coverage": [],
                        "error": "refusal", "refusal_category": category}
            raw_text = _extract_judge_text(resp)
            if not raw_text:
                raise ValueError("Judge returned empty content")
            parsed = _parse_json_response(raw_text)
            # The judge frequently names the accuracy score "accuracy_score"
            # instead of "accuracy" (mirroring the "accuracy_category" sibling).
            if "accuracy" not in parsed and "accuracy_score" in parsed:
                parsed["accuracy"] = parsed["accuracy_score"]
            _validate_qa_response(parsed)

            return parsed
        except Exception as e:
            snippet = raw_text[:400].replace("\n", " ")
            if attempt < max_retries - 1:
                wait = min(5.0 * (2 ** attempt) if "429" in str(e) else 2.0 * (attempt + 1), 30.0)
                log.warning(f"QA judge attempt {attempt+1}/{max_retries} failed: {type(e).__name__}: {e} "
                            f"| stop_reason={stop_reason} | raw[:400]={snippet!r}")
                await asyncio.sleep(wait)
            else:
                log.error(f"QA judge FAILED after {max_retries} attempts: {e} "
                          f"| stop_reason={stop_reason} | raw[:400]={snippet!r}")
                return {"accuracy": 1, "clarity": 1, "key_point_coverage": [], "error": str(e)}


async def _call_conv_judge(ground_truth, answer, raw_generation, client, max_retries=10):
    if not answer:

        return {"accuracy": 1, "behavior": 1, "compliance": 1}

    type_desc = ground_truth.get("type_descriptor", "")
    sub_index = ground_truth.get("sub_index", 1)
    total_subs = ground_truth.get("total_subs", 1)
    gold_response = ground_truth.get("gold_response", "")
    # Context is pre-computed in the dataset (ground_truth.context) or
    # injected by _build_conv_context (ground_truth._context)
    context = ground_truth.get("context",
              ground_truth.get("_context", "(No prior context)"))
    latest_user = ground_truth.get("latest_user",
                  ground_truth.get("_latest_user", ""))

    if context == "(No prior context)" and sub_index > 1:
        log.warning(f"Conv judge: sub_index={sub_index} but no prior context")

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
        stop_reason = None
        try:
            async with client.messages.stream(
                model=JUDGE_MODEL,
                system=CONV_JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=JUDGE_MAX_TOKENS,
            ) as stream:
                resp = await stream.get_final_message()
            stop_reason = getattr(resp, "stop_reason", None)
            if stop_reason == "refusal":
                # Deterministic for identical input — do NOT retry.
                category, explanation = _refusal_info(resp)
                log.warning(f"Conv judge REFUSED by safety filter "
                            f"(category={category}): {explanation}")
                _dump_refusal({
                    "eval_mode": "conversation",
                    "refusal_category": category,
                    "refusal_explanation": explanation,
                    "judge_model": JUDGE_MODEL,
                    "judge_system_prompt": CONV_JUDGE_SYSTEM_PROMPT,
                    "judge_user_prompt": prompt,          # full LLM prompt sent to the judge
                    "conversation_context": context,       # prior turns (original prompt)
                    "latest_user_message": latest_user,
                    "gold_response": gold_response,
                    "type_descriptor": type_desc,
                    "candidate_answer": answer,            # the extracted answer that was judged
                    "raw_generation": raw_generation,      # full policy output incl. <think>
                })
                return {"accuracy": 1, "behavior": 1, "compliance": 1,
                        "error": "refusal", "refusal_category": category}
            raw_text = _extract_judge_text(resp)
            if not raw_text:
                raise ValueError("Judge returned empty content")
            parsed = _parse_json_response(raw_text)
            # The judge frequently names the accuracy score "accuracy_score"
            # instead of "accuracy" (mirroring the "accuracy_category" sibling).
            if "accuracy" not in parsed and "accuracy_score" in parsed:
                parsed["accuracy"] = parsed["accuracy_score"]
            _validate_conv_response(parsed)

            return parsed
        except Exception as e:
            snippet = raw_text[:400].replace("\n", " ")
            if attempt < max_retries - 1:
                wait = min(5.0 * (2 ** attempt) if "429" in str(e) else 2.0 * (attempt + 1), 30.0)
                log.warning(f"Conv judge attempt {attempt+1}/{max_retries} failed: "
                            f"{type(e).__name__}: {e} | stop_reason={stop_reason} | raw[:400]={snippet!r}")
                await asyncio.sleep(wait)
            else:
                log.error(f"Conv judge FAILED after {max_retries} attempts: {e} "
                          f"| stop_reason={stop_reason} | raw[:400]={snippet!r}")
                return {"accuracy": 1, "behavior": 1, "compliance": 1, "error": str(e)}


# ============================================================================
# CONTEXT EXTRACTION FOR CONVERSATIONS
# ============================================================================
def _build_conv_context(prompt_messages, ground_truth):
    """Extract context and latest user message from prompt, inject into ground_truth."""
    gt = dict(ground_truth) if isinstance(ground_truth, dict) else {}

    # Handle numpy arrays and other container types from parquet
    try:
        if hasattr(prompt_messages, 'tolist'):
            msgs = prompt_messages.tolist()
        elif isinstance(prompt_messages, (list, tuple)):
            msgs = list(prompt_messages)
        else:
            log.error(f"Unexpected prompt_messages type: {type(prompt_messages).__name__}")
            gt["_context"] = "(Error: could not parse prompt messages)"
            gt["_latest_user"] = ""
            return gt

        if not msgs:
            log.warning("Empty prompt_messages for conversation")
            gt["_context"] = "(No context available — empty prompt)"
            gt["_latest_user"] = ""
            return gt

        # Ensure each message is a proper dict
        clean_msgs = []
        for i, m in enumerate(msgs):
            if isinstance(m, dict) and "role" in m and "content" in m:
                clean_msgs.append(m)
            else:
                log.warning(f"Malformed message at index {i}: type={type(m).__name__}, "
                           f"keys={list(m.keys()) if isinstance(m, dict) else 'N/A'}")

        if not clean_msgs:
            log.error("No valid messages found in prompt after cleaning")
            gt["_context"] = "(Error: no valid messages in prompt)"
            gt["_latest_user"] = ""
            return gt

        # Build context (all turns except the last)
        if len(clean_msgs) > 1:
            context_parts = []
            for m in clean_msgs[:-1]:
                role_label = "USER" if m["role"] == "user" else "ASSISTANT"
                context_parts.append(f"[{role_label}]: {m['content']}")
            gt["_context"] = "\n\n".join(context_parts)
        else:
            gt["_context"] = "(No prior context — this is the first turn)"

        # Latest user message
        last_msg = clean_msgs[-1]
        if last_msg["role"] == "user":
            gt["_latest_user"] = last_msg["content"]
        else:
            log.warning(f"Last message in prompt is '{last_msg['role']}', expected 'user'")
            gt["_latest_user"] = last_msg["content"]

    except Exception as e:
        log.error(f"Exception in _build_conv_context: {type(e).__name__}: {e}\n"
                 f"{traceback.format_exc()}")
        gt["_context"] = f"(Error extracting context: {e})"
        gt["_latest_user"] = ""

    return gt


# ============================================================================
# SINGLE-SAMPLE SCORING
# ============================================================================
def _score_single(solution_str, ground_truth, extra_info, judge_result, eval_mode):
    extracted = _extract_answer(solution_str)
    answer_to_grade = extracted if extracted else ""
    format_penalty = _compute_format_penalty(solution_str)

    think_match = re.search(
        rf"{re.escape(THINKING_START)}(.+?){re.escape(THINKING_END)}",
        solution_str, flags=re.DOTALL,
    )
    reasoning_len = len(think_match.group(1).split()) if think_match else 0
    answer_len = len(answer_to_grade.split()) if answer_to_grade else 0

    is_refusal = bool(judge_result) and judge_result.get("error") == "refusal"

    if eval_mode == "qa":
        key_points = ground_truth.get("key_points", [])
        if judge_result and judge_result.get("error") == "refusal":
            # Judge refused (safety filter) — couldn't grade; assign neutral reward.
            reward = {"raw_score": REFUSAL_REWARD, "penalized_score": REFUSAL_REWARD, "dimension_scores": {}}
            judge_score = REFUSAL_REWARD
            completeness_val = 1.0
        elif judge_result is not None and "error" not in judge_result:
            reward = _compute_qa_reward(judge_result, key_points)
            judge_score = reward["penalized_score"]
            completeness_val = _compute_completeness(judge_result, key_points)
        else:
            reward = {"raw_score": 0.0, "penalized_score": 0.0, "dimension_scores": {}}
            judge_score = 0.0
            completeness_val = 1.0

        final_score = max(0.0, judge_score + format_penalty)
        return {
            "score": final_score,
            "reward/judge_score": judge_score,
            "reward/raw_score": reward["raw_score"],
            "reward/format_penalty": format_penalty,
            # On refusal, report dimensions on their native scales: 0-10 for
            # accuracy/clarity, 1-5 for completeness.
            "reward/accuracy": REFUSAL_REWARD * 10 if is_refusal else (_safe_float(judge_result.get("accuracy", 0), 0, "qa_acc") if judge_result else 0),
            "reward/completeness": REFUSAL_REWARD * 5 if is_refusal else completeness_val,
            "reward/clarity": REFUSAL_REWARD * 10 if is_refusal else (_safe_float(judge_result.get("clarity", 0), 0, "qa_clar") if judge_result else 0),
            "reward/behavior": 0.0,
            "reward/compliance": 0.0,
            "reward/reasoning_length": reasoning_len,
            "reward/answer_length": answer_len,
            "reward/eval_mode": "qa",
        }
    else:
        if judge_result and judge_result.get("error") == "refusal":
            # Judge refused (safety filter) — couldn't grade; assign neutral reward.
            reward = {"raw_score": REFUSAL_REWARD, "penalized_score": REFUSAL_REWARD, "dimension_scores": {}}
            judge_score = REFUSAL_REWARD
        elif judge_result is not None and "error" not in judge_result:
            reward = _compute_conv_reward(judge_result)
            judge_score = reward["penalized_score"]
        else:
            reward = {"raw_score": 0.0, "penalized_score": 0.0, "dimension_scores": {}}
            judge_score = 0.0

        final_score = max(0.0, judge_score + format_penalty)
        return {
            "score": final_score,
            "reward/judge_score": judge_score,
            "reward/raw_score": reward["raw_score"],
            "reward/format_penalty": format_penalty,
            # On refusal, report dimensions on their native 0-10 scale.
            "reward/accuracy": REFUSAL_REWARD * 10 if is_refusal else (_safe_float(judge_result.get("accuracy", 0), 0, "conv_acc") if judge_result else 0),
            "reward/completeness": 0.0,
            "reward/clarity": 0.0,
            "reward/behavior": REFUSAL_REWARD * 10 if is_refusal else (_safe_float(judge_result.get("behavior", 0), 0, "conv_beh") if judge_result else 0),
            "reward/compliance": REFUSAL_REWARD * 10 if is_refusal else (_safe_float(judge_result.get("compliance", 0), 0, "conv_comp") if judge_result else 0),
            "reward/reasoning_length": reasoning_len,
            "reward/answer_length": answer_len,
            "reward/eval_mode": "conversation",
        }


# ============================================================================
# BATCHED JUDGE CALLS
# ============================================================================
async def _judge_one(eval_mode, *args):
    """Run a single judge call with its own client (for the single-sample path)."""
    async with _build_judge_client() as client:
        if eval_mode == "qa":
            return await _call_qa_judge(*args, client=client)
        return await _call_conv_judge(*args, client=client)


async def _call_judge_batch(items):
    # Client and semaphore are created inside this coroutine so they bind to the
    # event loop that asyncio.run() spins up for this batch — never reused across
    # loops (which is what caused "TCPTransport closed ... handler is closed").
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async with _build_judge_client() as client:
        async def _guarded(eval_mode, *args):
            async with sem:
                if eval_mode == "qa":
                    return await _call_qa_judge(*args, client=client)
                else:
                    return await _call_conv_judge(*args, client=client)

        tasks = [_guarded(*item) for item in items]
        return await asyncio.gather(*tasks)


# ============================================================================
# EVAL MODE DETECTION
# ============================================================================
def _determine_eval_mode(data_source, extra_info):
    if extra_info and isinstance(extra_info, dict):
        mode = extra_info.get("eval_mode")
        if mode:
            return mode
    if data_source == "medical_conv":
        return "conversation"
    return "qa"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def compute_score(
    data_source=None, solution_str=None, ground_truth=None, extra_info=None,
    data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None,
    **kwargs,
):
    if solution_strs is not None:
        return _compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos)
    else:
        return _compute_score_single(data_source, solution_str, ground_truth, extra_info)


def _compute_score_single(data_source, solution_str, ground_truth, extra_info=None):
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    eval_mode = _determine_eval_mode(data_source, extra_info)


    extracted = _extract_answer(solution_str)
    answer_to_grade = extracted if extracted else ""



    judge_result = None
    if answer_to_grade:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if eval_mode == "qa":
            question = extra_info.get("question", "") if extra_info and isinstance(extra_info, dict) else ""
            key_points = ground_truth.get("key_points", [])
            coro = _judge_one("qa", question, key_points, answer_to_grade, solution_str)
        else:
            # Build conversation context from the prompt
            prompt = None
            if extra_info and isinstance(extra_info, dict):
                prompt = extra_info.get("prompt", [])
            gt_with_context = _build_conv_context(prompt or [], ground_truth)
            coro = _judge_one("conversation", gt_with_context, answer_to_grade, solution_str)

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                judge_result = pool.submit(asyncio.run, coro).result()
        else:
            judge_result = asyncio.run(coro)

    return _score_single(solution_str, ground_truth, extra_info, judge_result, eval_mode)


def _compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None):
    n = len(solution_strs)
    if extra_infos is None:
        extra_infos = [{}] * n


    judge_items = []
    parsed_gts = []
    eval_modes = []
    qa_count = 0
    conv_count = 0

    for i in range(n):
        gt = ground_truths[i]
        if isinstance(gt, str):
            gt = json.loads(gt)
        parsed_gts.append(gt)

        ei = extra_infos[i] if extra_infos[i] else {}
        if not isinstance(ei, dict):
            try:
                ei = dict(ei)
            except Exception:
                ei = {}

        eval_mode = _determine_eval_mode(
            data_sources[i] if data_sources else None, ei
        )
        eval_modes.append(eval_mode)

        extracted = _extract_answer(solution_strs[i])
        answer_to_grade = extracted if extracted else ""

        if answer_to_grade:
            if eval_mode == "qa":
                question = ei.get("question", "") if isinstance(ei, dict) else ""
                key_points = gt.get("key_points", [])
                judge_items.append((i, "qa", question, key_points, answer_to_grade, solution_strs[i]))
                qa_count += 1
            else:
                prompt = ei.get("prompt", []) if isinstance(ei, dict) else []
                gt_with_context = _build_conv_context(prompt, gt)
                judge_items.append((i, "conversation", gt_with_context, answer_to_grade, solution_strs[i]))
                conv_count += 1

    judge_results = [None] * n
    if judge_items:
        call_args = []
        for item in judge_items:
            idx = item[0]
            mode = item[1]
            if mode == "qa":
                # (mode, question, key_points, answer, raw_generation)
                call_args.append(("qa", item[2], item[3], item[4], item[5]))
            else:
                # (mode, gt_with_context, answer, raw_generation)
                call_args.append(("conversation", item[2], item[3], item[4]))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                results = pool.submit(asyncio.run, _call_judge_batch(call_args)).result()
        else:
            results = asyncio.run(_call_judge_batch(call_args))

        for item, result in zip(judge_items, results):
            judge_results[item[0]] = result

    scores = []
    for i in range(n):
        scores.append(_score_single(
            solution_strs[i], parsed_gts[i], extra_infos[i], judge_results[i], eval_modes[i]
        ))

    return scores
