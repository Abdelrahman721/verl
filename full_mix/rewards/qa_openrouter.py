"""
Reward scoring for medical QA + conversations using LLM-as-a-judge (OpenRouter).

This is the OpenRouter variant of qa_bedrock.py — IDENTICAL scoring, IDENTICAL
prompts, IDENTICAL entry point (compute_score), IDENTICAL reward dict shape,
IDENTICAL retry / refusal / parse / repair / validation logic. The ONLY
thing that changes vs. qa_bedrock.py is the library used to contact the judge:
AsyncAnthropicBedrock -> AsyncOpenAI (pointed at OpenRouter's OpenAI-compatible
HTTP API at https://openrouter.ai/api/v1).

OpenRouter speaks the OpenAI protocol, so the rest of this module (refusal
detection via finish_reason / message.refusal, JSON parse, repair, retry,
score computation) is unchanged from qa_openai.py.

Additionally, both judge calls use OpenAI's structured-output strict mode
(response_format={"type": "json_schema", "strict": True, ...}) with the
schemas QA_RESPONSE_SCHEMA / CONV_RESPONSE_SCHEMA defined below. This
guarantees the returned content parses AND has every required field with
the correct type, eliminating the "missing bare numeric" and "unescaped
quote inside justification" parse failures we used to see. The
_repair_qa_parsed / _repair_conv_parsed chain stays in place as
belt-and-suspenders but should now rarely fire.

Required env vars (read at first judge call):
  QA_JUDGE_OPENROUTER_API_KEY   OpenRouter API key. If unset, falls back to
                                the standard OPENROUTER_API_KEY, so you can
                                reuse an existing key without exporting a
                                judge-specific one.
Optional:
  QA_JUDGE_OPENROUTER_BASE_URL  Override the OpenRouter base URL (proxy /
                                self-host). Default: https://openrouter.ai/api/v1
  QA_JUDGE_MODEL                OpenRouter model id, "<provider>/<model>"
                                (default: "openai/gpt-5.4-mini")
  QA_JUDGE_MAX_TOKENS           (default: 16384)
  QA_JUDGE_CONCURRENCY          (default: 8)
  QA_JUDGE_REFUSAL_REWARD       (default: 0.5)
  QA_JUDGE_REFUSAL_DIR          dump dir for refused judge calls (default: off)
  QA_LEN_PENALTY_THRESHOLD      (default: 2.0)
  QA_LEN_PENALTY_K              (default: 0.1)
  QA_LEN_PENALTY_MAX            (default: 0.25)

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

from openai import AsyncOpenAI

# ============================================================================
# LOGGING
# ============================================================================
log = logging.getLogger("qa_v2_reward_openrouter")
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
# LLM JUDGE CONFIG (OpenRouter)
# ============================================================================
JUDGE_MODEL = os.environ.get("QA_JUDGE_MODEL", "openai/gpt-5.4-mini")
JUDGE_MAX_TOKENS = int(os.environ.get("QA_JUDGE_MAX_TOKENS", "16384"))
JUDGE_CONCURRENCY = int(os.environ.get("QA_JUDGE_CONCURRENCY", "8"))
# Fallback reward when the judge refuses (safety filter) — we can't grade the
# sample, so assign a neutral score rather than punishing the policy with 0.
REFUSAL_REWARD = float(os.environ.get("QA_JUDGE_REFUSAL_REWARD", "0.5"))

# ─── Length penalty (mitigates the RL "longer == higher KP coverage" bias) ───
# Triggered post-hoc on the judge score. Penalty is computed from the ratio
# R = words(candidate_answer) / words(gold_text). No penalty if R <= threshold;
# linear ramp k * (R - threshold) above it, capped at LEN_PENALTY_MAX.
# Only over-long is penalized (under-length is already caught by completeness).
LEN_PENALTY_THRESHOLD = float(os.environ.get("QA_LEN_PENALTY_THRESHOLD", "14.0"))
LEN_PENALTY_K         = float(os.environ.get("QA_LEN_PENALTY_K",         "0.05"))
LEN_PENALTY_MAX       = float(os.environ.get("QA_LEN_PENALTY_MAX",       "0.2"))


def _build_judge_client():
    """Build a fresh AsyncOpenAI client pointed at OpenRouter.

    A new client must be created inside each event loop. The httpx connection
    pool / TCP transport binds to the loop it is first used on; reusing a cached
    client across the short-lived loops created by asyncio.run() (see the
    ThreadPoolExecutor bridge below) raises "TCPTransport closed ... handler is
    closed". Always use this under `async with` so the client closes with its loop.

    Key resolution order:
      1. QA_JUDGE_OPENROUTER_API_KEY  (judge-specific override)
      2. OPENROUTER_API_KEY           (lets you reuse an existing key)
    Base URL is overridable via QA_JUDGE_OPENROUTER_BASE_URL; default is
    OpenRouter's public endpoint (https://openrouter.ai/api/v1). Unlike
    qa_openai.py, base_url is REQUIRED on the client object because the
    OpenAI SDK would otherwise default to api.openai.com.
    """
    api_key = (
        os.environ.get("QA_JUDGE_OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or ""
    )
    base_url = (
        os.environ.get("QA_JUDGE_OPENROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )
    if not api_key:
        raise RuntimeError(
            "OpenRouter judge API key not set. Set QA_JUDGE_OPENROUTER_API_KEY "
            "(or OPENROUTER_API_KEY) before launching."
        )

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=4,
    )


def _extract_judge_text(resp) -> str:
    """Concatenate text from an OpenAI chat.completions response."""
    # OpenAI-compatible: response has .choices[0].message.content (str | None).
    try:
        content = resp.choices[0].message.content or ""
    except (AttributeError, IndexError):
        content = ""
    return content.strip()


def _refusal_info(resp):
    """Return (category, explanation) for a refused response.

    OpenAI-style APIs surface a model refusal either via the structured
    ``message.refusal`` field (newer SDKs) or via ``finish_reason=='content_filter'``.
    Either way we collapse into the same (category, explanation) shape that
    the rest of this module already expects, so downstream handling and the
    ``_dump_refusal`` record are byte-identical to qa_bedrock.
    """
    try:
        choice = resp.choices[0]
    except (AttributeError, IndexError):
        return None, None
    finish = getattr(choice, "finish_reason", None)
    msg = getattr(choice, "message", None)
    explanation = getattr(msg, "refusal", None) if msg is not None else None
    if finish == "content_filter":
        return "content_filter", explanation
    if explanation:
        return "refusal", explanation
    return None, None


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
You are an expert medical examiner evaluating a student's answer against a structured checklist of key points derived from a gold-standard reference.

You will receive:
1. A medical QUESTION
2. A KEY POINTS CHECKLIST (with importance: CORE or SUPPLEMENTARY)
3. The STUDENT'S ANSWER

━━━ CRITICAL: STYLE BLINDNESS ━━━
Do NOT penalize or reward based on:
- Stylistic choice of format (bullets vs prose, markdown vs plain text) when both communicate the answer equally well. HOWEVER: a complex multi-part answer presented as one dense paragraph with no structural breaks IS a clarity defect, graded under Clarity & Depth below.
- Structural similarity to any particular reference style
- Hedging language or confidence markers
- Whether the answer "reads like" a particular model's output
- Answer length per se — a long answer is not automatically better or worse than a short one. HOWEVER: padding, repetition, and off-topic elaboration ARE clarity defects and must be graded under Clarity & Depth below — not ignored as style.

ONLY evaluate factual content and answer quality.

━━━ SCORING DIMENSIONS ━━━

### Accuracy (two-step: classify then score 1-10)

Accuracy measures whether the student's claims are factually correct. It does NOT measure completeness (that is handled by key point coverage).

RULE 1 — Extra content is not automatically an error:
The student may discuss concepts not in the checklist. The checklist is not exhaustive. Do NOT penalize accuracy just because the student said something absent from the reference — only penalize when the student says something WRONG.

RULE 2 — But extra content CAN contain errors:
When the student adds content beyond the checklist, you must still verify it. If they cite a specific number, name a specific drug, describe a specific mechanism, or quote a specific study result, check whether it sounds correct. Confidently stated specifics that are wrong ARE accuracy errors, even if the checklist doesn't cover them.

RULE 3 — Active error detection:
Do NOT default to NO_ERRORS. Before classifying, actively scrutinize the student's answer. Read each specific claim and ask: "Is this medically correct?" Watch for:
- Wrong mechanism of action, drug class, or receptor
- Incorrect anatomy or physiology (e.g., saying the diaphragm is paralyzed in REM when it is spared)
- Confusing one condition or entity with another
- Wrong dosage, threshold, half-life, or normal range
- Fabricated brand names, study names, or statistics
- Stating something as fact when it contradicts established science
A well-structured, confident answer can still contain errors. Good formatting does not mean good accuracy — verify the claims.

Step 1 — Classify the error severity into ONE category:
- NO_ERRORS: You have actively checked the student's claims and everything stated is factually correct.
- TRIVIAL_IMPRECISION: Slightly imprecise wording but zero clinical consequence (e.g., "usually" vs "always", rounding a number).
- MINOR_ERROR: 1-2 small factual errors that do not affect clinical safety or decision-making.
- MODERATE_ERROR: A factual error with some clinical relevance — not dangerous, but a knowledgeable reader would notice.
- MAJOR_ERROR: Clinically meaningful error that could lead to wrong decisions (wrong mechanism, wrong drug class, wrong threshold).
- DANGEROUS: Misinformation that could directly cause patient harm.

Step 2 — Pick a score within the category's range:
- NO_ERRORS: 10
- TRIVIAL_IMPRECISION: 8-9
- MINOR_ERROR: 7
- MODERATE_ERROR: 4-6
- MAJOR_ERROR: 2-3
- DANGEROUS: 0-1

### Clarity & Depth (two-step: classify then score 1-10)

This dimension captures how well the answer communicates and how much useful depth it provides beyond the bare minimum.

Step 1 — Classify into ONE category:
- EXCEPTIONAL: Provides genuine clinical insight — helpful context, relevant mechanisms, practical considerations, or connections that enrich understanding. Well-structured AND well-calibrated.Do NOT punish appropriate elaboration. A thorough answer that earns its length is EXCEPTIONAL; brevity that omits useful detail is not.
- CLEAR: Answers the question clearly with good organization. Most competent answers belong here.
- BASIC: Understandable but exhibits one or more of: weak organization, missing relevant detail, *substantial padding or repetition that does not add clinical value* Coherent off-topic elaboration is still off-topic.
- DISORGANIZED: Hard to follow, poorly structured, too brief to convey the answer, or *so verbose / off-scope that the actual answer is hard to extract from the surrounding content*.
- INCOMPREHENSIBLE: Cannot be meaningfully parsed.

Step 2 — Pick a score:
- EXCEPTIONAL: 9-10
- CLEAR: 7-8
- BASIC: 4-6
- DISORGANIZED: 2-3
- INCOMPREHENSIBLE: 1

━━━ KEY POINT COVERAGE ━━━

Key points represent MEDICAL CONCEPTS, not exact phrases. Determine whether the student conveyed each concept — not whether they used the same words or level of specificity as the reference.

For each key point, mark:
- COVERED: The medical concept is addressed correctly. Different terminology, different examples, or less granular detail is fine. If a key point mentions a specific number (e.g., "20-day half-life") and the student conveys the concept qualitatively (e.g., "prolonged half-life of weeks"), that is COVERED. HOWEVER: merely mentioning the topic area is NOT enough. The student must convey the KEY INSIGHT. E.g., if the key point is "aliskiren + valsartan is contraindicated in diabetes due to dual RAAS blockade," discussing RAAS in general without stating the contraindication is PARTIAL, not COVERED.
- PARTIAL: The student touches the right topic but misses the specific clinical insight that makes the key point important. The omission must matter to a medical expert's understanding — not just a minor detail.
- MISSING: Not addressed at all.

Do NOT mark PARTIAL for:
- Using different but equivalent medical terminology
- Providing the concept at a different level of specificity
- Omitting a minor detail that doesn't change the clinical meaning
- Adding extra content alongside the key point

Test: "Would a medical expert say this concept is adequately addressed?" If yes → COVERED.

━━━ OUTPUT FORMAT ━━━
Respond in JSON only. No markdown fences. No preamble.

{
  "accuracy_category": "NO_ERRORS|TRIVIAL_IMPRECISION|MINOR_ERROR|MODERATE_ERROR|MAJOR_ERROR|DANGEROUS",
  "accuracy": <1-10>,
  "accuracy_justification": "<Quote or describe each specific error found. If NO_ERRORS, state what you checked. If MINOR/MODERATE/MAJOR, identify the exact claim that is wrong and what the correct fact is. E.g.: 'The student states the half-life is 7 hours — the correct value is 47 hours. This is a clinically meaningful error.' If no errors: 'All claims checked against the key points and general medical knowledge appear correct.'>",
  "clarity_category": "EXCEPTIONAL|CLEAR|BASIC|DISORGANIZED|INCOMPREHENSIBLE",
  "clarity": <1-10>,
  "clarity_justification": "<Explain why this category. If EXCEPTIONAL, what specific insight elevates it beyond CLEAR? If BASIC or lower, what makes it hard to follow? E.g.: 'Well-organized with numbered points, but excessively verbose — repeats the same concept across multiple paragraphs without adding value. CLEAR, not EXCEPTIONAL.'>",
  "key_point_coverage": [
    {"id": <int>, "status": "COVERED|PARTIAL|MISSING", "note": "<brief explanation>"}
  ],
  "overall_justification": "<3-5 sentence summary tying together accuracy, completeness, and clarity assessments. Explain what the student got right, what they missed, and the overall quality of the response.>"
}"""

QA_JUDGE_USER_TEMPLATE = """\
QUESTION:
{question}

KEY POINTS CHECKLIST:
{key_points_json}

STUDENT'S ANSWER:
{answer}

Evaluate the student's answer:"""

# Structured-output schema for the QA judge — paired with
# response_format={"type": "json_schema", "strict": True, ...} on the chat
# completion. Mirrors the JSON the system prompt asks for, field-for-field.
# strict mode requires:
#   - additionalProperties: False on every object
#   - every property listed under `required`
#   - no range/format constraints (so 1-10 bounds stay enforced only by the
#     prompt, but _safe_float + _compute_qa_reward clamp downstream anyway)
QA_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "accuracy_category", "accuracy", "accuracy_justification",
        "clarity_category",  "clarity",  "clarity_justification",
        "key_point_coverage", "overall_justification",
    ],
    "properties": {
        "accuracy_category": {
            "type": "string",
            "enum": ["NO_ERRORS", "TRIVIAL_IMPRECISION", "MINOR_ERROR",
                     "MODERATE_ERROR", "MAJOR_ERROR", "DANGEROUS"],
        },
        "accuracy":              {"type": "integer"},
        "accuracy_justification": {"type": "string"},
        "clarity_category": {
            "type": "string",
            "enum": ["EXCEPTIONAL", "CLEAR", "BASIC",
                     "DISORGANIZED", "INCOMPREHENSIBLE"],
        },
        "clarity":              {"type": "integer"},
        "clarity_justification": {"type": "string"},
        "key_point_coverage": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "status", "note"],
                "properties": {
                    "id":     {"type": "integer"},
                    "status": {"type": "string",
                               "enum": ["COVERED", "PARTIAL", "MISSING"]},
                    "note":   {"type": "string"},
                },
            },
        },
        "overall_justification": {"type": "string"},
    },
}

# ============================================================================
# CONVERSATION JUDGE PROMPT
# ============================================================================
CONV_JUDGE_SYSTEM_PROMPT = """\
You are an expert medical communication evaluator. You are evaluating a single response from a medical AI assistant within an ongoing conversation.

You will receive:
1. The EXPECTED BEHAVIOR for this conversation type (what good looks like)
2. The CONVERSATION CONTEXT (prior turns leading to this response)
3. The LATEST USER MESSAGE (what the user just said)
4. A GOLD REFERENCE RESPONSE (a high-quality example of what the response should be)
5. The CANDIDATE RESPONSE (the response you are evaluating)
6. The TURN POSITION (which response this is in the conversation)

━━━ CRITICAL: WHAT YOU ARE EVALUATING ━━━

You are NOT evaluating a standalone medical answer. You are evaluating one turn in a conversation. The quality of a response depends on:
- What came before (the conversation context)
- Who is talking (inferred from the context and conversation type)
- What the conversation type expects at this point
- Whether medical content is accurate

A response can be medically accurate but behaviorally wrong (e.g., giving a physician unnecessary disclaimers, or failing to flag an emergency for a patient). A response can be behaviorally perfect but medically wrong. Both matter.

━━━ CRITICAL: TURN POSITION CALIBRATION ━━━

The GOLD REFERENCE shows what THIS specific turn needs to accomplish. Use it as your anchor. Do not judge the candidate by whether it would work as a standalone answer — judge it by whether it does what the gold reference does at this point in the conversation.

━━━ SCORING DIMENSIONS ━━━

### Accuracy (two-step: classify then score 0-10)

Does the response contain correct medical information?

This is the same standard regardless of conversation type. Facts must be right whether the user is a physician or a patient. The PRESENTATION changes by audience — the ACCURACY standard does not.

RULE — Active error detection:
Do NOT default to NO_ERRORS. Scrutinize each medical claim. Watch for:
- Wrong mechanism, drug class, dosage, threshold, or normal range
- Confusing one condition with another
- Stating something as fact that contradicts established science
- Inventing specific statistics, brand names, or study results
A well-written, confident response can still contain errors.

RULE — Accuracy vs. simplification:
When the response intentionally simplifies for a lay audience, simplification is NOT an accuracy error. "Your blood is having trouble clotting" is an acceptable simplification of coagulopathy. But "aspirin thins the blood by dissolving clots" IS an accuracy error — it misdescribes the mechanism.

Step 1 — Classify:
- NO_ERRORS: All medical claims are factually correct.
- TRIVIAL_IMPRECISION: Slightly imprecise but zero clinical consequence.
- MINOR_ERROR: 1-2 small factual errors, no clinical impact.
- MODERATE_ERROR: A noticeable error with some clinical relevance.
- MAJOR_ERROR: Could lead to wrong clinical decisions.
- DANGEROUS: Could directly cause patient harm.

Step 2 — Score:
- NO_ERRORS: 10
- TRIVIAL_IMPRECISION: 8-9
- MINOR_ERROR: 7
- MODERATE_ERROR: 4-6
- MAJOR_ERROR: 2-3
- DANGEROUS: 0-1

### Behavioral Appropriateness (two-step: classify then score 0-10)

Does the response match the expected behavior for this conversation type and turn position?

Read the EXPECTED BEHAVIOR description carefully. It defines what "appropriate" means for this specific interaction. Then evaluate whether the candidate response matches that expectation.

Things to consider:
- TONE: Does it match the user's level? (Peer-level for professionals, warm and plain for patients, pedagogical for students)
- SAFETY POSTURE: For professionals — is it direct without unnecessary hedging? For patients — does it include appropriate AI disclaimers and referral guidance? For emergencies — is urgency flagged FIRST?
- TURN AWARENESS: This is critical for mid-conversation turns. Evaluate:
  • Does the response build on what was already discussed, or does it unnecessarily repeat prior content?
  • If the user provided new information, does the response integrate it and update the picture? ("Given what you just mentioned about X, this changes things because...")
  • Does it answer the CURRENT question directly, or does it rehash the entire topic from scratch?
  • For later turns, the response should be more focused and specific than the opening turn — the broad groundwork was already laid.
  • If the user references something from an earlier turn ("you mentioned X"), does the response show continuity?
- SCOPE: For specialists, does it stay within their actionable scope? More broadly, does the response stay focused on what was actually asked on this turn, or does it drift into tangentially related topics — even correct, coherent ones — that weren't requested? Off-topic elaboration and padding beyond the question are behavioral defects, not added value. Right-sized depth that earns its length is fine; covering neighboring topics "while we're here" is not.
- CONVERSATIONAL PROGRESSION: A good mid-conversation response feels like a natural continuation — it doesn't read like a standalone answer that ignores everything that came before. Penalize responses that would make sense as a turn-1 answer but ignore the established context.

Step 1 — Classify:
- EXEMPLARY: Matches the expected behavior precisely. Tone, safety posture, turn awareness, scope, and length calibration are all exactly right — right-sized for the turn, no padding, no off-topic elaboration, no rehashing. A training example of how to handle this type of interaction.
- APPROPRIATE: Generally matches expectations with minor gaps. The response would serve the user well. Most good responses belong here.
- PARTIALLY_APPROPRIATE: Gets some behavioral aspects right but misses others. E.g., correct tone but fails to flag urgency, appropriate depth but wrong safety posture, or *substantially padded / drifts into tangential topics that weren't asked about* (even if those topics are themselves correct and coherent).
- INAPPROPRIATE: Significant behavioral mismatch. E.g., giving a physician disclaimers, refusing to discuss diagnoses with a patient, burying an emergency flag, or treating a specialist like a student.
- HARMFUL_BEHAVIOR: The behavioral mismatch could cause harm. E.g., telling a patient to adjust their own medication dosing, or failing to flag a life-threatening emergency.

Step 2 — Score:
- EXEMPLARY: 9-10
- APPROPRIATE: 7-8
- PARTIALLY_APPROPRIATE: 4-6
- INAPPROPRIATE: 2-3
- HARMFUL_BEHAVIOR: 0-1

### Instruction Compliance (two-step: classify then score 0-10)

Did the response follow any explicit or implicit instructions from the user or the conversation type?

Explicit instructions: "give me 3 items," "keep it short," "skip the pathophysiology," "format as a SOAP note," "explain it simpler."

Implicit instructions from the conversation type: the EXPECTED BEHAVIOR description may specify structural or content requirements (e.g., "produce a ranked differential," "lead with priorities," "translate into plain language").

Step 1 — Classify:
- FULLY_COMPLIANT: All instructions (explicit and implicit) followed.
- MOSTLY_COMPLIANT: Main instructions followed, minor deviations. E.g., asked for 3 items and got 4, or asked for brief and got slightly long.
- PARTIALLY_COMPLIANT: Some instructions followed, others missed. E.g., correct format but wrong scope, or right content but wrong structure.
- NON_COMPLIANT: Instructions largely ignored. The response may be medically fine but doesn't do what was asked.
- NOT_APPLICABLE: No specific instructions — the conversation type is open-ended. Score 8 by default.

Step 2 — Score:
- FULLY_COMPLIANT: 9-10
- MOSTLY_COMPLIANT: 7-8
- PARTIALLY_COMPLIANT: 4-6
- NON_COMPLIANT: 2-3
- NOT_APPLICABLE: 8

━━━ OUTPUT FORMAT ━━━
Respond in JSON only. No markdown fences. No preamble.

{
  "accuracy_category": "NO_ERRORS|TRIVIAL_IMPRECISION|MINOR_ERROR|MODERATE_ERROR|MAJOR_ERROR|DANGEROUS",
  "accuracy": <0-10>,
  "accuracy_justification": "<Identify specific errors or confirm what was checked.>",
  "behavior_category": "EXEMPLARY|APPROPRIATE|PARTIALLY_APPROPRIATE|INAPPROPRIATE|HARMFUL_BEHAVIOR",
  "behavior": <0-10>,
  "behavior_justification": "<Explain how the response matches or deviates from expected behavior. Reference specific aspects: tone, safety posture, turn awareness, scope.>",
  "compliance_category": "FULLY_COMPLIANT|MOSTLY_COMPLIANT|PARTIALLY_COMPLIANT|NON_COMPLIANT|NOT_APPLICABLE",
  "compliance": <0-10>,
  "compliance_justification": "<What instructions existed (explicit or from the type description)? Which were followed, which weren't?>",
  "overall_justification": "<3-5 sentence summary. What did the response get right, what did it miss, and how does it compare to the gold reference?>"
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

# Structured-output schema for the conversation judge — same strict-mode
# rules as QA_RESPONSE_SCHEMA above. The bare numeric fields stay typed as
# integers; the prompt's 0-10 range remains enforced by the prompt + the
# clamp inside _compute_conv_reward.
CONV_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "accuracy_category",   "accuracy",   "accuracy_justification",
        "behavior_category",   "behavior",   "behavior_justification",
        "compliance_category", "compliance", "compliance_justification",
        "overall_justification",
    ],
    "properties": {
        "accuracy_category": {
            "type": "string",
            "enum": ["NO_ERRORS", "TRIVIAL_IMPRECISION", "MINOR_ERROR",
                     "MODERATE_ERROR", "MAJOR_ERROR", "DANGEROUS"],
        },
        "accuracy":              {"type": "integer"},
        "accuracy_justification": {"type": "string"},
        "behavior_category": {
            "type": "string",
            "enum": ["EXEMPLARY", "APPROPRIATE", "PARTIALLY_APPROPRIATE",
                     "INAPPROPRIATE", "HARMFUL_BEHAVIOR"],
        },
        "behavior":              {"type": "integer"},
        "behavior_justification": {"type": "string"},
        "compliance_category": {
            "type": "string",
            "enum": ["FULLY_COMPLIANT", "MOSTLY_COMPLIANT",
                     "PARTIALLY_COMPLIANT", "NON_COMPLIANT", "NOT_APPLICABLE"],
        },
        "compliance":              {"type": "integer"},
        "compliance_justification": {"type": "string"},
        "overall_justification":    {"type": "string"},
    },
}


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
# PARTIAL-RESPONSE REPAIR
# ============================================================================
# Observed failure mode under load: the judge emits "<dim>_category" and
# "<dim>_justification" but skips the bare "<dim>: <int>" field between them.
# Rather than burning 10 retries on a deterministic LLM mistake, derive the
# bare numeric from the category's midpoint when only the bare field is
# missing. The judge already committed to the category; the numeric is just
# a transcription drop.

_QA_ACCURACY_CATEGORY_SCORES = {
    "NO_ERRORS":            10.0,
    "TRIVIAL_IMPRECISION":   8.5,   # band: 8-9
    "MINOR_ERROR":           7.0,
    "MODERATE_ERROR":        5.0,   # band: 4-6
    "MAJOR_ERROR":           2.5,   # band: 2-3
    "DANGEROUS":             0.5,   # band: 0-1
}
_QA_CLARITY_CATEGORY_SCORES = {
    "EXCEPTIONAL":           9.5,   # band: 9-10
    "CLEAR":                 7.5,   # band: 7-8
    "BASIC":                 5.0,   # band: 4-6
    "DISORGANIZED":          2.5,   # band: 2-3
    "INCOMPREHENSIBLE":      1.0,
}
_CONV_BEHAVIOR_CATEGORY_SCORES = {
    "EXEMPLARY":             9.5,
    "APPROPRIATE":           7.5,
    "PARTIALLY_APPROPRIATE": 5.0,
    "INAPPROPRIATE":         2.5,
    "HARMFUL_BEHAVIOR":      0.5,
}
_CONV_COMPLIANCE_CATEGORY_SCORES = {
    "FULLY_COMPLIANT":       9.5,
    "MOSTLY_COMPLIANT":      7.5,
    "PARTIALLY_COMPLIANT":   5.0,
    "NON_COMPLIANT":         2.5,
    "NOT_APPLICABLE":        8.0,   # rubric specifies "score 8 by default"
}


def _fill_from_category(parsed: dict, bare_key: str, cat_key: str, table: dict) -> None:
    """If bare_key is missing but cat_key holds a known category, fill bare_key
    from the table. No-op if already present or category isn't recognised."""
    if not isinstance(parsed, dict):
        return
    if bare_key in parsed:
        return
    cat = parsed.get(cat_key)
    if isinstance(cat, str) and cat in table:
        parsed[bare_key] = table[cat]
        log.warning(
            f"Judge omitted '{bare_key}' bare field; derived "
            f"{parsed[bare_key]} from {cat_key}='{cat}'"
        )


def _repair_qa_parsed(parsed: dict) -> dict:
    """Patch common omissions in a QA judge response in place.

    Handles two patterns observed in production:
      1. Renamed bare key: "accuracy_score" / "clarity_score" instead of
         "accuracy" / "clarity".
      2. Skipped bare numeric: "accuracy_category" + "accuracy_justification"
         emitted but no bare "accuracy" between them.
    """
    if not isinstance(parsed, dict):
        return parsed
    # 1. Renamed bare key.
    if "accuracy" not in parsed and "accuracy_score" in parsed:
        parsed["accuracy"] = parsed["accuracy_score"]
    if "clarity" not in parsed and "clarity_score" in parsed:
        parsed["clarity"] = parsed["clarity_score"]
    # 2. Category fallback.
    _fill_from_category(parsed, "accuracy", "accuracy_category", _QA_ACCURACY_CATEGORY_SCORES)
    _fill_from_category(parsed, "clarity",  "clarity_category",  _QA_CLARITY_CATEGORY_SCORES)
    return parsed


def _repair_conv_parsed(parsed: dict) -> dict:
    """Patch common omissions in a conversation judge response in place."""
    if not isinstance(parsed, dict):
        return parsed
    # 1. Renamed bare key.
    for bare in ("accuracy", "behavior", "compliance"):
        alt = f"{bare}_score"
        if bare not in parsed and alt in parsed:
            parsed[bare] = parsed[alt]
    # 2. Category fallback.
    _fill_from_category(parsed, "accuracy",   "accuracy_category",   _QA_ACCURACY_CATEGORY_SCORES)
    _fill_from_category(parsed, "behavior",   "behavior_category",   _CONV_BEHAVIOR_CATEGORY_SCORES)
    _fill_from_category(parsed, "compliance", "compliance_category", _CONV_COMPLIANCE_CATEGORY_SCORES)
    return parsed


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


def _compute_length_penalty(answer: str, gold_text: str) -> tuple[float, float]:
    """Compute a length-bias penalty against the gold response length.

    Returns (penalty, ratio) where penalty <= 0. Skipped (0.0, 1.0) if either
    side is empty. Only over-long is penalized; under-length is already covered
    by the completeness dimension.
    """
    if not answer or not gold_text:
        return 0.0, 1.0
    cand_words = len(answer.split())
    gold_words = max(1, len(gold_text.split()))
    ratio = cand_words / gold_words
    if ratio <= LEN_PENALTY_THRESHOLD:
        return 0.0, ratio
    excess = ratio - LEN_PENALTY_THRESHOLD
    penalty = min(LEN_PENALTY_MAX, LEN_PENALTY_K * excess)
    return -penalty, ratio


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
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": QA_JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_completion_tokens=JUDGE_MAX_TOKENS,
                temperature=0.0,
                # Strict structured output — guarantees the returned content
                # parses AND matches QA_RESPONSE_SCHEMA. Eliminates the
                # "missing accuracy/clarity bare numeric" and "unescaped quote
                # inside justification" failure modes upstream. _repair_qa_parsed
                # stays in place as belt-and-suspenders but should rarely fire.
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name":   "qa_verdict",
                        "strict": True,
                        "schema": QA_RESPONSE_SCHEMA,
                    },
                },
            )
            # OpenRouter occasionally returns 200 OK with no choices when the
            # upstream provider rate-limits or rejects the request; the actual
            # reason is under resp.error. Surface as ValueError so the retry
            # loop picks it up.
            if not getattr(resp, "choices", None):
                upstream_err = getattr(resp, "error", None)
                resp_id = getattr(resp, "id", None)
                raise ValueError(
                    f"OpenRouter returned no choices "
                    f"(resp.id={resp_id!r}, error={upstream_err!r})"
                )
            # stop_reason mirrors the bedrock variable name; for OpenAI-style
            # APIs the equivalent field is finish_reason. Kept under the same
            # name so retry/log lines are byte-identical to qa_bedrock.
            stop_reason = getattr(resp.choices[0], "finish_reason", None)
            category, explanation = _refusal_info(resp)
            if category is not None:
                # Deterministic for identical input — do NOT retry.
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
            # Repair common judge omissions before validation: renamed bare
            # keys (accuracy_score → accuracy) and skipped bare numerics
            # when only the *_category and *_justification appear.
            parsed = _repair_qa_parsed(parsed)
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
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": CONV_JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_completion_tokens=JUDGE_MAX_TOKENS,
                temperature=0.0,
                # Strict structured output — see _call_qa_judge for the
                # rationale. _repair_conv_parsed stays as belt-and-suspenders.
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name":   "conv_verdict",
                        "strict": True,
                        "schema": CONV_RESPONSE_SCHEMA,
                    },
                },
            )
            # OpenRouter occasionally returns 200 OK with no choices when the
            # upstream provider rate-limits or rejects the request; the actual
            # reason is under resp.error. Surface as ValueError so the retry
            # loop picks it up.
            if not getattr(resp, "choices", None):
                upstream_err = getattr(resp, "error", None)
                resp_id = getattr(resp, "id", None)
                raise ValueError(
                    f"OpenRouter returned no choices "
                    f"(resp.id={resp_id!r}, error={upstream_err!r})"
                )
            # stop_reason mirrors the bedrock variable name; for OpenAI-style
            # APIs the equivalent field is finish_reason. Kept under the same
            # name so retry/log lines are byte-identical to qa_bedrock.
            stop_reason = getattr(resp.choices[0], "finish_reason", None)
            category, explanation = _refusal_info(resp)
            if category is not None:
                # Deterministic for identical input — do NOT retry.
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
            # Repair common judge omissions before validation: renamed bare
            # keys (accuracy_score → accuracy, etc.) and skipped bare
            # numerics when only the *_category and *_justification appear.
            parsed = _repair_conv_parsed(parsed)
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
        gold_text = ground_truth.get("gold_answer") or ""
        length_penalty, length_ratio = _compute_length_penalty(answer_to_grade, gold_text)
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

        final_score = max(0.0, judge_score + format_penalty + length_penalty)
        return {
            "score": final_score,
            "reward/judge_score": judge_score,
            "reward/raw_score": reward["raw_score"],
            "reward/format_penalty": format_penalty,
            "reward/length_penalty": length_penalty,
            "reward/length_ratio": length_ratio,
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
        gold_text = ground_truth.get("gold_response") or ""
        length_penalty, length_ratio = _compute_length_penalty(answer_to_grade, gold_text)
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

        final_score = max(0.0, judge_score + format_penalty + length_penalty)
        return {
            "score": final_score,
            "reward/judge_score": judge_score,
            "reward/raw_score": reward["raw_score"],
            "reward/format_penalty": format_penalty,
            "reward/length_penalty": length_penalty,
            "reward/length_ratio": length_ratio,
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
