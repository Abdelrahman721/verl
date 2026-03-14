"""
Reward scoring for medical QA using LLM-as-a-judge (Gemini Flash).

The dataset stores ground_truth as a dict:
    {"gold_answer": str, "key_points": list[dict]}

Each key_point dict has: {"id": int, "point": str, "importance": "CORE"|"SUPPLEMENTARY"}

The compute_score function is the single entry point for verl's reward manager.

Expected model format: <think>reasoning</think> then the answer directly after.

Reward structure:
    - score: 0.0 to 1.0 (multi-dimensional: 40% accuracy + 40% completeness + 20% clarity)
    - format_penalty: -0.5 to 0.0 (applied on top of score, floored at 0.0)

Scoring dimensions:
    - Accuracy (1-10): categorical error severity → numeric score
    - Completeness (1-5): computed from key point coverage (CORE vs SUPPLEMENTARY)
    - Clarity (1-10): categorical quality → numeric score
    - Flag penalties: dangerous_error, fabrication, contradiction (multiplicative)
"""

import asyncio
import json
import os
import re

from openai import AsyncOpenAI

# ============================================================================
# FORMAT TAGS
# ============================================================================
THINKING_START = "<think>"
THINKING_END = "</think>"

# ============================================================================
# LLM JUDGE CONFIG — Vertex AI with service account credentials
# ============================================================================
_JUDGE_CLIENT = None
_JUDGE_CREDENTIALS = None
_JUDGE_TOKEN = None

JUDGE_CREDENTIALS_FILE = os.environ.get(
    "QA_JUDGE_CREDENTIALS_FILE", "/data/hazem/creds/vertex-sa.json"
)
JUDGE_BASE_URL = os.environ.get(
    "QA_JUDGE_BASE_URL",
    "https://aiplatform.googleapis.com/v1beta1/projects/project-a8ff85a9-571d-4e15-841/locations/global/endpoints/openapi/",
)
JUDGE_MODEL = os.environ.get("QA_JUDGE_MODEL", "gemini-3-flash-preview")


def _get_judge_client():
    """Return an AsyncOpenAI client authenticated via Vertex AI service account.

    Tokens are refreshed automatically when they expire (~60 min).
    """
    global _JUDGE_CLIENT, _JUDGE_CREDENTIALS, _JUDGE_TOKEN

    from google.oauth2 import service_account
    import google.auth.transport.requests

    if _JUDGE_CREDENTIALS is None:
        _JUDGE_CREDENTIALS = service_account.Credentials.from_service_account_file(
            JUDGE_CREDENTIALS_FILE,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    if _JUDGE_CREDENTIALS.expired or not _JUDGE_CREDENTIALS.token:
        _JUDGE_CREDENTIALS.refresh(google.auth.transport.requests.Request())

    if _JUDGE_CREDENTIALS.token != _JUDGE_TOKEN:
        _JUDGE_TOKEN = _JUDGE_CREDENTIALS.token
        _JUDGE_CLIENT = AsyncOpenAI(
            api_key=_JUDGE_TOKEN,
            base_url=JUDGE_BASE_URL,
        )

    return _JUDGE_CLIENT


# ============================================================================
# SCORING CONFIG
# ============================================================================
DIMENSION_WEIGHTS = {
    "accuracy": 0.40,
    "completeness": 0.40,
    "clarity": 0.20,
}

FLAG_PENALTIES = {
    "dangerous_error": 0.50,
    "fabrication": 0.70,
    "contradiction": 0.85,
}

# ============================================================================
# JUDGE PROMPT — same as evaluate_v2.py
# ============================================================================
JUDGE_SYSTEM_PROMPT = """\
You are an expert medical examiner evaluating a student's answer against \
a structured checklist of key points derived from a gold-standard reference.

You will receive:
1. A medical QUESTION
2. A KEY POINTS CHECKLIST (with importance: CORE or SUPPLEMENTARY)
3. The STUDENT'S ANSWER

━━━ CRITICAL: STYLE BLINDNESS ━━━
Do NOT penalize or reward based on:
- Formatting (bullets vs prose, markdown vs plain text)
- Structural similarity to any particular reference style
- Hedging language or confidence markers
- Whether the answer "reads like" a particular model's output

ONLY evaluate factual content and answer quality.

━━━ SCORING DIMENSIONS ━━━

### Accuracy (two-step: classify then score 1-10)

Step 1 — Classify the error severity into ONE category:
- NO_ERRORS: Everything stated is factually correct.
- TRIVIAL_IMPRECISION: Slightly imprecise wording but zero clinical \
consequence (e.g., "usually" vs "always", rounding a number).
- MINOR_ERROR: 1-2 small factual errors that do not affect clinical \
safety or decision-making.
- MODERATE_ERROR: A factual error with some clinical relevance — not \
dangerous, but a knowledgeable reader would notice.
- MAJOR_ERROR: Clinically meaningful error that could lead to wrong \
decisions (wrong mechanism, wrong drug class, wrong threshold).
- DANGEROUS: Misinformation that could directly cause patient harm.

Step 2 — Pick a score within the category's range:
- NO_ERRORS: 10
- TRIVIAL_IMPRECISION: 8-9
- MINOR_ERROR: 6-7
- MODERATE_ERROR: 4-5
- MAJOR_ERROR: 2-3
- DANGEROUS: 1

### Clarity & Depth (two-step: classify then score 1-10)

This dimension captures both how well the answer communicates AND how \
much useful depth it provides beyond the bare minimum.

Step 1 — Classify the answer quality into ONE category:
- EXCEPTIONAL: Goes above and beyond — provides helpful clinical context, \
relevant mechanisms, practical considerations, or insightful connections \
that enrich understanding. Well-structured and easy to follow.
- CLEAR: Answers the question clearly with good organization. Easy to follow. \
May not add extra depth but communicates the content well.
- BASIC: Understandable but could be better organized or more detailed. \
Gets the point across without being particularly clear or insightful.
- DISORGANIZED: Hard to follow, poorly structured, or so brief that key \
information is difficult to extract.
- INCOMPREHENSIBLE: Cannot be meaningfully parsed.

Step 2 — Pick a score within the category's range:
- EXCEPTIONAL: 9-10
- CLEAR: 7-8
- BASIC: 4-6
- DISORGANIZED: 2-3
- INCOMPREHENSIBLE: 1

━━━ FACTUAL ERROR FLAGS (true/false each) ━━━
- dangerous_error: Contains advice that could cause patient harm if followed.
- fabrication: Contains a made-up fact, citation, statistic, or mechanism.
- contradiction: Contradicts itself on a material point.

━━━ KEY POINT COVERAGE ━━━
For each key point in the checklist, think carefully and mark:
- COVERED: The concept is addressed correctly. Equivalent medical \
terminology or procedures count — do not require exact wording.
- PARTIAL: The concept is genuinely incomplete — the student started \
addressing it but left out a meaningful part. Using different but \
equivalent terminology is NOT partial — that is COVERED.
- MISSING: Not addressed at all.

Before marking a key point as PARTIAL or MISSING, ask yourself: would a \
medical expert reading the student's answer consider this concept \
adequately addressed? If yes, mark COVERED.

━━━ OUTPUT FORMAT ━━━
Respond in JSON only. No markdown fences. No preamble.

{
  "accuracy_category": "NO_ERRORS|TRIVIAL_IMPRECISION|MINOR_ERROR|MODERATE_ERROR|MAJOR_ERROR|DANGEROUS",
  "accuracy": <1-10>,
  "clarity_category": "EXCEPTIONAL|CLEAR|BASIC|DISORGANIZED|INCOMPREHENSIBLE",
  "clarity": <1-10>,
  "dangerous_error": <true|false>,
  "fabrication": <true|false>,
  "contradiction": <true|false>,
  "key_point_coverage": [
    {"id": <int>, "status": "COVERED|PARTIAL|MISSING", \
"note": "<brief explanation>"}
  ],
  "justification": "<2-4 sentence summary>"
}"""

JUDGE_USER_TEMPLATE = """\
QUESTION:
{question}

KEY POINTS CHECKLIST:
{key_points_json}

STUDENT'S ANSWER:
{answer}

Evaluate the student's answer:"""


# ============================================================================
# JSON PARSING
# ============================================================================
def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON:\n{cleaned[:500]}")


# ============================================================================
# COMPLETENESS FROM KEY POINTS
# ============================================================================
def _compute_completeness(judge_result: dict, key_points: list) -> float:
    """Compute completeness (1-5) from key point coverage weighted by importance.

    Formula:
        completeness = 0.5 + (core_score * 3.5) + (core_score * supp_score * 1)
        clamped to [1, 5]
    """
    KP_SCORE = {"COVERED": 1.0, "PARTIAL": 0.5, "MISSING": 0.0, "WRONG": 0.0}

    importance_by_id = {kp["id"]: kp.get("importance", "CORE") for kp in key_points}
    coverage = judge_result.get("key_point_coverage", [])

    core_scores = []
    supp_scores = []
    for kp in coverage:
        score = KP_SCORE.get(kp.get("status", "MISSING"), 0.0)
        imp = importance_by_id.get(kp["id"], "CORE")
        if imp == "CORE":
            core_scores.append(score)
        else:
            supp_scores.append(score)

    core_avg = sum(core_scores) / len(core_scores) if core_scores else 1.0
    supp_avg = sum(supp_scores) / len(supp_scores) if supp_scores else 1.0

    completeness = 0.5 + (core_avg * 3.5) + (core_avg * supp_avg * 1.0)
    return max(1.0, min(5.0, completeness))


# ============================================================================
# MULTI-DIMENSIONAL SCORE → 0-1
# ============================================================================
def _compute_reward(judge_result: dict, key_points: list) -> dict:
    """Compute weighted 0-1 score from judge dimensions + flag penalties."""
    completeness_val = _compute_completeness(judge_result, key_points)

    dims = {}
    for d in DIMENSION_WEIGHTS:
        if d == "completeness":
            val = max(1.0, min(5.0, completeness_val))
            dims[d] = (val - 1) / 4  # 1→0.0, 5→1.0
        else:
            val = min(10, max(1, judge_result.get(d, 1)))
            dims[d] = (val - 1) / 9  # 1→0.0, 10→1.0

    raw = sum(dims[k] * DIMENSION_WEIGHTS[k] for k in DIMENSION_WEIGHTS)

    penalized = raw
    for flag, penalty in FLAG_PENALTIES.items():
        if judge_result.get(flag, False):
            penalized *= penalty

    return {
        "raw_score": round(raw, 4),
        "penalized_score": round(penalized, 4),
        "dimension_scores": dims,
    }


# ============================================================================
# FORMAT SCORING — penalty only, no bonus
# ============================================================================
_match_format_perfect = re.compile(
    rf"^\s*"
    rf"{re.escape(THINKING_START)}.+?{re.escape(THINKING_END)}\s*"
    rf".+\s*\Z",
    flags=re.DOTALL,
)

_extract_after_think = re.compile(
    rf"{re.escape(THINKING_END)}\s*(.+)",
    flags=re.DOTALL,
)


def _extract_answer(response: str) -> str | None:
    """Extract everything after </think> as the answer."""
    m = _extract_after_think.search(response)
    return m.group(1).strip() if m else None


def _compute_format_penalty(response: str) -> float:
    """
    Format penalty — 0.0 for correct format, negative for broken format.
    """
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
# LLM JUDGE — multi-dimensional call
# ============================================================================
async def _call_judge(
    question: str,
    key_points: list,
    answer: str,
    max_retries: int = 3,
) -> dict:
    """Call the LLM judge and return parsed JSON result."""
    if not answer:
        return {
            "accuracy": 1, "clarity": 1,
            "key_point_coverage": [],
            "dangerous_error": False, "fabrication": False, "contradiction": False,
        }

    kp_json = json.dumps(key_points, indent=2)
    prompt = JUDGE_USER_TEMPLATE.format(
        question=question, key_points_json=kp_json, answer=answer
    )
    client = _get_judge_client()

    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=4096,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            parsed = _parse_json_response(text)

            for field in ["accuracy", "clarity"]:
                if field not in parsed:
                    raise ValueError(f"Missing field: {field}")

            return parsed
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2.0 * (attempt + 1))
            else:
                print(f"[QA REWARD] LLM judge failed: {e}")
                return {
                    "accuracy": 1, "clarity": 1,
                    "key_point_coverage": [],
                    "dangerous_error": False, "fabrication": False,
                    "contradiction": False, "error": str(e),
                }


# ============================================================================
# SINGLE-SAMPLE SCORING
# ============================================================================
def _score_single(solution_str, ground_truth, extra_info, judge_result):
    """Build the reward dict for one sample given its judge result."""
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    key_points = ground_truth.get("key_points", [])

    # Extract answer after </think>
    extracted = _extract_answer(solution_str)
    answer_to_grade = extracted if extracted else ""

    # Format penalty
    format_penalty = _compute_format_penalty(solution_str)

    if judge_result is not None:
        reward = _compute_reward(judge_result, key_points)
        judge_score = reward["penalized_score"]
        completeness_val = _compute_completeness(judge_result, key_points)
    else:
        judge_result = {}
        reward = {"raw_score": 0.0, "penalized_score": 0.0, "dimension_scores": {}}
        judge_score = 0.0
        completeness_val = 1.0

    # Length stats for monitoring
    think_match = re.search(
        rf"{re.escape(THINKING_START)}(.+?){re.escape(THINKING_END)}",
        solution_str,
        flags=re.DOTALL,
    )
    reasoning_len = len(think_match.group(1).split()) if think_match else 0
    answer_len = len(answer_to_grade.split()) if answer_to_grade else 0

    # Final score: judge reward (0-1) + format penalty, floored at 0
    final_score = max(0.0, judge_score + format_penalty)

    return {
        "score": final_score,
        "reward/judge_score": judge_score,
        "reward/raw_score": reward["raw_score"],
        "reward/format_penalty": format_penalty,
        "reward/accuracy": judge_result.get("accuracy", 0),
        "reward/completeness": completeness_val,
        "reward/clarity": judge_result.get("clarity", 0),
        "reward/reasoning_length": reasoning_len,
        "reward/answer_length": answer_len,
    }


# ============================================================================
# BATCHED JUDGE CALLS
# ============================================================================
_JUDGE_SEMAPHORE = None


async def _call_judge_batch(items):
    """Call the judge for all items concurrently with a concurrency limit."""
    global _JUDGE_SEMAPHORE
    if _JUDGE_SEMAPHORE is None:
        _JUDGE_SEMAPHORE = asyncio.Semaphore(32)

    async def _guarded(question, key_points, answer):
        async with _JUDGE_SEMAPHORE:
            return await _call_judge(question, key_points, answer)

    tasks = [_guarded(q, kp, a) for q, kp, a in items]
    return await asyncio.gather(*tasks)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def compute_score(
    data_source=None, solution_str=None, ground_truth=None, extra_info=None,
    # Plural (batched) kwargs — used by BatchRewardManager
    data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None,
    **kwargs,
):
    """
    Compute reward scores for medical QA.

    Handles two calling conventions:
    - NaiveRewardManager: singular kwargs (data_source, solution_str, ground_truth, extra_info)
    - BatchRewardManager: plural kwargs (data_sources, solution_strs, ground_truths, extra_infos)

    Returns:
        Single call: dict with "score" and component metrics
        Batched call: list of such dicts
    """
    # Detect which calling convention was used
    if solution_strs is not None:
        # Batched call
        return _compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos)
    else:
        # Single call
        return _compute_score_single(data_source, solution_str, ground_truth, extra_info)


def _compute_score_single(data_source, solution_str, ground_truth, extra_info=None):
    """Score a single sample (called by NaiveRewardManager)."""
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    gold_answer = ground_truth.get("gold_answer", "")
    key_points = ground_truth.get("key_points", [])

    question = ""
    if extra_info and isinstance(extra_info, dict):
        question = extra_info.get("question", "")

    extracted = _extract_answer(solution_str)
    answer_to_grade = extracted if extracted else ""

    # Call judge for this single sample
    judge_result = None
    if gold_answer and answer_to_grade:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                judge_result = pool.submit(
                    asyncio.run,
                    _call_judge(question, key_points, answer_to_grade),
                ).result()
        else:
            judge_result = asyncio.run(
                _call_judge(question, key_points, answer_to_grade)
            )

    return _score_single(solution_str, ground_truth, extra_info, judge_result)


def _compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None):
    """Score a batch of samples (called by BatchRewardManager)."""
    n = len(solution_strs)
    if extra_infos is None:
        extra_infos = [{}] * n

    # Parse ground truths and build judge call items
    judge_items = []  # (index, question, key_points, answer)
    parsed_gts = []
    for i in range(n):
        gt = ground_truths[i]
        if isinstance(gt, str):
            gt = json.loads(gt)
        parsed_gts.append(gt)

        gold_answer = gt.get("gold_answer", "")
        key_points = gt.get("key_points", [])

        question = ""
        ei = extra_infos[i]
        if ei and isinstance(ei, dict):
            question = ei.get("question", "")

        extracted = _extract_answer(solution_strs[i])
        answer_to_grade = extracted if extracted else ""

        if gold_answer and answer_to_grade:
            judge_items.append((i, question, key_points, answer_to_grade))

    # Run all judge calls concurrently
    judge_results = [None] * n
    if judge_items:
        call_args = [(q, kp, a) for _, q, kp, a in judge_items]

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

        for (idx, _, _, _), result in zip(judge_items, results):
            judge_results[idx] = result

    # Build per-sample score dicts
    scores = []
    for i in range(n):
        scores.append(_score_single(
            solution_strs[i], parsed_gts[i], extra_infos[i], judge_results[i]
        ))

    return scores
