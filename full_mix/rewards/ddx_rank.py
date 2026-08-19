"""
DDx-rank reward (LLM-adjudicated F1 + NDCG) — standalone, single-route variant.

This is a trimmed-down sibling of qa_openrouter.py that keeps ONLY the
``ddx_rank`` scoring path (ranked differential-diagnosis, graded by an LLM
"terminology adjudicator" via OpenRouter). All other eval modes (qa,
conversation, mcq, numeric, medec_hybrid, text_judge_conv, ranked_differential)
have been removed for clarity. Scoring math, prompts, parsing, retry/refusal
handling and the reward-dict shape for ddx_rank are byte-identical to the canonical.

Entry point (verl `custom_reward_function.name`): ``compute_score`` — supports both
the single-sample and the batched agent-loop calling conventions.

Routing (kept conditional so this module stays correct if imported/reused later):
an item is scored as ddx_rank when ANY of these hold —
  extra_info["eval_mode"] == "ddx_rank"  OR
  extra_info["task_type"] == "ddx_rank"  OR
  extra_info["helm_scenario"] == "ddx_rank"  OR
  data_source == "medical_benchmark_ddx".
Anything else is returned a neutral (score 0.0) reward with a warning instead of
being mis-scored or crashing.

Uniform return dict: EVERY returned score dict carries the identical canonical key
set (mode-specific fields default to neutral values). verl's agent-loop reward path
derives per-batch columns from the first sample then asserts every rollout worker
produced the SAME non_tensor keys; a ragged key set crashes DataProto.concat with an
AssertionError. Do not drop keys from `_CANONICAL_REWARD_DEFAULTS`.

Required env:
  QA_JUDGE_OPENROUTER_API_KEY   OpenRouter API key (falls back to OPENROUTER_API_KEY)
Optional:
  QA_JUDGE_OPENROUTER_BASE_URL  (default https://openrouter.ai/api/v1)
  QA_JUDGE_MODEL                (default "openai/gpt-5.4-mini")
  QA_JUDGE_MAX_TOKENS           (default 16384)
  QA_JUDGE_CONCURRENCY          (default 8)
  QA_JUDGE_REFUSAL_REWARD       (default 0.5)
  QA_JUDGE_REFUSAL_DIR          dump dir for refused judge calls (default off)
  QA_DDX_F1_WEIGHT              (default 0.7)
  QA_DDX_NDCG_WEIGHT            (default 0.3)
  QA_DDX_TOP3_PENALTY           (default 0.05)
  QA_DDX_TOP_MISS_PENALTY       (default 0.15)
  QA_DDX_MAX_CANDIDATES         (default 12)
"""

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid

from openai import AsyncOpenAI

# ============================================================================
# LOGGING
# ============================================================================
log = logging.getLogger("ddx_rank_reward")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(handler)
    log.setLevel(logging.WARNING)

# ============================================================================
# FORMAT TAGS + JUDGE CONFIG
# ============================================================================
THINKING_START = "<think>"
THINKING_END = "</think>"

JUDGE_MODEL = os.environ.get("QA_JUDGE_MODEL", "openai/gpt-5.4-mini")
JUDGE_MAX_TOKENS = int(os.environ.get("QA_JUDGE_MAX_TOKENS", "16384"))
JUDGE_CONCURRENCY = int(os.environ.get("QA_JUDGE_CONCURRENCY", "8"))
# Neutral reward when the judge refuses (safety filter) — we can't grade the sample.
REFUSAL_REWARD = float(os.environ.get("QA_JUDGE_REFUSAL_REWARD", "0.5"))

# DDx reward weights / penalties (env-overridable).
DDX_F1_WEIGHT        = float(os.environ.get("QA_DDX_F1_WEIGHT",        "0.7"))
DDX_NDCG_WEIGHT      = float(os.environ.get("QA_DDX_NDCG_WEIGHT",      "0.3"))
DDX_TOP3_PENALTY     = float(os.environ.get("QA_DDX_TOP3_PENALTY",     "0.05"))
DDX_TOP_MISS_PENALTY = float(os.environ.get("QA_DDX_TOP_MISS_PENALTY", "0.15"))
DDX_MAX_CANDIDATES   = int(os.environ.get("QA_DDX_MAX_CANDIDATES",     "12"))

REFUSAL_DUMP_DIR = os.environ.get("QA_JUDGE_REFUSAL_DIR", "")


# ============================================================================
# JUDGE CLIENT + RESPONSE HELPERS
# ============================================================================
def _build_judge_client():
    """Fresh AsyncOpenAI client pointed at OpenRouter.

    A new client must be created inside each event loop (the httpx transport binds
    to the loop it is first used on; reuse across asyncio.run() loops raises
    "TCPTransport closed"). Always use under `async with`.
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
    return AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=4)


def _extract_judge_text(resp) -> str:
    """Concatenate text from an OpenAI chat.completions response."""
    try:
        content = resp.choices[0].message.content or ""
    except (AttributeError, IndexError):
        content = ""
    return content.strip()


def _refusal_info(resp):
    """Return (category, explanation) for a refused response, else (None, None)."""
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


def _dump_refusal(record: dict) -> None:
    """Persist one refused judge call to its own JSON file under REFUSAL_DUMP_DIR."""
    if not REFUSAL_DUMP_DIR:
        return
    try:
        os.makedirs(REFUSAL_DUMP_DIR, exist_ok=True)
        fname = f"refusal_{int(time.time() * 1000)}_{os.getpid()}_{uuid.uuid4().hex[:8]}.json"
        with open(os.path.join(REFUSAL_DUMP_DIR, fname), "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        log.warning(f"Failed to dump refusal record: {type(e).__name__}: {e}")


# ============================================================================
# FORMAT SCORING (<think>...</think> answer)
# ============================================================================
_match_format_perfect = re.compile(
    rf"^\s*{re.escape(THINKING_START)}.+?{re.escape(THINKING_END)}\s*.+\s*\Z",
    flags=re.DOTALL,
)
_extract_after_think = re.compile(rf"{re.escape(THINKING_END)}\s*(.+)", flags=re.DOTALL)


def _extract_answer(response: str) -> str | None:
    m = _extract_after_think.search(response or "")
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
# DDx ADJUDICATOR PROMPT (verbatim)
# ============================================================================
DDX_JUDGE_SYSTEM_PROMPT = """\
ROLE
You are a clinical terminology adjudicator.

INPUT
You are given two RANKED differential-diagnosis lists, GOLD and CANDIDATE, each a list of
(name, rank) tuples with ranks counted from 1,
e.g. [('Aortic dissection', 1), ('Myocardial infarction', 2), ('Unstable angina', 3)]

WHAT MATCHES
A CANDIDATE matches a GOLD when it denotes:
- the SAME clinical condition — including synonyms, abbreviations, lay terms, and common
  alternate spellings (e.g. "flu" = "influenza"; "heart attack" = "myocardial infarction" = "MI"); OR
- Whether a more-specific term also counts depends on the GOLD as the follows:
  - GENERIC gold  -> matched by the generic term itself OR any of its subtypes
    (gold "pneumonia" is matched by "pneumonia", "bacterial pneumonia", or "viral pneumonia").
  - SPECIFIC gold -> matched ONLY by that exact condition or a more specific condition than the gold
    (gold "bacterial pneumonia" is matched only by "bacterial pneumonia").

WHAT DOES NOT MATCH (output 0)
- A vaguer term or a sibling subtype of a SPECIFIC gold
  (gold "bacterial pneumonia" is NOT matched by "pneumonia" or "viral pneumonia").
- Conditions that are merely related, or that differ by a clinically meaningful qualifier into
  a DIFFERENT diagnosis — different type, cause, acuity, or anatomical site
  (e.g. "heart attack" is NOT "heart failure"; "type 1 diabetes" is NOT "type 2 diabetes";
  "bacterial pneumonia" is NOT "viral pneumonia").

Several CANDIDATE items MAY map to the SAME gold (e.g. two subtypes of one generic gold).

TASK
For EVERY item in CANDIDATE (in candidate order), report "matched":
- "matched" = the rank number paired with the matching condition in the GOLD list — i.e. the
  second element of that (name, rank) tuple. Copy that number directly; do not recount positions.
- "matched" = 0 means that candidate corresponds to NO gold condition.
- Only use a number that actually appears next to a GOLD item; never invent one.

OUTPUT
Be concise. Do NOT explain. Return ONLY a JSON array, one object per candidate item, in
candidate order, and nothing else:
[{"candidate": "<candidate name>", "matched": <int>}]

EXAMPLES
- GOLD:      [('Myocardial infarction', 1), ('Pulmonary embolism', 2), ('Pneumonia', 3)]
  CANDIDATE: [('heart attack', 1), ('PE', 2), ('bacterial pneumonia', 3), ('viral pneumonia', 4), ('asthma', 5)]
  -> [{"candidate": "heart attack", "matched": 1}, {"candidate": "PE", "matched": 2},
      {"candidate": "bacterial pneumonia", "matched": 3}, {"candidate": "viral pneumonia", "matched": 3},
      {"candidate": "asthma", "matched": 0}]
     (heart attack = MI -> 1; PE = pulmonary embolism -> 2; bacterial and viral pneumonia are
      both kinds of the generic gold "Pneumonia" -> both 3; asthma is absent -> 0)

- GOLD:      [('bacterial pneumonia', 1), ('type 2 diabetes', 2)]
  CANDIDATE: [('pneumonia', 1), ('type 1 diabetes', 2)]
  -> [{"candidate": "pneumonia", "matched": 0}, {"candidate": "type 1 diabetes", "matched": 0}]
     (specific golds need the exact condition: "pneumonia" is vaguer than "bacterial pneumonia" -> 0;
      "type 1 diabetes" is a different type than "type 2 diabetes" -> 0)

Now label every CANDIDATE for the lists below. Return ONLY the JSON array."""

DDX_JUDGE_USER_TEMPLATE = """\
GOLD: {gold_list}
CANDIDATE: {candidate_list}"""


# ============================================================================
# CANDIDATE / GOLD PARSING
# ============================================================================
_DDX_NUMBERED_RE = re.compile(r"^\(?\s*\d+\s*[\.\)\-:]\s*(.+)$")
_DDX_BULLET_RE   = re.compile(r"^[-*•]\s*(.+)$")


def _parse_ddx_list(text: str, max_items: int = DDX_MAX_CANDIDATES) -> list[str]:
    """Best-effort parse of a ranked differential from free text.

    Priority: a JSON array of strings; numbered lines (``1. dx`` / ``1) dx``);
    bulleted lines; finally a newline/semicolon split (NEVER commas). De-duplicated
    case-insensitively, order preserved, capped at ``max_items``.
    """
    if not text:
        return []
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    items: list[str] = []
    m = re.search(r"\[.*\]", t, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                items = v
        except (json.JSONDecodeError, ValueError):
            items = []
    if not items:
        for line in t.splitlines():
            line = line.strip()
            if not line:
                continue
            mm = _DDX_NUMBERED_RE.match(line) or _DDX_BULLET_RE.match(line)
            if mm:
                items.append(mm.group(1))
    if not items:
        items = [p.strip() for p in re.split(r"[\n;]+", t) if p.strip()]

    cleaned: list[str] = []
    seen: set[str] = set()
    for x in items:
        x = re.sub(r"\s+", " ", x).strip().strip(".").strip()
        x = re.sub(r"\s*\(\s*\d+\s*\)\s*$", "", x).strip()   # drop trailing "(1)" rank artifact
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        cleaned.append(x)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _format_ddx_tuples(pairs) -> str:
    """Render [(name, rank), ...] as the python-tuple style the prompt shows."""
    return "[" + ", ".join(f"({str(n)!r}, {int(r)})" for n, r in pairs) + "]"


def _parse_json_array(text: str) -> list:
    """Parse a top-level JSON array from the adjudicator response."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        v = json.loads(t)
        if isinstance(v, list):
            return v
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", t, re.S)
    if m:
        v = json.loads(m.group(0))
        if isinstance(v, list):
            return v
    raise ValueError(f"Could not parse JSON array:\n{t[:400]}")


def _align_ddx_matches(arr: list, candidate_list: list) -> list[int]:
    """Project the adjudicator array onto candidate order -> list of ints.

    The judge returns one object per candidate IN candidate order, so we align by
    position. Anything missing / unparseable becomes 0 (no match).
    """
    matches: list[int] = []
    for i in range(len(candidate_list)):
        m = 0
        if i < len(arr) and isinstance(arr[i], dict):
            try:
                m = int(arr[i].get("matched", 0))
            except (TypeError, ValueError):
                m = 0
        matches.append(m)
    return matches


def _ddx_gold_list(ground_truth: dict) -> list:
    """Extract the ordered gold differential as a python list (numpy-safe).

    Parquet hands the differential back as a numpy array, so `x or default` would
    raise on the ambiguous truth value — handle None / tolist explicitly.
    """
    gold = ground_truth.get("differential")
    if gold is None:
        gold = ground_truth.get("gold_differential")
    if gold is None:
        return []
    if hasattr(gold, "tolist"):          # numpy array from parquet
        gold = gold.tolist()
    return list(gold)


def _bench_ddx_judge_args(ground_truth: dict, answer_to_grade: str):
    """Build (gold_list, candidate_list) tuples-of-(name, rank) for the judge."""
    gold = _ddx_gold_list(ground_truth)
    gold_list = [(str(d), i + 1) for i, d in enumerate(gold)]
    cand_names = _parse_ddx_list(answer_to_grade)
    cand_list = [(str(d), i + 1) for i, d in enumerate(cand_names)]
    return gold_list, cand_list


# ============================================================================
# THE ADJUDICATOR JUDGE CALL
# ============================================================================
async def _call_ddx_judge(gold_list, candidate_list, client, max_retries=10):
    if not candidate_list:
        return {"matches": []}
    if not gold_list:
        return {"matches": [0] * len(candidate_list)}

    prompt = DDX_JUDGE_USER_TEMPLATE.format(
        gold_list=_format_ddx_tuples(gold_list),
        candidate_list=_format_ddx_tuples(candidate_list),
    )

    for attempt in range(max_retries):
        raw_text = ""
        stop_reason = None
        try:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": DDX_JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_completion_tokens=JUDGE_MAX_TOKENS,
                temperature=0.0,
            )
            if not getattr(resp, "choices", None):
                upstream_err = getattr(resp, "error", None)
                resp_id = getattr(resp, "id", None)
                raise ValueError(
                    f"OpenRouter returned no choices (resp.id={resp_id!r}, error={upstream_err!r})"
                )
            stop_reason = getattr(resp.choices[0], "finish_reason", None)
            category, explanation = _refusal_info(resp)
            if category is not None:
                log.warning(f"DDX judge REFUSED by safety filter (category={category}): {explanation}")
                _dump_refusal({
                    "eval_mode": "ddx_rank",
                    "refusal_category": category,
                    "refusal_explanation": explanation,
                    "judge_model": JUDGE_MODEL,
                    "judge_system_prompt": DDX_JUDGE_SYSTEM_PROMPT,
                    "judge_user_prompt": prompt,
                    "gold_list": gold_list,
                    "candidate_list": candidate_list,
                })
                return {"matches": [], "error": "refusal", "refusal_category": category}
            raw_text = _extract_judge_text(resp)
            if not raw_text:
                raise ValueError("Judge returned empty content")
            arr = _parse_json_array(raw_text)
            matches = _align_ddx_matches(arr, candidate_list)
            return {"matches": matches, "raw": arr}
        except Exception as e:
            snippet = raw_text[:400].replace("\n", " ")
            if attempt < max_retries - 1:
                wait = min(5.0 * (2 ** attempt) if "429" in str(e) else 2.0 * (attempt + 1), 30.0)
                log.warning(f"DDX judge attempt {attempt+1}/{max_retries} failed: {type(e).__name__}: {e} "
                            f"| stop_reason={stop_reason} | raw[:400]={snippet!r}")
                await asyncio.sleep(wait)
            else:
                log.error(f"DDX judge FAILED after {max_retries} attempts: {e} "
                          f"| stop_reason={stop_reason} | raw[:400]={snippet!r}")
                return {"matches": [], "error": str(e)}


async def _judge_one(gold_list, candidate_list):
    """Single ddx judge call with its own client (single-sample path)."""
    async with _build_judge_client() as client:
        return await _call_ddx_judge(gold_list, candidate_list, client=client)


async def _call_judge_batch(items):
    """Batched ddx judge calls. `items` = list of (idx, gold_list, candidate_list)."""
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    async with _build_judge_client() as client:
        async def _guarded(gold_list, candidate_list):
            async with sem:
                return await _call_ddx_judge(gold_list, candidate_list, client=client)
        return await asyncio.gather(*[_guarded(g, c) for (_i, g, c) in items])


# ============================================================================
# METRICS
# ============================================================================
def _ddx_f1(matches: list[int], n_gold: int):
    """Set-level precision / recall / F1 from the match vector.

    precision = candidates that hit some gold / candidates produced
    recall    = distinct gold conditions hit / gold conditions
    """
    n_cand = len(matches)
    if n_cand == 0 or n_gold == 0:
        return 0.0, 0.0, 0.0
    matched_cands = sum(1 for m in matches if m)
    distinct_gold = len({m for m in matches if m})
    precision = matched_cands / n_cand
    recall = distinct_gold / n_gold
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def _ddx_ndcg(matches: list[int], n_gold: int) -> float:
    """Rank-aware NDCG with graded relevance by GOLD importance.

    A candidate matching gold rank r earns gain (n_gold - r + 1), so hitting gold #1
    is worth most. Each gold counts once (first candidate position that hits it).
    """
    if n_gold == 0:
        return 0.0

    def gain(r: int) -> int:
        return (n_gold - r + 1) if (1 <= r <= n_gold) else 0

    seen: set[int] = set()
    dcg = 0.0
    for pos, r in enumerate(matches, start=1):
        if r and r not in seen and 1 <= r <= n_gold:
            seen.add(r)
            dcg += gain(r) / math.log2(pos + 1)
    idcg = sum(gain(r) / math.log2(i + 1) for i, r in enumerate(range(1, n_gold + 1), start=1))
    return (dcg / idcg) if idcg > 0 else 0.0


# ============================================================================
# UNIFORM REWARD DICT (keep the FULL canonical key set — see module docstring)
# ============================================================================
_CANONICAL_REWARD_DEFAULTS: dict = {
    "score": 0.0,
    "reward/judge_score": 0.0,
    "reward/raw_score": 0.0,
    "reward/format_penalty": 0.0,
    "reward/length_penalty": 0.0,
    "reward/length_ratio": 1.0,
    "reward/accuracy": 0.0,
    "reward/completeness": 0.0,
    "reward/clarity": 0.0,
    "reward/behavior": 0.0,
    "reward/compliance": 0.0,
    "reward/reasoning_length": 0,
    "reward/answer_length": 0,
    "reward/eval_mode": "",
    # mcq / numeric (unused here; kept for return-shape parity)
    "reward/correct": 0,
    "reward/numeric_abs_diff": 0.0,
    "reward/numeric_rel_diff": 0.0,
    # medec (unused here)
    "reward/medec_line_match": 0.0,
    # ddx_rank
    "reward/ddx_f1": 0.0,
    "reward/ddx_ndcg": 0.0,
    "reward/ddx_precision": 0.0,
    "reward/ddx_recall": 0.0,
    "reward/ddx_top_penalty": 0.0,
    "reward/ddx_top_status": "",
    "reward/ddx_n_gold": 0,
    "reward/ddx_n_candidates": 0,
}


def _normalize_score_dict(d: dict) -> dict:
    """Fill any missing canonical key with its neutral default (in place)."""
    for k, v in _CANONICAL_REWARD_DEFAULTS.items():
        d.setdefault(k, v)
    return d


def _empty_score_dict(eval_mode: str, format_penalty: float,
                      reasoning_len: int, answer_len: int) -> dict:
    """Full canonical key set with neutral defaults + the common fields filled."""
    out = dict(_CANONICAL_REWARD_DEFAULTS)
    out["reward/format_penalty"] = format_penalty
    out["reward/reasoning_length"] = reasoning_len
    out["reward/answer_length"] = answer_len
    out["reward/eval_mode"] = eval_mode
    return out


def _score_ddx_rank(solution_str, ground_truth, extra_info,
                    format_penalty, reasoning_len, answer_len, answer_to_grade,
                    judge_result):
    """reward = w_f1*F1 + w_ndcg*NDCG  +  top-diagnosis position penalty."""
    out = _empty_score_dict("ddx_rank", format_penalty, reasoning_len, answer_len)

    n_gold = len(_ddx_gold_list(ground_truth))

    # Judge refused (safety filter) — couldn't adjudicate; assign neutral reward.
    if judge_result and judge_result.get("error") == "refusal":
        out["reward/raw_score"] = REFUSAL_REWARD
        out["reward/judge_score"] = REFUSAL_REWARD
        out["reward/accuracy"] = REFUSAL_REWARD * 10
        out["score"] = max(0.0, REFUSAL_REWARD + format_penalty)
        return out

    matches = []
    if judge_result is not None and "error" not in judge_result:
        matches = judge_result.get("matches", []) or []
    matches = [m if (isinstance(m, int) and 1 <= m <= n_gold) else 0 for m in matches]

    f1, precision, recall = _ddx_f1(matches, n_gold)
    ndcg = _ddx_ndcg(matches, n_gold)
    core = DDX_F1_WEIGHT * f1 + DDX_NDCG_WEIGHT * ndcg

    # Top-diagnosis position penalty (gold rank 1 == the leading diagnosis).
    top_pos = next((pos for pos, m in enumerate(matches, start=1) if m == 1), None)
    if top_pos == 1:
        top_penalty, top_status = 0.0, "top1"
    elif top_pos is not None and top_pos <= 3:
        top_penalty, top_status = -DDX_TOP3_PENALTY, "top3"
    else:
        top_penalty, top_status = -DDX_TOP_MISS_PENALTY, "miss"

    final_score = max(0.0, core + top_penalty + format_penalty)

    out["reward/raw_score"] = round(core, 4)
    out["reward/judge_score"] = round(core, 4)
    out["reward/ddx_f1"] = round(f1, 4)
    out["reward/ddx_ndcg"] = round(ndcg, 4)
    out["reward/ddx_precision"] = round(precision, 4)
    out["reward/ddx_recall"] = round(recall, 4)
    out["reward/ddx_top_penalty"] = top_penalty
    out["reward/ddx_top_status"] = top_status
    out["reward/ddx_n_gold"] = n_gold
    out["reward/ddx_n_candidates"] = len(matches)
    out["reward/accuracy"] = round(core * 10, 4)   # 0-10 telemetry parity
    out["score"] = final_score
    return out


# ============================================================================
# ROUTING (conditional — kept so the module stays correct if reused later)
# ============================================================================
def _is_ddx_rank(data_source, extra_info) -> bool:
    if isinstance(extra_info, dict):
        if extra_info.get("eval_mode") == "ddx_rank":
            return True
        if extra_info.get("task_type") == "ddx_rank":
            return True
        if extra_info.get("helm_scenario") == "ddx_rank":
            return True
    return data_source == "medical_benchmark_ddx"


def _determine_eval_mode(data_source, extra_info) -> str:
    """ddx_rank when routed there; else echo the requested mode (or 'unknown')."""
    if _is_ddx_rank(data_source, extra_info):
        return "ddx_rank"
    if isinstance(extra_info, dict):
        return extra_info.get("eval_mode") or extra_info.get("task_type") or "unknown"
    return "unknown"


def _score_one(solution_str, ground_truth, extra_info, judge_result, eval_mode):
    """Build per-sample reward dict; ddx_rank scored, anything else neutral."""
    answer_to_grade = _extract_answer(solution_str) or ""
    format_penalty = _compute_format_penalty(solution_str or "")
    think_match = re.search(
        rf"{re.escape(THINKING_START)}(.+?){re.escape(THINKING_END)}",
        solution_str or "", flags=re.DOTALL,
    )
    reasoning_len = len(think_match.group(1).split()) if think_match else 0
    answer_len = len(answer_to_grade.split()) if answer_to_grade else 0

    if eval_mode != "ddx_rank":
        # Out of scope for this reward — neutral score, never crash.
        log.warning(f"ddx_rank reward received non-ddx eval_mode={eval_mode!r}; returning neutral score.")
        return _normalize_score_dict(_empty_score_dict(eval_mode, format_penalty, reasoning_len, answer_len))

    return _normalize_score_dict(_score_ddx_rank(
        solution_str, ground_truth, extra_info,
        format_penalty, reasoning_len, answer_len, answer_to_grade, judge_result,
    ))


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def _run_coro(coro):
    """Run a coroutine whether or not an event loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def compute_score(
    data_source=None, solution_str=None, ground_truth=None, extra_info=None,
    data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None,
    **kwargs,
):
    if solution_strs is not None:
        return _compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos)
    return _compute_score_single(data_source, solution_str, ground_truth, extra_info)


def _compute_score_single(data_source, solution_str, ground_truth, extra_info=None):
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    eval_mode = _determine_eval_mode(data_source, extra_info)
    answer_to_grade = _extract_answer(solution_str) or ""

    judge_result = None
    if eval_mode == "ddx_rank" and answer_to_grade:
        gold_list, cand_list = _bench_ddx_judge_args(ground_truth, answer_to_grade)
        judge_result = _run_coro(_judge_one(gold_list, cand_list))

    return _score_one(solution_str, ground_truth, extra_info, judge_result, eval_mode)


def _compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None):
    n = len(solution_strs)
    if extra_infos is None:
        extra_infos = [{}] * n

    parsed_gts = []
    eval_modes = []
    judge_items = []   # (idx, gold_list, candidate_list)

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

        eval_mode = _determine_eval_mode(data_sources[i] if data_sources else None, ei)
        eval_modes.append(eval_mode)

        answer_to_grade = _extract_answer(solution_strs[i]) or ""
        if eval_mode == "ddx_rank" and answer_to_grade:
            gold_list, cand_list = _bench_ddx_judge_args(gt, answer_to_grade)
            judge_items.append((i, gold_list, cand_list))

    judge_results = [None] * n
    if judge_items:
        results = _run_coro(_call_judge_batch(judge_items))
        for (idx, _g, _c), result in zip(judge_items, results):
            judge_results[idx] = result

    return [
        _score_one(solution_strs[i], parsed_gts[i], extra_infos[i], judge_results[i], eval_modes[i])
        for i in range(n)
    ]
