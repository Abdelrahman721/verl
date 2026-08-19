"""IFEval reward with anti-hacking gate.

Why
---
The vanilla IFEval rule scorer is purely structural — it checks whether the
response satisfies the constraints listed in `ground_truth` (length bounds,
required keywords, format, etc.). A model that learns to game this can score
1.0 by emitting *anything* that mechanically satisfies the constraints,
e.g. repeating the constraint keywords, copy-pasting the prompt, or producing
gibberish padded to the right length. That's reward hacking.

What this does
--------------
This wraps the existing rule scorer with a quality gate from an LLM judge:

    final_reward = ifeval_rule_score * quality_multiplier

The judge is asked one question: "is the candidate response a GENUINE
attempt to address the user request, or is it gibberish/padding/off-topic
content that exists only to satisfy structural constraints?" The judge's
verdict maps to a multiplier in {0.0, 0.3, 1.0}.

To save compute, the judge is only called when the rule score is high
enough that hacking is plausible (default threshold 0.5; configurable via
``FULL_MIX_IFEVAL_GATE_THRESHOLD``). When rule_score < threshold, we skip
the judge and just return rule_score — the model isn't getting a high
reward anyway, so there's nothing to hack-gate.

Env knobs
---------
    FULL_MIX_IFEVAL_GATE_ENABLED      "1" (default) / "0"
    FULL_MIX_IFEVAL_GATE_THRESHOLD    rule_score below which we skip judge.
                                      Default 0.5.
    FULL_MIX_IFEVAL_GATE_MULT_GENUINE     default 1.0
    FULL_MIX_IFEVAL_GATE_MULT_BORDERLINE  default 0.3
    FULL_MIX_IFEVAL_GATE_MULT_GIBBERISH   default 0.0
"""

import logging
import os

from full_mix import ifeval_reward as _ifeval_rule
from full_mix.common.judge_client import JudgeUnavailable, call_judge
from full_mix.common.think import strip_think


logger = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = (
    "You are CoherenceJudge. You will see a CANDIDATE_RESPONSE produced by an "
    "AI model, and (optionally) the CONSTRAINT_HINT describing the structural "
    "requirement it was supposed to satisfy.\n\n"
    "Decide whether the candidate is a GENUINE substantive piece of content, "
    "or whether it's gibberish / padding / repetition / random tokens / "
    "copy-pasted prompt content that exists only to satisfy a structural "
    "constraint without saying anything meaningful.\n\n"
    "DO NOT evaluate whether the candidate satisfies the constraint — that's "
    "checked separately. DO NOT penalize the candidate for being short, long, "
    "formatted oddly, or written in a particular style — only intrinsic "
    "quality matters here.\n\n"
    "Categories (pick exactly one):\n"
    "  genuine    — Coherent, well-formed natural-language content. Reads as a "
    "real attempt to communicate something substantive. Code, lists, prose, "
    "etc. all count as genuine if they're coherent.\n"
    "  borderline — Mostly coherent but a meaningful portion is filler, "
    "repetition of the same phrase, restating the prompt back without adding "
    "value, or off-topic rambling.\n"
    "  gibberish  — Random characters, repeated single tokens, copy-pasted "
    "prompt verbatim with no addition, lorem-ipsum-style filler, or otherwise "
    "non-substantive content that exists only to fill space.\n\n"
    "Return exactly one JSON object:\n"
    '{"category": "genuine"|"borderline"|"gibberish", "reason": "<at most 25 words>"}\n\n'
    "Output valid JSON only. No markdown. No extra commentary."
)

JUDGE_USER_TEMPLATE_WITH_HINT = (
    "[CONSTRAINT_HINT]\n{constraint_hint}\n[/CONSTRAINT_HINT]\n\n"
    "[CANDIDATE_RESPONSE]\n{candidate_response}\n[/CANDIDATE_RESPONSE]"
)
JUDGE_USER_TEMPLATE_NO_HINT = (
    "[CANDIDATE_RESPONSE]\n{candidate_response}\n[/CANDIDATE_RESPONSE]"
)


_VALID_CATEGORIES = {"genuine", "borderline", "gibberish"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid float for %s=%r; using default %f", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _multipliers() -> dict[str, float]:
    return {
        "genuine":    _env_float("FULL_MIX_IFEVAL_GATE_MULT_GENUINE",    1.0),
        "borderline": _env_float("FULL_MIX_IFEVAL_GATE_MULT_BORDERLINE", 0.3),
        "gibberish":  _env_float("FULL_MIX_IFEVAL_GATE_MULT_GIBBERISH",  0.0),
    }


def _is_quality_verdict(obj: dict) -> bool:
    cat = obj.get("category")
    return isinstance(cat, str) and cat in _VALID_CATEGORIES


def _ask_judge(constraint_hint: str, candidate: str) -> str:
    """Returns one of {"genuine", "borderline", "gibberish"} on success.

    On any failure path returns "genuine" — i.e. fail OPEN. We don't want a
    flaky judge to silently zero out the reward for legitimate responses;
    the worst-case fallback is "no gate". Failures are logged.
    """
    if constraint_hint:
        user_msg = JUDGE_USER_TEMPLATE_WITH_HINT.format(
            constraint_hint=constraint_hint,
            candidate_response=candidate,
        )
    else:
        user_msg = JUDGE_USER_TEMPLATE_NO_HINT.format(candidate_response=candidate)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    try:
        obj = call_judge(messages, validate=_is_quality_verdict)
    except JudgeUnavailable as e:
        logger.warning("ifeval quality judge unavailable; failing open (%s)", e)
        return "genuine"
    except Exception:
        logger.warning("ifeval quality judge raised; failing open", exc_info=True)
        return "genuine"
    return obj["category"]


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Quality-gated IFEval reward in [0, 1].

    Same signature as ``full_mix.ifeval_reward.compute_score`` so it can be
    swapped in by the IFEval-only dispatcher.
    """
    rule_score = float(
        _ifeval_rule.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    )

    if not _env_bool("FULL_MIX_IFEVAL_GATE_ENABLED", True):
        return rule_score

    threshold = _env_float("FULL_MIX_IFEVAL_GATE_THRESHOLD", 0.5)
    if rule_score < threshold:
        # Reward already low; no point spending a judge call. The rule-based
        # signal is enough to push the model away.
        return rule_score

    if not solution_str or not solution_str.strip():
        return 0.0
    candidate = strip_think(solution_str) if solution_str else solution_str
    if not candidate or not candidate.strip():
        # Model emitted only thinking tokens, nothing visible — no quality
        # to judge; this is hacking by definition.
        return 0.0

    constraint_hint = ""
    if extra_info and isinstance(extra_info, dict):
        # IFEval rows don't carry the full user prompt in extra_info, but they
        # do carry the constraint sentence — pass it as context. If neither is
        # available, the judge will judge intrinsic coherence with no hint.
        constraint_hint = (extra_info.get("user_prompt") or "").strip()
        if not constraint_hint:
            constraint_hint = (extra_info.get("constraint") or "").strip()

    category = _ask_judge(constraint_hint, candidate)
    mult = _multipliers().get(category, 1.0)
    return rule_score * mult
