"""Continuous length cost, shared by every qlcm reward dispatcher.

    penalty = -COST_PER_1K * total_words / 1000

Linear in total words (think + answer). No threshold, no cap, no dead zone:
every extra token costs the same everywhere, so there is always a gradient
toward brevity.

Why linear, and why no threshold
--------------------------------
This pipeline trains with ``norm_adv_by_std_in_grpo=False``, so the advantage
is ``r_i - mean(r_group)`` with no division by the group std. An additive term
therefore contributes EXACTLY

    -COST_PER_1K * (L_i - mean_group(L)) / 1000

to the advantage: the constant part cancels against GRPO's own baseline and
only the deviation from the group survives. That gives the property we want for
free, with no grouping logic in the reward function at all —

  * the reference length is the group's own mean, so there is no arbitrary
    threshold to pick and nothing for the policy to pile up just underneath;
  * it is per-prompt adaptive. A hard question where every sample legitimately
    runs long incurs no net pressure, because all samples shift together and
    the mean moves with them. Only being longer THAN YOUR PEERS ON THE SAME
    QUESTION costs anything;
  * it never saturates, so the longest rollouts — the ones most worth
    shortening — feel the pressure most, not least.

The previous design (a penalty that switched on at 2,000 words and saturated at
2,800) left 67.2% of real rollouts with an exactly-zero length gradient: 60.9%
sat below the knee and felt no pressure to shorten at all, and the 6.2% at the
cap felt nothing either. It steered inside an 800-word band and was flat
everywhere else, which is a constraint rather than an incentive.

Why not multiplicative (``score * exp(-L/L0)``)
-----------------------------------------------
That form is quality-gated for free — a zero-scoring answer gains nothing by
being short — which protects against a "give up quickly" attractor on prompts
the model cannot solve. Measured on 2,481 real prompt groups, that attractor
does not exist here: 0 groups score all-zero, 0 groups have a score spread
under 0.01, and only 0.5% have a best sample below 0.20. The medical QA judge
returns graded accuracy/clarity/completeness rather than a binary, so there is
always within-group quality variance for the length term to sit underneath.
Meanwhile, with std-normalisation off, the multiplicative form's ~20% shrink of
the reward scale is a straight ~20% cut to the advantage magnitude — a stealth
learning-rate drop. Linear leaves the quality signal's scale untouched.

Calibration
-----------
Measured over the last quarter of the first stage-1 run (39,678 rollouts,
2,481 prompt groups): within-group score std 0.084, within-group length std
297 words. At 0.07 per 1,000 words the length signal is 0.021, i.e. 0.25x the
task-score spread — enough to break ties between equally-good samples, far too
little to overturn a real quality difference. It cuts 149 words per group for
0.0042 of judge score, against the old design's 152 words for 0.0094: the same
compression at 45% of the cost, and now with pressure everywhere instead of in
one narrow band.

No explicit clamp is needed. vLLM hard-stops generation at
``max_response_length`` (8,192 tokens, roughly 5,900 words), which bounds the
term at about -0.41 on its own.

Scores may go negative, and that is intended — it is how verl's own DAPO
manager applies its overlong penalty (``reward += overlong_reward``, no floor).
Clamping at zero would reintroduce exactly the dead zone this design removes.

Env knobs (read at import):
    QLCM_LEN_PENALTY_ENABLE     1/0, default 1  — kill switch
    QLCM_LEN_COST_PER_1K_WORDS  default 0.07    — reward cost per 1,000 words
"""

from __future__ import annotations

import os
import re

THINKING_START = "<think>"
THINKING_END = "</think>"

_THINK_RE = re.compile(
    rf"{re.escape(THINKING_START)}(.+?){re.escape(THINKING_END)}", re.DOTALL
)


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "")


ENABLED = _env_flag("QLCM_LEN_PENALTY_ENABLE", "1")
COST_PER_1K = float(os.environ.get("QLCM_LEN_COST_PER_1K_WORDS", "0.07"))

# Written into every scored row so the term is visible in the rollout dumps and
# the per-source wandb panels without having to re-derive it.
PENALTY_KEY = "reward/length_cost"
WORDS_KEY = "reward/total_words"

# Bookkeeping flag, 0.0 until this row has actually been charged.
#
# It exists because the two invariants collide. Every row must carry an
# identical key set (verl's DataProto concat requires it), so PENALTY_KEY has
# to appear in the dispatchers' padding defaults — which means "the key is
# present" cannot also mean "this row was already charged". The dispatchers pad
# BEFORE they call apply(), so a presence check skips every row and the whole
# mechanism goes silently inert. Charging is therefore tracked explicitly, and
# survives padding in both orders: `_pad` builds from the defaults and then
# does `out.update(result)`, so a 1.0 set by an inner apply() is preserved.
CHARGED_KEY = "reward/length_charged"

UNION_DEFAULTS: dict = {PENALTY_KEY: 0.0, WORDS_KEY: 0, CHARGED_KEY: 0.0}


def penalty_for(total_words: float) -> float:
    """The length cost. Linear, non-positive, unbounded below by design."""
    if not ENABLED:
        return 0.0
    return -COST_PER_1K * max(0.0, float(total_words)) / 1000.0


def count_words(solution_str: str) -> int:
    """Total words in think + answer, mirroring qa_openrouter's accounting.

    The delimiters themselves are excluded. An unterminated or absent think
    block leaves the whole string as answer text, which is the conservative
    reading — it counts every word the model actually emitted.
    """
    if not solution_str:
        return 0
    m = _THINK_RE.search(solution_str)
    if not m:
        return len(solution_str.split())
    think_words = len(m.group(1).split())
    rest = solution_str[: m.start()] + solution_str[m.end():]
    return think_words + len(rest.split())


def apply(result: dict, solution_str: str) -> dict:
    """Fold the length cost into ``result`` in place and return it.

    Idempotent via ``CHARGED_KEY``, not via key presence — the dispatchers pad
    the key set before calling this, so presence proves nothing. The
    dispatchers nest (medical_mix calls qa_openrouter_bench), so without the
    guard a row could be charged twice.

    ``reward/reasoning_length`` + ``reward/answer_length`` are reused when the
    scorer already computed them, so this agrees exactly with the accounting
    the stage-1 dumps were measured with; otherwise words are counted from the
    raw generation.
    """
    if not isinstance(result, dict) or result.get(CHARGED_KEY):
        return result

    rl = result.get("reward/reasoning_length") or 0
    al = result.get("reward/answer_length") or 0
    total = (rl + al) if (rl or al) else count_words(solution_str or "")

    pen = penalty_for(total)
    result[WORDS_KEY] = int(total)
    result[PENALTY_KEY] = pen
    result[CHARGED_KEY] = 1.0
    if pen and "score" in result:
        # Deliberately NOT floored at zero — see the module docstring.
        result["score"] = float(result["score"]) + pen
    return result
