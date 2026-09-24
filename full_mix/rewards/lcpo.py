"""LCPO reward math (arXiv 2503.04697), adapted for a three-mode model.

Pure functions over plain numbers — no torch, no ray, no verl imports — so the
whole reward surface is unit-testable without a cluster. The reward manager does
the token accounting and calls in here.

The three modes and their rules:

    runaway  reward = -runaway_penalty            (any mode, either stage; checked first)
    nothink  reward = task_score * format_gate
    free     reward = task_score
    budget   Stage A:  task_score - alpha * |budget - n_len|
             Stage B:  task_score * clip(alpha_eff * (budget - n_len) + delta, 0, 1)

A response is a RUNAWAY when it did not terminate: it was cut by
max_response_length, or it never emitted </think>. Either way no answer was
produced. It scores a flat -runaway_penalty (default 1.0) before any mode rule
runs, so a finished wrong answer (0) always beats a loop (-1), while nothing in
the rule rewards being shorter than any other finished answer. A response that
closed the tag but has an empty answer is malformed, not a runaway, and keeps
the score-0 treatment. Why a flat rule and not a length term: 5-15% of the
stage-b step-500 checkpoint's generations, free mode included, were verbatim
repetition loops running into the 16,384 cap, and under task*mult a capped
sample scored 0 — identical to a finished wrong answer, so the group-relative
advantage never learned that looping is worse than being wrong.

`alpha_eff` in stage B is alpha with a budget-proportional floor, see
`effective_alpha`: the multiplier reaches zero no later than
`max_slack_frac * budget` tokens past the budget, instead of a fixed
delta/alpha tokens at every budget.

`n_len` is the CONTROLLED LENGTH, selected by `length_target`:

    "total"  n_think + n_answer   (default)
    "think"  n_think only         (the original behaviour)

Default "total" because penalizing only the think span leaves the answer body
unconstrained, and the cheapest way to satisfy a think-budget is then to
relocate the reasoning rather than compress it. Observed in the stage-a run of
2026-08-19: budgeted math rollouts went to `<think>\n\n</think>` (n_think
8045 -> 3) while n_answer grew 399 -> 1376 and the task score recovered as it
did. Whatever quantity is penalized is the only one that gets compressed.

Stage A is the paper's LCPO-Exact and teaches the number -> length mapping; it
punishes coming in under the target as hard as going over. Stage B is LCPO-Max
and is what we ship: a bigger budget strictly enlarges the feasible set, so the
best achievable reward is non-decreasing in budget, which is the monotonicity
property we actually want. Stage B is trained FROM a Stage A checkpoint, because
on its own it has a collapse-to-short attractor — at budget 200 an empty think
block still earns clip(0.06 + 0.5) = 0.56 of the task score.

Still deliberately absent: no length REWARD — anything that pays for n_len
approaching the budget from below is satisfiable with filler. The penalty is
one-sided in that sense: it only ever subtracts.

A minimum-think floor applies in budget mode (see `min_think_tokens`). Without
it, an empty think block is a reachable fixed point that the group-relative
advantage cannot escape: once every sample in a GRPO group emits the same
near-zero n_think, the length term has zero within-group variance and cancels
out of the advantage entirely, so it stops being a learning signal at all.
"""

from __future__ import annotations

DEFAULT_ALPHA = 0.0003   # the paper's value; anchor for math, calibrated per source
DEFAULT_DELTA = 0.5      # the paper's value
# Stage B only. The multiplier hits zero no later than max_slack_frac * budget
# tokens over budget. 0 or None disables; then the slack is the fixed
# delta/alpha tokens at every budget (pre-2026-09-21 behaviour).
DEFAULT_MAX_SLACK_FRAC = 1.0
# Flat reward for a response that never terminated (cut by the cap, or no
# </think>). Applies in every mode and both stages. 0 restores the old
# behaviour, where a runaway scored the same 0 as a finished wrong answer.
DEFAULT_RUNAWAY_PENALTY = 1.0

# Which length the budget governs. See the module docstring.
LENGTH_TARGET_TOTAL = "total"
LENGTH_TARGET_THINK = "think"
DEFAULT_LENGTH_TARGET = LENGTH_TARGET_TOTAL

# Minimum reasoning required of a budgeted response, as max(floor, frac*budget).
# The floor keeps small budgets from demanding a share that rounds to nothing;
# the fraction keeps large budgets from being satisfiable by a token or two.
DEFAULT_MIN_THINK_FLOOR = 32
DEFAULT_MIN_THINK_FRAC = 0.10

MODE_BUDGET = "budget"
MODE_FREE = "free"
MODE_NOTHINK = "nothink"


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def nothink_format_ok(think_span_text: str) -> bool:
    """True when the think block holds no reasoning.

    Whitespace-only, NOT empty. The model emits `<think>\\n\\n</think>` because
    that is exactly what the SFT data taught it, and newlines in there are
    correct behaviour, not a violation. Counting tokens would fail every one of
    those rows. A single non-whitespace character fails.
    """
    return think_span_text.strip() == ""


def controlled_length(n_think: int, n_total: int, length_target: str) -> int:
    """The length the budget applies to. See the module docstring."""
    if length_target == LENGTH_TARGET_TOTAL:
        return int(n_total)
    if length_target == LENGTH_TARGET_THINK:
        return int(n_think)
    raise ValueError(
        f"unknown length_target {length_target!r}, expected "
        f"{LENGTH_TARGET_TOTAL!r} or {LENGTH_TARGET_THINK!r}"
    )


DEFAULT_ANSWER_CAP_MULT = 1.5
DEFAULT_ANSWER_CAP_FLOOR = 64


def answer_cap_tokens(ref_len: float | None, mult: float = DEFAULT_ANSWER_CAP_MULT,
                      floor: int = DEFAULT_ANSWER_CAP_FLOOR) -> int | None:
    """Answer-body ceiling for the leak guard: max(floor, mult * ref_len).

    `ref_len` is the source's free-mode answer length (a measured fact, kept in
    a per-source table); `mult` is the slack allowed above it. None means no
    ceiling for this source.
    """
    if ref_len is None:
        return None
    return int(max(floor, round(float(mult) * float(ref_len))))


def answer_cap_multiplier(n_answer: int, cap: int | None) -> float:
    """Leak guard for length_target='think': 1.0 up to `cap` answer tokens,
    then linear to 0.0 at 2*cap. With the think block budgeted and the answer
    free, relocating the reasoning into the answer is the cheapest way to
    comply; this makes an answer body far longer than the source's free-mode
    norm cost reward. Off (1.0) when cap is None.
    """
    if cap is None or cap <= 0:
        return 1.0
    if n_answer <= cap:
        return 1.0
    return max(0.0, 1.0 - (float(n_answer) - cap) / float(cap))


def min_think_tokens(
    budget: int,
    floor: int = DEFAULT_MIN_THINK_FLOOR,
    frac: float = DEFAULT_MIN_THINK_FRAC,
) -> int:
    """Fewest think tokens a budgeted response must spend to count as an attempt."""
    return max(int(floor), int(float(frac) * int(budget)))


def stage_a_multiplier_terms(budget: int, n_len: int, alpha: float) -> float:
    """Length penalty for Stage A: symmetric, grows with distance from target."""
    return alpha * abs(int(budget) - int(n_len))


def effective_alpha(budget: int, alpha: float, delta: float, max_slack_frac: float | None) -> float:
    """alpha with a budget-proportional floor: max(alpha, delta / (max_slack_frac * budget)).

    With the floor, the stage-B multiplier reaches zero at
    n = budget + max_slack_frac * budget whenever that is tighter than the fixed
    delta/alpha slack, and is byte-identical to the unfloored reward otherwise.
    At alpha 3.5e-4, delta 0.85, frac 1.0 the crossover is budget = 2429: below
    it the floor binds, at and above it nothing changes.
    """
    if not max_slack_frac or max_slack_frac <= 0 or int(budget) <= 0:
        return float(alpha)
    return max(float(alpha), float(delta) / (float(max_slack_frac) * int(budget)))


def stage_b_multiplier(
    budget: int,
    n_len: int,
    alpha: float,
    delta: float = DEFAULT_DELTA,
    max_slack_frac: float | None = DEFAULT_MAX_SLACK_FRAC,
) -> float:
    """Stage B multiplier: 1 well under budget, delta at budget, 0 well over.

    The slope is `effective_alpha`. Without the floor the grace zone is a fixed
    token count (delta/alpha, 2429 tokens at the shipped alpha 3.5e-4 and delta
    0.85) at every budget, so at small budgets it is wide relative to the
    budget: measured on the stage-b step-500 checkpoint, a 512-token budget kept
    67% of the reward at double its length and the within-budget rate at 512
    was 53% on MATH-500. With the proportional floor (max_slack_frac 1.0) the
    zero point is 2N for N < 2429 and unchanged above; delta keeps its meaning
    as the multiplier at exactly n == budget.
    """
    a = effective_alpha(budget, alpha, delta, max_slack_frac)
    return clip01(a * (int(budget) - int(n_len)) + delta)


def compute_reward(
    *,
    mode: str,
    stage: str,
    task_score: float,
    wellformed: bool,
    n_think: int,
    budget: int,
    n_total: int | None = None,
    nothink_ok: bool = True,
    alpha: float = DEFAULT_ALPHA,
    delta: float = DEFAULT_DELTA,
    length_target: str = DEFAULT_LENGTH_TARGET,
    min_think_floor: int = DEFAULT_MIN_THINK_FLOOR,
    min_think_frac: float = DEFAULT_MIN_THINK_FRAC,
    max_slack_frac: float | None = DEFAULT_MAX_SLACK_FRAC,
    terminated: bool = True,
    runaway_penalty: float = DEFAULT_RUNAWAY_PENALTY,
    n_answer: int | None = None,
    answer_cap: int | None = None,
) -> dict:
    """Return {"reward", "len_mult", "len_penalty", "task_score_used", "min_think_ok",
    "runaway", "alpha_eff", "answer_cap_mult", "answer_over_cap"}.

    `answer_cap` (leak guard, budget mode only): ceiling on `n_answer` tokens
    above which the reward is scaled by `answer_cap_multiplier`. None = off.
    `answer_cap_mult` is the factor applied (1.0 when off or under the cap),
    `answer_over_cap` is 1.0 when the answer exceeded the cap.

    `terminated` is False for a runaway: the response was cut by the length cap
    or never emitted </think>. That is checked before any mode branch and scores
    a flat -runaway_penalty with the length term left unevaluated. `alpha_eff`
    is the slope actually used; it equals `alpha` outside budget mode / stage B
    so the key is always present.

    `wellformed` means: a </think> was emitted and something non-empty follows.
    A malformed response has its task score forced to 0 — it has no answer, so
    any score it received came from grading the reasoning itself.

    `n_total` is the whole response (think + answer). Required when
    `length_target` is "total"; passing None there is a bug in the caller, not a
    reason to silently fall back to n_think — that fallback is precisely the
    behaviour we are trying to remove.

    In budget mode a response that spends fewer than `min_think_tokens(budget)`
    tokens reasoning has its task score forced to 0, the same treatment as a
    malformed one. It kept the tags but did not do the work.
    """
    score = float(task_score) if wellformed else 0.0
    out = {"reward": 0.0, "len_mult": 1.0, "len_penalty": 0.0,
           "task_score_used": score, "min_think_ok": 1.0,
           "runaway": 0.0, "alpha_eff": float(alpha),
           "answer_cap_mult": 1.0, "answer_over_cap": 0.0}

    # Did not terminate: no answer exists, whatever the mode. Worse than wrong.
    if not terminated:
        out.update(reward=-float(runaway_penalty), len_mult=0.0,
                   task_score_used=0.0, runaway=1.0)
        return out

    if mode == MODE_NOTHINK:
        gate = 1.0 if nothink_ok else 0.0
        out["len_mult"] = gate
        out["reward"] = score * gate
        return out

    if mode == MODE_FREE:
        # No length term in either stage. That is what "no budget" means; any
        # length shaping here would make it a fourth, hidden budget. The only
        # length pressure is the malformed rule above.
        out["reward"] = score
        return out

    if mode != MODE_BUDGET:
        raise ValueError(f"unknown mode {mode!r}")

    if length_target == LENGTH_TARGET_TOTAL and n_total is None:
        raise ValueError("length_target='total' requires n_total")
    n_len = controlled_length(n_think, n_think if n_total is None else n_total, length_target)

    # Kept the tags, skipped the reasoning. Scored as an empty attempt so the
    # degenerate `<think></think>` output cannot be the cheapest way to comply.
    if int(n_think) < min_think_tokens(budget, min_think_floor, min_think_frac):
        out["min_think_ok"] = 0.0
        score = 0.0
        out["task_score_used"] = score

    # Leak guard on the answer body (budget mode only). Scales the task score,
    # so a wrong answer stays 0 and a runaway is already handled above.
    if answer_cap is not None and n_answer is not None:
        cap_mult = answer_cap_multiplier(int(n_answer), int(answer_cap))
        out["answer_cap_mult"] = cap_mult
        out["answer_over_cap"] = 1.0 if int(n_answer) > int(answer_cap) else 0.0
        score = score * cap_mult
        out["task_score_used"] = score

    if stage == "a":
        penalty = stage_a_multiplier_terms(budget, n_len, alpha)
        out["len_penalty"] = penalty
        out["reward"] = score - penalty
        return out

    if stage == "b":
        # Multiplicative: a wrong answer scores 0 regardless of length, so
        # malformed and merely-wrong collapse together here. That is the paper's
        # form and it is acceptable because Stage B starts from a Stage A
        # checkpoint that has already learned to close the tag.
        out["alpha_eff"] = effective_alpha(budget, alpha, delta, max_slack_frac)
        mult = stage_b_multiplier(budget, n_len, alpha, delta, max_slack_frac)
        out["len_mult"] = mult
        out["reward"] = score * mult
        return out

    raise ValueError(f"unknown stage {stage!r}, expected 'a' or 'b'")
