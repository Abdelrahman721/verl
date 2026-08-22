"""LCPO reward math (arXiv 2503.04697), adapted for a three-mode model.

Pure functions over plain numbers — no torch, no ray, no verl imports — so the
whole reward surface is unit-testable without a cluster. The reward manager does
the token accounting and calls in here.

The three modes and their rules:

    nothink  reward = task_score * format_gate
    free     reward = task_score
    budget   Stage A:  task_score - alpha * |budget - n_len|
             Stage B:  task_score * clip(alpha * (budget - n_len) + delta, 0, 1)

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


def stage_b_multiplier(budget: int, n_len: int, alpha: float, delta: float = DEFAULT_DELTA) -> float:
    """Stage B multiplier: 1 well under budget, delta at budget, 0 well over.

    Note the grace zone is a fixed token count (delta/alpha ~= 1667 at the
    paper's alpha), so at small budgets it is wide relative to the budget: at
    budget 200 a response ~1700 tokens over still keeps ~44% of the task score.
    Watch percentage length error at the 256 and 512 rungs; if they do not
    converge, switch the term to (budget - n_think) / budget, which gives the
    same grace zone in percentage terms at every scale.
    """
    return clip01(alpha * (int(budget) - int(n_len)) + delta)


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
) -> dict:
    """Return {"reward", "len_mult", "len_penalty", "task_score_used", "min_think_ok"}.

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
           "task_score_used": score, "min_think_ok": 1.0}

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
        mult = stage_b_multiplier(budget, n_len, alpha, delta)
        out["len_mult"] = mult
        out["reward"] = score * mult
        return out

    raise ValueError(f"unknown stage {stage!r}, expected 'a' or 'b'")
