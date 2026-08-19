"""LCPO reward math (arXiv 2503.04697), adapted for a three-mode model.

Pure functions over plain numbers — no torch, no ray, no verl imports — so the
whole reward surface is unit-testable without a cluster. The reward manager does
the token accounting and calls in here.

The three modes and their rules:

    nothink  reward = task_score * format_gate
    free     reward = task_score
    budget   Stage A:  task_score - alpha * |budget - n_think|
             Stage B:  task_score * clip(alpha * (budget - n_think) + delta, 0, 1)

Stage A is the paper's LCPO-Exact and teaches the number -> length mapping; it
punishes coming in under the target as hard as going over. Stage B is LCPO-Max
and is what we ship: a bigger budget strictly enlarges the feasible set, so the
best achievable reward is non-decreasing in budget, which is the monotonicity
property we actually want. Stage B is trained FROM a Stage A checkpoint, because
on its own it has a collapse-to-short attractor — at budget 200 an empty think
block still earns clip(0.06 + 0.5) = 0.56 of the task score.

Deliberately absent, per review: no answer-length penalty (we measure answer
length instead and can add one if it moves), and no length reward — anything
that pays for n_think approaching the budget is satisfiable with filler.
"""

DEFAULT_ALPHA = 0.0003   # the paper's value; anchor for math, calibrated per source
DEFAULT_DELTA = 0.5      # the paper's value

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


def stage_a_multiplier_terms(budget: int, n_think: int, alpha: float) -> float:
    """Length penalty for Stage A: symmetric, grows with distance from target."""
    return alpha * abs(int(budget) - int(n_think))


def stage_b_multiplier(budget: int, n_think: int, alpha: float, delta: float = DEFAULT_DELTA) -> float:
    """Stage B multiplier: 1 well under budget, delta at budget, 0 well over.

    Note the grace zone is a fixed token count (delta/alpha ~= 1667 at the
    paper's alpha), so at small budgets it is wide relative to the budget: at
    budget 200 a response ~1700 tokens over still keeps ~44% of the task score.
    Watch percentage length error at the 256 and 512 rungs; if they do not
    converge, switch the term to (budget - n_think) / budget, which gives the
    same grace zone in percentage terms at every scale.
    """
    return clip01(alpha * (int(budget) - int(n_think)) + delta)


def compute_reward(
    *,
    mode: str,
    stage: str,
    task_score: float,
    wellformed: bool,
    n_think: int,
    budget: int,
    nothink_ok: bool = True,
    alpha: float = DEFAULT_ALPHA,
    delta: float = DEFAULT_DELTA,
) -> dict:
    """Return {"reward", "len_mult", "len_penalty", "task_score_used"}.

    `wellformed` means: a </think> was emitted and something non-empty follows.
    A malformed response has its task score forced to 0 — it has no answer, so
    any score it received came from grading the reasoning itself.
    """
    score = float(task_score) if wellformed else 0.0
    out = {"reward": 0.0, "len_mult": 1.0, "len_penalty": 0.0, "task_score_used": score}

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

    if stage == "a":
        penalty = stage_a_multiplier_terms(budget, n_think, alpha)
        out["len_penalty"] = penalty
        out["reward"] = score - penalty
        return out

    if stage == "b":
        # Multiplicative: a wrong answer scores 0 regardless of length, so
        # malformed and merely-wrong collapse together here. That is the paper's
        # form and it is acceptable because Stage B starts from a Stage A
        # checkpoint that has already learned to close the tag.
        mult = stage_b_multiplier(budget, n_think, alpha, delta)
        out["len_mult"] = mult
        out["reward"] = score * mult
        return out

    raise ValueError(f"unknown stage {stage!r}, expected 'a' or 'b'")
