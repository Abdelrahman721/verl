"""Gate G1: the LCPO reward surface, offline and without a GPU.

Runs standalone (`python -m full_mix.lcpo.test_lcpo`) or under pytest. The only
dependency is the pure-python reward module, so this is the check to run after
touching anything in full_mix/rewards/lcpo.py or the prompt format.

Covers the three modes x {well-formed, cut off, answer padded with reasoning},
plus the two properties the whole project rests on:
  * a bigger budget never lowers the best achievable reward (stage b)
  * the unconstrained reward does not depend on reasoning length at all
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from full_mix.common import budget_prompt as bp  # noqa: E402
from full_mix.rewards import lcpo  # noqa: E402

ALPHA = lcpo.DEFAULT_ALPHA


def _r(mode, stage, score, wellformed, n_think, budget, nothink_ok=True,
       n_answer=0, length_target=lcpo.DEFAULT_LENGTH_TARGET, terminated=True,
       max_slack_frac=lcpo.DEFAULT_MAX_SLACK_FRAC,
       runaway_penalty=lcpo.DEFAULT_RUNAWAY_PENALTY, delta=lcpo.DEFAULT_DELTA,
       alpha=ALPHA):
    return lcpo.compute_reward(
        mode=mode, stage=stage, task_score=score, wellformed=wellformed,
        n_think=n_think, n_total=n_think + n_answer, budget=budget,
        nothink_ok=nothink_ok, alpha=alpha, delta=delta, length_target=length_target,
        terminated=terminated, max_slack_frac=max_slack_frac,
        runaway_penalty=runaway_penalty,
    )["reward"]


# The shipped stage-b settings (train_lcpo_grpo.sh), used by the floor tests.
SHIP_ALPHA, SHIP_DELTA, SHIP_FRAC = 3.5e-4, 0.85, 1.0


def test_nothink_gate_accepts_whitespace():
    """Newlines inside the block are what SFT taught; they must not be penalized."""
    for span in ("", "\n", "\n\n", "   \n \t "):
        assert lcpo.nothink_format_ok(span), f"whitespace span {span!r} must pass"


def test_nothink_gate_rejects_content():
    for span in ("Okay", "\n\nOk\n\n", "\nlet me think\n", "."):
        assert not lcpo.nothink_format_ok(span), f"content span {span!r} must fail"


def test_nothink_reward():
    assert _r("nothink", "a", 0.8, True, 3, -1, nothink_ok=True) == 0.8
    assert _r("nothink", "a", 0.8, True, 40, -1, nothink_ok=False) == 0.0
    # malformed zero-mode row: no answer after the block
    assert _r("nothink", "a", 0.8, False, 3, -1, nothink_ok=True) == 0.0


def test_free_reward_is_length_independent():
    for stage in ("a", "b"):
        short = _r("free", stage, 0.7, True, 50, -1)
        long_ = _r("free", stage, 0.7, True, 9000, -1)
        assert short == long_ == 0.7, f"stage {stage}: free mode must ignore length"


def test_free_malformed_scores_zero():
    assert _r("free", "a", 1.0, False, 9000, -1) == 0.0
    assert _r("free", "b", 1.0, False, 9000, -1) == 0.0


def test_stage_a_is_symmetric():
    over = _r("budget", "a", 1.0, True, 1800, 800)
    under = _r("budget", "a", 1.0, True, 800, 1800)
    assert abs(over - under) < 1e-9, "Exact must punish over and under equally"
    assert abs(_r("budget", "a", 1.0, True, 800, 800) - 1.0) < 1e-9


def test_stage_a_malformed_keeps_the_length_penalty():
    """Score zeroed AND the penalty applied, so never closing the tag is worse."""
    r = _r("budget", "a", 1.0, False, 5000, 800)
    assert r < 0, f"expected a negative reward, got {r}"
    assert abs(r - (0.0 - ALPHA * 4200)) < 1e-9


def test_stage_b_multiplier_shape():
    m = lambda nt, b: lcpo.stage_b_multiplier(b, nt, ALPHA)
    assert m(0, 4000) == 1.0            # well under budget -> free
    assert abs(m(800, 800) - 0.5) < 1e-9  # exactly on budget
    assert m(4000, 800) == 0.0          # far over -> nothing


def test_stage_b_is_monotone_in_budget():
    """The property the whole project is for: more budget is never worse.

    Checked with the proportional floor ON (the default), and in the stronger
    form the floor could have broken: at a FIXED length, the multiplier must be
    non-decreasing in the budget across the whole trained grid.
    """
    grid = list(range(256, 6145, 256))
    for frac in (lcpo.DEFAULT_MAX_SLACK_FRAC, None):
        best = [max(_r("budget", "b", 1.0, True, nt, b, max_slack_frac=frac)
                    for nt in range(0, 6001, 50)) for b in (256, 512, 1024, 2048, 4000, 6000)]
        assert all(x <= y + 1e-9 for x, y in zip(best, best[1:])), f"not monotone (frac={frac}): {best}"
        for n in (100, 300, 600, 1000, 1500, 2500, 4000, 6000, 9000):
            ms = [lcpo.stage_b_multiplier(b, n, SHIP_ALPHA, SHIP_DELTA, frac) for b in grid]
            assert all(x <= y + 1e-9 for x, y in zip(ms, ms[1:])), f"fixed n={n} not monotone (frac={frac}): {ms}"


def test_effective_alpha_floor():
    ea = lambda b, frac=SHIP_FRAC: lcpo.effective_alpha(b, SHIP_ALPHA, SHIP_DELTA, frac)
    assert abs(ea(512) - SHIP_DELTA / 512) < 1e-12
    assert ea(4096) == SHIP_ALPHA
    assert abs(ea(2429) - SHIP_ALPHA) < 1e-6          # crossover: delta/alpha = 2428.6
    assert ea(512, None) == SHIP_ALPHA
    assert ea(512, 0) == SHIP_ALPHA
    assert lcpo.effective_alpha(-1, SHIP_ALPHA, SHIP_DELTA, SHIP_FRAC) == SHIP_ALPHA


def test_stage_b_zero_point_is_proportional_at_small_budgets():
    m = lambda b, n, frac=SHIP_FRAC: lcpo.stage_b_multiplier(b, n, SHIP_ALPHA, SHIP_DELTA, frac)
    for N in (256, 512, 1024):
        assert m(N, 2 * N) == 0.0, f"N={N}: zero point must be 2N"
        assert abs(m(N, N) - SHIP_DELTA) < 1e-9, f"N={N}: delta at n == N"
        assert abs(m(N, int(1.5 * N)) - SHIP_DELTA / 2) < 1e-9, f"N={N}: half of delta at 1.5N"
        assert m(N, 2 * N - 1) > 0.0
    # Large budgets: byte-identical to the fixed slack.
    assert m(4096, 4096 + 2429) == 0.0
    assert m(4096, 4096 + 1000) == m(4096, 4096 + 1000, None)
    assert abs(m(4096, 4096 + 1000) - (SHIP_DELTA - SHIP_ALPHA * 1000)) < 1e-9
    assert m(6144, 6144 - 429) == 1.0


def test_runaway_is_below_wrong_in_every_mode():
    """A loop scores -1.0 even with task score 1.0, and a finished wrong answer
    always beats it. Outside stage-a budget mode the wrong answer scores exactly
    0; in stage-a budget mode it also carries the symmetric length term, so it is
    slightly negative there but still far above -1."""
    for stage in ("a", "b"):
        for mode, budget in (("nothink", -1), ("free", -1), ("budget", 1024)):
            loop = _r(mode, stage, 1.0, True, 800, budget, terminated=False)
            wrong = _r(mode, stage, 0.0, True, 800, budget)
            assert loop == -1.0, f"{mode}/{stage}: runaway must score -1.0, got {loop}"
            assert wrong > loop, f"{mode}/{stage}: a finished wrong answer must beat a loop"
            if not (mode == "budget" and stage == "a"):
                assert wrong == 0.0, f"{mode}/{stage}: a finished wrong answer scores 0, got {wrong}"
            else:
                assert wrong == -ALPHA * abs(1024 - 800)


def test_runaway_ignores_length_term():
    out = lcpo.compute_reward(mode="budget", stage="b", task_score=1.0, wellformed=True,
                              n_think=50, n_total=60, budget=6144, alpha=SHIP_ALPHA,
                              delta=SHIP_DELTA, terminated=False)
    assert out["reward"] == -1.0 and out["len_mult"] == 0.0 and out["runaway"] == 1.0
    assert out["task_score_used"] == 0.0


def test_runaway_penalty_is_configurable():
    assert _r("free", "b", 1.0, True, 9000, -1, terminated=False, runaway_penalty=0.0) == 0.0
    assert _r("budget", "b", 1.0, True, 800, 512, terminated=False, runaway_penalty=0.25) == -0.25


def test_malformed_but_terminated_is_not_a_runaway():
    for stage in ("a", "b"):
        out = lcpo.compute_reward(mode="free", stage=stage, task_score=1.0, wellformed=False,
                                  n_think=300, n_total=300, budget=-1, alpha=ALPHA, terminated=True)
        assert out["reward"] == 0.0 and out["runaway"] == 0.0


def test_reward_dict_always_carries_the_new_keys():
    for mode, stage, budget in (("nothink", "a", -1), ("free", "b", -1), ("budget", "a", 512), ("budget", "b", 512)):
        out = lcpo.compute_reward(mode=mode, stage=stage, task_score=1.0, wellformed=True,
                                  n_think=100, n_total=120, budget=budget, alpha=ALPHA)
        assert "runaway" in out and "alpha_eff" in out
        if not (mode == "budget" and stage == "b"):
            assert out["alpha_eff"] == ALPHA


def test_reasoning_in_the_answer_is_not_rewarded_by_the_length_term():
    """Relocating reasoning into the answer must not pay.

    This inverts the contract this test used to pin. Under length_target="think"
    a 2000-token thinker was beaten by a 10-token thinker that moved the same
    2000 tokens into its answer, and the stage-a run of 2026-08-19 found that
    exploit within ~4 steps: budgeted math went to `<think>\n\n</think>` with
    n_think 8045 -> 3 while n_answer grew 399 -> 1376.
    """
    # Same total work, split differently. Under "total" they are equivalent...
    honest = _r("budget", "b", 1.0, True, 2000, 800, n_answer=100)
    leaked = _r("budget", "b", 1.0, True, 100, 800, n_answer=2000)
    assert leaked <= honest, "relocating reasoning into the answer must not score better"

    # ...and the old target still exhibits the exploit, which is why it is not
    # the default. Keeping it covered documents exactly what "think" costs you.
    honest_think = _r("budget", "b", 1.0, True, 2000, 800, n_answer=100,
                      length_target="think")
    leaked_think = _r("budget", "b", 1.0, True, 100, 800, n_answer=2000,
                      length_target="think")
    assert leaked_think > honest_think, "length_target='think' is the exploitable one"


def test_empty_think_block_is_gated_in_budget_mode():
    """`<think></think>` + a long answer must not out-score a real attempt."""
    r = lcpo.compute_reward(
        mode="budget", stage="a", task_score=1.0, wellformed=True,
        n_think=3, n_total=2000, budget=2000, alpha=ALPHA,
    )
    assert r["min_think_ok"] == 0.0
    assert r["task_score_used"] == 0.0, "a skipped think block scores as no attempt"

    real = lcpo.compute_reward(
        mode="budget", stage="a", task_score=1.0, wellformed=True,
        n_think=1900, n_total=2000, budget=2000, alpha=ALPHA,
    )
    assert real["min_think_ok"] == 1.0
    assert real["reward"] > r["reward"]


def test_min_think_floor_scales_with_budget():
    assert lcpo.min_think_tokens(100) == 32      # floor dominates at small budgets
    assert lcpo.min_think_tokens(6000) == 600    # fraction dominates at large ones


def test_free_and_nothink_are_not_gated_on_think_length():
    """The floor is a budget-mode rule only; /no_think must stay legal."""
    assert _r("nothink", "a", 1.0, True, 0, -1) == 1.0
    assert _r("free", "a", 1.0, True, 0, -1) == 1.0


def test_prompt_round_trip():
    for mode, budget in (("budget", 256), ("budget", 6000), ("free", -1), ("nothink", -1)):
        text = bp.render_prompt("Some task", mode, budget)
        assert bp.parse_suffix(text) == (mode, budget)
        marker = bp.MARKER_NOTHINK if mode == "nothink" else bp.MARKER_THINK
        assert text.endswith(marker)
        assert text.count(marker) == 1


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
