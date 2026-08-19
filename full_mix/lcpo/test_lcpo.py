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


def _r(mode, stage, score, wellformed, n_think, budget, nothink_ok=True):
    return lcpo.compute_reward(
        mode=mode, stage=stage, task_score=score, wellformed=wellformed,
        n_think=n_think, budget=budget, nothink_ok=nothink_ok, alpha=ALPHA,
    )["reward"]


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
    """The property the whole project is for: more budget is never worse."""
    best = []
    for budget in (256, 512, 1024, 2048, 4000, 6000):
        best.append(max(_r("budget", "b", 1.0, True, nt, budget)
                        for nt in range(0, 6001, 50)))
    assert all(x <= y + 1e-9 for x, y in zip(best, best[1:])), f"not monotone: {best}"


def test_reasoning_in_the_answer_is_not_rewarded_by_the_length_term():
    """Leaking is only detectable via n_answer; the reward must not encourage it.

    We have no answer penalty right now (deliberately), so this pins the current
    contract: moving reasoning into the answer DOES improve the stage-b
    multiplier. That is exactly why n_answer is logged per budget bucket, and
    this test is the reminder to add a penalty if that metric starts moving.
    """
    honest = _r("budget", "b", 1.0, True, 2000, 800)
    leaked = _r("budget", "b", 1.0, True, 10, 800)
    assert leaked > honest, "documented gap: leaking currently scores better"


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
