"""Offline checks for the qlcm stage-9 LCPO pieces. No GPU, no judges.

Run from the repo root inside the training container (the reward wrapper
imports the qlcm scorers, which need the container's packages):

    QLCM_LEN_PENALTY_ENABLE=0 python -m qlcm.tests.test_lcpo_stage9
"""

from __future__ import annotations

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qlcm.common import budget_prompt as bp  # noqa: E402
from qlcm.curriculum import build_lcpo_stage9 as build  # noqa: E402


def test_render_and_parse_round_trip():
    for mode, budget in ((bp.MODE_BUDGET, 256), (bp.MODE_BUDGET, 4096), (bp.MODE_FREE, -1)):
        text = bp.render_prompt_plain("Some task\n", mode, budget)
        assert bp.parse_suffix_plain(text) == (mode, budget if mode == bp.MODE_BUDGET else bp.NO_BUDGET)
        assert "/think" not in text
    assert bp.render_prompt_plain("Some task", bp.MODE_FREE) == "Some task"
    assert bp.render_prompt_plain("Some task\n\n", bp.MODE_BUDGET, 512) == "Some task\n\nThink for a maximum of 512 tokens."


def test_no_think_mode_is_rejected():
    try:
        bp.render_suffix_plain("nothink", -1)
    except ValueError:
        return
    raise AssertionError("nothink must not render for the qlcm model")


def test_only_last_user_turn_is_touched():
    conv = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"}]
    out = bp.apply_to_messages_plain(conv, bp.MODE_BUDGET, 1024)
    assert out[0] == conv[0] and out[1] == conv[1]
    assert out[2]["content"] == "second\n\nThink for a maximum of 1024 tokens."
    assert conv[2]["content"] == "second", "input must not be mutated"
    assert bp.apply_to_messages_plain(conv, bp.MODE_FREE) == conv


def test_slot_pattern_is_exactly_70_30():
    pids = [f"{i:08x}" + "0" * 24 for i in range(10)]      # slots 0..9
    slots = [build.slot_pattern(p) for p in pids]
    n_budget = sum(m == bp.MODE_BUDGET for pat in slots for m in pat)
    assert n_budget == 21 and len(slots) * 3 == 30
    assert all(sum(m == bp.MODE_BUDGET for m in pat) >= 2 for pat in slots)


def test_budgets_never_repeat_within_a_prompt_and_are_reproducible():
    for i in range(200):
        pid = f"{i:032x}"
        b = build.draw_budgets(42, "mlb", pid, 3)
        assert len(set(b)) == 3 and all(x in build.GRID for x in b)
        assert b == build.draw_budgets(42, "mlb", pid, 3)
    assert build.draw_budgets(42, "mlb", "a" * 32, 2) != build.draw_budgets(43, "mlb", "a" * 32, 2)


def test_grid():
    assert build.GRID[0] == 256 and build.GRID[-1] == 4096 and len(build.GRID) == 16
    assert all(b % 256 == 0 for b in build.GRID)


def test_wrapper_strips_tags_and_stamps_src_id():
    os.environ["QLCM_LEN_PENALTY_ENABLE"] = "0"
    from qlcm.rewards import compute_score_lcpo as w

    seen = []

    def fake(*args, **kwargs):
        ds = kwargs.get("data_source", args[0] if args else None)
        seen.append(ds)
        return {"score": 1.0, "reward/eval_mode": "x"}

    orig = w._mix.compute_score
    w._mix.compute_score = fake
    try:
        for tagged, base, sid in (("mlb@b00256", "mlb", 1), ("medical_qa@free", "medical_qa", 9),
                                  ("local/dolci-chat-32b@b04096", "local/dolci-chat-32b", 12), ("mlb", "mlb", 1)):
            out = w.compute_score(data_source=tagged, solution_str="<think>x</think>y", ground_truth="g", extra_info={})
            assert seen[-1] == base, (tagged, seen[-1])
            assert out["src_id"] == float(sid)
        out = w.compute_score(tagged, "<think>x</think>y", "g", {})           # positional form
        assert seen[-1] == "mlb"
        # batch form
        w._mix.compute_score = lambda *a, **k: [{"score": 0.0} for _ in (k.get("data_sources") or a[0])]
        outs = w.compute_score(data_sources=["mlb@b00512", "if@free"], solution_strs=["a", "b"], ground_truths=["g", "g"], extra_infos=[{}, {}])
        assert [o["src_id"] for o in outs] == [1.0, 3.0]
    finally:
        w._mix.compute_score = orig


def test_wrapper_refuses_length_cost():
    """Importing the wrapper with the qlcm length cost enabled must fail loudly."""
    import importlib
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); "
            "import qlcm.rewards.compute_score_lcpo" % _REPO_ROOT)
    env = dict(os.environ, QLCM_LEN_PENALTY_ENABLE="1")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "QLCM_LEN_PENALTY_ENABLE" in r.stderr, r.stderr[-400:]
    importlib.invalidate_caches()


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
