"""Prompt pairing in the paired leak report, offline and without a GPU.

Runs standalone (`python -m full_mix.lcpo.test_leak_report`) or under pytest.
The report is the stop rule for the think-target runs, so its pairing has to
hold for the sources whose rows all carry the same `prompt_uid` sentinel.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from full_mix.common import budget_prompt as bp  # noqa: E402
from full_mix.lcpo import leak_report as lr  # noqa: E402

TASK = "Given the following clinical note, output all the ICD-10 codes.\n\nNote: ..."


def _qlcm(task, budget=None):
    """A qlcm stage-9 input as the dumps decode it (special tokens skipped)."""
    line = f"\n\n{bp.budget_line(budget)}" if budget else ""
    return f"user\n{task}{line}\nassistant\n"


def _full_mix(task, mode, budget=bp.NO_BUDGET, style=bp.DEFAULT_BUDGET_STYLE):
    text = bp.render_prompt(task, mode, budget, style)
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def _row(inp, budget, n_answer, src_id=3, prompt_uid=0.0):
    return {"input": inp, "think_budget": float(budget), "n_answer": float(n_answer), "n_think": 100.0,
            "is_nothink_mode": 0.0, "src_id": float(src_id), "prompt_uid": prompt_uid}


def test_qlcm_budget_and_free_share_a_key():
    free = lr.prompt_key(_qlcm(TASK))
    for b in (256, 512, 1024, 2048, 4096):
        assert lr.prompt_key(_qlcm(TASK, b)) == free, b


def test_full_mix_budget_and_free_share_a_key_in_both_styles():
    free = lr.prompt_key(_full_mix(TASK, bp.MODE_FREE))
    for style in bp.BUDGET_STYLES:
        assert lr.prompt_key(_full_mix(TASK, bp.MODE_BUDGET, 777, style)) == free, style


def test_task_ending_in_newlines_pairs():
    # Budgeted rows rstrip the task before the sentence; qlcm free rows keep it.
    task = TASK + "\nFollowup Instructions:\n___\n"
    budgeted = f"user\n{task.rstrip()}\n\n{bp.budget_line(256)}\nassistant\n"
    assert lr.prompt_key(budgeted) == lr.prompt_key(_qlcm(task))


def test_different_prompts_get_different_keys():
    assert lr.prompt_key(_qlcm(TASK, 256)) != lr.prompt_key(_qlcm(TASK + " Second note.", 256))


def test_budget_sentence_inside_the_task_is_kept():
    inner = f"Quote: {bp.budget_line(5)}\n\n{bp.budget_line(5)}\n\nNow answer."
    assert lr.prompt_key(_qlcm(inner)) != lr.prompt_key(_qlcm("Quote: \n\nNow answer."))
    assert lr.prompt_key(_qlcm(inner, 256)) == lr.prompt_key(_qlcm(inner))


def test_rows_sharing_prompt_uid_pair_with_their_own_free_answer():
    # Two prompts of one source, both stamped prompt_uid=0 as the coding scorers do.
    a, b = TASK, TASK + " Second note."
    rows = [_row(_qlcm(a), -1, 100), _row(_qlcm(b), -1, 10),
            _row(_qlcm(a, 256), 256, 120), _row(_qlcm(b, 256), 256, 40)]
    per, free, _, _ = lr.pair_rows(rows)
    got = sorted((round(d), leaked) for d, leaked, _, _ in per[("if", 256)])
    # a: 120 vs its own 100 -> +20, no leak; b: 40 vs its own 10 -> +30, leak (> 1.5x).
    assert got == [(20, False), (30, True)], got
    assert len(free) == 2


def test_several_free_rows_use_their_median():
    rows = [_row(_qlcm(TASK), -1, n) for n in (10, 20, 90)] + [_row(_qlcm(TASK, 512), 512, 50)]
    per, _, _, _ = lr.pair_rows(rows)
    assert [round(d) for d, *_ in per[("if", 512)]] == [30]


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
