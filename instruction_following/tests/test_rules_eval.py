"""Unit tests for rule evaluators.

Fast tests mock judges (default). Opt-in live tests call DeepSeek when
``IF_USE_LLM_JUDGES=1`` and ``DEEPSEEK_API_KEY`` are set — see ``TestLiveJudgesIntegration``.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from instruction_following.judges import set_judge_fn, set_v09_extract_fn, verdict_yes
from instruction_following.rules_eval import (
    RULE_EVALUATORS,
    CodesNotInLabelsError,
    V17InputError,
    evaluate_row,
)
from instruction_following.types import EvalResult
from instruction_following.utils import f1_set


class TestRuleEval(unittest.TestCase):
    def tearDown(self) -> None:
        set_judge_fn(None)
        set_v09_extract_fn(None)

    def _log_scores(
        self,
        case: str,
        r: EvalResult,
        *,
        expected_instruction: object,
        expected_accuracy: object,
    ) -> None:
        """Print test id, expected vs got instruction/accuracy (for debugging test runs)."""
        suffix = f" [{case}]" if case else ""
        print(f"\nTest: {self.id()}{suffix}")
        print(f"  expected instruction_score: {expected_instruction!r}")
        print(f"  expected accuracy_score:    {expected_accuracy!r}")
        print(f"  got instruction_score:      {r.instruction_score!r}")
        print(f"  got accuracy_score:         {r.accuracy_score!r}")

    def test_dispatch_unknown_rule(self) -> None:
        with self.assertRaises(KeyError):
            evaluate_row("v99_unknown", "x", [], [])

    def test_v01_match(self) -> None:
        r = evaluate_row(
            "v01_direct_coding",
            "Codes: 96050003 and 20316001",
            ["96050003", "20316001"],
            ["96050003", "20316001"],
        )
        self._log_scores("", r, expected_instruction=None, expected_accuracy=1.0)
        self.assertIsNone(r.instruction_score)
        self.assertEqual(r.accuracy_score, 1.0)

    def test_v04_ids_and_letters(self) -> None:
        r_ok = evaluate_row(
            "v04_id_plus_description",
            "96050003 (Cefpodoxime proxetil (substance))",
            ["96050003"],
            ["96050003"],
        )
        self._log_scores("ids+letters", r_ok, expected_instruction=1.0, expected_accuracy=1.0)
        self.assertEqual(r_ok.instruction_score, 1.0)

        no_letters = evaluate_row("v04_id_plus_description", "96050003", ["96050003"], ["96050003"])
        self._log_scores("no_letters", no_letters, expected_instruction=0.0, expected_accuracy=1.0)
        self.assertEqual(no_letters.instruction_score, 0.0)

        no_ids = evaluate_row(
            "v04_id_plus_description",
            "Cefpodoxime proxetil only",
            ["96050003"],
            ["96050003"],
        )
        self._log_scores("no_ids", no_ids, expected_instruction=0.0, expected_accuracy=0.0)
        self.assertEqual(no_ids.instruction_score, 0.0)

    def test_v02_no_letters(self) -> None:
        r = evaluate_row("v02_id_only", "96050003 20316001", ["96050003", "20316001"], [])
        self._log_scores("no_letters_ok", r, expected_instruction=1.0, expected_accuracy=1.0)
        self.assertEqual(r.instruction_score, 1.0)
        self.assertEqual(r.accuracy_score, 1.0)

        r2 = evaluate_row("v02_id_only", "96050003 code", ["96050003"], ["96050003"])
        self._log_scores("has_letters", r2, expected_instruction=0.0, expected_accuracy=1.0)
        self.assertEqual(r2.instruction_score, 0.0)

    def test_v05_json(self) -> None:
        r = evaluate_row(
            "v05_json_output",
            '{"concepts": [{"sct_id": "96050003", "description": "x"}]}',
            ["96050003"],
            ["96050003"],
        )
        self._log_scores("valid_json", r, expected_instruction=1.0, expected_accuracy=1.0)
        self.assertEqual(r.instruction_score, 1.0)
        self.assertEqual(r.accuracy_score, 1.0)

        bad = evaluate_row("v05_json_output", "not json", ["96050003"], ["96050003"])
        self._log_scores("invalid_json", bad, expected_instruction=0.0, expected_accuracy=0.0)
        self.assertEqual(bad.instruction_score, 0.0)

    def test_v06_order(self) -> None:
        # IDs from B1 labels; ascending numeric order is 96050003 then 443859009
        r_ok = evaluate_row(
            "v06_sorted_multilabel",
            "First then second: 96050003 443859009",
            ["96050003", "443859009"],
            ["96050003", "443859009"],
        )
        self._log_scores("r_ok", r_ok, expected_instruction=1.0, expected_accuracy=1.0)
        self.assertEqual(r_ok.accuracy_score, 1.0)
        self.assertEqual(r_ok.instruction_score, 1.0)

        r_bad = evaluate_row(
            "v06_sorted_multilabel",
            "443859009 before 96050003 wrong order 443859009 96050003",
            ["96050003", "443859009"],
            ["96050003", "443859009"],
        )
        self._log_scores("r_bad", r_bad, expected_instruction=0.0, expected_accuracy=1.0)
        self.assertEqual(r_bad.accuracy_score, 1.0)
        self.assertEqual(r_bad.instruction_score, 0.0)

    def test_v07_count(self) -> None:
        r = evaluate_row(
            "v07_count_prediction",
            " 3 ",
            ["96050003", "20316001", "443859009"],
            ["96050003", "20316001", "443859009"],
        )
        self._log_scores("", r, expected_instruction=1.0, expected_accuracy=1.0)
        self.assertEqual(r.instruction_score, 1.0)
        self.assertEqual(r.accuracy_score, 1.0)

    def test_v08_heuristic_agree(self) -> None:
        import os

        os.environ.pop("IF_USE_LLM_JUDGES", None)
        r = evaluate_row(
            "v08_code_validation",
            "Yes, the candidate is correct for this note.",
            ["96050003"],
            ["96050003"],
            clinical_note="note text",
        )
        self._log_scores("", r, expected_instruction=1.0, expected_accuracy=1.0)
        self.assertEqual(r.instruction_score, 1.0)
        self.assertEqual(r.accuracy_score, 1.0)

    def test_v08_judge_mock(self) -> None:
        set_judge_fn(lambda s, u: "yes")
        r = evaluate_row(
            "v08_code_validation",
            "Anything.",
            ["96050003"],
            ["20316001"],
            clinical_note="note",
        )
        self._log_scores("", r, expected_instruction=1.0, expected_accuracy=1.0)
        self.assertEqual(r.accuracy_score, 1.0)
        self.assertEqual(r.model_response, "yes")

    def test_v09(self) -> None:
        # Simulate DeepSeek output: only final chosen IDs
        set_v09_extract_fn(lambda _: "96050003 20316001")
        r = evaluate_row(
            "v09_multiple_choice",
            "Pick only: 96050003 20316001",
            ["96050003", "20316001", "722052006"],
            ["96050003", "20316001"],
        )
        self._log_scores("", r, expected_instruction=1.0, expected_accuracy=1.0)
        self.assertEqual(r.instruction_score, 1.0)
        self.assertEqual(r.accuracy_score, 1.0)
        self.assertEqual(r.model_response, "96050003 20316001")

    def test_v09_instruction_requires_subset_of_processed(self) -> None:
        # Simulated extractor returns three IDs (e.g. did not strip a distractor)
        set_v09_extract_fn(lambda _: "96050003 20316001 722052006")
        r = evaluate_row(
            "v09_multiple_choice",
            "96050003 20316001 722052006",
            ["96050003", "20316001"],
            ["96050003", "20316001"],
        )
        exp_acc = f1_set({"96050003", "20316001", "722052006"}, {"96050003", "20316001"})
        self._log_scores("", r, expected_instruction=0.0, expected_accuracy=exp_acc)
        self.assertEqual(r.instruction_score, 0.0)
        self.assertAlmostEqual(r.accuracy_score, exp_acc)

    def test_f1_set(self) -> None:
        self.assertEqual(f1_set(set(), set()), 1.0)
        self.assertEqual(f1_set({"a"}, {"a", "b"}), 2.0 / 3.0)

    def test_v10_requires_prompt(self) -> None:
        r = evaluate_row(
            "v10_parent_child_specificity",
            "720617006",
            ["720617006"],
            ["720617006"],
            prompt=None,
        )
        self._log_scores("", r, expected_instruction=None, expected_accuracy=0.0)
        self.assertIsNone(r.instruction_score)
        self.assertEqual(r.accuracy_score, 0.0)

    def test_v10_valid(self) -> None:
        r = evaluate_row(
            "v10_parent_child_specificity",
            "The specific code is 720617006 for this case.",
            ["720617006"],
            ["720617006"],
            prompt="Using parent 118356003 (Metal vapor laser device), code the note: {clinical_note}",
        )
        self._log_scores("", r, expected_instruction=None, expected_accuracy=1.0)
        self.assertIsNone(r.instruction_score)
        self.assertEqual(r.accuracy_score, 1.0)

    def test_v11_judge_mock(self) -> None:
        set_judge_fn(lambda s, u: "yes")
        r = evaluate_row(
            "v11_code_desc_consistency",
            "The pair is consistent.",
            ["96050003"],
            ["96050003"],
        )
        self._log_scores("", r, expected_instruction=None, expected_accuracy=1.0)
        self.assertIsNone(r.instruction_score)
        self.assertEqual(r.accuracy_score, 1.0)

    def test_codes_not_in_labels_raises(self) -> None:
        bad = ["99999999999999999"]
        for rule in ("v01_direct_coding", "v11_code_desc_consistency"):
            with self.subTest(rule=rule):
                with self.assertRaises(CodesNotInLabelsError):
                    evaluate_row(rule, "x", bad, bad)

    def test_v12_judge_mock(self) -> None:
        set_judge_fn(lambda s, u: "no")
        r = evaluate_row(
            "v12_set_correction",
            "The set is wrong.",
            ["96050003", "20316001"],
            ["96050003", "223936005"],
        )
        self._log_scores("", r, expected_instruction=None, expected_accuracy=0.0)
        self.assertIsNone(r.instruction_score)
        self.assertEqual(r.accuracy_score, 0.0)
        self.assertEqual(r.model_response, "no")

    def test_v14_one_code(self) -> None:
        r = evaluate_row(
            "v14_policy_conditioned_one_code_only",
            "20316001",
            ["20316001", "96050003"],
            ["20316001"],
        )
        self._log_scores("", r, expected_instruction=1.0, expected_accuracy=1.0)
        self.assertEqual(r.instruction_score, 1.0)
        self.assertEqual(r.accuracy_score, 1.0)

    def test_v16_single_id(self) -> None:
        r = evaluate_row(
            "v16_code_from_description", "Code 720617006", ["720617006"], ["720617006"]
        )
        self._log_scores("", r, expected_instruction=None, expected_accuracy=1.0)
        self.assertIsNone(r.instruction_score)
        self.assertEqual(r.accuracy_score, 1.0)

    def test_v13_instruction_none(self) -> None:
        r = evaluate_row("v13_surface_perturbation", "96050003", ["96050003"], ["96050003"])
        self._log_scores("", r, expected_instruction=None, expected_accuracy=1.0)
        self.assertIsNone(r.instruction_score)

    def test_v15_instruction_none(self) -> None:
        r = evaluate_row(
            "v15_description_from_code",
            "Cefpodoxime proxetil (substance)",
            ["96050003"],
            ["96050003"],
        )
        self._log_scores("", r, expected_instruction=None, expected_accuracy=1.0)
        self.assertIsNone(r.instruction_score)
        self.assertEqual(r.accuracy_score, 1.0)

    def test_v17_negation(self) -> None:
        r = evaluate_row(
            "v17_exclude_negation",
            "96050003",
            ["96050003", "440565004"],
            ["96050003"],
        )
        self._log_scores("", r, expected_instruction=None, expected_accuracy=1.0)
        self.assertIsNone(r.instruction_score)
        self.assertEqual(r.accuracy_score, 1.0)

    def test_v17_raises_without_negation_in_original(self) -> None:
        with self.assertRaises(V17InputError):
            evaluate_row(
                "v17_exclude_negation",
                "96050003 20316001",
                ["96050003", "20316001"],
                ["96050003", "20316001"],
            )

    def test_v17_raises_when_processed_has_negation(self) -> None:
        with self.assertRaises(V17InputError):
            evaluate_row(
                "v17_exclude_negation",
                "96050003",
                ["96050003", "440565004"],
                ["96050003", "440565004"],
            )

    def test_all_rules_registered(self) -> None:
        expected = {
            "v01_direct_coding",
            "v13_surface_perturbation",
            "v02_id_only",
            "v03_descriptions_only",
            "v04_id_plus_description",
            "v05_json_output",
            "v06_sorted_multilabel",
            "v07_count_prediction",
            "v08_code_validation",
            "v09_multiple_choice",
            "v10_parent_child_specificity",
            "v11_code_desc_consistency",
            "v12_set_correction",
            "v14_policy_conditioned_one_code_only",
            "v15_description_from_code",
            "v16_code_from_description",
            "v17_exclude_negation",
        }
        self.assertEqual(set(RULE_EVALUATORS.keys()), expected)

    def test_answer_tags_preference(self) -> None:
        r = evaluate_row(
            "v01_direct_coding",
            "<redacted_thinking>x</redacted_thinking><answer>96050003</answer>",
            ["96050003"],
            ["96050003"],
        )
        self._log_scores("", r, expected_instruction=None, expected_accuracy=1.0)
        self.assertIsNone(r.instruction_score)
        self.assertEqual(r.accuracy_score, 1.0)


def _live_judges_configured() -> bool:
    """True when live DeepSeek calls can run (key + client). Tests set ``IF_USE_LLM_JUDGES`` via ``patch``."""
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return False
    try:
        from openai import OpenAI  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


_LIVE_SKIP = "Set DEEPSEEK_API_KEY and install openai — optional live DeepSeek judge integration tests"


@unittest.skipUnless(_live_judges_configured(), _LIVE_SKIP)
class TestLiveJudgesIntegration(unittest.TestCase):
    """Real ``call_judge`` / ``call_v09_pred_extraction`` via DeepSeek (no mocks).

    Responses are chosen so a strict yes/no grader should agree the assistant did the right thing.
    Judge output must be exactly ``yes`` or ``no`` (see ``verdict_yes``); occasional API variance may
    require re-running or adjusting prompts.
    """

    def tearDown(self) -> None:
        set_judge_fn(None)
        set_v09_extract_fn(None)

    @patch.dict(os.environ, {"IF_USE_LLM_JUDGES": "1"}, clear=False)
    def test_live_v08_judge_when_codes_match(self) -> None:
        # True and proposed are both 96050003; assistant explicitly agrees they are the same.
        r = evaluate_row(
            "v08_code_validation",
            "The proposed SNOMED CT code 96050003 is identical to the true code 96050003; "
            "the candidate correctly matches the ground truth.",
            ["96050003"],
            ["96050003"],
            clinical_note="Synthetic note for validation.",
        )
        self.assertIsNotNone(r.model_response)
        self.assertTrue(verdict_yes(r.model_response), msg=f"judge raw: {r.model_response!r}")
        self.assertEqual(r.accuracy_score, 1.0)

    @patch.dict(os.environ, {"IF_USE_LLM_JUDGES": "1"}, clear=False)
    def test_live_v08_judge_when_codes_differ(self) -> None:
        # True 96050003 vs proposed 20316001; assistant must reflect that they do not match.
        r = evaluate_row(
            "v08_code_validation",
            "The true code is 96050003 (Cefpodoxime proxetil) and the proposed code is 20316001; "
            "they are different concepts, so the candidate does not match the true code.",
            ["96050003"],
            ["20316001"],
            clinical_note="Synthetic note for validation.",
        )
        self.assertIsNotNone(r.model_response)
        self.assertTrue(verdict_yes(r.model_response), msg=f"judge raw: {r.model_response!r}")
        self.assertEqual(r.accuracy_score, 1.0)

    @patch.dict(os.environ, {"IF_USE_LLM_JUDGES": "1"}, clear=False)
    def test_live_v11_same_id_description_matches_ground_truth(self) -> None:
        # original == processed → shown pair equals ground truth; assistant confirms consistency.
        r = evaluate_row(
            "v11_code_desc_consistency",
            "The ID and description shown match the official SNOMED CT label for this concept; "
            "the pair is consistent with SNOMED CT.",
            ["96050003"],
            ["96050003"],
        )
        self.assertIsNotNone(r.model_response)
        self.assertTrue(verdict_yes(r.model_response), msg=f"judge raw: {r.model_response!r}")
        self.assertEqual(r.accuracy_score, 1.0)

    @patch.dict(os.environ, {"IF_USE_LLM_JUDGES": "1"}, clear=False)
    def test_live_v12_corrects_proposed_set_to_ground_truth(self) -> None:
        # Ground truth: 96050003 + 20316001; proposed swaps second code for 223936005; assistant fixes it.
        r = evaluate_row(
            "v12_set_correction",
            "Ground truth for this note is two concepts: 96050003 (Cefpodoxime proxetil) and "
            "20316001 (Amoxicillin). The proposed set included 223936005 instead of 20316001; "
            "that was wrong. The corrected set matching the note is 96050003 and 20316001.",
            ["96050003", "20316001"],
            ["96050003", "223936005"],
        )
        self.assertIsNotNone(r.model_response)
        self.assertTrue(verdict_yes(r.model_response), msg=f"judge raw: {r.model_response!r}")
        self.assertEqual(r.accuracy_score, 1.0)

    @patch.dict(os.environ, {"IF_USE_LLM_JUDGES": "1"}, clear=False)
    def test_live_v09_extractor_isolates_final_chosen_ids(self) -> None:
        # Options include a distractor; answer states final chosen IDs only (no extraction mock).
        r = evaluate_row(
            "v09_multiple_choice",
            "I considered all options. For my final answer I select only these SNOMED CT concepts: "
            "96050003 and 20316001. I am not choosing 722052006.",
            ["96050003", "20316001", "722052006"],
            ["96050003", "20316001"],
        )
        self.assertIsNotNone(r.model_response)
        self.assertEqual(r.instruction_score, 1.0)
        self.assertEqual(r.accuracy_score, 1.0)


if __name__ == "__main__":
    unittest.main()
