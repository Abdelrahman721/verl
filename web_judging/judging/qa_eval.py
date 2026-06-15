"""Shared qa-judge orchestrator.

Runs ONE qa-judge backend (either the verbatim ``qa_bedrock`` Sonnet module or
the ``qa_openrouter`` gpt-5.4-mini module — they expose the same
``_build_judge_client`` / ``_call_qa_judge`` / ``_call_conv_judge`` surface) and
returns both the raw judge verdict (so the UI can show justifications) and the
numeric score breakdown computed by qa_bedrock's own ``_score_single``.

We deliberately call the judge helpers directly rather than the top-level
``compute_score``, because ``compute_score`` discards the parsed verdict — and
the whole point of the app is to *show the user the judges' reasoning*.
"""

from .qa_bedrock import _extract_answer, _build_conv_context, _score_single


async def evaluate_qa(backend, *, question, key_points, candidate,
                      ground_truth, extra_info, eval_mode):
    """Grade ``candidate`` with the given backend module.

    backend: module exposing _build_judge_client / _call_qa_judge /
             _call_conv_judge (qa_bedrock or qa_openrouter).
    Returns {"verdict": <parsed judge JSON>, "breakdown": <reward/* dict>}.
    """
    extracted = _extract_answer(candidate)
    answer_to_grade = extracted if extracted else ""

    verdict = None
    if answer_to_grade:
        async with backend._build_judge_client() as client:
            if eval_mode == "qa":
                verdict = await backend._call_qa_judge(
                    question, key_points, answer_to_grade, candidate, client
                )
            else:
                prompt = extra_info.get("prompt", []) if isinstance(extra_info, dict) else []
                gt_with_context = _build_conv_context(prompt, ground_truth)
                verdict = await backend._call_conv_judge(
                    gt_with_context, answer_to_grade, candidate, client
                )

    breakdown = _score_single(candidate, ground_truth, extra_info, verdict, eval_mode)
    return {"verdict": verdict, "breakdown": breakdown}
