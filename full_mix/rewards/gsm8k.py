"""GSM8K reward — a whitespace-tolerant replacement for verl's #### extractor.

verl's "strict" method is `re.findall("#### (\\-?[0-9\\.\\,]+)", s)`, which needs
the marker, then EXACTLY ONE SPACE, then the number. Every other shape the model
actually produces scores zero even when the arithmetic is right. Measured on the
stage-a validation dumps, these four all failed:

    13  \\n####      answer first, marker last, two trailing spaces
    34####          answer first, no separator
    ####  \\n11      marker first, two spaces and a newline
    ####\\n\\n5       marker first, blank line

By the step-420 eval that was **48% of all gsm8k rows** — the grader reported
0.403 where the true accuracy was 0.908, and the gap grew all run as the model's
formatting drifted. That made the gsm8k ladder unreadable and, worse, looked for
two reports like a real accuracy collapse at the large budgets.

The fix keeps the marker's intent (the answer is whatever sits next to `####`)
but stops grading whitespace:

    1. last `####` in the tail -> nearest number AFTER it, else immediately BEFORE it
    2. no `####` at all        -> last number in the tail

Validated against 600 real budgeted rows at step 480: agrees with the old strict
scorer on 100% of the rows it marked correct, and rescues the rest. verl's
built-in "flexible" method would also rescue them, but it ignores `####`
entirely and reads the last number anywhere in a 300-char window, which happily
picks up a number from the middle of a sentence; this stays anchored to the
marker whenever one is present.

Only `openai/gsm8k` routes here (see `_GSM8K_SOURCES` in compute_score.py). The
training math slice is `local/dolci-math-7b`, which is `\\boxed{}` and unaffected.
"""

import logging
import re

logger = logging.getLogger(__name__)

# The answer is at the end. Matching the whole response is slow and invites
# false positives from the reasoning; verl clips to 300 chars for the same
# reason. A little wider here because the marker can trail the number.
_TAIL_CHARS = 400

_NUM = r"-?\d[\d,]*\.?\d*"
_AFTER = re.compile(r"####\D{0,8}?(" + _NUM + r")")
_BEFORE = re.compile(r"(" + _NUM + r")\D{0,8}?####")
_ANY = re.compile(_NUM)


def _clean(tok: str) -> str:
    return tok.replace(",", "").replace("$", "").rstrip(".")


def extract_answer(solution_str: str) -> str | None:
    """The model's final answer, tolerant of how it punctuated the marker."""
    if not solution_str:
        return None
    tail = solution_str[-_TAIL_CHARS:]

    if "####" in tail:
        # Anchor on the LAST marker: earlier ones can appear inside reasoning.
        head, _, after = tail.rpartition("####")
        m = _AFTER.search("####" + after)
        if m:
            return _clean(m.group(1))
        m = None
        for m in _BEFORE.finditer(head + "####"):
            pass
        if m:
            return _clean(m.group(1))

    nums = _ANY.findall(tail)
    return _clean(nums[-1]) if nums else None


def _equal(pred: str, truth) -> bool:
    t = str(truth).strip().replace(",", "").replace("$", "").rstrip(".")
    if pred == t:
        return True
    try:
        return abs(float(pred) - float(t)) < 1e-6
    except (TypeError, ValueError):
        return False


def compute_score(solution_str: str, ground_truth) -> float:
    if not solution_str or not solution_str.strip():
        return 0.0
    try:
        pred = extract_answer(solution_str)
        if pred is None:
            return 0.0
        return 1.0 if _equal(pred, ground_truth) else 0.0
    except Exception:
        logger.debug("gsm8k scorer error", exc_info=True)
        return 0.0
