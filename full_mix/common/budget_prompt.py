"""Single source of truth for how a reasoning-budget prompt is written.

Three modes, distinguished only by what is appended to the last user turn:

    budget  {task}\\n\\nThink for {n} tokens.\\n\\n/think
    free    {task}\\n\\n/think
    nothink {task}\\n\\n/no_think

`free` and `budget` end with the identical `/think` marker; the only difference
is the budget line above it. That keeps unconstrained rollouts byte-identical to
what the dual-mode SFT stage trained and what the Aug 9 benchmark run measured,
so the budgeted mode is added without disturbing the mode we already have.

The marker stays LAST, in the slot `sft_modes/export_for_sfttrainer.py` put it.
About 20% of the ifeval slice carries copy/repeat-family constraints, ~400 rows
of which ask the model to repeat the request verbatim against a stored copy of
the original text. The SFT data already taught the model to leave the trailing
marker out when repeating, so extending that same slot is the low-risk position
for the budget line — but the pass rate on those rows is worth watching.

Both the training-data builder and the eval chat template import from here so
the two framings cannot drift apart.
"""

MARKER_THINK = "/think"
MARKER_NOTHINK = "/no_think"
SEP = "\n\n"

MODE_BUDGET = "budget"
MODE_FREE = "free"
MODE_NOTHINK = "nothink"
MODES = (MODE_BUDGET, MODE_FREE, MODE_NOTHINK)

# think_budget value stored for rows that carry no budget. Not None: the column
# has to stay a plain int64 so it survives verl's non_tensor_batch round-trip
# and so the reward manager can emit a fixed, all-numeric key set.
NO_BUDGET = -1


# How the budget sentence is phrased. The two stages enforce different rules and
# the prompt has to say which one is in force, or the model cannot tell them
# apart and stage b simply overwrites stage a's behaviour on identical text.
#
#   "exact"  stage a / LCPO-Exact: reward = task - alpha*|budget - n|
#            Coming in under is punished as hard as going over, so the
#            instruction is a target to hit.
#   "max"    stage b / LCPO-Max: reward = task * clip(alpha*(budget - n) + delta)
#            Only overshoot is punished, so the instruction is a ceiling.
STYLE_EXACT = "exact"
STYLE_MAX = "max"
BUDGET_STYLES = (STYLE_EXACT, STYLE_MAX)

# Default is "max": stage b is what ships. Stage-a data already exists and is
# rebuilt with style="exact" if it is ever needed again.
DEFAULT_BUDGET_STYLE = STYLE_MAX

_PREFIX = {STYLE_EXACT: "Think for ", STYLE_MAX: "Think for a maximum of "}
_SUFFIX = " tokens."


def budget_line(n_tokens: int, style: str = DEFAULT_BUDGET_STYLE) -> str:
    """The one sentence that states the budget. Keep this exact wording."""
    if style not in BUDGET_STYLES:
        raise ValueError(f"unknown budget style {style!r}, expected one of {BUDGET_STYLES}")
    return f"{_PREFIX[style]}{int(n_tokens)}{_SUFFIX}"


def render_suffix(mode: str, budget: int = NO_BUDGET, style: str = DEFAULT_BUDGET_STYLE) -> str:
    """The text appended after the task, starting with the separator."""
    if mode == MODE_NOTHINK:
        return SEP + MARKER_NOTHINK
    if mode == MODE_FREE:
        return SEP + MARKER_THINK
    if mode == MODE_BUDGET:
        if budget is None or int(budget) <= 0:
            raise ValueError(f"budget mode needs a positive budget, got {budget!r}")
        return SEP + budget_line(budget, style) + SEP + MARKER_THINK
    raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")


def render_prompt(task_text: str, mode: str, budget: int = NO_BUDGET,
                  style: str = DEFAULT_BUDGET_STYLE) -> str:
    """Full user-turn text for one row."""
    return task_text.rstrip() + render_suffix(mode, budget, style)


def apply_to_messages(messages: list[dict], mode: str, budget: int = NO_BUDGET,
                      style: str = DEFAULT_BUDGET_STYLE) -> list[dict]:
    """Return a copy of `messages` with the suffix on the LAST non-assistant turn."""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") != "assistant":
            m["content"] = render_prompt(m["content"], mode, budget, style)
            return out
    raise ValueError("no non-assistant turn to attach a mode marker to")


def parse_suffix(user_text: str) -> tuple[str, int]:
    """Recover (mode, budget) from a rendered prompt. For tests and analysis."""
    stripped = user_text.rstrip()
    if stripped.endswith(MARKER_NOTHINK):
        return MODE_NOTHINK, NO_BUDGET
    if stripped.endswith(MARKER_THINK):
        body = stripped[: -len(MARKER_THINK)].rstrip()
        last = body.rsplit("\n", 1)[-1].strip()
        # Longest prefix first: "Think for a maximum of N tokens." also starts
        # with "Think for ", so checking the exact style first would try to
        # int() the words "a maximum of N".
        for style in (STYLE_MAX, STYLE_EXACT):
            pre = _PREFIX[style]
            if last.startswith(pre) and last.endswith(_SUFFIX):
                inner = last[len(pre):-len(_SUFFIX)]
                if inner.isdigit():
                    return MODE_BUDGET, int(inner)
        return MODE_FREE, NO_BUDGET
    raise ValueError("prompt does not end with a mode marker")


def parse_budget_style(user_text: str) -> str | None:
    """Which wording a rendered budget prompt uses, or None if it has no budget."""
    stripped = user_text.rstrip()
    if not stripped.endswith(MARKER_THINK):
        return None
    body = stripped[: -len(MARKER_THINK)].rstrip()
    last = body.rsplit("\n", 1)[-1].strip()
    for style in (STYLE_MAX, STYLE_EXACT):
        pre = _PREFIX[style]
        if last.startswith(pre) and last.endswith(_SUFFIX) and last[len(pre):-len(_SUFFIX)].isdigit():
            return style
    return None
