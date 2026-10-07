# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Reward for data_source "olive_rl300k" (examples/data_preprocess/olive_rl300k_preprocess.py).

FORMAT GATE (nemotron_unified_v4.classify, shared with the v4 reward, applied before anything
is scored). The response must be exactly one "<think>...</think>" block, starting at character
zero, followed by either prose or whitespace-separated "<tool_call>{"name", "arguments": {...}}
</tool_call>" blocks and nothing else. Nothing is recovered from broken markup - an unclosed
tag, a stray tag, text around or between blocks, arguments given as a string, extra or
duplicate keys - and one bad block voids the others. This deliberately replaces
nemotron_pivot.extract_action, which mimicked vLLM's hermes parser and still credited an
unclosed trailing <tool_call>; on the v3 run 3.9% of rollouts collected tool reward that way.

  * think block broken -> THINK_ERROR_SCORE (-1.0), no judge call, think_error=1.
  * tool markup broken -> the response IS prose, with format_error=1.

Prose rows (expected message):
  * a valid tool call where prose was expected          -> -1.0, no judge call
  * otherwise the LLM judge (same prompt, same provider) -> +1.0 pass / -1.0 fail
    judge unreachable                                  -> NEMOTRON_JUDGE_FALLBACK (default 0.0)
    (a reply carrying broken tool markup is judged like any other prose)

Tool rows (expected call or batch):
  * prose, or no valid call                            -> -1.0
  * otherwise each EXPECTED call is paired with its best unused emitted call of the same
    name and scored on one of three tiers:
        1.0  name, argument keys and every value match
        0.5  name matches and the emitted key set is one the tool would accept
        0.0  anything else (wrong name, key set the tool would reject, call never made)
    and the row score is the mean over EXPECTED calls (recall: a missing call costs its full
    share), then scaled by matched/emitted when surplus calls were made (precision: one junk
    call beside one correct call halves the row, a spray of five keeps a fifth). A perfect
    response is untouched and the scaling never pushes below zero.

Key sets, when the tool schema is readable from the prompt (schemas_from_prompt):
  * an optional param left at its declared default is stripped from BOTH sides before the key
    sets are compared, so supplying or omitting it is free. Measured on the val split, 77% of
    call rows carry an optional argument the expert supplied and 32% are run_terminal, whose
    optional "duration" the expert sets on every single call.
  * when the key sets still differ, the 0.5 tier is earned by a schema-VALID key set - every
    required param present, nothing undeclared - rather than by exact equality. Without this a
    correct call that skipped one optional param scored 0.0, the same as calling a completely
    different tool, and below the 0.5 that a call with entirely wrong arguments collects.
  * with no readable schema the key sets must match exactly, as before.

Value matching. Non-strings follow nemotron_pivot._arguments_match exactly: same JSON type,
dicts need equal key sets and recurse, lists need equal length and recurse, floats within
1e-6, everything else by equality. STRINGS use the rules below, tried in order; the first
rule whose parser accepts both sides decides. Every string is first normalised: strip,
strip one layer of surrounding quotes, collapse whitespace, casefold.

  1. equal after normalisation                                   -> match
  2. parameter declares an enum (schema read from the prompt)    -> exact only
  3. both parse as JSON objects/arrays                           -> recursive value match
  4. both are http(s) URLs with a query                          -> scheme+host+path exact,
                                                                    query keys equal, values
                                                                    by these string rules
  5. both are key=value strings (a=b&c=d / ; / ,)               -> same as 4's query rule
  6. command keys (keystrokes, command, cmd, code, script, ...), or the call's LONGEST
     string argument when the row's verifier is "freeform-command" -> verb multiset Jaccard
                                                                    >= OLIVE_RL300K_JACCARD_CMD (0.5)
  7. search keys (query, q, keyword, ...)                        -> quoted phrases and
                                                                    site:/AND/OR/NOT operators
                                                                    must match as sets; the
                                                                    remainder by rule 9
  8. SQL statements                                              -> token-sequence ratio
                                                                    >= OLIVE_RL300K_SIM_SQL (0.8)
  9. by the shape of the expected value:
        single token (no whitespace), path-like -> exact, or basenames equal
        single token                            -> exact
        2..15 words                             -> word-set Jaccard >= OLIVE_RL300K_JACCARD_SHORT (0.5)
        16+ words                               -> word-set Jaccard >= OLIVE_RL300K_JACCARD_LONG (0.3)

Returned dict keys are fixed (see RESULT_KEYS) so reward_extra_info stacks across items:
score, think_error, format_error, is_call, type_match, name_match, judged, judge_score,
judge_error, n_expected_calls, n_emitted_calls, calls_full, calls_half, calls_zero,
n_surplus_calls.
"""
from __future__ import annotations

import difflib
import json
import os
import posixpath
import re
from collections import Counter
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import nemotron_pivot, nemotron_unified_v4
from .nemotron_pivot_judge import _env, judge_prose

RESULT_KEYS = ("score", "think_error", "format_error", "is_call", "type_match", "name_match",
               "judged", "judge_score", "judge_error", "n_expected_calls", "n_emitted_calls",
               "calls_full", "calls_half", "calls_zero", "n_surplus_calls")

# A broken think block is scored like prose on a tool row: below every call tier (worst 0.0)
# and level with a failed judge, so it is never the cheap way out of making a call.
THINK_ERROR_SCORE = -1.0

COMMAND_KEYS = frozenset({"keystrokes", "command", "cmd", "commands", "code", "script", "bash", "shell_command"})
SEARCH_KEYS = frozenset({"query", "q", "search_query", "searchquery", "search", "keyword", "keywords",
                         "search_term", "term"})
_SHELL_NOISE = frozenset({"sudo", "cd", "then", "do", "else", "elif", "fi", "done", "time", "exec",
                          "nohup", "env", "echo", "printf", "true", "false", "exit", "return"})
_SEARCH_OPS = re.compile(r'^(?:site|filetype|intitle|inurl|inanchor|before|after):\S+$|^(?:AND|OR|NOT)$')
_SQL_HEAD = re.compile(r"^\s*(select|insert|update|delete|with|create|alter|drop|replace)\b", re.I)
_SQL_BODY = re.compile(r"\b(from|into|set|table|where|values|join|index|view)\b", re.I)
_KV_RE = re.compile(r"^\s*[\w.\-\[\]]+\s*=\s*[^&;,\n]*(?:\s*[&;,]\s*[\w.\-\[\]]+\s*=\s*[^&;,\n]*)*\s*$")
_WORD = re.compile(r"\w+", re.UNICODE)


def _thr(name: str, default: float) -> float:
    return float(_env(name, str(default)))


# --------------------------------------------------------------------------- strings

def normalise(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        s = s[1:-1].strip()
    return re.sub(r"\s+", " ", s).casefold()


def words(s: str) -> list[str]:
    return _WORD.findall(s.casefold())


def jaccard(a, b) -> float:
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return 1.0 if union == 0 else inter / union


def _parse_json_struct(s: str):
    try:
        v = json.loads(s.strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return v if isinstance(v, (dict, list)) else None


def _parse_url(s: str):
    s = s.strip()
    if not re.match(r"^https?://", s, re.I):
        return None
    u = urlsplit(s)
    if not u.netloc:
        return None
    return u


def _parse_kv(s: str) -> dict[str, str] | None:
    if not _KV_RE.match(s):
        return None
    out: dict[str, str] = {}
    for part in re.split(r"\s*[&;,]\s*", s.strip()):
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    return out


def command_verbs(text: str) -> list[str]:
    verbs: list[str] = []
    for line in text.splitlines():
        for seg in re.split(r"\|\||&&|\||;", line):
            toks = seg.strip().split()
            while toks and (re.match(r"^\w+=", toks[0]) or toks[0].strip("\"'`") in _SHELL_NOISE):
                toks = toks[1:]
            if not toks:
                continue
            t = toks[0].strip("\"'`()")
            if not re.match(r"^[A-Za-z0-9_./\-]+$", t):
                continue
            verbs.append(posixpath.basename(t.rstrip("/")) or t)
    return verbs


def _search_parts(s: str) -> tuple[set[str], set[str], list[str]]:
    quoted = {normalise(q) for q in re.findall(r'"([^"]+)"', s)}
    rest = re.sub(r'"[^"]+"', " ", s)
    ops, plain = set(), []
    for tok in rest.split():
        if _SEARCH_OPS.match(tok):
            ops.add(tok.casefold())
        else:
            plain.extend(words(tok))
    return quoted, ops, plain


def _is_sql(s: str) -> bool:
    return bool(_SQL_HEAD.match(s) and _SQL_BODY.search(s))


def _sql_tokens(s: str) -> list[str]:
    return [t.casefold() for t in re.findall(r"\w+|[^\w\s]", s)]


def string_match(expected: str, actual: str, key: str = "", enum: set[str] | None = None,
                 is_command: bool = False) -> bool:
    ne, na = normalise(expected), normalise(actual)
    if ne == na:                                                            # 1
        return True
    if enum:                                                                # 2
        return False

    je, ja = _parse_json_struct(expected), _parse_json_struct(actual)       # 3
    if je is not None and ja is not None:
        return value_match(je, ja)

    ue, ua = _parse_url(expected), _parse_url(actual)                       # 4
    if ue is not None and ua is not None and (ue.query or ua.query):
        if (ue.scheme.lower(), ue.netloc.lower(), ue.path.rstrip("/")) != \
                (ua.scheme.lower(), ua.netloc.lower(), ua.path.rstrip("/")):
            return False
        return _kv_match(parse_qs(ue.query, keep_blank_values=True),
                         parse_qs(ua.query, keep_blank_values=True))

    ke, ka = _parse_kv(expected), _parse_kv(actual)                         # 5
    if ke is not None and ka is not None:
        return _kv_match({k: [v] for k, v in ke.items()}, {k: [v] for k, v in ka.items()})

    if is_command or key.casefold() in COMMAND_KEYS:                        # 6
        ve, va = command_verbs(expected), command_verbs(actual)
        if not ve and not va:
            return ne == na
        return jaccard(ve, va) >= _thr("OLIVE_RL300K_JACCARD_CMD", 0.5)

    if key.casefold() in SEARCH_KEYS:                                       # 7
        qe, oe, pe = _search_parts(expected)
        qa, oa, pa = _search_parts(actual)
        if qe != qa or oe != oa:
            return False
        if not pe and not pa:
            return True
        return jaccard(pe, pa) >= _thr("OLIVE_RL300K_JACCARD_SHORT", 0.5)

    if _is_sql(expected) and _is_sql(actual):                               # 8
        ratio = difflib.SequenceMatcher(None, _sql_tokens(expected), _sql_tokens(actual)).ratio()
        return ratio >= _thr("OLIVE_RL300K_SIM_SQL", 0.8)

    if " " not in ne:                                                      # 9: single token
        if ("/" in ne or "\\" in ne) and " " not in na:
            be = posixpath.basename(ne.replace("\\", "/").rstrip("/"))
            ba = posixpath.basename(na.replace("\\", "/").rstrip("/"))
            return bool(be) and be == ba
        return False
    we, wa = words(expected), words(actual)
    if len(we) < 16:
        return jaccard(we, wa) >= _thr("OLIVE_RL300K_JACCARD_SHORT", 0.5)
    return jaccard(we, wa) >= _thr("OLIVE_RL300K_JACCARD_LONG", 0.3)


def _kv_match(exp: dict[str, list[str]], act: dict[str, list[str]]) -> bool:
    if set(exp) != set(act):
        return False
    for k, vs in exp.items():
        va = act[k]
        if len(vs) != len(va):
            return False
        if not all(string_match(e, a, key=k) for e, a in zip(vs, va)):
            return False
    return True


# --------------------------------------------------------------------------- values / calls

def value_match(expected: Any, actual: Any, key: str = "", enums: dict[str, set[str]] | None = None,
                is_command: bool = False) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual or expected == actual and type(expected) is type(actual)
    if not isinstance(actual, type(expected)):
        return False
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return False
        return all(value_match(v, actual[k], key=k, enums=enums, is_command=is_command)
                   for k, v in expected.items())
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return False
        return all(value_match(e, a, key=key, enums=enums, is_command=is_command)
                   for e, a in zip(expected, actual))
    if isinstance(expected, float):
        return abs(actual - expected) < nemotron_pivot.FLOAT_TOLERANCE
    if isinstance(expected, str):
        return string_match(expected, actual, key=key, enum=(enums or {}).get(key), is_command=is_command)
    return expected == actual


def _same_literal(a: Any, b: Any) -> bool:
    """Equality that does not collapse True==1 or 1==1.0, for comparing against schema defaults."""
    return type(a) is type(b) and a == b


def _strip_default_optionals(args: dict, schema: dict | None) -> dict:
    """Drop optional params left at their declared default.

    Supplying or omitting such a param says nothing about the decision: the tool behaves the
    same either way. Both sides are stripped identically, so an exact copy of the expert action
    is unaffected; only a disagreement that is purely about a documented default stops costing
    anything. Without a schema nothing is stripped.
    """
    props = (schema or {}).get("properties") or {}
    if not props:
        return args
    req = set((schema or {}).get("required") or [])
    return {k: v for k, v in args.items()
            if k in req
            or not (isinstance(props.get(k), dict) and "default" in props[k]
                    and _same_literal(props[k]["default"], v))}


def _key_set_is_valid(args: dict, schema: dict | None) -> bool:
    """True when the emitted key set is one the tool would accept: every required param present
    and nothing undeclared. This is what earns the 0.5 tier when the key sets differ, so a
    correct call that skips an optional param no longer scores what a wrong tool scores.
    Without a schema there is nothing to check against and the key sets must match exactly.
    """
    props = (schema or {}).get("properties") or {}
    if not props:
        return False
    return set((schema or {}).get("required") or []) <= set(args) <= set(props)


def _enums_of(schema: dict | None) -> dict[str, set[str]]:
    return {k: {str(x) for x in p["enum"]}
            for k, p in ((schema or {}).get("properties") or {}).items()
            if isinstance(p, dict) and isinstance(p.get("enum"), list) and p["enum"]}


def call_score(exp: dict, act: dict, schema: dict | None = None, is_command: bool = False) -> float:
    """1.0 name + keys + values, 0.5 name + a schema-valid key set, 0.0 otherwise."""
    if exp["name"] != act["name"]:
        return 0.0
    try:
        ea, aa = json.loads(exp["arguments"]), json.loads(act["arguments"])
    except (json.JSONDecodeError, TypeError):
        return 0.0
    enums = _enums_of(schema)
    if not isinstance(ea, dict) or not isinstance(aa, dict):
        return 1.0 if value_match(ea, aa, enums=enums, is_command=is_command) else 0.0
    se, sa = _strip_default_optionals(ea, schema), _strip_default_optionals(aa, schema)
    if set(se) != set(sa):
        return 0.5 if _key_set_is_valid(aa, schema) else 0.0
    # A "freeform-command" row was classified by the source on its LONGEST string value, so
    # only that argument (plus the explicit command keys) is matched on shell verbs; the
    # call's other string arguments (a prose "goal", a path) keep the ordinary rules.
    cmd_key = None
    if is_command:
        strs = {k: len(v) for k, v in se.items() if isinstance(v, str)}
        cmd_key = max(strs, key=strs.get) if strs else None
    return 1.0 if all(value_match(v, sa[k], key=k, enums=enums, is_command=(is_command and k == cmd_key))
                      for k, v in se.items()) else 0.5


def tool_score(expected: dict, actual: dict, schemas: dict[str, dict] | None = None,
               is_command: bool = False) -> tuple[float, Counter]:
    """Mean tier over expected calls, name-grouped best matching; -1 when no call was emitted."""
    exp_calls = nemotron_pivot._calls_of(expected)
    act_calls = nemotron_pivot._calls_of(actual)
    tally = Counter(n_expected_calls=len(exp_calls), n_emitted_calls=len(act_calls))
    if not exp_calls:
        return -1.0, tally
    if not act_calls:
        tally["calls_zero"] = len(exp_calls)
        return -1.0, tally

    exp_by: dict[str, list[int]] = {}
    act_by: dict[str, list[int]] = {}
    for i, e in enumerate(exp_calls):
        exp_by.setdefault(e["name"], []).append(i)
    for j, a in enumerate(act_calls):
        act_by.setdefault(a["name"], []).append(j)

    # Within each name group, assign pairs best-first over the whole score matrix, so an
    # exact emitted call is never spent on a half-matching expected call that came earlier.
    per_expected = [0.0] * len(exp_calls)
    matched_actual: set[int] = set()
    for name, eis in exp_by.items():
        ajs = act_by.get(name, [])
        pairs = sorted(
            ((call_score(exp_calls[i], act_calls[j], schema=(schemas or {}).get(name), is_command=is_command), i, j)
             for i in eis for j in ajs),
            key=lambda t: (-t[0], t[1], t[2]),
        )
        done_e: set[int] = set()
        for v, i, j in pairs:
            if i in done_e or j in matched_actual:
                continue
            done_e.add(i)
            matched_actual.add(j)
            per_expected[i] = v
    for v in per_expected:
        tally["calls_full" if v == 1.0 else "calls_half" if v == 0.5 else "calls_zero"] += 1

    # Recall over the ground truth: a missing call already costs its full share.
    score = sum(per_expected) / len(exp_calls)
    # Precision over what was emitted: surplus calls scale the score down by matched/emitted,
    # so one junk call next to one correct call halves the row and a spray of five keeps a
    # fifth. A perfect response is untouched; nothing here can push below zero.
    if len(act_calls) > len(matched_actual):
        score *= len(matched_actual) / len(act_calls)
    tally["n_surplus_calls"] = len(act_calls) - len(matched_actual)
    return max(-1.0, min(1.0, score)), tally


# --------------------------------------------------------------------------- schema enums from the prompt

_TOOLS_BLOCK = re.compile(r"<tools>\n(.*?)\n</tools>", re.DOTALL)


def schemas_from_prompt(raw_prompt: Any) -> dict[str, dict]:
    """{tool name: its JSON-Schema `parameters`} parsed off the system message's tools block.

    The block is rendered one JSON object per line by the tooling chat template, which is what
    the preprocess bakes into the system message. Anything else yields {} and the reward falls
    back to exact key-set matching.
    """
    out: dict[str, dict] = {}
    if raw_prompt is None:
        return out
    seq = list(raw_prompt)
    if not seq:
        return out
    first = seq[0]
    get = first.get if isinstance(first, dict) else lambda k, d=None: getattr(first, k, d)
    content, role = get("content"), get("role")
    if not isinstance(content, str) or role != "system":
        return out
    m = _TOOLS_BLOCK.search(content)
    if not m:
        return out
    for line in m.group(1).split("\n"):
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        fn = t.get("function", t) if isinstance(t, dict) else {}
        params = fn.get("parameters")
        if isinstance(fn.get("name"), str) and isinstance(params, dict):
            out[fn["name"]] = params
    return out


def enums_from_prompt(raw_prompt: Any) -> dict[str, dict[str, set[str]]]:
    """{tool name: {param key: set of enum strings}}; the enum slice of schemas_from_prompt."""
    out: dict[str, dict[str, set[str]]] = {}
    for name, params in schemas_from_prompt(raw_prompt).items():
        per = _enums_of(params)
        if per:
            out[name] = per
    return out


# --------------------------------------------------------------------------- entry point

def _base() -> dict:
    return {k: 0.0 for k in RESULT_KEYS}


def _as_action(calls: list[dict]) -> dict:
    if not calls:
        return {"type": "message", "content": ""}
    if len(calls) == 1:
        return calls[0]
    return {"type": "function_call_batch", "calls": calls}


def compute_score(solution_str: str, ground_truth: str, extra_info: dict | None = None,
                  last_user_message: str = "", raw_prompt: Any = None, **kwargs: Any) -> dict:
    del kwargs
    out = _base()
    try:
        expected = json.loads(ground_truth or "")
    except json.JSONDecodeError:
        return out
    if not isinstance(expected, dict) or "type" not in expected:
        return out

    # v4's gate, not nemotron_pivot.extract_action: the response must be exactly one <think>
    # block followed by prose or by whitespace-separated, well-formed <tool_call> blocks.
    # Nothing is recovered from broken markup; one bad block voids the others.
    c = nemotron_unified_v4.classify(solution_str or "")
    out["format_error"] = float(c["format_error"])
    if not c["think_ok"]:
        out["think_error"] = 1.0
        out["score"] = THINK_ERROR_SCORE
        return out
    out["is_call"] = float(bool(c["calls"]))

    if expected["type"] == "message":
        if c["calls"]:
            out["score"] = -1.0
            return out
        out["type_match"] = 1.0
        verdict, err = judge_prose(last_user_message, c["tail"].strip())
        out["judged"] = 1.0
        if verdict is None:
            out["judge_error"] = 1.0
            out["score"] = float(_env("NEMOTRON_JUDGE_FALLBACK", "0.0"))
            out["judge_score"] = out["score"]
            if os.getenv("NEMOTRON_JUDGE_VERBOSE"):
                print(f"[olive_rl300k] judge failed: {err}")
            return out
        out["judge_score"] = verdict
        out["score"] = 1.0 if verdict == 1.0 else -1.0
        return out

    # expected a call or a batch
    is_command = (extra_info or {}).get("verifier") == "freeform-command"
    actual = _as_action(c["calls"])
    score, tally = tool_score(expected, actual, schemas=schemas_from_prompt(raw_prompt),
                              is_command=is_command)
    out.update({k: float(v) for k, v in tally.items()})
    out["type_match"] = out["is_call"]
    exp_names = {x["name"] for x in nemotron_pivot._calls_of(expected)}
    act_names = {x["name"] for x in nemotron_pivot._calls_of(actual)}
    out["name_match"] = 1.0 if exp_names & act_names else 0.0
    out["score"] = score
    return out
