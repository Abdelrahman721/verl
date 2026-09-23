# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Tests for verl/utils/reward_score/olive_rl300k.py.

Runs under pytest, or directly with any Python 3 (`python tests/utils/reward_score/
test_olive_rl300k_on_cpu.py`): the reward package is loaded by path so the top-level verl
package (and torch) is never imported. The LLM judge is monkeypatched; no network is used.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PKG_DIR = REPO / "verl" / "utils" / "reward_score"


def _load():
    """Import verl.utils.reward_score.{nemotron_pivot,nemotron_pivot_judge,olive_rl300k}
    without executing verl/__init__.py."""
    try:
        return importlib.import_module("verl.utils.reward_score.olive_rl300k")
    except Exception:
        pass
    for name in ("verl", "verl.utils", "verl.utils.reward_score"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [str(REPO / name.replace(".", "/"))]
            sys.modules[name] = m
    for mod in ("nemotron_pivot", "nemotron_pivot_judge", "olive_rl300k"):
        full = f"verl.utils.reward_score.{mod}"
        if full in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(full, PKG_DIR / f"{mod}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
    return sys.modules["verl.utils.reward_score.olive_rl300k"]


R = _load()
T = "<think>\nplan\n</think>\n\n"


def call(name, **args):
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}\n</tool_call>'


def gt_call(name, **args):
    return json.dumps({"type": "function_call", "name": name, "arguments": json.dumps(args, ensure_ascii=False)})


def gt_batch(*calls):
    return json.dumps({"type": "function_call_batch",
                       "calls": [{"type": "function_call", "name": n, "arguments": json.dumps(a)} for n, a in calls]})


def gt_msg(text="hello"):
    return json.dumps({"type": "message", "content": text})


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------------- string rules

def test_normalisation_equal():
    assert R.string_match("Spanish", "  spanish ")
    assert R.string_match('"Paris"', "paris")
    assert R.string_match("a   b", "A B")


def test_enum_is_exact():
    # without an enum, 2 of 3 words overlap -> match; with the enum it must be exact
    assert R.string_match("customer rating", "customer rating high")
    assert not R.string_match("customer rating", "customer rating high", enum={"customer rating", "price"})
    assert R.string_match("customer rating", "Customer Rating", enum={"customer rating"})


def test_single_token_exact():
    assert not R.string_match("CL123456", "CL123457")
    assert not R.string_match("2025-09-04", "2025-9-4")
    assert not R.string_match("Spanish", "French")


def test_path_basename():
    assert R.string_match("/app/solution.py", "solution.py", key="path")
    assert R.string_match("/app/solution.py", "./solution.py")
    assert not R.string_match("/app/a.py", "/app/b.py")


def test_short_text_jaccard():
    assert R.string_match("Robert Lindstedt tennis player", "Robert Lindstedt tennis")
    assert not R.string_match("Robert Lindstedt tennis", "Paris weather forecast")
    with env(OLIVE_RL300K_JACCARD_SHORT="0.9"):
        assert not R.string_match("Robert Lindstedt tennis player", "Robert Lindstedt tennis")


def test_long_text_jaccard():
    a = ("The customer would like to change the delivery address of order 4471 to the new "
         "apartment on Main Street and asks whether the delivery date can stay the same")
    b = ("Customer requests updating order 4471 delivery address to the Main Street apartment "
         "and wants to keep the delivery date unchanged")
    assert len(R.words(a)) >= 16
    assert R.string_match(a, b, key="summary")
    c = "Please book a table for four at the Italian restaurant downtown on Friday evening at eight"
    assert not R.string_match(a, c, key="summary")


def test_command_verbs():
    exp = "ls -la\nfind /app -name '*.txt' | head -20\ncat README.md"
    assert R.command_verbs(exp) == ["ls", "find", "head", "cat"]
    assert R.string_match(exp, "ls\nfind . -type f\ncat notes.txt", key="keystrokes")
    assert not R.string_match(exp, "python solve.py", key="keystrokes")
    # shell noise and env assignments are skipped, paths reduced to basenames
    assert R.string_match("cd /app && sudo apt install x", "FOO=1 /usr/bin/apt install y", key="command")
    # a non-command key with commands in it only counts as a command when the row says so
    assert R.string_match("ls -la; cat x", "ls; cat y", key="input", is_command=True)
    assert not R.string_match("ls -la; cat x", "ls; cat y", key="input", is_command=False)


def test_json_in_string():
    assert R.string_match('{"a": 1, "b": [1, 2]}', '{"b":[1,2],"a":1}', key="params")
    assert not R.string_match('{"a": 1}', '{"a": 2}', key="params")
    assert R.string_match('["x y", "z"]', '["x y", "z"]')


def test_url_query():
    assert R.string_match("https://www.amazon.co.jp/s?k=ceramic+tray&ref=x",
                          "https://www.amazon.co.jp/s?ref=x&k=ceramic+tray", key="url")
    assert not R.string_match("https://www.amazon.co.jp/s?k=tray", "https://www.amazon.com/s?k=tray", key="url")
    assert not R.string_match("https://a.com/s?k=tray&ref=x", "https://a.com/s?k=tray", key="url")
    assert not R.string_match("https://a.com/s?k=ceramic+tray+garden", "https://a.com/s?k=laptop", key="url")


def test_key_value_strings():
    assert R.string_match("title=round&max_price=5000&clarity=VVS1",
                          "clarity=VVS1;title=round;max_price=5000", key="filter")
    assert not R.string_match("title=round&max_price=5000", "title=round", key="filter")
    assert not R.string_match("title=round&max_price=5000", "title=round&max_price=9000", key="filter")


def test_search_operators():
    e = '"trauma-informed classroom" AND strategies for teachers 2023'
    assert R.string_match(e, 'strategies teachers 2023 AND "trauma-informed classroom"', key="query")
    assert not R.string_match(e, "trauma-informed classroom strategies for teachers 2023", key="query")
    assert not R.string_match("site:nytimes.com climate report", "site:bbc.co.uk climate report", key="q")
    assert R.string_match("Robert Lindstedt tennis player", "Robert Lindstedt tennis", key="query")


def test_sql():
    a = "SELECT city, AVG(hotel_price) AS avg_price FROM default_db WHERE city IN ('Paris', 'Lyon') GROUP BY city"
    b = "select city, avg(hotel_price) as avg_price\nfrom default_db\nwhere city in ('Paris','Lyon') group by city"
    assert R.string_match(a, b, key="query")
    assert not R.string_match(a, "SELECT name FROM users WHERE id = 3", key="query")


# --------------------------------------------------------------------------- values and calls

def test_non_string_values_follow_pivot_rules():
    assert R.value_match(3, 3) and not R.value_match(3, 3.0) and not R.value_match(3, "3")
    assert R.value_match(1.0, 1.0 + 1e-9) and not R.value_match(1.0, 1.1)
    assert R.value_match(True, True) and not R.value_match(True, 1)
    assert R.value_match([1, "a b c"], [1, "a b d"]) and not R.value_match([1, 2], [1])
    assert R.value_match({"x": {"y": "hello world"}}, {"x": {"y": "hello world"}})
    assert not R.value_match({"x": 1}, {"x": 1, "z": 2})


def test_call_tiers():
    e = {"name": "f", "arguments": json.dumps({"client_id": "C1", "days": 42})}
    assert R.call_score(e, {"name": "f", "arguments": json.dumps({"client_id": "C1", "days": 42})}) == 1.0
    assert R.call_score(e, {"name": "f", "arguments": json.dumps({"client_id": "C1", "days": 41})}) == 0.5
    assert R.call_score(e, {"name": "f", "arguments": json.dumps({"client_id": "C1"})}) == 0.0
    assert R.call_score(e, {"name": "f", "arguments": json.dumps({"client_id": "C1", "days": 42, "x": 1})}) == 0.0
    assert R.call_score(e, {"name": "g", "arguments": json.dumps({"client_id": "C1", "days": 42})}) == 0.0
    assert R.call_score(e, {"name": "f", "arguments": "{not json"}) == 0.0
    z = {"name": "noargs", "arguments": "{}"}
    assert R.call_score(z, {"name": "noargs", "arguments": "{}"}) == 1.0


def test_tool_score_batch_average_and_matching():
    exp = json.loads(gt_batch(("f", {"id": "A"}), ("f", {"id": "B"}), ("g", {"n": 1})))
    act = R.nemotron_pivot.extract_action(T + call("f", id="B") + "\n" + call("f", id="X") + "\n" + call("g", n=1))
    score, tally = R.tool_score(exp, act)
    # f(B) full, f(A) paired with f(X) -> keys match, value wrong -> 0.5, g full
    assert abs(score - (1.0 + 0.5 + 1.0) / 3) < 1e-9
    assert tally["calls_full"] == 2 and tally["calls_half"] == 1 and tally["calls_zero"] == 0
    # only one of three made
    act = R.nemotron_pivot.extract_action(T + call("g", n=1))
    score, tally = R.tool_score(exp, act)
    assert abs(score - 1.0 / 3) < 1e-9 and tally["calls_zero"] == 2
    # prose where calls were expected
    score, _ = R.tool_score(exp, R.nemotron_pivot.extract_action(T + "Sure, done."))
    assert score == -1.0


def test_surplus_calls_scale_by_precision():
    exp = json.loads(gt_call("f", id="A"))
    one = R.nemotron_pivot.extract_action(T + call("f", id="A"))
    assert R.tool_score(exp, one)[0] == 1.0                                   # perfect: untouched
    two = R.nemotron_pivot.extract_action(T + call("f", id="A") + "\n" + call("h", z=1))
    s, tally = R.tool_score(exp, two)
    assert s == 0.5 and tally["n_surplus_calls"] == 1                         # 1 * 1/2
    five = R.nemotron_pivot.extract_action(T + call("f", id="A") + "\n" + "\n".join(call("h", z=i) for i in range(4)))
    assert abs(R.tool_score(exp, five)[0] - 0.2) < 1e-9                       # 1 * 1/5
    # a keys-only match beside junk: 0.5 * 1/2
    half = R.nemotron_pivot.extract_action(T + call("f", id="Z") + "\n" + call("h", z=1))
    assert R.tool_score(exp, half)[0] == 0.25
    # repeats of an expected name with wrong arguments are surplus too: fishing does not pay
    fish = R.nemotron_pivot.extract_action(T + "\n".join(call("f", id=x) for x in "AQRST"))
    assert abs(R.tool_score(exp, fish)[0] - 0.2) < 1e-9
    # batch: 2 of 3 expected made, plus one junk -> recall 2/3, precision 2/3
    exp3 = json.loads(gt_batch(("f", {"id": "A"}), ("f", {"id": "B"}), ("g", {"n": 1})))
    act = R.nemotron_pivot.extract_action(T + call("f", id="A") + "\n" + call("g", n=1) + "\n" + call("h", z=1))
    assert abs(R.tool_score(exp3, act)[0] - (2 / 3) * (2 / 3)) < 1e-9


# --------------------------------------------------------------------------- compute_score

def _no_judge(*a, **k):
    raise AssertionError("judge must not be called")


def test_prose_expected():
    with patched(R, "judge_prose", _no_judge):
        r = R.compute_score(T + call("f", x=1), gt_msg())
        assert r["score"] == -1.0 and r["is_call"] == 1.0 and r["judged"] == 0.0
    with patched(R, "judge_prose", lambda u, c: (1.0, "")):
        r = R.compute_score(T + "Here you go.", gt_msg(), last_user_message="hi")
        assert r["score"] == 1.0 and r["judged"] == 1.0 and r["judge_score"] == 1.0
    with patched(R, "judge_prose", lambda u, c: (0.0, "")):
        assert R.compute_score(T + "meh", gt_msg())["score"] == -1.0
    with patched(R, "judge_prose", lambda u, c: (None, "HTTP 429")), env(NEMOTRON_JUDGE_FALLBACK="0.0"):
        r = R.compute_score(T + "meh", gt_msg())
        assert r["score"] == 0.0 and r["judge_error"] == 1.0


def test_call_expected_end_to_end():
    with patched(R, "judge_prose", _no_judge):
        gt = gt_call("check_coverage", client_id="CL123456", service_type="SkilledNursing", duration=42)
        assert R.compute_score(T + call("check_coverage", client_id="CL123456", service_type="SkilledNursing", duration=42), gt)["score"] == 1.0
        r = R.compute_score(T + call("check_coverage", client_id="CL123456", service_type="SkilledNursing", duration=41), gt)
        assert r["score"] == 0.5 and r["name_match"] == 1.0 and r["type_match"] == 1.0
        r = R.compute_score(T + "I will check that.", gt)
        assert r["score"] == -1.0 and r["is_call"] == 0.0 and r["type_match"] == 0.0
        r = R.compute_score(T + '<tool_call>\n{"name": check_coverage}\n</tool_call>', gt)
        assert r["score"] == -1.0 and r["format_error"] == 1.0
        assert set(r) == set(R.RESULT_KEYS)


def test_verifier_and_enums_from_prompt():
    with patched(R, "judge_prose", _no_judge):
        gt = gt_call("run", input="ls -la; cat x")
        emitted = T + call("run", input="ls; cat y")
        assert R.compute_score(emitted, gt)["score"] == 0.5
        assert R.compute_score(emitted, gt, extra_info={"verifier": "freeform-command"})["score"] == 1.0

        tool = {"type": "function", "function": {"name": "sort_items", "description": "", "parameters": {
            "type": "object", "properties": {"by": {"type": "string", "enum": ["customer rating", "price"]}}}}}
        system = "You are a helpful assistant.\n\n# Tools\n<tools>\n" + json.dumps(tool) + "\n</tools>\nend"
        raw_prompt = [{"role": "system", "content": system}, {"role": "user", "content": "sort"}]
        assert R.enums_from_prompt(raw_prompt) == {"sort_items": {"by": {"customer rating", "price"}}}
        gt = gt_call("sort_items", by="customer rating")
        emitted = T + call("sort_items", by="customer rating high")
        assert R.compute_score(emitted, gt)["score"] == 1.0                      # no schema: jaccard passes
        assert R.compute_score(emitted, gt, raw_prompt=raw_prompt)["score"] == 0.5  # enum: exact only


def test_expert_actions_score_one_on_corpus():
    """Every call-row ground truth in the val parquet scores 1.0 when fed back verbatim."""
    path = REPO / "rl-data" / "olive_rl300k" / "olive_rl300k_all_val.parquet"
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return
    if not path.exists():
        return
    n = 0
    with patched(R, "judge_prose", _no_judge):
        for r in pq.read_table(str(path), columns=["reward_model", "extra_info"]).to_pylist():
            gt = json.loads(r["reward_model"]["ground_truth"])
            if gt["type"] == "message":
                continue
            calls = gt["calls"] if gt["type"] == "function_call_batch" else [gt]
            resp = T + "\n".join(f'<tool_call>\n{json.dumps({"name": c["name"], "arguments": json.loads(c["arguments"])}, ensure_ascii=False)}\n</tool_call>' for c in calls)
            out = R.compute_score(resp, r["reward_model"]["ground_truth"], extra_info=r["extra_info"])
            assert out["score"] == 1.0, (r["extra_info"]["row_id"], out)
            n += 1
    assert n > 0


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print("ALL OK" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
