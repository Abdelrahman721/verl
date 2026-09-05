# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Preprocess nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1 to parquet for verl.

Each source row is one decision point: a Responses-API conversation prefix, the row's own tool
list, and the expert's next action. This script turns it into a single-turn verl prompt whose
reward is computed by comparing the policy's action against `expected_action`
(data_source "nemotron_pivot"; the scorer lives in verl/utils/reward_score/nemotron_pivot.py).

What the prompt column contains, and why it is shaped this way:

  * verl renders prompts inside the agent loop with the checkpoint tokenizer's chat template and
    only a *global* tool list, so per-row tools are baked into the system message here, using the
    exact "# Tools ... <tools>...</tools> ..." preamble Qwen's template emits. `--check_render N`
    proves the two renderings are byte-identical on N rows.
  * Prior assistant turns are written in the same form the SFT converters write: a
    "<think>\\n{trace}\\n</think>" prefix, then prose or tool calls as {"name", "arguments"}.
    Tool outputs become "tool" messages. Tool schemas are normalised to the OpenAI nested form and
    missing "type": "object" is repaired, as in llm-pretrainer/dataset-converters/.
  * `--reasoning all` keeps every prior turn's trace (what flat SFT taught the model);
    `--reasoning official` keeps traces only on assistant turns after the last user message
    (what Qwen3's official template shows at serving time). NOTE: "all" is only honoured by a
    template that renders assistant content verbatim, such as chat_template_tooling.jinja that
    the SFT checkpoints carry; Qwen3's official template parses <think> out of content and
    drops it before the last user turn regardless of what is written here.

Filters (all rows are dropped, never patched):
  * any prior assistant action without a reasoning item          (~5% of rows)
  * control strings (<tool_call>, <think>, ...) as literal text in context or expected action
  * expected call whose name the row does not offer, or whose arguments do not parse (0 in source)

Ground truth is the JSON of the expected action, normalised to one of:
  {"type": "function_call", "name": ..., "arguments": "<json string>"}
  {"type": "message", "content": ...}
extra_info carries trajectory_id, depth, the two difficulty signals and the offered tool names.

Rows are split by trajectory_id so no conversation appears in both train and val.

Example:
  python examples/data_preprocess/nemotron_pivot_preprocess.py \\
      --local_save_dir rl-data/nemotron_pivot --reasoning all --check_render 200 --token_sample 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import datasets

DATA_SOURCE = "nemotron_pivot"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT = (
    REPO_ROOT.parent / "tooling-research" / "external-data"
    / "Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1" / "train.jsonl"
)
DEFAULT_TOKENIZER = REPO_ROOT.parent / "llm-pretrainer" / "models" / "Qwen3-4B-Base"
DEFAULT_KEPT_TEMPLATE = REPO_ROOT.parent / "llm-pretrainer" / "chat_template_tooling.jinja"

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
CONTROL_STRINGS = ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
                   "<think>", "</think>", "<|im_start|>", "<|im_end|>")

# Byte-identical to the tools block in Qwen3's official template and chat_template_tooling.jinja.
TOOLS_HEAD = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_TAIL = (
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, "
    "\"arguments\": <args-json-object>}\n</tool_call>"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="train.jsonl from the HF snapshot")
    p.add_argument("--local_save_dir", default="/workspace/verl/rl-data/", help="Directory for the parquet files")
    p.add_argument("--reasoning", choices=["all", "official"], default="all",
                   help="Which prior-turn traces stay in context (see module docstring)")
    p.add_argument("--val_trajectories", type=int, default=1000, help="Trajectories held out for validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Stop after this many source rows (smoke test)")
    p.add_argument("--keep_unreasoned", action="store_true", help="Keep rows with unreasoned prior actions")
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER,
                   help="Tokenizer used for --check_render and --token_sample")
    p.add_argument("--kept_template", type=Path, default=DEFAULT_KEPT_TEMPLATE,
                   help="Verbatim-content template used to measure token lengths under --reasoning all")
    p.add_argument("--check_render", type=int, default=200,
                   help="Rows on which to prove baked tools == template tools kwarg (0 disables)")
    p.add_argument("--token_sample", type=int, default=2000,
                   help="Rows on which to report rendered prompt length (0 disables)")
    return p.parse_args()


# --------------------------------------------------------------------------- tools

def repair_schema(node):
    """Add a missing "type": "object" wherever "properties" is present; recurse. Nothing else."""
    if isinstance(node, dict):
        out = {k: repair_schema(v) for k, v in node.items()}
        if "properties" in out and "type" not in out:
            out["type"] = "object"
        return out
    if isinstance(node, list):
        return [repair_schema(x) for x in node]
    return node


def normalise_tool(tool: dict) -> dict:
    """Responses flat form {type, name, description, parameters, strict} -> OpenAI nested form."""
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    params = fn.get("parameters")
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}
    out = {"name": fn["name"], "description": fn.get("description", ""), "parameters": repair_schema(params)}
    return {"type": "function", "function": out}


def tools_block(tools: list[dict]) -> str:
    lines = [TOOLS_HEAD]
    for t in tools:
        lines.append("\n" + json.dumps(t, ensure_ascii=False))
    lines.append(TOOLS_TAIL)
    return "".join(lines)


# --------------------------------------------------------------------------- messages

def text_of(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return ""


def has_control(s) -> bool:
    return isinstance(s, str) and any(cs in s for cs in CONTROL_STRINGS)


class Drop(Exception):
    pass


def group_history(items: list[dict]) -> tuple[str, list[dict]]:
    """Responses items -> (system_text, turns). Each turn is
    {"role": "user"/"tool", "content"} or
    {"role": "assistant", "reasoning": str|None, "text": str, "calls": [{"name","arguments"}]}."""
    system = None
    turns: list[dict] = []
    cur = None

    def flush():
        nonlocal cur
        if cur is not None:
            turns.append(cur)
            cur = None

    def new_group(reasoning=None):
        nonlocal cur
        flush()
        cur = {"role": "assistant", "reasoning": reasoning, "text": "", "calls": []}

    for m in items:
        t = m.get("type") or "message"
        if t == "message":
            role = m.get("role")
            if role == "system":
                if system is not None:
                    raise Drop("two system messages")
                system = text_of(m)
            elif role == "user":
                flush()
                turns.append({"role": "user", "content": text_of(m)})
            elif role == "assistant":
                if cur is None or cur["text"] or cur["calls"]:
                    new_group()
                cur["text"] = text_of(m)
                if has_control(cur["text"]):
                    raise Drop("control string in assistant text")
                flush()
            else:
                raise Drop(f"unknown role {role}")
        elif t == "reasoning":
            txt = "".join(s.get("text", "") for s in m.get("summary", []) if isinstance(s, dict))
            if has_control(txt):
                raise Drop("control string in reasoning")
            new_group(txt if txt.strip() else None)
        elif t == "function_call":
            if cur is None or cur["text"]:
                new_group()
            args = m.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            try:
                json.loads(args)
            except json.JSONDecodeError as e:
                raise Drop("context call arguments unparseable") from e
            cur["calls"].append({"name": m["name"], "arguments": args})
        elif t == "function_call_output":
            flush()
            out = m.get("output")
            if not isinstance(out, str):
                out = json.dumps(out, ensure_ascii=False)
            if has_control(out):
                raise Drop("control string in tool output")
            turns.append({"role": "tool", "content": out})
        else:
            raise Drop(f"unknown item type {t}")
    flush()
    if system is None:
        raise Drop("no system message")
    if not turns or turns[0]["role"] != "user":
        raise Drop("history does not start with a user message")
    if turns[-1]["role"] == "assistant":
        raise Drop("history ends on an assistant turn")
    return system, turns


def render_turns(turns: list[dict], reasoning_mode: str) -> list[dict]:
    """Assistant groups -> {"role","content","tool_calls"} in the SFT converters' exact spacing."""
    last_user = max(i for i, t in enumerate(turns) if t["role"] == "user")
    out = []
    for i, t in enumerate(turns):
        if t["role"] != "assistant":
            out.append({"role": t["role"], "content": t["content"]})
            continue
        keep = reasoning_mode == "all" or i > last_user
        prefix = f"{THINK_OPEN}\n{t['reasoning']}\n{THINK_CLOSE}" if (keep and t["reasoning"]) else ""
        body = t["text"]
        if t["calls"]:
            content = f"{prefix}\n{body}" if body else (f"{prefix}\n" if prefix else "")
        else:
            content = f"{prefix}\n\n{body}" if prefix else body
        msg = {"role": "assistant", "content": content}
        if t["calls"]:
            msg["tool_calls"] = t["calls"]
        out.append(msg)
    return out


def normalise_expected(exp: dict, offered: set[str]) -> dict:
    et = exp.get("type")
    if et == "message":
        content = exp.get("content", "")
        if has_control(content):
            raise Drop("control string in expected message")
        return {"type": "message", "content": content}
    if et == "function_call":
        name, args = exp.get("name"), exp.get("arguments")
        if name not in offered:
            raise Drop("expected call not offered")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        try:
            json.loads(args)
        except json.JSONDecodeError as e:
            raise Drop("expected arguments unparseable") from e
        return {"type": "function_call", "name": name, "arguments": args}
    if et == "function_call_batch":
        calls = []
        for c in exp.get("calls", []):
            calls.append(normalise_expected({"type": "function_call", **c}, offered))
        return {"type": "function_call_batch", "calls": calls}
    raise Drop(f"unknown expected type {et}")


# --------------------------------------------------------------------------- rows

def convert_row(r: dict, idx: int, args) -> tuple[dict, dict]:
    rcp = r["responses_create_params"]
    tools = [normalise_tool(t) for t in rcp.get("tools") or []]
    if not tools:
        raise Drop("no tools")
    offered = {t["function"]["name"] for t in tools}
    system, turns = group_history(rcp["input"])
    if has_control(system):
        raise Drop("control string in system prompt")

    assistant_groups = [t for t in turns if t["role"] == "assistant"]
    if not args.keep_unreasoned and any(t["reasoning"] is None for t in assistant_groups):
        raise Drop("unreasoned prior action")

    expected = normalise_expected(r["expected_action"], offered)

    prompt = [{"role": "system", "content": system + "\n\n" + tools_block(tools)}]
    prompt += render_turns(turns, args.reasoning)

    mi = r.get("meta_info") or {}
    qi = r.get("qwen_235b_info") or {}
    extra = {
        "index": idx,
        "trajectory_id": int(r["trajectory_id"]),
        "assistant_depth": int(mi.get("assistant_depth", len(assistant_groups))),
        "user_turns": sum(1 for t in turns if t["role"] == "user"),
        "expected_type": expected["type"],
        "pass_rate": float(r["pass_rate"]) if r.get("pass_rate") is not None else None,
        "qwen_235b_reward_mean": float(qi["reward_mean"]) if qi.get("reward_mean") is not None else None,
        "offered_tools": sorted(offered),
        "reasoning_mode": args.reasoning,
    }
    row = {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": "tool_use",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(expected, ensure_ascii=False)},
        "extra_info": extra,
    }
    # kept aside for --check_render; not written to parquet
    aux = {"system_text": system, "tools": tools, "turns": prompt[1:]}
    return row, aux


def check_render(samples: list[tuple[dict, dict]], tokenizer) -> None:
    """Prove that baking tools into the system message renders byte-identically to passing
    them through the template's `tools` kwarg, under the tokenizer's own (official) template."""
    for row, aux in samples:
        baked = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False)
        via_kwarg = tokenizer.apply_chat_template(
            [{"role": "system", "content": aux["system_text"]}] + aux["turns"],
            tools=aux["tools"], add_generation_prompt=True, tokenize=False,
        )
        if baked != via_kwarg:
            a, b = baked, via_kwarg
            k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            raise SystemExit(
                f"render mismatch at trajectory {row['extra_info']['trajectory_id']} char {k}:\n"
                f"  baked : {a[max(0, k - 80):k + 80]!r}\n  kwarg : {b[max(0, k - 80):k + 80]!r}"
            )
    print(f"check_render: baked tools == template tools kwarg on {len(samples)} rows")


def token_report(rows: list[dict], tokenizer, kept_template: str | None, mode: str) -> None:
    lens, think = [], []
    for row in rows:
        kw = {"chat_template": kept_template} if (mode == "all" and kept_template) else {}
        txt = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False, **kw)
        lens.append(len(tokenizer(txt, add_special_tokens=False).input_ids))
        think.append(txt.count(THINK_OPEN))
    lens.sort()
    q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
    print(f"prompt tokens ({len(lens)} rows, reasoning={mode}): mean={statistics.fmean(lens):,.0f} "
          f"p50={q(.5):,} p90={q(.9):,} p99={q(.99):,} max={lens[-1]:,}")
    for cap in (4096, 8192, 16384):
        print(f"  over {cap:,}: {100 * sum(x > cap for x in lens) / len(lens):.2f}%")
    print(f"  <think> blocks rendered per prompt: mean {statistics.fmean(think):.2f}")


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    save_dir = Path(os.path.expanduser(args.local_save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    render_samples: list[tuple[dict, dict]] = []
    drops = Counter()
    n = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if args.limit is not None and n >= args.limit:
                break
            r = json.loads(line)
            n += 1
            try:
                row, aux = convert_row(r, n - 1, args)
            except Drop as e:
                drops[str(e)] += 1
                continue
            rows.append(row)
            if args.check_render and len(render_samples) < args.check_render and n % 97 == 0:
                render_samples.append((row, aux))

    kept = len(rows)
    print(f"source rows {n:,}   kept {kept:,} ({100 * kept / n:.2f}%)   dropped {n - kept:,}")
    for reason, c in drops.most_common():
        print(f"  {c:>7,}  {reason}")
    print(f"expected action types: {dict(Counter(r['extra_info']['expected_type'] for r in rows))}")

    tokenizer = None
    if args.check_render or args.token_sample:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    if args.check_render:
        check_render(render_samples, tokenizer)
    if args.token_sample:
        stride = max(1, kept // args.token_sample)
        kept_template = None
        if args.reasoning == "all":
            kept_template = args.kept_template.read_text(encoding="utf-8")
            kept_template = kept_template.replace("{% generation %}", "").replace("{% endgeneration %}", "")
        token_report(rows[::stride][: args.token_sample], tokenizer, kept_template, args.reasoning)

    # split by trajectory so a conversation never straddles train and val
    traj_ids = sorted({r["extra_info"]["trajectory_id"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(traj_ids)
    val_ids = set(traj_ids[: args.val_trajectories])
    train_rows = [r for r in rows if r["extra_info"]["trajectory_id"] not in val_ids]
    val_rows = [r for r in rows if r["extra_info"]["trajectory_id"] in val_ids]

    for name, subset in (("train", train_rows), ("val", val_rows)):
        if not subset:
            continue
        ds = datasets.Dataset.from_list(subset)
        path = save_dir / f"{DATA_SOURCE}_{args.reasoning}_{name}.parquet"
        ds.to_parquet(str(path))
        print(f"wrote {len(subset):>7,} rows -> {path}")

    manifest = {
        "input": str(args.input),
        "input_md5_first_row": hashlib.md5(open(args.input, "rb").readline()).hexdigest(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "source_rows": n,
        "kept": kept,
        "drops": dict(drops),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "val_trajectories": len(val_ids),
    }
    (save_dir / f"{DATA_SOURCE}_{args.reasoning}_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
