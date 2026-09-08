# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Build ONE single-step tool-use RL parquet from two sources, in the exact row format
nemotron_pivot_preprocess.py produces (data_source "nemotron_pivot", same scorer):

  1. the Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1 rows ("pivot"), converted by the
     original script's `convert_row`, unchanged;
  2. the Nemotron-Post-Training tool_calling SFT conversations ("sft", the HF dataset written by
     llm-pretrainer/dataset-converters/convert_nemotron_to_sft.py), re-cut so that EVERY assistant
     turn becomes one decision row: the context is everything before it, the expected action is
     the turn itself — a single call, a parallel batch when the turn made 2+ calls, or a message.

Every SFT assistant message is parsed into (reasoning, text, calls) and re-rendered through the
pivot script's `render_turns`, so history turns come out byte-identical to the SFT form the model
was trained on; the parse is verified by re-rendering under `--reasoning all` and comparing to the
original content, and any turn that does not round-trip drops its conversation (counted, never
patched). Tools are baked into the system message with the same tools block, verified by
`--check_render` against the tokenizer's tools kwarg on rows from BOTH sources.

--balance (default on) equalises three buckets by count:
    prose     : expected message, from the pivot rows ONLY (--prose_sources both|sft widens it)
    single    : expected single call, from the pivot rows ONLY (sft single-call turns are dropped)
    parallel  : expected batch of 2+ calls, from the sft rows (the pivot set has none)
The smallest bucket is taken whole; the other two are sampled without replacement to the same
size with --seed. --no-balance keeps every row from both sources.

Rows are split into train/val by conversation (pivot trajectory_id or sft row index), so no
conversation straddles the split. Output is streamed to parquet with a fixed Arrow schema, so a
1.5 M-row build never has to sit in memory as Python objects.

extra_info gains `source` ("pivot"|"sft"), `turn_index` (assistant-turn ordinal within the
conversation) and `n_calls`; pivot-only fields (pass_rate, qwen_235b_reward_mean) are null on sft
rows. sft trajectory ids are offset by SFT_TRAJ_OFFSET so they never collide with pivot ids.

Example:
  python examples/data_preprocess/nemotron_unified_preprocess.py \\
      --local_save_dir rl-data/nemotron_unified --reasoning all
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from nemotron_pivot_preprocess import (  # noqa: E402
    DATA_SOURCE, DEFAULT_INPUT, DEFAULT_KEPT_TEMPLATE, DEFAULT_TOKENIZER, THINK_CLOSE, THINK_OPEN,
    Drop, check_render, convert_row, has_control, render_turns, repair_schema, token_report, tools_block,
)

REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_SFT = REPO_ROOT.parent / "llm-pretrainer" / "sft_datasets" / "nemotron-tooling-sft"
DEFAULT_SYSTEM = "You are a helpful assistant."
SFT_TRAJ_OFFSET = 10_000_000
THINK_RE = re.compile(r"^" + re.escape(THINK_OPEN) + r"\n(.*?)\n" + re.escape(THINK_CLOSE) + r"(.*)$", re.DOTALL)

SCHEMA = pa.schema([
    ("data_source", pa.string()),
    ("prompt", pa.list_(pa.struct([
        ("content", pa.string()),
        ("role", pa.string()),
        ("tool_calls", pa.list_(pa.struct([("arguments", pa.string()), ("name", pa.string())]))),
    ]))),
    ("ability", pa.string()),
    ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
    ("extra_info", pa.struct([
        ("assistant_depth", pa.int64()),
        ("expected_type", pa.string()),
        ("index", pa.int64()),
        ("n_calls", pa.int64()),
        ("offered_tools", pa.list_(pa.string())),
        ("pass_rate", pa.float64()),
        ("qwen_235b_reward_mean", pa.float64()),
        ("reasoning_mode", pa.string()),
        ("source", pa.string()),
        ("trajectory_id", pa.int64()),
        ("turn_index", pa.int64()),
        ("user_turns", pa.int64()),
    ])),
])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pivot_input", type=Path, default=DEFAULT_INPUT, help="Pivot train.jsonl")
    p.add_argument("--sft_input", type=Path, default=DEFAULT_SFT, help="HF dataset dir of the converted SFT split")
    p.add_argument("--local_save_dir", default="/workspace/verl/rl-data/")
    p.add_argument("--reasoning", choices=["all", "official"], default="all")
    p.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True,
                   help="Equalise prose / single(pivot only) / parallel(sft) buckets (default on)")
    p.add_argument("--prose_sources", choices=["pivot", "both", "sft"], default="pivot",
                   help="Where the prose bucket is drawn from when balancing (default pivot: the RL source only; "
                        "'both' would be ~96%% sft prose because that pool is ~24x larger)")
    p.add_argument("--val_trajectories", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit_pivot", type=int, default=None, help="Smoke test: first N pivot rows")
    p.add_argument("--limit_sft", type=int, default=None, help="Smoke test: first N sft conversations")
    p.add_argument("--keep_unreasoned", action="store_true")
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--kept_template", type=Path, default=DEFAULT_KEPT_TEMPLATE)
    p.add_argument("--check_render", type=int, default=200, help="Rows per source to prove baked tools == tools kwarg")
    p.add_argument("--token_sample", type=int, default=2000)
    p.add_argument("--write_chunk", type=int, default=20_000)
    return p.parse_args()


# --------------------------------------------------------------------------- sft side

def parse_sft_assistant(m: dict) -> dict:
    """SFT assistant message -> {"role": "assistant", "reasoning", "text", "calls"} for render_turns."""
    content = m.get("content") or ""
    calls = [{"name": c["name"], "arguments": c["arguments"]} for c in (m.get("tool_calls") or [])]
    mt = THINK_RE.match(content)
    if mt:
        reasoning, rest = mt.group(1), mt.group(2)
        # undo the converter's spacing: '\n' + body (calls) / '\n\n' + body (prose) / '\n' alone (calls, no body)
        if calls:
            body = rest[1:] if rest.startswith("\n") else rest
        else:
            body = rest[2:] if rest.startswith("\n\n") else rest
    else:
        reasoning, body = None, content
    if reasoning is not None and not reasoning.strip():
        reasoning = None
    return {"role": "assistant", "reasoning": reasoning, "text": body, "calls": calls}


def sft_turns(messages: list[dict]) -> tuple[str, list[dict]]:
    """Whole SFT conversation -> (system_text, turns) where assistant turns are parsed dicts."""
    system = DEFAULT_SYSTEM
    turns = []
    for i, m in enumerate(messages):
        role = m["role"]
        if role == "system":
            if i != 0:
                raise Drop("system message not first")
            system = m.get("content") or DEFAULT_SYSTEM
        elif role in ("user", "tool"):
            turns.append({"role": role, "content": m.get("content") or ""})
        elif role == "assistant":
            turns.append(parse_sft_assistant(m))
        else:
            raise Drop(f"unknown role {role}")
    if not turns or turns[0]["role"] != "user":
        raise Drop("conversation does not start with a user message")
    return system, turns


def sft_roundtrip_ok(turns: list[dict], messages: list[dict]) -> bool:
    """Re-render every assistant turn under 'all' and compare with the original SFT content."""
    rendered = render_turns(turns, "all")
    originals = [m for m in messages if m["role"] != "system"]
    for r, o in zip(rendered, originals):
        if r["role"] == "assistant" and r["content"] != (o.get("content") or ""):
            return False
    return True


def sft_expected(turn: dict) -> tuple[dict, str]:
    calls = turn["calls"]
    if not calls:
        if has_control(turn["text"]):
            raise Drop("control string in expected message")
        return {"type": "message", "content": turn["text"]}, "prose"
    for c in calls:
        json.loads(c["arguments"])  # converter guarantees this; raise loudly otherwise
    if len(calls) == 1:
        return {"type": "function_call", "name": calls[0]["name"], "arguments": calls[0]["arguments"]}, "single"
    return {"type": "function_call_batch",
            "calls": [{"type": "function_call", "name": c["name"], "arguments": c["arguments"]} for c in calls]}, "parallel"


def sft_conversation_plan(conv_idx: int, messages: list[dict], args) -> tuple[str, list[dict], list[tuple[int, str]]]:
    """Parse once; return (system, turns, [(turn_position, category)]) for every assistant turn that
    is a valid decision row. Raises Drop for conversation-level defects."""
    system, turns = sft_turns(messages)
    if has_control(system):
        raise Drop("control string in system prompt")
    if not sft_roundtrip_ok(turns, messages):
        raise Drop("assistant turn does not round-trip through render_turns")
    plan = []
    reasoned_so_far = True
    for pos, t in enumerate(turns):
        if t["role"] != "assistant":
            continue
        if not args.keep_unreasoned and not reasoned_so_far:
            # an earlier assistant turn had no trace: every later row would carry it in context
            break
        try:
            _, cat = sft_expected(t)
        except Drop:
            cat = None
        if cat is not None:
            plan.append((pos, cat))
        if t["reasoning"] is None:
            reasoned_so_far = False
    return system, turns, plan


def build_sft_row(conv_idx: int, system: str, tools: list[dict], turns: list[dict], pos: int,
                  turn_index: int, running_index: int, args) -> tuple[dict, dict]:
    offered = {t["function"]["name"] for t in tools}
    expected, cat = sft_expected(turns[pos])
    history = turns[:pos]
    prompt = [{"role": "system", "content": system + "\n\n" + tools_block(tools)}]
    prompt += render_turns(history, args.reasoning)
    row = {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": "tool_use",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(expected, ensure_ascii=False)},
        "extra_info": {
            "index": running_index,
            "trajectory_id": SFT_TRAJ_OFFSET + conv_idx,
            "assistant_depth": sum(1 for t in history if t["role"] == "assistant"),
            "user_turns": sum(1 for t in history if t["role"] == "user"),
            "expected_type": expected["type"],
            "n_calls": len(turns[pos]["calls"]),
            "pass_rate": None,
            "qwen_235b_reward_mean": None,
            "offered_tools": sorted(offered),
            "reasoning_mode": args.reasoning,
            "source": "sft",
            "turn_index": turn_index,
        },
    }
    aux = {"system_text": system, "tools": tools, "turns": prompt[1:]}
    return row, aux


def load_sft_tools(tools_json: str) -> list[dict]:
    tools = json.loads(tools_json)
    out = []
    for t in tools:
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        out.append({"type": "function", "function": {
            "name": fn["name"], "description": fn.get("description", ""),
            "parameters": repair_schema(fn.get("parameters") or {"type": "object", "properties": {}})}})
    if not out:
        raise Drop("no tools")
    return out


# --------------------------------------------------------------------------- writer

class ShardedWriter:
    """Streams rows into train/val parquet files with a fixed schema."""

    def __init__(self, save_dir: Path, stem: str, chunk: int):
        self.paths = {name: save_dir / f"{stem}_{name}.parquet" for name in ("train", "val")}
        self.writers = {name: pq.ParquetWriter(str(p), SCHEMA) for name, p in self.paths.items()}
        self.buffers = {name: [] for name in self.paths}
        self.counts = Counter()
        self.chunk = chunk

    def add(self, name: str, row: dict):
        self.buffers[name].append(row)
        self.counts[name] += 1
        if len(self.buffers[name]) >= self.chunk:
            self.flush(name)

    def flush(self, name: str):
        if self.buffers[name]:
            self.writers[name].write_table(pa.Table.from_pylist(self.buffers[name], schema=SCHEMA))
            self.buffers[name] = []

    def close(self):
        for name in self.writers:
            self.flush(name)
            self.writers[name].close()


# --------------------------------------------------------------------------- main

def main():
    args = parse_args()
    for p in (args.pivot_input, args.sft_input):
        if not p.exists():
            raise FileNotFoundError(p)
    save_dir = Path(os.path.expanduser(args.local_save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- pass 1a: pivot rows (small enough to hold; they are needed whole anyway)
    pivot_rows: list[tuple[dict, dict]] = []
    pivot_drops = Counter()
    n_pivot = 0
    with open(args.pivot_input, encoding="utf-8") as f:
        for line in f:
            if args.limit_pivot is not None and n_pivot >= args.limit_pivot:
                break
            n_pivot += 1
            try:
                row, aux = convert_row(json.loads(line), n_pivot - 1, args)
            except Drop as e:
                pivot_drops[str(e)] += 1
                continue
            ei = row["extra_info"]
            ei.update({"source": "pivot", "turn_index": ei["assistant_depth"],
                       "n_calls": 1 if ei["expected_type"] == "function_call" else 0})
            pivot_rows.append((row, aux))
    pivot_cat = {"function_call": "single", "message": "prose"}
    pivot_by_cat = defaultdict(list)
    for i, (row, _) in enumerate(pivot_rows):
        pivot_by_cat[pivot_cat[row["extra_info"]["expected_type"]]].append(i)
    print(f"pivot: source rows {n_pivot:,}  kept {len(pivot_rows):,}  dropped {n_pivot - len(pivot_rows):,}  "
          f"{ {k: len(v) for k, v in pivot_by_cat.items()} }")
    for reason, c in pivot_drops.most_common():
        print(f"  {c:>7,}  {reason}")

    # ---- pass 1b: sft plan (parse every conversation once, keep only the plan)
    from datasets import load_from_disk
    sft = load_from_disk(str(args.sft_input))
    n_sft = sft.num_rows if args.limit_sft is None else min(args.limit_sft, sft.num_rows)
    sft_plan: dict[int, list[tuple[int, str]]] = {}
    sft_by_cat = defaultdict(list)  # cat -> [(conv_idx, pos)]
    sft_drops = Counter()
    sft_turns_seen = 0
    for conv_idx in range(n_sft):
        rec = sft[conv_idx]
        try:
            _, _, plan = sft_conversation_plan(conv_idx, rec["messages"], args)
            load_sft_tools(rec["tools"])
        except Drop as e:
            sft_drops[str(e)] += 1
            continue
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            sft_drops[f"unparseable: {type(e).__name__}"] += 1
            continue
        sft_turns_seen += sum(1 for m in rec["messages"] if m["role"] == "assistant")
        if not plan:
            sft_drops["no usable assistant turn"] += 1
            continue
        sft_plan[conv_idx] = plan
        for pos, cat in plan:
            sft_by_cat[cat].append((conv_idx, pos))
    print(f"sft: conversations {n_sft:,}  usable {len(sft_plan):,}  assistant turns seen {sft_turns_seen:,}  "
          f"decision rows available { {k: len(v) for k, v in sft_by_cat.items()} }")
    for reason, c in sft_drops.most_common():
        print(f"  {c:>7,}  {reason}")

    # ---- selection
    if args.balance:
        pools = {
            "prose": ([("pivot", i) for i in pivot_by_cat["prose"]] if args.prose_sources in ("both", "pivot") else [])
                     + ([("sft", k) for k in sft_by_cat["prose"]] if args.prose_sources in ("both", "sft") else []),
            "single": [("pivot", i) for i in pivot_by_cat["single"]],
            "parallel": [("sft", k) for k in sft_by_cat["parallel"]],
        }
        n_target = min(len(v) for v in pools.values())
        selected = []
        for cat, pool in pools.items():
            take = pool if len(pool) == n_target else rng.sample(pool, n_target)
            selected += [(src, key, cat) for src, key in take]
        print(f"balance: pools { {k: len(v) for k, v in pools.items()} } -> {n_target:,} each, "
              f"{len(selected):,} rows; sft single-call turns excluded: {len(sft_by_cat['single']):,}")
    else:
        selected = [("pivot", i, pivot_cat[pivot_rows[i][0]["extra_info"]["expected_type"]]) for i in range(len(pivot_rows))]
        selected += [("sft", k, cat) for cat, keys in sft_by_cat.items() for k in keys]
        print(f"no balance: {len(selected):,} rows")

    # ---- val split by conversation
    conv_key = lambda src, key: ("pivot", pivot_rows[key][0]["extra_info"]["trajectory_id"]) if src == "pivot" else ("sft", key[0])
    convs = sorted({conv_key(src, key) for src, key, _ in selected})
    rng.shuffle(convs)
    val_convs = set(convs[: args.val_trajectories])

    # ---- pass 2: materialise in source order (sft conversations parsed once each)
    stem = f"nemotron_unified_{'balanced' if args.balance else 'full'}_{args.reasoning}"
    writer = ShardedWriter(save_dir, stem, args.write_chunk)
    render_samples = {"pivot": [], "sft": []}
    token_rows = []
    cat_counts = Counter()
    running = 0
    stride = max(1, len(selected) // args.token_sample) if args.token_sample else None

    sel_pivot = [(key, cat) for src, key, cat in selected if src == "pivot"]
    for i, cat in sel_pivot:
        row, aux = pivot_rows[i]
        row["extra_info"]["index"] = running
        split = "val" if conv_key("pivot", i) in val_convs else "train"
        writer.add(split, row)
        cat_counts[(split, cat)] += 1
        if args.check_render and len(render_samples["pivot"]) < args.check_render and running % 97 == 0:
            render_samples["pivot"].append((row, aux))
        if stride and running % stride == 0:
            token_rows.append(row)
        running += 1
    del pivot_rows

    sel_sft = defaultdict(list)
    for src, key, cat in selected:
        if src == "sft":
            sel_sft[key[0]].append((key[1], cat))
    for conv_idx in sorted(sel_sft):
        rec = sft[conv_idx]
        system, turns, _ = sft_conversation_plan(conv_idx, rec["messages"], args)
        tools = load_sft_tools(rec["tools"])
        turn_ordinal = {pos: k for k, pos in enumerate(p for p, _ in sft_plan[conv_idx])}
        split = "val" if ("sft", conv_idx) in val_convs else "train"
        for pos, cat in sorted(sel_sft[conv_idx]):
            row, aux = build_sft_row(conv_idx, system, tools, turns, pos, turn_ordinal[pos], running, args)
            writer.add(split, row)
            cat_counts[(split, cat)] += 1
            if args.check_render and len(render_samples["sft"]) < args.check_render and running % 97 == 0:
                render_samples["sft"].append((row, aux))
            if stride and running % stride == 0:
                token_rows.append(row)
            running += 1
    writer.close()

    print(f"wrote {writer.counts['train']:,} train / {writer.counts['val']:,} val rows -> {save_dir}/{stem}_*.parquet")
    for (split, cat), c in sorted(cat_counts.items()):
        print(f"  {split:<5} {cat:<8} {c:>9,}")

    tokenizer = None
    if args.check_render or args.token_sample:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    if args.check_render:
        for src, samples in render_samples.items():
            print(f"[{src}] ", end="")
            check_render(samples, tokenizer)
    if args.token_sample and token_rows:
        kept_template = None
        if args.reasoning == "all":
            kept_template = args.kept_template.read_text(encoding="utf-8")
            kept_template = kept_template.replace("{% generation %}", "").replace("{% endgeneration %}", "")
        token_report(token_rows[: args.token_sample], tokenizer, kept_template, args.reasoning)

    manifest = {
        "pivot_input": str(args.pivot_input), "sft_input": str(args.sft_input),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "pivot": {"source_rows": n_pivot, "kept": sum(len(v) for v in pivot_by_cat.values()), "drops": dict(pivot_drops),
                  "by_category": {k: len(v) for k, v in pivot_by_cat.items()}},
        "sft": {"conversations": n_sft, "usable": len(sft_plan), "assistant_turns_seen": sft_turns_seen,
                "drops": dict(sft_drops), "decision_rows_available": {k: len(v) for k, v in sft_by_cat.items()}},
        "selected": len(selected),
        "written": {f"{s}/{c}": n for (s, c), n in cat_counts.items()},
        "val_conversations": len(val_convs),
    }
    (save_dir / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
