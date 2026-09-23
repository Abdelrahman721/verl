# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Preprocess the unified-dataset RL cut (tooling-research/unified-dataset/rl-300k, copied to
Datasets/rl-300k) to verl parquet, in the same row format as nemotron_pivot_preprocess.py but under
its OWN data_source, "olive_rl300k", so it can be given its own reward.

The source is already one decision per row: `messages` is the prefix (roles user / assistant /
tool, with reasoning, content and tool_calls as separate fields), `expected_action` is the expert's
next turn (message, one call, or a parallel batch), `tools` is the row's own tool list, `system` is
the source's own system prompt or "" where it was harness protocol. Seven corpora, 300,000 rows.

What this script does to each row:

  * bakes the row's tools into the system message with Qwen's exact tools preamble (verl renders
    with a global tool list only). Rows that deliberately offer no tools (the "no-tools" class)
    get the bare preamble, exactly as Qwen's template renders `tools=None`. Proved byte-identical
    to the template's tools kwarg by --check_render on rows from EVERY source.
  * re-inlines each prior assistant turn's reasoning as "<think>\\n{trace}\\n</think>" followed by
    prose or tool calls, in the SFT converters' exact spacing (render_turns from the pivot script).
    --reasoning all (default) keeps every prior trace; --reasoning official keeps traces only after
    the last user message, which is what Qwen3's serving template shows.
  * repairs tool schemas missing "type": "object" (322 instances), keeps duplicate tool names
    (a source quirk the SFT model also saw).
  * writes ground_truth as {"type": "message", "content"} | {"type": "function_call", name,
    arguments} | {"type": "function_call_batch", calls: [...]}, the shapes the pivot scorer reads.

Rows are DROPPED, never patched, when: the prefix does not start on a user turn or ends on an
assistant turn; any prior assistant turn has no reasoning (0 in this build; --keep_unreasoned
keeps them); a control string (<tool_call>, <think>, ...) appears anywhere in the prefix, system
prompt or expected action; the expected message is empty; an expected call names a tool the row
does not offer or has unparseable arguments; the tool list does not parse.

extra_info carries the source's own metadata so a reward can branch on it: `source`, `subset`,
`leg`, `verifier` (argument-comparison | freeform-command | message), `labels`, `chain_depth`,
`turn_index`, `n_assistant_total`, `task_id`, `conv_id`. `trajectory_id` is a 63-bit hash of
`conv_id`, and train/val are split on it so no conversation straddles the split.

--render_dir writes one rendered prompt per source, exactly as the policy sees it at rollout time
(tooling template, generation tags stripped, generation prompt appended), followed by the hidden
expected action.

Example:
  python examples/data_preprocess/olive_rl300k_preprocess.py \\
      --local_save_dir rl-data/olive_rl300k --render_dir rl-data/olive_rl300k/renders
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from nemotron_pivot_preprocess import (  # noqa: E402
    DEFAULT_KEPT_TEMPLATE, DEFAULT_TOKENIZER, THINK_OPEN, Drop, check_render, has_control,
    render_turns, repair_schema, tools_block,
)

DATA_SOURCE = "olive_rl300k"
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT = REPO_ROOT / "Datasets" / "rl-300k"
DEFAULT_SYSTEM = "You are a helpful assistant."

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
        ("chain_depth", pa.int64()),
        ("conv_id", pa.string()),
        ("expected_type", pa.string()),
        ("index", pa.int64()),
        ("labels", pa.list_(pa.string())),
        ("leg", pa.string()),
        ("n_assistant_total", pa.int64()),
        ("n_calls", pa.int64()),
        ("offered_tools", pa.list_(pa.string())),
        ("reasoning_mode", pa.string()),
        ("row_id", pa.string()),
        ("source", pa.string()),
        ("source_est_tokens", pa.int64()),
        ("subset", pa.string()),
        ("task_id", pa.string()),
        ("trajectory_id", pa.int64()),
        ("turn_index", pa.int64()),
        ("user_turns", pa.int64()),
        ("verifier", pa.string()),
    ])),
])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="rl-300k directory (one sub-dir per source)")
    p.add_argument("--local_save_dir", default="/workspace/verl/rl-data/")
    p.add_argument("--reasoning", choices=["all", "official"], default="all")
    p.add_argument("--sources", nargs="*", default=None, help="Subset of source directories (default all)")
    p.add_argument("--limit_per_source", type=int, default=None, help="Smoke test: first N rows per source")
    p.add_argument("--val_conversations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep_unreasoned", action="store_true")
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--kept_template", type=Path, default=DEFAULT_KEPT_TEMPLATE)
    p.add_argument("--check_render", type=int, default=100, help="Rows PER SOURCE to prove baked tools == tools kwarg")
    p.add_argument("--token_sample", type=int, default=3000)
    p.add_argument("--render_dir", type=Path, default=None, help="Write one rendered prompt per source here")
    p.add_argument("--render_max_tokens", type=int, default=6000, help="Prefer a render row under this many tokens")
    p.add_argument("--write_chunk", type=int, default=20_000)
    return p.parse_args()


# --------------------------------------------------------------------------- row conversion

def traj_id(conv_id: str) -> int:
    return int.from_bytes(hashlib.md5(conv_id.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def load_tools(tools_json: str) -> list[dict]:
    try:
        tools = json.loads(tools_json)
    except json.JSONDecodeError as e:
        raise Drop("tools do not parse") from e
    out = []
    for t in tools:
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if not isinstance(fn.get("name"), str):
            raise Drop("tool without a name")
        out.append({"type": "function", "function": {
            "name": fn["name"], "description": fn.get("description", "") or "",
            "parameters": repair_schema(fn.get("parameters") or {"type": "object", "properties": {}})}})
    return out


def parse_calls(raw) -> list[dict]:
    calls = raw if isinstance(raw, list) else (json.loads(raw) if raw else [])
    out = []
    for c in calls:
        args = c.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args if args is not None else {}, ensure_ascii=False)
        try:
            json.loads(args)
        except json.JSONDecodeError as e:
            raise Drop("call arguments unparseable") from e
        if not isinstance(c.get("name"), str) or not c["name"]:
            raise Drop("call without a name")
        out.append({"name": c["name"], "arguments": args})
    return out


def build_turns(messages: list[dict], args) -> list[dict]:
    turns = []
    for m in messages:
        role = m["role"]
        if role == "user":
            if has_control(m["content"]):
                raise Drop("control string in user turn")
            turns.append({"role": "user", "content": m["content"]})
        elif role == "tool":
            if has_control(m["content"]):
                raise Drop("control string in tool output")
            turns.append({"role": "tool", "content": m["content"]})
        elif role == "assistant":
            reasoning = (m.get("reasoning") or "").strip()
            text = (m.get("content") or "").strip()
            if has_control(reasoning) or has_control(text):
                raise Drop("control string in assistant turn")
            if not reasoning and not args.keep_unreasoned:
                raise Drop("unreasoned prior assistant turn")
            turns.append({"role": "assistant", "reasoning": reasoning or None, "text": text,
                          "calls": parse_calls(m.get("tool_calls"))})
        else:
            raise Drop(f"unknown role {role}")
    if not turns or turns[0]["role"] != "user":
        raise Drop("prefix does not start on a user turn")
    if turns[-1]["role"] == "assistant":
        raise Drop("prefix ends on an assistant turn")
    return turns


def build_expected(ea: dict, offered: set[str]) -> tuple[dict, int]:
    calls = parse_calls(ea.get("tool_calls"))
    if has_control(ea.get("content") or ""):
        raise Drop("control string in expected action")
    if not calls:
        content = (ea.get("content") or "").strip()
        if not content:
            raise Drop("empty expected message")
        return {"type": "message", "content": content}, 0
    for c in calls:
        if c["name"] not in offered:
            raise Drop("expected call not offered")
    if len(calls) == 1:
        return {"type": "function_call", "name": calls[0]["name"], "arguments": calls[0]["arguments"]}, 1
    return {"type": "function_call_batch",
            "calls": [{"type": "function_call", "name": c["name"], "arguments": c["arguments"]} for c in calls]}, len(calls)


def convert_row(r: dict, running_index: int, args) -> tuple[dict, dict]:
    system = (r.get("system") or "").strip() or DEFAULT_SYSTEM
    if has_control(system):
        raise Drop("control string in system prompt")
    tools = load_tools(r["tools"])
    offered = {t["function"]["name"] for t in tools}
    turns = build_turns(r["messages"], args)
    try:
        ea = json.loads(r["expected_action"])
    except json.JSONDecodeError as e:
        raise Drop("expected_action does not parse") from e
    expected, n_calls = build_expected(ea, offered)

    system_content = system + "\n\n" + tools_block(tools) if tools else system
    prompt = [{"role": "system", "content": system_content}] + render_turns(turns, args.reasoning)
    row = {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": "tool_use",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(expected, ensure_ascii=False)},
        "extra_info": {
            "assistant_depth": sum(1 for t in turns if t["role"] == "assistant"),
            "chain_depth": int(r.get("chain_depth") or 0),
            "conv_id": r["conv_id"],
            "expected_type": expected["type"],
            "index": running_index,
            "labels": list(r.get("labels") or []),
            "leg": r.get("leg") or "",
            "n_assistant_total": int(r.get("n_assistant_total") or 0),
            "n_calls": n_calls,
            "offered_tools": sorted(offered),
            "reasoning_mode": args.reasoning,
            "row_id": r["id"],
            "source": r["source"],
            "source_est_tokens": int(r.get("est_tokens") or 0),
            "subset": r.get("subset") or "",
            "task_id": r.get("task_id") or "",
            "trajectory_id": traj_id(r["conv_id"]),
            "turn_index": int(r.get("turn_index") or 0),
            "user_turns": sum(1 for t in turns if t["role"] == "user"),
            "verifier": r.get("verifier") or "",
        },
    }
    aux = {"system_text": system, "tools": tools, "turns": prompt[1:]}
    return row, aux


# --------------------------------------------------------------------------- output

class ShardedWriter:
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


def token_report(rows: list[dict], tokenizer, kept_template: str | None, mode: str) -> list[int]:
    lens, think = [], []
    for row in rows:
        kw = {"chat_template": kept_template} if (mode == "all" and kept_template) else {}
        txt = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False, **kw)
        lens.append(len(tokenizer(txt, add_special_tokens=False).input_ids))
        think.append(txt.count(THINK_OPEN))
    s = sorted(lens)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    print(f"prompt tokens ({len(s)} rows, reasoning={mode}): mean={statistics.fmean(s):,.0f} "
          f"p50={q(.5):,} p90={q(.9):,} p99={q(.99):,} max={s[-1]:,}")
    for cap in (4096, 8192, 16384, 32768):
        print(f"  over {cap:,}: {100 * sum(x > cap for x in s) / len(s):.2f}%")
    print(f"  <think> blocks rendered per prompt: mean {statistics.fmean(think):.2f}")
    return lens


def write_render(path: Path, row: dict, tokenizer, kept_template: str | None, mode: str) -> int:
    kw = {"chat_template": kept_template} if (mode == "all" and kept_template) else {}
    txt = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False, **kw)
    n_tok = len(tokenizer(txt, add_special_tokens=False).input_ids)
    ei = row["extra_info"]
    gt = json.loads(row["reward_model"]["ground_truth"])
    head = (f"# source={ei['source']} subset={ei['subset']} row_id={ei['row_id']}\n"
            f"# prior assistant turns={ei['assistant_depth']} user turns={ei['user_turns']} "
            f"labels={ei['labels']} verifier={ei['verifier']} prompt tokens={n_tok:,}\n"
            f"# ---- WHAT THE POLICY SEES (prompt, ends at the generation header) ----\n")
    tail = ("\n# ---- HIDDEN FROM THE POLICY: expected action the reward scores against ----\n"
            + json.dumps(gt, indent=2, ensure_ascii=False) + "\n")
    path.write_text(head + txt + tail, encoding="utf-8")
    return n_tok


def main():
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(args.input)
    save_dir = Path(os.path.expanduser(args.local_save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    sources = sorted(d.name for d in args.input.iterdir() if d.is_dir())
    if args.sources:
        sources = [s for s in sources if s in set(args.sources)]

    # pass 1: convert everything, keep rows in memory (300k rows ≈ 1 GB of JSON, fits), collect stats
    kept: list[tuple[dict, dict]] = []
    drops = defaultdict(Counter)
    n_src = Counter()
    for src in sources:
        n = 0
        for f in sorted(glob.glob(str(args.input / src / "*.parquet"))):
            for r in pq.read_table(f).to_pylist():
                if args.limit_per_source is not None and n >= args.limit_per_source:
                    break
                n += 1
                try:
                    kept.append(convert_row(r, len(kept), args))
                except Drop as e:
                    drops[src][str(e)] += 1
        n_src[src] = n
        print(f"{src:<24} read {n:>8,}  kept {n - sum(drops[src].values()):>8,}  dropped {sum(drops[src].values()):>6,}"
              + ("" if not drops[src] else "  " + ", ".join(f"{v} {k}" for k, v in drops[src].most_common())))
    print(f"total read {sum(n_src.values()):,}  kept {len(kept):,}")

    # split by conversation
    convs = sorted({row["extra_info"]["trajectory_id"] for row, _ in kept})
    rng.shuffle(convs)
    val_convs = set(convs[: args.val_conversations])

    # pass 2: write
    stem = f"{DATA_SOURCE}_{args.reasoning}"
    writer = ShardedWriter(save_dir, stem, args.write_chunk)
    by_split = defaultdict(Counter)
    render_samples = defaultdict(list)
    token_rows = []
    stride = max(1, len(kept) // args.token_sample) if args.token_sample else None
    for i, (row, aux) in enumerate(kept):
        row["extra_info"]["index"] = i
        ei = row["extra_info"]
        split = "val" if ei["trajectory_id"] in val_convs else "train"
        writer.add(split, row)
        by_split[split][ei["expected_type"]] += 1
        by_split[split]["source=" + ei["source"]] += 1
        for lb in ei["labels"]:
            by_split[split]["label=" + lb] += 1
        if args.check_render and len(render_samples[ei["source"]]) < args.check_render and i % 7 == 0:
            render_samples[ei["source"]].append((row, aux))
        if stride and i % stride == 0:
            token_rows.append(row)
    writer.close()
    print(f"\nwrote {writer.counts['train']:,} train / {writer.counts['val']:,} val rows -> {save_dir}/{stem}_*.parquet")
    for split in ("train", "val"):
        c = by_split[split]
        print(f"  {split}: " + ", ".join(f"{k}={v:,}" for k, v in sorted(c.items()) if "=" not in k))
        print(f"        " + ", ".join(f"{k.split('=', 1)[1]}={v:,}" for k, v in sorted(c.items()) if k.startswith("source=")))
        print(f"        " + ", ".join(f"{k.split('=', 1)[1]}={v:,}" for k, v in sorted(c.items()) if k.startswith("label=")))

    tokenizer = None
    kept_template = None
    if args.check_render or args.token_sample or args.render_dir:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
        if args.reasoning == "all":
            kept_template = args.kept_template.read_text(encoding="utf-8")
            kept_template = kept_template.replace("{% generation %}", "").replace("{% endgeneration %}", "")
    if args.check_render:
        for src, samples in render_samples.items():
            print(f"[{src}] ", end="")
            check_render(samples, tokenizer)
    if args.token_sample and token_rows:
        token_report(token_rows[: args.token_sample], tokenizer, kept_template, args.reasoning)

    if args.render_dir:
        args.render_dir.mkdir(parents=True, exist_ok=True)
        # per source: prefer ≥2 prior assistant turns, a tool result in context, then the smallest prompt
        best = {}
        for row, _ in kept:
            ei = row["extra_info"]
            has_tool_msg = any(m["role"] == "tool" for m in row["prompt"])
            key = (ei["assistant_depth"] >= 2, has_tool_msg, ei["expected_type"] != "message",
                   -max(ei["source_est_tokens"], 1))
            if ei["source"] not in best or key > best[ei["source"]][0]:
                if ei["source_est_tokens"] <= args.render_max_tokens or ei["source"] not in best:
                    best[ei["source"]] = (key, row)
        for src, (_, row) in sorted(best.items()):
            n_tok = write_render(args.render_dir / f"{src}.txt", row, tokenizer, kept_template, args.reasoning)
            print(f"render: {args.render_dir / (src + '.txt')}  ({n_tok:,} tokens)")

    manifest = {
        "input": str(args.input),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "read_per_source": dict(n_src),
        "drops_per_source": {s: dict(c) for s, c in drops.items()},
        "kept": len(kept),
        "train_rows": writer.counts["train"],
        "val_rows": writer.counts["val"],
        "val_conversations": len(val_convs),
        "by_split": {s: dict(c) for s, c in by_split.items()},
    }
    (save_dir / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
