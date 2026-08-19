"""Export the dual-mode SFT records into the trainer's on-disk dataset format.

Target schema, matched against
`/data/abdelrahman/qwen-sft/full_scale/data/general/combined_sft_max32k_clean_v2`:

    messages     : list[{content: str, role: str}]   <- ONLY these two keys
    id           : str    "<source>_<n>"
    source       : str
    token_length : int64

The reference set carries the reasoning INSIDE the assistant `content`
(`<think>...</think>\\n\\n<answer>`), not in a separate field, so this exporter
folds `reasoning_content` into `content`. Instant-mode rows get the empty block
`<think>\\n\\n</think>\\n\\n` in exactly the same position. That keeps the data
trainable with the plain minimal chat template — no template change needed for
SFT — while `chat_template_dual_mode.jinja` remains useful at INFERENCE time for
forcing a mode via `enable_thinking`.

Note the reference set's own framing is inconsistent (`<think>\\nOkay`,
`<think>Okay`, `<think> Okay`, and both `</think>\\n` and `</think>\\n\\n`). Ours
is normalized to a single form, per the requirement that thinking and
non-thinking traces share one format.

Each prompt ends with a mode marker — `/think` when the response carries a
reasoning trace, `/no_think` when it does not — appended to the last
non-assistant turn. The chat template stays open-ended (it never pre-fills a
think block), so the model learns to emit `<think>...</think>` or the empty
`<think>\n\n</think>` itself in response to the marker. Disable with
--no_markers.

`token_length` is computed with the SFT model's tokenizer and its minimal chat
template — the rendering actually trained on. The reference's own numbers run
2-3 tokens higher (a different template/tokenizer revision), which is
immaterial for a length filter but means the columns are not bit-comparable.

Usage:
    python -m full_mix.sft_modes.export_for_sfttrainer
    python -m full_mix.sft_modes.export_for_sfttrainer --sources ifeval --max_tokens 32768
"""

import argparse
import json
import os

import pyarrow.parquet as pq

THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n\n"
EMPTY_BLOCK = "<think>\n\n</think>\n\n"

# data_source in the parquet -> the `source` string written to the dataset
SOURCE_NAMES = {
    "local/dolci-ifeval-32b": "dual-mode-ifeval-rl60",
    "local/dolci-chat-32b": "dual-mode-chat-rl60",
}


def append_mode_marker(messages: list[dict], thinking: bool, marker_think: str,
                       marker_no_think: str, sep: str) -> list[dict]:
    """Append /think or /no_think to the LAST non-assistant turn.

    The marker states the mode the response is in, so at inference the caller
    picks the mode by appending the same token — the chat template stays
    open-ended and the model emits its own think block either way.

    Separated onto its own line rather than inline because ~3.7% of the ifeval
    prompts say "first repeat the request word for word"; a marker on its own
    line reads as a control token rather than part of the request. The stored
    responses predate the markers and repeat the request WITHOUT one, so these
    rows actively teach the model to exclude it.
    """
    marker = marker_think if thinking else marker_no_think
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m["role"] != "assistant":
            m["content"] = m["content"].rstrip() + sep + marker
            return out
    raise ValueError("no non-assistant turn to attach a mode marker to")


def fold_reasoning(messages: list[dict]) -> list[dict]:
    """Collapse {content, reasoning_content} assistant turns into content-only.

    Thinking turns become  <think>\\n{reasoning}\\n</think>\\n\\n{answer}
    Instant turns become   <think>\\n\\n</think>\\n\\n{answer}
    Non-assistant turns are copied with only role/content retained.
    """
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            out.append({"role": m["role"], "content": m["content"]})
            continue
        reasoning = (m.get("reasoning_content") or "").strip()
        prefix = THINK_OPEN + reasoning + THINK_CLOSE if reasoning else EMPTY_BLOCK
        out.append({"role": "assistant", "content": prefix + m["content"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="/data/abdelrahman/verl/data/sft_dual_mode")
    ap.add_argument("--out", default="/data/abdelrahman/verl/data/sft_dual_mode/hf_dataset")
    ap.add_argument("--sources", nargs="+", default=["ifeval", "chat"])
    ap.add_argument("--model_path", default="/data/abdelrahman/qwen-sft/full_scale/experiment/final")
    ap.add_argument("--max_tokens", type=int, default=32768,
                    help="drop rows longer than this, matching the reference's max32k filter")
    ap.add_argument("--think_marker", default="/think")
    ap.add_argument("--no_think_marker", default="/no_think")
    ap.add_argument("--marker_sep", default="\n\n",
                    help="separator between the prompt and the mode marker; must match inference")
    ap.add_argument("--no_markers", action="store_true",
                    help="omit /think and /no_think (the pre-marker dataset format)")
    ap.add_argument("--shuffle_seed", type=int, default=42)
    ap.add_argument("--num_proc", type=int, default=32)
    args = ap.parse_args()

    import datasets
    from transformers import AutoTokenizer

    datasets.disable_progress_bars()
    tok = AutoTokenizer.from_pretrained(args.model_path)

    rows: list[dict] = []
    per_source: dict[str, int] = {}
    n_relabelled = 0
    for src in args.sources:
        path = os.path.join(args.in_dir, f"{src}_sft.parquet")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        recs = pq.read_table(path).to_pylist()
        name = None
        for i, rec in enumerate(recs):
            name = SOURCE_NAMES.get(rec["data_source"], rec["data_source"])
            msgs = fold_reasoning(rec["messages"])
            # Derive the mode from the RENDERED content, not rec["thinking"].
            # The two can disagree: parse_response only requires a non-empty
            # ANSWER, so a sample with an empty <think></think> is accepted and
            # select_and_split may still label it thinking by position. Reading
            # the content back makes marker and response agree by construction.
            asst = next(m for m in reversed(msgs) if m["role"] == "assistant")
            is_thinking = not asst["content"].startswith(EMPTY_BLOCK)
            if is_thinking != bool(rec["thinking"]):
                n_relabelled += 1
            if not args.no_markers:
                msgs = append_mode_marker(
                    msgs, is_thinking,
                    args.think_marker, args.no_think_marker, args.marker_sep,
                )
            rows.append(
                {
                    "messages": msgs,
                    "id": f"{name}_{i}",
                    "source": name,
                    "thinking": is_thinking,
                }
            )
        per_source[name or src] = len(recs)
        print(f"  {src}: {len(recs)} records from {path}")

    if not rows:
        raise SystemExit("no *_sft.parquet inputs found — run build_sft_dataset first")

    ds = datasets.Dataset.from_list(rows)

    def add_len(batch):
        texts = [tok.apply_chat_template(m, tokenize=False) for m in batch["messages"]]
        return {"token_length": [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]}

    ds = ds.map(add_len, batched=True, batch_size=256, num_proc=args.num_proc)

    n_before = len(ds)
    ds = ds.filter(lambda r: r["token_length"] <= args.max_tokens, num_proc=args.num_proc)
    n_dropped = n_before - len(ds)

    thinking = sum(ds["thinking"])
    ds = ds.remove_columns("thinking")  # not part of the target schema
    ds = ds.shuffle(seed=args.shuffle_seed)
    # Column order matches the reference dataset.
    ds = ds.select_columns(["messages", "id", "source", "token_length"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ds.save_to_disk(args.out)

    lengths = ds["token_length"]
    if n_relabelled:
        print(f"  relabelled        : {n_relabelled} record(s) whose `thinking` flag "
              f"disagreed with the rendered think block")
    print(f"\n  rows              : {len(ds)}  ({n_dropped} dropped over {args.max_tokens} tokens)")
    print(f"  thinking / instant: {thinking} / {len(ds) - thinking}")
    print(f"  token_length      : min {min(lengths)}  mean {sum(lengths) // len(lengths)}  max {max(lengths)}")
    print(f"  features          : {ds.features}")
    print(f"  saved -> {args.out}")

    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(
            {
                "kind": "sft",
                "rows": len(ds),
                "columns": ["messages", "id", "source", "token_length"],
                "thinking_rows": thinking,
                "instant_rows": len(ds) - thinking,
                "max_tokens": args.max_tokens,
                "dropped_over_max_tokens": n_dropped,
                "relabelled_from_content": n_relabelled,
                "think_marker": args.think_marker,
                "no_think_marker": args.no_think_marker,
                "marker_sep": args.marker_sep,
                "token_length_tokenizer": args.model_path,
                "sources": [{"source": k, "records_in": v} for k, v in per_source.items()],
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
