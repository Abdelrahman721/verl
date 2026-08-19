"""Sample medical prompts, generate candidate responses, and write the exact
text qa_bedrock would send to the judge — for manual paste-into-LLM inspection.

The script reuses qa_bedrock's prompt templates verbatim (imported, not copied)
so the manual inspection matches the production judge byte-for-byte.

Usage:
    # default: 10 rows, generate responses with DeepSeek-V3.2 via OpenRouter
    python -m full_mix.curriculum.check_medical_qa_judge_prompts

    # 5 conversation rows only, use the dataset's gold_response as the answer
    # (sanity check — should produce near-perfect judge scores)
    python -m full_mix.curriculum.check_medical_qa_judge_prompts \
        --n 5 --conv-only --response-source gold

    # leave a placeholder for the answer; fill in manually before pasting
    python -m full_mix.curriculum.check_medical_qa_judge_prompts \
        --response-source placeholder

Outputs:
    One <output-dir>/case_NN_<eval_mode>.txt per sample, with:
      [METADATA]                        — eval_mode, data_source, row index, etc.
      [ORIGINAL POLICY PROMPT]          — what the trained model would see
      [CANDIDATE RESPONSE]              — openrouter-generated / gold / placeholder
      [JUDGE PROMPT — SYSTEM / USER]    — paste into Claude exactly as-is

Env vars (for --response-source=openrouter, which is the default):
    RESPONSE_API_BASE  (default: https://openrouter.ai/api/v1)
    RESPONSE_API_KEY   (required; falls back to FULL_MIX_JUDGE_API_KEY if unset)
    RESPONSE_MODEL     (default: deepseek/deepseek-v3.2-exp)
    RESPONSE_MAX_TOKENS (default: 2048)
"""

import argparse
import asyncio
import json
import os
import random
import sys
from typing import Any

import pyarrow.parquet as pq


# Lazy imports so --response-source=placeholder works without anthropic installed
# (qa_bedrock imports anthropic at module load — only needed for the templates,
# but we still take the hit unless the user runs in the training container).
def _import_qa_bedrock_templates():
    from full_mix.rewards.qa_bedrock import (
        QA_JUDGE_SYSTEM_PROMPT,
        QA_JUDGE_USER_TEMPLATE,
        CONV_JUDGE_SYSTEM_PROMPT,
        CONV_JUDGE_USER_TEMPLATE,
    )
    return (
        QA_JUDGE_SYSTEM_PROMPT,
        QA_JUDGE_USER_TEMPLATE,
        CONV_JUDGE_SYSTEM_PROMPT,
        CONV_JUDGE_USER_TEMPLATE,
    )


# ---------------------------------------------------------------------------
# Response generation (OpenAI-compatible: OpenRouter / DeepSeek-V3.2)
# ---------------------------------------------------------------------------

RESPONSE_SYSTEM_PROMPT = (
    "You are an expert medical AI assistant. Answer the user's question "
    "accurately and concisely. Provide just the medical answer — do not "
    "include any meta-commentary about your reasoning process."
)


def _build_openai_client():
    """Construct an AsyncOpenAI client pointed at the response endpoint
    (OpenRouter by default). Reuses FULL_MIX_JUDGE_API_KEY if RESPONSE_API_KEY
    is unset, since the user has the OpenRouter key already wired up there."""
    from openai import AsyncOpenAI
    base_url = os.environ.get("RESPONSE_API_BASE", "https://openrouter.ai/api/v1")
    api_key = (
        os.environ.get("RESPONSE_API_KEY")
        or os.environ.get("FULL_MIX_JUDGE_API_KEY")
        or ""
    )
    if not api_key:
        raise RuntimeError(
            "no API key for response generation: set RESPONSE_API_KEY (or "
            "FULL_MIX_JUDGE_API_KEY) to your OpenRouter key (or pass "
            "--response-source placeholder / gold to skip generation)."
        )
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120.0)


async def _generate_one(client, prompt_messages: list[dict], max_tokens: int) -> str:
    """Generate a response for one prompt via the OpenAI-compatible endpoint.

    `prompt_messages` is the parquet row's `prompt` column — a list of
    {role, content} dicts. For QA rows this is a single user turn; for conv
    rows it's alternating user/assistant/user turns ending with a user
    message that the response should reply to.
    """
    model = os.environ.get("RESPONSE_MODEL", "deepseek/deepseek-v3.2-exp")
    msgs: list[dict] = [{"role": "system", "content": RESPONSE_SYSTEM_PROMPT}]
    for m in prompt_messages:
        msgs.append({"role": m["role"], "content": m["content"]})
    resp = await client.chat.completions.create(
        model=model,
        messages=msgs,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    choice = resp.choices[0]
    return (choice.message.content or "").strip()


async def _generate_all_openrouter(rows: list[dict], max_tokens: int) -> list[str]:
    client = _build_openai_client()
    try:
        # Modest concurrency to avoid hammering the endpoint and tripping rate
        # limits (OpenRouter applies per-key + per-model caps).
        sem = asyncio.Semaphore(8)
        async def _guarded(row):
            async with sem:
                return await _generate_one(client, row["prompt"], max_tokens)
        return await asyncio.gather(*(_guarded(r) for r in rows))
    finally:
        await client.close()


def _gold_response_for(row: dict) -> str:
    """Pick the dataset's gold-standard answer for sanity checking."""
    gt = row["reward_model"]["ground_truth"]
    if row["extra_info"]["eval_mode"] == "qa":
        return gt.get("gold_answer", "") or gt.get("gold_response", "")
    return gt.get("gold_response", "") or gt.get("gold_answer", "")


# ---------------------------------------------------------------------------
# Judge prompt formatting  (mirrors qa_bedrock exactly)
# ---------------------------------------------------------------------------

def _format_qa_judge(row: dict, answer: str, templates: tuple) -> tuple[str, str]:
    QA_SYS, QA_USR, _, _ = templates
    question = row["extra_info"].get("question", "") or row["prompt"][0]["content"]
    key_points = row["reward_model"]["ground_truth"].get("key_points", [])
    user_text = QA_USR.format(
        question=question,
        key_points_json=json.dumps(key_points, indent=2),
        answer=answer,
    )
    return QA_SYS, user_text


def _format_conv_judge(row: dict, answer: str, templates: tuple) -> tuple[str, str]:
    _, _, CONV_SYS, CONV_USR = templates
    gt = row["reward_model"]["ground_truth"]
    # Same resolution order as qa_bedrock._call_conv_judge: prefer the parquet's
    # pre-computed `context` / `latest_user` over the synthesized `_context` /
    # `_latest_user` (which would only matter if extra_info["prompt"] were set).
    context = gt.get("context", gt.get("_context", "(No prior context)"))
    latest_user = gt.get("latest_user", gt.get("_latest_user", ""))
    type_desc = gt.get("type_descriptor", "") or "No type descriptor available."
    user_text = CONV_USR.format(
        type_descriptor=type_desc,
        sub_index=gt.get("sub_index", 1),
        total_subs=gt.get("total_subs", 1),
        context=context,
        latest_user_message=latest_user,
        gold_response=gt.get("gold_response", ""),
        candidate_response=answer,
    )
    return CONV_SYS, user_text


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------

def _write_case(out_path: str, case_num: int, row: dict, response: str,
                response_source: str, templates: tuple) -> None:
    eval_mode = row["extra_info"]["eval_mode"]
    data_source = row["data_source"]
    if eval_mode == "qa":
        sys_prompt, user_text = _format_qa_judge(row, response, templates)
    else:
        sys_prompt, user_text = _format_conv_judge(row, response, templates)

    bar = "=" * 80
    lines: list[str] = []
    lines.append(bar)
    lines.append(f"CASE {case_num}   eval_mode: {eval_mode}   data_source: {data_source}")
    lines.append(f"row index: {row['extra_info'].get('index', '?')}   "
                 f"question_id: {row['extra_info'].get('question_id', '?')}")
    lines.append(bar)
    lines.append("")

    # Metadata
    lines.append("[METADATA]")
    for k in ("specialty", "difficulty", "audience", "source", "topic_group",
              "parent_conv_id", "sub_index", "total_subs"):
        v = row["extra_info"].get(k)
        if v not in (None, "", 0):
            lines.append(f"  {k}: {v}")
    lines.append("")

    # Original prompt (multi-turn for conv)
    lines.append("[ORIGINAL POLICY PROMPT]")
    for m in row["prompt"]:
        lines.append(f"  [{m['role'].upper()}]")
        for ln in m["content"].splitlines():
            lines.append(f"    {ln}")
        lines.append("")

    # Candidate response
    lines.append(f"[CANDIDATE RESPONSE — source: {response_source}]")
    for ln in (response or "(empty)").splitlines():
        lines.append(f"  {ln}")
    lines.append("")

    # The judge prompt itself
    lines.append(bar)
    lines.append("JUDGE PROMPT — paste the SYSTEM + USER blocks below into your LLM")
    lines.append(bar)
    lines.append("")
    lines.append(">>> SYSTEM <<<")
    lines.append(sys_prompt)
    lines.append("")
    lines.append(">>> USER <<<")
    lines.append(user_text)
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _sample_rows(t, n: int, seed: int, qa_only: bool, conv_only: bool) -> list[dict]:
    rng = random.Random(seed)
    order = list(range(t.num_rows))
    rng.shuffle(order)
    selected: list[dict] = []
    # Read in slices for efficiency — random rows from a 300k-row table.
    for i in order:
        row = t.slice(i, 1).to_pylist()[0]
        mode = row["extra_info"]["eval_mode"]
        if qa_only and mode != "qa":
            continue
        if conv_only and mode != "conversation":
            continue
        selected.append(row)
        if len(selected) >= n:
            break
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default="/data/abdelrahman/verl/data/medical_qa/train.parquet",
        help="Source parquet to sample from.",
    )
    ap.add_argument("--n", type=int, default=10, help="Number of cases to write.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for sampling.")
    ap.add_argument(
        "--output-dir", default="/tmp/medical_judge_prompts",
        help="Directory to write per-case text files into (created if missing).",
    )
    ap.add_argument(
        "--response-source", choices=["openrouter", "gold", "placeholder"],
        default="openrouter",
        help=(
            "Source of the candidate response. 'openrouter' generates fresh "
            "via DeepSeek-V3.2 by default (configurable via RESPONSE_MODEL / "
            "RESPONSE_API_BASE / RESPONSE_API_KEY). 'gold' uses the dataset's "
            "gold_response (sanity check). 'placeholder' writes a literal "
            "stub you replace manually."
        ),
    )
    ap.add_argument(
        "--max-tokens", type=int,
        default=int(os.environ.get("RESPONSE_MAX_TOKENS", "2048")),
        help="max_tokens for response generation (openrouter only).",
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--qa-only", action="store_true", help="Sample qa rows only.")
    g.add_argument("--conv-only", action="store_true", help="Sample conversation rows only.")
    args = ap.parse_args()

    if not os.path.exists(args.parquet):
        print(f"missing parquet: {args.parquet}", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)

    # Load templates verbatim from qa_bedrock so the formatted text matches what
    # the production judge sees byte-for-byte.
    templates = _import_qa_bedrock_templates()

    print(f"Loading {args.parquet}…")
    t = pq.read_table(args.parquet)
    print(f"  {t.num_rows} rows total")

    print(f"Sampling {args.n} rows (seed={args.seed})"
          f"{' [qa-only]' if args.qa_only else ''}"
          f"{' [conv-only]' if args.conv_only else ''}…")
    rows = _sample_rows(t, args.n, args.seed, args.qa_only, args.conv_only)
    if not rows:
        print("no rows matched the filter", file=sys.stderr)
        return 2
    n_qa = sum(1 for r in rows if r["extra_info"]["eval_mode"] == "qa")
    n_conv = len(rows) - n_qa
    print(f"  selected {len(rows)} rows: {n_qa} qa, {n_conv} conv")

    # Build the candidate responses
    print(f"Building responses (source={args.response_source})…")
    if args.response_source == "openrouter":
        model = os.environ.get("RESPONSE_MODEL", "deepseek/deepseek-v3.2")
        base = os.environ.get("RESPONSE_API_BASE", "https://openrouter.ai/api/v1")
        print(f"  model: {model}  endpoint: {base}")
        responses = asyncio.run(_generate_all_openrouter(rows, args.max_tokens))
    elif args.response_source == "gold":
        responses = [_gold_response_for(r) for r in rows]
    else:  # placeholder
        responses = [
            "<<< PASTE YOUR CANDIDATE RESPONSE HERE — this is what the trained "
            "model would generate for the user prompt above. The judge will "
            "score this text against the rubric below. >>>"
            for _ in rows
        ]

    # Write one file per case
    for i, (row, resp) in enumerate(zip(rows, responses), start=1):
        fname = f"case_{i:02d}_{row['extra_info']['eval_mode']}.txt"
        path = os.path.join(args.output_dir, fname)
        _write_case(path, i, row, resp, args.response_source, templates)
        print(f"  wrote {path}")

    print(f"\nDone — {len(rows)} cases written to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
