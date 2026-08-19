"""Generate N sampled responses per prompt from a vLLM server.

Reads the held-out prompt parquets (data/chat_ifeval_remaining by default) and
writes one JSONL line per PROMPT holding all N samples, so the downstream
verifier can pick correct ones per prompt.

Requests go to /v1/chat/completions so the SERVER applies the model's own
chat template — the same rendering the model was RL'd under. Do NOT point the
server at chat_template_dual_mode.jinja; that template is for the SFT output
only, and using it here would change the generation prompt.

Resumable: prompt_uids already present in the output file are skipped, so an
interrupted run continues where it stopped. Safe to re-run.

Usage (server must already be up — see serve_dp8.sh):
    python -m full_mix.sft_modes.generate_responses --source ifeval
    python -m full_mix.sft_modes.generate_responses --source chat --limit 200
"""

import argparse
import asyncio
import json
import os
import time

import pyarrow.parquet as pq

from full_mix.curriculum.encoding import DATASET_ID, UID_DATASET_STRIDE, base_data_source

SOURCES = {
    "chat": "chat_with_baseline_train.parquet",
    "ifeval": "ifeval_train.parquet",
}

# Samples per prompt, per source. ifeval draws 16 because "correct" there means
# ALL constraints satisfied at once — 77% of prompts produced nothing at n=8,
# so the extra draws buy coverage. chat's bar (beat the baseline) is met far
# more often, and each chat sample costs a judge call, so it stays at 8.
DEFAULT_N = {"chat": 8, "ifeval": 16}


def load_prompts(path: str, limit: int | None = None) -> list[dict]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    out = []
    for row in rows:
        info = row.get("extra_info") or {}
        base = base_data_source(row["data_source"])
        uid = DATASET_ID[base] * UID_DATASET_STRIDE + int(info["index"])
        out.append(
            {
                "prompt_uid": uid,
                "data_source": row["data_source"],
                "messages": [
                    {"role": m["role"], "content": m["content"]} for m in row["prompt"]
                ],
                "ground_truth": row["reward_model"]["ground_truth"],
                "user_prompt": info.get("user_prompt") or row["prompt"][-1]["content"],
                "baseline_response": info.get("baseline_response"),
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def load_done_uids(path: str) -> set[int]:
    """prompt_uids already written. Truncated final lines are ignored."""
    done: set[int] = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["prompt_uid"]))
            except Exception:
                continue
    return done


async def generate_one(client, sem, rec: dict, args) -> dict | None:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=args.model,
                messages=rec["messages"],
                n=args.n,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                # REQUIRED. Without an explicit seed this server returns n
                # byte-identical samples regardless of temperature (verified:
                # 12 distinct out of 96 at temperature=1.0, and still 1/4 at
                # temperature=1.5). Seeding per prompt restores full diversity
                # (8/8 distinct) and keeps the run reproducible.
                seed=args.seed + int(rec["prompt_uid"]),
            )
        except Exception as e:
            return {"prompt_uid": rec["prompt_uid"], "error": f"{type(e).__name__}: {e}"[:300]}
    responses = [c.message.content or "" for c in resp.choices]
    return {
        "n_distinct": len(set(responses)),
        "prompt_uid": rec["prompt_uid"],
        "data_source": rec["data_source"],
        "user_prompt": rec["user_prompt"],
        "ground_truth": rec["ground_truth"],
        "baseline_response": rec["baseline_response"],
        "messages": rec["messages"],
        "responses": responses,
        "finish_reasons": [c.finish_reason for c in resp.choices],
    }


async def run(args) -> int:
    from openai import AsyncOpenAI

    src_path = os.path.join(args.data_dir, SOURCES[args.source])
    prompts = load_prompts(src_path)
    total = len(prompts)
    # Interleaved sharding (i::N), not contiguous blocks: prompt difficulty is
    # correlated with position in these parquets, so contiguous halves would
    # finish at different times. This workload needs no cross-machine
    # communication — each shard is a fully independent server + client.
    if args.num_shards > 1:
        prompts = prompts[args.shard_index :: args.num_shards]
    if args.limit:
        prompts = prompts[: args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = load_done_uids(args.out)
    todo = [p for p in prompts if p["prompt_uid"] not in done]

    print(f"source        : {src_path}")
    print(f"shard         : {args.shard_index}/{args.num_shards}  "
          f"({len(prompts)} of {total} prompts)")
    print(f"output        : {args.out}")
    print(f"prompts       : {len(prompts)}  ({len(done)} already done, {len(todo)} to go)")
    print(f"samples/prompt: {args.n}   -> {len(todo) * args.n} generations")
    if not todo:
        print("nothing to do")
        return 0

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    sem = asyncio.Semaphore(args.concurrency)
    start = time.time()
    n_err = 0

    with open(args.out, "a") as fout:
        tasks = [asyncio.create_task(generate_one(client, sem, r, args)) for r in todo]
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            rec = await fut
            if rec is None:
                continue
            if "error" in rec:
                n_err += 1
                if n_err <= 5:
                    print(f"  [error] uid={rec['prompt_uid']}: {rec['error']}")
                continue
            fout.write(json.dumps(rec) + "\n")
            if i % 50 == 0:
                fout.flush()
                el = time.time() - start
                rate = i / el
                eta = (len(todo) - i) / rate / 3600 if rate else 0
                print(f"  {i}/{len(todo)} prompts  {rate:.2f} prompt/s  eta {eta:.1f}h  errors={n_err}", flush=True)

    print(f"\ndone: {len(todo) - n_err} prompts written, {n_err} errors, "
          f"{(time.time() - start) / 60:.1f} min -> {args.out}")
    return 1 if n_err else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data/abdelrahman/verl/data/chat_ifeval_remaining")
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    ap.add_argument("--out", default=None, help="default: <out_dir>/<source>_generations.jsonl")
    ap.add_argument("--out_dir", default="/data/abdelrahman/verl/data/sft_dual_mode/raw")
    ap.add_argument("--model", default="/data/abdelrahman/verl/checkpoints/RL-Exps/chat-ifeval-sync-4b/global_step_60/merged_hf_model")
    ap.add_argument("--base_url", default="http://127.0.0.1:8300/v1")
    ap.add_argument("--n", type=int, default=None,
                    help=f"samples per prompt; defaults per source {DEFAULT_N}")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234,
                    help="base seed; each prompt uses seed+prompt_uid (see generate_one)")
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--concurrency", type=int, default=64,
                    help="in-flight PROMPTS (each is n samples), not requests")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=1,
                    help="split the prompt set across this many machines")
    ap.add_argument("--shard_index", type=int, default=0,
                    help="which shard THIS machine handles, 0-based")
    args = ap.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit(f"shard_index must be in [0, {args.num_shards}); got {args.shard_index}")
    if args.n is None:
        args.n = DEFAULT_N[args.source]
    if args.out is None:
        # /data is a shared network mount, so both machines see the same path —
        # the shard suffix is what keeps them from clobbering each other.
        suffix = "" if args.num_shards == 1 else f".shard{args.shard_index}of{args.num_shards}"
        args.out = os.path.join(args.out_dir, f"{args.source}_generations{suffix}.jsonl")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
