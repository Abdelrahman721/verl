"""Stats for the stage-5 retention pool (consumes stage-4 coding rollouts).

Walks ``/data/abdelrahman/verl/coding_rollouts/medical_qa_stage4_rollouts/*.jsonl``,
groups by chat-templated ``input``, computes per-prompt mean ``score``, and
matches each prompt against TWO source parquets:

  - data/medical_qa_stage4/train_coding_only.parquet  (25 000 stage-4 coding rows)
        → ``coding_source_index`` for the **stage-4 unmastered pool**
          (mean_score < threshold defines unmastered).

  - data/medical_qa_stage3/train.parquet (bench rows only, 17 193 of 21 193)
        → ``bench_source_index`` for the **bench-unseen intersection**
          (stage-5 bench pool must be unseen in BOTH bench rollouts AND
          stage-4 coding rollouts; this gives us the second set).

Rollouts that match neither map (general / medical retention exposed during
stage-4) are recorded as orphans — they're not needed for stage-5 sampling,
but counted for sanity.

Reuses verbatim:
  - render_chat / build_rendered_index / _hash16 from stats_medical_qa_next.py

Usage:
    python -m full_mix.curriculum.stats_medical_qa_stage5_retention \\
        --rollout_dir   /data/abdelrahman/verl/coding_rollouts/medical_qa_stage4_rollouts \\
        --coding_source /data/abdelrahman/verl/data/medical_qa_stage4/train_coding_only.parquet \\
        --bench_source  /data/abdelrahman/verl/data/medical_qa_stage3/train.parquet \\
        --tokenizer     /data/abdelrahman/verl/checkpoints/RL-Exps/medical-qa-fresh/global_step_500/merged_hf_model \\
        --out           /data/abdelrahman/verl/data/medical_qa_stage5/_stage4_retention_cache.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from collections import Counter


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from full_mix.curriculum.stats_medical_qa_next import (  # noqa: E402
    build_rendered_index,
    render_chat,
)


_BENCH_DATA_SOURCES = frozenset({
    "medical_benchmark_mcq",
    "medical_benchmark_numeric",
    "medical_benchmark_medec",
    "medical_benchmark_text",
})

# Stage-4 emits these as ``reward/eval_mode`` for the coding pool. We label
# prompts as coding ONLY when EVERY observed eval_mode is in this set — that
# way a prompt tagged across coding + bench (shouldn't happen, but defensive)
# doesn't accidentally land in the unmastered pool.
_CODING_EVAL_MODES = frozenset({"icd_multi_label", "snomed_single_label"})

# Bench eval_modes ACTUALLY emitted by qa_openrouter_bench. Includes the
# hybrid medec variant and the text-judge conversation variant — both show
# up in real rollouts even though the user spec only lists the short names.
_BENCH_EVAL_MODES = frozenset({"mcq", "numeric", "medec", "medec_hybrid",
                                "text", "text_judge_conv"})


def _render_bench_only(tok, bench_parquet: str) -> tuple[dict[str, int], list[int], list[str]]:
    """Return (rendered_to_source_idx, bench_source_idxs, bench_data_sources).

    Restricts rendering to ``data_source ∈ _BENCH_DATA_SOURCES`` (≈17 k rows
    instead of all 21 k), then maps each rendered string back to its GLOBAL
    row index in ``bench_parquet``. Collisions keep the first occurrence,
    same behaviour as the existing bench retention stats script."""
    import pyarrow.parquet as pq
    t = pq.read_table(bench_parquet, columns=["prompt", "data_source"])
    prompts = t.column("prompt").to_pylist()
    sources = t.column("data_source").to_pylist()
    rendered: list[str] = []
    kept_sources: list[str] = []
    kept_idxs: list[int] = []
    for i, (p, ds) in enumerate(zip(prompts, sources)):
        if ds not in _BENCH_DATA_SOURCES:
            continue
        rendered.append(render_chat(tok, p))
        kept_sources.append(ds)
        kept_idxs.append(i)
    rendered_to_local, collisions = build_rendered_index(rendered)
    rendered_to_source = {k: kept_idxs[local] for k, local in rendered_to_local.items()}
    print(f"  bench rendered: {len(rendered)} rows  → distinct keys: "
          f"{len(rendered_to_source)} (collisions={collisions})")
    return rendered_to_source, kept_idxs, kept_sources


def _render_coding(tok, coding_parquet: str) -> tuple[dict[str, int], int]:
    """Return (rendered_to_source_idx, n_source_rows). All rows kept."""
    import pyarrow.parquet as pq
    t = pq.read_table(coding_parquet, columns=["prompt"])
    prompts = t.column("prompt").to_pylist()
    rendered = [render_chat(tok, p) for p in prompts]
    rendered_to_idx, collisions = build_rendered_index(rendered)
    print(f"  coding rendered: {len(rendered)} rows  → distinct keys: "
          f"{len(rendered_to_idx)} (collisions={collisions})")
    return rendered_to_idx, len(rendered)


def _walk_coding_rollouts(rollout_dir: str) -> tuple[int, int, list[dict]]:
    """Walk every rollout row and group by ``input``. Returns
    (n_total, n_kept, per_prompt)."""
    by_input: dict[str, dict] = {}
    n_total = 0
    n_kept = 0
    files = sorted(glob.glob(os.path.join(rollout_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no *.jsonl files under {rollout_dir}")
    for fp in files:
        try:
            step = int(os.path.basename(fp).split(".")[0])
        except ValueError:
            step = -1
        with open(fp) as f:
            for line in f:
                n_total += 1
                r = json.loads(line)
                inp = r.get("input")
                if not inp:
                    continue
                n_kept += 1
                em = r.get("reward/eval_mode") or ""
                s = by_input.get(inp)
                if s is None:
                    s = by_input[inp] = {"scores": [], "em": set(), "steps": set()}
                s["scores"].append(float(r["score"]))
                if em:
                    s["em"].add(em)
                s["steps"].add(step)
    per_prompt = [
        {
            "input":        inp,
            "mean_score":   sum(d["scores"]) / len(d["scores"]),
            "n_samples":    len(d["scores"]),
            "n_steps_seen": len(d["steps"]),
            "eval_mode":    sorted(d["em"]),
        }
        for inp, d in by_input.items()
    ]
    return n_total, n_kept, per_prompt


def _is_coding_prompt(em_list: list[str]) -> bool:
    return bool(em_list) and all(em in _CODING_EVAL_MODES for em in em_list)


def _is_bench_prompt(em_list: list[str]) -> bool:
    return bool(em_list) and all(em in _BENCH_EVAL_MODES for em in em_list)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rollout_dir",   required=True)
    ap.add_argument("--coding_source", required=True,
                    help="data/medical_qa_stage4/train_coding_only.parquet")
    ap.add_argument("--bench_source",  required=True,
                    help="data/medical_qa_stage3/train.parquet (bench rows are kept)")
    ap.add_argument("--tokenizer",     required=True)
    ap.add_argument("--out",           required=True)
    ap.add_argument("--thresholds",    default="0.5,0.6,0.7,0.8")
    args = ap.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]

    print(f"rollout_dir   : {args.rollout_dir}")
    print(f"coding_source : {args.coding_source}")
    print(f"bench_source  : {args.bench_source}")
    print(f"tokenizer     : {args.tokenizer}")

    # ----- Phase 1: walk rollouts -----
    print("\n=== Phase 1: walk stage-4 coding rollouts ===")
    t0 = time.time()
    n_total, n_kept, per_prompt = _walk_coding_rollouts(args.rollout_dir)
    em_counts = Counter(",".join(p["eval_mode"]) for p in per_prompt)
    print(f"  total rollout rows: {n_total}   kept (input present): {n_kept}")
    print(f"  distinct rollout inputs: {len(per_prompt)}")
    print(f"  eval_mode breakdown (distinct prompts):")
    for em, c in sorted(em_counts.items()):
        print(f"    {em!r:<40} {c}")
    print(f"  elapsed: {time.time() - t0:.1f}s")

    # ----- Phase 2: render source parquets -----
    print("\n=== Phase 2: render source parquets ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    t0 = time.time()
    print("  coding source...")
    coding_to_src, n_coding_rows = _render_coding(tok, args.coding_source)
    print(f"  coding render elapsed: {time.time() - t0:.1f}s")

    t0 = time.time()
    print("  bench source...")
    bench_to_src, bench_idxs, bench_sources = _render_bench_only(tok, args.bench_source)
    print(f"  bench render elapsed:  {time.time() - t0:.1f}s")

    # ----- Phase 3: match rollouts → source -----
    print("\n=== Phase 3: match rollouts → source rows ===")
    n_coding_matched = 0
    n_bench_matched = 0
    n_orphans = 0
    n_ambiguous = 0
    seen_coding_idxs: set[int] = set()
    seen_bench_idxs: set[int]  = set()
    for p in per_prompt:
        ci = coding_to_src.get(p["input"])
        bi = bench_to_src.get(p["input"])
        p["coding_source_index"] = ci
        p["bench_source_index"]  = bi
        if ci is not None and bi is not None:
            # Same chat-rendered string maps to BOTH a coding row and a bench
            # row — extremely unlikely given different prompt prefixes, but
            # log if it happens.
            n_ambiguous += 1
        if ci is not None:
            n_coding_matched += 1
            seen_coding_idxs.add(ci)
        if bi is not None:
            n_bench_matched += 1
            seen_bench_idxs.add(bi)
        if ci is None and bi is None:
            n_orphans += 1
    print(f"  coding-matched: {n_coding_matched}  distinct coding rows seen: "
          f"{len(seen_coding_idxs)} / {n_coding_rows}")
    print(f"  bench-matched : {n_bench_matched}   distinct bench rows seen:  "
          f"{len(seen_bench_idxs)} / {len(bench_idxs)}")
    print(f"  orphans (general/medical retention or unrecognised): {n_orphans}")
    if n_ambiguous:
        print(f"  WARNING: {n_ambiguous} prompts matched BOTH coding and bench source maps")

    # ----- Phase 4: threshold what-ifs over the coding pool -----
    print("\n=== Phase 4: threshold what-ifs (coding pool, mean_score < threshold ⇒ unmastered) ===")
    coding_pp = [p for p in per_prompt
                 if p["coding_source_index"] is not None and _is_coding_prompt(p["eval_mode"])]
    print(f"  coding prompts with matched source AND pure-coding eval_modes: {len(coding_pp)}")
    for thr in thresholds:
        carry = sum(1 for p in coding_pp if p["mean_score"] <  thr)
        mast  = sum(1 for p in coding_pp if p["mean_score"] >= thr)
        print(f"  threshold={thr:>5.2f}:  unmastered={carry:>5}   mastered={mast:>5}")

    if coding_pp:
        scores = [p["mean_score"] for p in coding_pp]
        qs = statistics.quantiles(scores, n=100) if len(scores) >= 100 else None
        print(f"  coding mean_score across seen: {statistics.fmean(scores):.4f}")
        if qs:
            print(f"  percentiles: P10={qs[9]:.3f} P50={qs[49]:.3f} P90={qs[89]:.3f}")

    # ----- Phase 5: emit cache -----
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Separate seen sets by eval_mode bucket so the build script can use
    # them directly without re-filtering.
    seen_bench_strict = sorted({
        p["bench_source_index"] for p in per_prompt
        if p["bench_source_index"] is not None and _is_bench_prompt(p["eval_mode"])
    })
    seen_coding_strict = sorted({
        p["coding_source_index"] for p in per_prompt
        if p["coding_source_index"] is not None and _is_coding_prompt(p["eval_mode"])
    })

    cache = {
        "meta": {
            "rollout_dir":           args.rollout_dir,
            "coding_source_parquet": args.coding_source,
            "bench_source_parquet":  args.bench_source,
            "tokenizer":             args.tokenizer,
            "n_rollout_rows":        n_total,
            "n_kept_rollout_rows":   n_kept,
            "n_distinct_prompts":    len(per_prompt),
            "n_coding_matched":      n_coding_matched,
            "n_bench_matched":       n_bench_matched,
            "n_orphans":             n_orphans,
            "n_ambiguous":           n_ambiguous,
            "n_coding_rows_in_source": n_coding_rows,
            "n_bench_rows_in_source":  len(bench_idxs),
            "thresholds_reported":   thresholds,
        },
        # Per-prompt records. input_hash for dedup; source indices for slicing.
        "per_prompt": [
            {
                "input_hash":          _hash16(p["input"]),
                "mean_score":          round(p["mean_score"], 4),
                "n_samples":           p["n_samples"],
                "n_steps_seen":        p["n_steps_seen"],
                "eval_mode":           p["eval_mode"],
                "coding_source_index": p["coding_source_index"],
                "bench_source_index":  p["bench_source_index"],
            }
            for p in per_prompt
        ],
        # Convenience sets (strict by eval_mode) for the build script.
        "seen_coding_source_indices_strict": seen_coding_strict,
        "seen_bench_source_indices_strict":  seen_bench_strict,
        # Permissive seen sets (ANY rollout match, regardless of eval_mode).
        # Used by the build's bench-unseen intersection — safer to also exclude
        # bench rows the model saw under a misleading eval_mode tag.
        "seen_coding_source_indices_any":    sorted(seen_coding_idxs),
        "seen_bench_source_indices_any":     sorted(seen_bench_idxs),
    }
    with open(args.out, "w") as f:
        json.dump(cache, f, indent=2)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\nwrote {args.out}  ({size_mb:.1f} MB)")
    return 0


def _hash16(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    sys.exit(main())
