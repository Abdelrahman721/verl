"""Stats for assembling the next medical_qa training stage.

Walks `--rollout_dir` to compute per-prompt mean success scores, maps each
rollout `input` string back to its row in `--source` (medical_qa/train.parquet)
by reproducing the chat template render verl uses, and reports candidate-pool
sizes for the build step that follows.

The script is read-only and idempotent. Output is a JSON cache (`--out`) that
`build_medical_qa_next.py` consumes; the printed report lets the user pick a
success threshold and a conv-keep portion before invoking the build script.

Why the chat-template round-trip:
    verl dumps each rollout's `input` as `tokenizer.decode(prompt_ids,
    skip_special_tokens=True)` (verl/trainer/main_ppo_sync.py:1044). For the
    medical branch of the reward dispatcher the `gts` field is always `None`
    and `prompt_uid` is padded to 0, so the chat-templated `input` string is
    the only stable per-prompt key shared between rollouts and the source
    parquet. We reproduce the same string for every source row by applying
    the model's chat template with `tokenize=False` and stripping the
    `<|im_start|>` / `<|im_end|>` markers manually — bit-identical to the
    tokenize→decode round trip but ~250× faster.

Usage:
    python -m full_mix.curriculum.stats_medical_qa_next \\
        --rollout_dir /data/abdelrahman/verl/medical_qa_rollouts \\
        --source /data/abdelrahman/verl/data/medical_qa/train.parquet \\
        --tokenizer /data/abdelrahman/verl/checkpoints/RL-Exps/medical-qa-fresh/global_step_500/merged_hf_model \\
        --carve /data/hazem/medical-data-gen/processed/rl/combined/v1/carve_clinical_reasoning.parquet \\
        --carve /data/hazem/medical-data-gen/processed/rl/combined/v1/carve_lay_patient_tier.parquet \\
        --out   /data/abdelrahman/verl/data/medical_qa_stage2/_stats_cache.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict


# ============================================================================
# Chat-template renderer
# ============================================================================
_SPECIAL_PATTERN = re.compile(r"<\|im_start\|>|<\|im_end\|>")


def render_chat(tok, messages) -> str:
    """Mirror verl's tokenize→decode(skip_special_tokens=True) pipeline.

    Confirmed byte-identical to that round-trip on Qwen3-4B-Base. Faster
    because it skips the actual tokenization step — we render the raw
    template string and strip Qwen's two special-token markers.
    """
    s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return _SPECIAL_PATTERN.sub("", s)


def _hash16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ============================================================================
# Rollout walking
# ============================================================================
def walk_rollouts(rollout_dir: str):
    """Yield (step, row) for every JSONL row across `rollout_dir`.

    Step is parsed from the filename (`N.jsonl`).
    """
    files = sorted(glob.glob(os.path.join(rollout_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no *.jsonl files under {rollout_dir}")
    for fp in files:
        step = int(os.path.basename(fp).split(".")[0])
        with open(fp) as f:
            for line in f:
                yield step, json.loads(line)


def aggregate_rollouts(rollout_dir: str):
    """Walk every rollout row; keep only medical (eval_mode in {qa, conversation}).

    Group by the chat-templated `input` string. Returns
    (n_rollout_rows_total, n_medical_rows, list_of_per_prompt_dicts).

    Each per_prompt dict has:
        input, mean_score, n_samples, n_steps_seen, eval_mode (set→list)
    """
    by_input: dict[str, dict] = {}
    n_total = 0
    n_medical = 0
    for step, r in walk_rollouts(rollout_dir):
        n_total += 1
        em = r.get("reward/eval_mode") or ""
        if em not in ("qa", "conversation"):
            continue
        n_medical += 1
        s = by_input.get(r["input"])
        if s is None:
            s = by_input[r["input"]] = {
                "scores": [],
                "eval_modes": set(),
                "steps": set(),
            }
        s["scores"].append(float(r["score"]))
        s["eval_modes"].add(em)
        s["steps"].add(step)
    per_prompt = []
    for inp, s in by_input.items():
        per_prompt.append({
            "input": inp,
            "mean_score": sum(s["scores"]) / len(s["scores"]),
            "n_samples": len(s["scores"]),
            "n_steps_seen": len(s["steps"]),
            "eval_mode": sorted(s["eval_modes"]),
        })
    return n_total, n_medical, per_prompt


# ============================================================================
# Source / carve rendering
# ============================================================================
def render_parquet_prompts(tok, parquet_path: str):
    """Load `prompt` + `data_source` from a parquet, render each row's prompt.

    Returns:
        rendered_strings: list[str]  — one per source row, in row order
        data_sources:     list[str]  — one per source row, in row order
    """
    import pyarrow.parquet as pq

    t = pq.read_table(parquet_path, columns=["prompt", "data_source"])
    prompts = t.column("prompt").to_pylist()
    data_sources = t.column("data_source").to_pylist()
    rendered = [render_chat(tok, p) for p in prompts]
    return rendered, data_sources


def build_rendered_index(rendered):
    """Map rendered string → first row index. Track collisions defensively.

    Two rows producing the same rendered string would mean the underlying
    prompts are identical — uncommon but possible (boilerplate template +
    duplicate question). We keep the first occurrence and report the count.
    """
    out: dict[str, int] = {}
    collisions = 0
    for i, r in enumerate(rendered):
        if r in out:
            collisions += 1
        else:
            out[r] = i
    return out, collisions


# ============================================================================
# CLI
# ============================================================================
def _print_histogram(scores, label="mean_score"):
    if not scores:
        print(f"  no {label} values to summarise")
        return
    qs = statistics.quantiles(scores, n=100)
    pcts = {p: round(qs[p - 1], 4) for p in (5, 10, 25, 50, 75, 90, 95)}
    print(f"  n={len(scores)}  mean={statistics.fmean(scores):.4f}  "
          f"min={min(scores):.4f}  max={max(scores):.4f}")
    print(f"  percentiles: " + "  ".join(f"P{p}={v}" for p, v in pcts.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout_dir", required=True,
                    help="Dir of *.jsonl rollout dumps (one file per step).")
    ap.add_argument("--source", required=True,
                    help="data/medical_qa/train.parquet (used in the previous stage).")
    ap.add_argument("--tokenizer", required=True,
                    help="Path to a HF tokenizer dir whose chat template matches the run.")
    ap.add_argument("--carve", action="append", default=[],
                    help="Carve parquet to dedup-check against seen prompts (repeatable).")
    ap.add_argument("--out", required=True,
                    help="JSON cache path the build script will consume.")
    ap.add_argument("--thresholds", default="0.5,0.6,0.7,0.8",
                    help="Comma-separated success-rate thresholds for what-if counts.")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]

    print(f"[stats_medical_qa_next] rollout_dir = {args.rollout_dir}")
    print(f"[stats_medical_qa_next] source      = {args.source}")
    print(f"[stats_medical_qa_next] tokenizer   = {args.tokenizer}")
    print(f"[stats_medical_qa_next] carve files = {len(args.carve)}")

    # ----- Phase 1: walk rollouts -----
    print("\n=== Phase 1: walk rollouts ===")
    t0 = time.time()
    n_total, n_medical, per_prompt = aggregate_rollouts(args.rollout_dir)
    print(f"  total rollout rows scanned: {n_total}")
    print(f"  medical rollout rows kept:  {n_medical}  "
          f"(retention rows skipped: {n_total - n_medical})")
    print(f"  distinct medical prompts:   {len(per_prompt)}")
    em_counts = Counter(",".join(p["eval_mode"]) for p in per_prompt)
    for em, c in sorted(em_counts.items()):
        print(f"    eval_mode={em!r}: {c} distinct prompts")
    print(f"  elapsed: {time.time() - t0:.1f}s")

    # ----- Phase 2: render source -----
    print("\n=== Phase 2: render source parquet ===")
    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"  tokenizer loaded ({time.time() - t0:.1f}s)")
    t0 = time.time()
    src_rendered, src_data_sources = render_parquet_prompts(tok, args.source)
    print(f"  source rows: {len(src_rendered)}  rendered in {time.time() - t0:.1f}s")
    print(f"  data_source counts: {dict(Counter(src_data_sources))}")
    src_idx, src_collisions = build_rendered_index(src_rendered)
    print(f"  distinct rendered keys: {len(src_idx)}  collisions (kept first): {src_collisions}")

    # ----- Phase 3: match rollouts → source -----
    print("\n=== Phase 3: match rollouts → source rows ===")
    seen_indices: set[int] = set()
    orphans = 0
    for p in per_prompt:
        idx = src_idx.get(p["input"])
        if idx is None:
            orphans += 1
            p["source_index"] = None
        else:
            p["source_index"] = idx
            seen_indices.add(idx)
    matched = len(per_prompt) - orphans
    print(f"  matched: {matched} / {len(per_prompt)} distinct prompts")
    if orphans:
        print(f"  orphans: {orphans} rollout inputs had no source match")
        print("    (likely chat-template drift between training run and current tokenizer;")
        print("     review the orphan keys in the cache if this is > 0.)")
    print(f"  distinct source rows seen: {len(seen_indices)}")

    # Conv pool composition
    seen_conv = sum(1 for i in seen_indices if src_data_sources[i] == "medical_conv")
    total_conv = sum(1 for ds in src_data_sources if ds == "medical_conv")
    unseen_conv_indices = sorted(
        i for i, ds in enumerate(src_data_sources)
        if ds == "medical_conv" and i not in seen_indices
    )
    print(f"\n  medical_conv total in source : {total_conv}")
    print(f"    seen in rollouts            : {seen_conv}")
    print(f"    UNSEEN (build-script pool)  : {len(unseen_conv_indices)}")
    assert seen_conv + len(unseen_conv_indices) == total_conv, \
        "conv sanity check failed"

    # ----- Phase 4: per-prompt score summary + threshold what-ifs -----
    print("\n=== Phase 4: success-rate distribution & threshold what-ifs ===")
    matched_pp = [p for p in per_prompt if p["source_index"] is not None]
    _print_histogram([p["mean_score"] for p in matched_pp], "mean score (matched prompts)")
    by_em = defaultdict(list)
    for p in matched_pp:
        by_em[",".join(p["eval_mode"])].append(p["mean_score"])
    for em, vals in sorted(by_em.items()):
        print(f"\n  -- eval_mode={em!r} ({len(vals)} prompts) --")
        _print_histogram(vals)

    print("\n  threshold what-ifs (mean_score < threshold ⇒ carry forward):")
    for thr in thresholds:
        carry = sum(1 for p in matched_pp if p["mean_score"] < thr)
        mastered = len(matched_pp) - carry
        print(f"    threshold={thr:>5.2f}:  carry={carry:>6}   mastered={mastered:>6}")

    # ----- Phase 5: carve file dedup -----
    print("\n=== Phase 5: carve file dedup ===")
    seen_rendered = {p["input"] for p in matched_pp}
    carve_summary: dict[str, dict] = {}
    for cp in args.carve:
        t0 = time.time()
        cr, cds = render_parquet_prompts(tok, cp)
        survivor_idxs = [i for i, r in enumerate(cr) if r not in seen_rendered]
        carve_summary[cp] = {
            "total_rows": len(cr),
            "rows_seen_in_rollouts": len(cr) - len(survivor_idxs),
            "rows_after_dedup": len(survivor_idxs),
            "after_dedup_carve_row_indices": survivor_idxs,
            "data_source_counts": dict(Counter(cds)),
        }
        print(f"  {os.path.basename(cp)}: "
              f"total={len(cr)}, seen_in_rollouts={len(cr) - len(survivor_idxs)}, "
              f"after_dedup={len(survivor_idxs)}  ({time.time() - t0:.1f}s)")

    # ----- Phase 6: emit cache -----
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cache = {
        "meta": {
            "rollout_dir": args.rollout_dir,
            "source_parquet": args.source,
            "tokenizer": args.tokenizer,
            "n_rollout_rows": n_total,
            "n_medical_rollout_rows": n_medical,
            "n_distinct_medical_prompts": len(per_prompt),
            "n_orphans": orphans,
            "thresholds_reported": thresholds,
        },
        # One entry per distinct medical prompt seen in rollouts. `input_hash`
        # lets the build script verify integrity without storing all the raw
        # text; `source_index` is the parquet row index in --source.
        "per_prompt": [
            {
                "input_hash": _hash16(p["input"]),
                "mean_score": round(p["mean_score"], 4),
                "n_samples": p["n_samples"],
                "n_steps_seen": p["n_steps_seen"],
                "eval_mode": p["eval_mode"],
                "source_index": p["source_index"],
            }
            for p in per_prompt
        ],
        # Convenience indices (computable from per_prompt, but the build
        # script avoids re-walking the parquet by reading these directly).
        "seen_source_indices": sorted(seen_indices),
        "unseen_conv_source_indices": unseen_conv_indices,
        "carve": carve_summary,
    }
    with open(args.out, "w") as f:
        json.dump(cache, f, indent=2)
    cache_size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\nwrote {args.out}  ({cache_size_mb:.1f} MB)")

    if matched_pp:
        s = matched_pp[0]
        print("\n  sample matched prompt:")
        print(f"    source_index={s['source_index']}, mean_score={s['mean_score']:.3f}, "
              f"n_samples={s['n_samples']}, eval_mode={s['eval_mode']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
