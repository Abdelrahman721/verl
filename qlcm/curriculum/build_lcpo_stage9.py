"""Build the qlcm stage-9 LCPO mix and its validation ladder.

Stage 9 tests LCPO-Max budget following on the stage-8 model across the
domains it was trained on. This script samples prompts from every training
pool, materialises each K times in budget / free modes, and renders the
budget as a plain trailing sentence (see qlcm/common/budget_prompt.py):

    budget  {task}\\n\\nThink for a maximum of {n} tokens.
    free    {task}

Budgets come from a 16-value grid, 256..4096 in steps of 256, drawn per prompt
with a seeded RNG and never repeated within a prompt. The mode pattern is
exact: prompt slot = md5(prompt) % 10; slot 0 gets (B, B, B), every other
slot (B, B, F), which is 21 of 30 slots budgeted = 70% / 30% and at least two
budgeted rows per prompt.

Prompt identity is the md5 of the messages, stored as `extra_info.lcpo_pid`.
`extra_info.index` is NOT usable: all stage-8 mlb rows share one index.

Everything verl loads as one dataset must share one Arrow schema, so all
selected tables — train AND val — are reconciled to a single extra_info
struct first (qlcm/curriculum/harmonise_train_files.py's
`_unified_extra_info_type` / `_reconcile`, with conflicting field types
dropped and reported), and only then are the top-level `think_mode` /
`think_budget` columns added. The harmoniser CLI itself must NOT be run on
the outputs afterwards: it drops non-canonical columns and would delete
those two.

Overlong prompts are dropped at build time: the last user turn is rendered
with the LONGEST budget sentence, passed through the model's chat template
with the generation prompt, and tokenized with the stage-8 tokenizer; anything
over --max_prompt_tokens is skipped, so no budget value can push a row over
verl's prompt cap.

Validation: the same base rows rendered once per rung (256, 512, 1024, 2048,
4096) plus free, with data_source tagged "<base>@b00256" / "<base>@free" so
verl reports one series per rung. The tag is stripped again by
qlcm/rewards/compute_score_lcpo.py before routing.

Usage (inside the training container — the host has no transformers):
    python -m qlcm.curriculum.build_lcpo_stage9 \\
        --tokenizer /data/abdelrahman/verl/checkpoints/QLCM/medical-qa-stage8-coding/global_step_160/merged_hf_model \\
        --out_dir /data/abdelrahman/verl/data/qlcm/medical_qa_stage9 --seed 42 --k 3
    # scale a source:  --n mlb=300 --n medical_qa=600
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_QLCM_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_QLCM_DIR)
for _p in (_REPO_ROOT, _QLCM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from full_mix.lcpo.build_val_ladder import variant_tag  # noqa: E402
from qlcm.common.budget_prompt import (  # noqa: E402
    MODE_BUDGET,
    MODE_FREE,
    NO_BUDGET,
    apply_to_messages_plain,
    parse_suffix_plain,
)
from qlcm.curriculum.harmonise_train_files import (  # noqa: E402
    _reconcile,
    _unified_extra_info_type,
)

DATA = "/data/abdelrahman/verl/data"
GRID = list(range(256, 4097, 256))          # 16 budgets
VAL_RUNGS = [256, 512, 1024, 2048, 4096]

# domain, data_source, pool, default prompt count (None = every unique prompt)
TRAIN_SOURCES = [
    ("coding", "mlb",        f"{DATA}/qlcm/medical_qa_stage8/train.parquet", 700),
    ("coding", "ml",         f"{DATA}/qlcm/medical_qa_stage5/train.parquet", 600),
    ("coding", "if",         f"{DATA}/qlcm/medical_qa_stage5/train.parquet", 400),
    ("coding", "snomed_ml",  f"{DATA}/qlcm/medical_qa_stage5/train.parquet", 500),
    ("coding", "snomed_ml",  f"{DATA}/qlcm/medical_qa_stage8/train.parquet", None),   # the 103 real notes
    ("coding", "snomed_if",  f"{DATA}/qlcm/medical_qa_stage5/train.parquet", 400),
    ("coding", "sl",         f"{DATA}/qlcm/medical_qa_stage4/train_coding_only.parquet", 200),
    ("coding", "snomed_sl",  f"{DATA}/qlcm/medical_qa_stage4/train_coding_only.parquet", 200),
    ("coding", "slb",        f"{DATA}/qlcm/medical_qa_stage6/train.parquet", 100),
    ("medical", "medical_qa",   f"{DATA}/qlcm/medical_qa_stage2/train.parquet", 1200),
    ("medical", "medical_conv", f"{DATA}/qlcm/medical_qa_stage2/train.parquet", 1200),
    ("general", "local/dolci-ifeval-32b",     f"{DATA}/qlcm/train/ifeval_train.parquet", 900),
    ("general", "local/dolci-chat-32b",       f"{DATA}/qlcm/train/chat_with_baseline_train.parquet", 800),
    ("general", "local/safety-dpo-reference", f"{DATA}/qlcm/train/safety_train.parquet", 500),
    ("general", "local/avey-identity",        f"{DATA}/qlcm/train/identity_train.parquet", 200),
]

# pool, data_source filter (None = all rows), prompt count (None = all)
VAL_SOURCES = [
    (f"{DATA}/qlcm/medical_qa_stage8/val.parquet", None, None),        # mlb 77 + snomed_ml 68 (real notes)
    (f"{DATA}/qlcm/medical_qa_stage5/val.parquet", None, None),        # ml / if / snomed_ml / snomed_if, 200
    (f"{DATA}/medical_qa/val.parquet", None, None),                    # medical_qa 40 + medical_conv 8 (judge)
    (f"{DATA}/qlcm/eval/ifeval_eval.parquet", None, None),             # google/IFEval 100
    (f"{DATA}/qlcm/eval/gsm8k_eval.parquet", None, 50),                # never trained on: clean slope
    (f"{DATA}/qlcm/eval/math500_eval.parquet", None, 50),
    ("/data/abdelrahman/verl/handoff/datasets/general_chat_unseen.parquet", None, 40),
    ("/data/abdelrahman/verl/handoff/datasets/general_safety_unseen.parquet", None, 40),
]


def prompt_id(messages: list[dict]) -> str:
    key = json.dumps([(m.get("role"), m.get("content")) for m in messages], ensure_ascii=False)
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def slot_pattern(pid: str) -> tuple[str, ...]:
    """(B,B,B) for one prompt in ten, (B,B,F) for the rest: exactly 70/30."""
    return (MODE_BUDGET,) * 3 if int(pid[:8], 16) % 10 == 0 else (MODE_BUDGET, MODE_BUDGET, MODE_FREE)


def draw_budgets(seed: int, data_source: str, pid: str, n: int) -> list[int]:
    rng = random.Random(f"{seed}:{data_source}:{pid}")
    return rng.sample(GRID, k=n)


class LengthGate:
    """Drops prompts that would exceed the prompt cap with the longest budget sentence."""

    def __init__(self, tokenizer_path: str, max_tokens: int):
        from transformers import AutoTokenizer  # container only
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_tokens = max_tokens

    def tokens(self, messages: list[dict]) -> int:
        rendered = apply_to_messages_plain(messages, MODE_BUDGET, max(GRID))
        text = self.tok.apply_chat_template(rendered, add_generation_prompt=True, tokenize=False)
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def ok(self, messages: list[dict]) -> bool:
        return self.tokens(messages) <= self.max_tokens


def select_rows(table: pa.Table, data_source: str | None, n: int | None, seed: int,
                gate: LengthGate, seen: set, stats: dict) -> pa.Table:
    """Dedup by prompt md5 (across every call, via `seen`), drop overlong, sample n."""
    if data_source is not None:
        mask = pc.equal(table.column("data_source"), data_source)
        table = table.filter(mask)
    prompts = table.column("prompt").to_pylist()
    order = list(range(table.num_rows))
    random.Random(f"{seed}:select:{data_source}:{table.num_rows}").shuffle(order)
    keep, dropped_dup, dropped_long = [], 0, 0
    for i in order:
        pid = prompt_id(prompts[i])
        if pid in seen:
            dropped_dup += 1
            continue
        if not gate.ok(prompts[i]):
            dropped_long += 1
            continue
        seen.add(pid)
        keep.append(i)
        if n is not None and len(keep) >= n:
            break
    stats.update(pool_rows=table.num_rows, duplicates_skipped=dropped_dup,
                 overlong_dropped=dropped_long, selected=len(keep))
    return table.take(pa.array(sorted(keep)))


def materialise(rows: list[dict], data_source_override: str | None, mode_fn, ei_fields: list[str]) -> list[dict]:
    """Expand base rows into LCPO rows. `mode_fn(pid) -> list[(mode, budget)]`."""
    out = []
    for r in rows:
        pid = prompt_id(r["prompt"])
        ei = dict(r["extra_info"] or {})
        ei["lcpo_pid"] = pid
        ei = {k: ei.get(k) for k in ei_fields}
        for mode, budget in mode_fn(pid, r):
            out.append({
                "data_source": data_source_override or r["data_source"],
                "prompt": apply_to_messages_plain(r["prompt"], mode, budget),
                "ability": r.get("ability"),
                "reward_model": r["reward_model"],
                "extra_info": ei,
                "think_mode": mode,
                "think_budget": int(budget),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="stage-8 merged model dir (tokenizer + chat template)")
    ap.add_argument("--out_dir", default=f"{DATA}/qlcm/medical_qa_stage9")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=3, help="rows per prompt; the pattern assumes 3")
    ap.add_argument("--max_prompt_tokens", type=int, default=8192)
    ap.add_argument("--n", action="append", default=[], metavar="SOURCE=N",
                    help="override a train source's prompt count, e.g. --n mlb=300 (repeatable)")
    ap.add_argument("--force", action="store_true", help="write into a non-empty --out_dir")
    args = ap.parse_args()
    if args.k != 3:
        raise SystemExit("the slot pattern is defined for k=3; change slot_pattern() before using another k")

    overrides = {}
    for item in args.n:
        src, val = item.split("=", 1)
        overrides[src] = int(val)

    if os.path.isdir(args.out_dir) and any(f.endswith(".parquet") for f in os.listdir(args.out_dir)) and not args.force:
        raise SystemExit(f"{args.out_dir} already holds parquets; pass --force to overwrite")
    os.makedirs(args.out_dir, exist_ok=True)

    gate = LengthGate(args.tokenizer, args.max_prompt_tokens)
    report = {"seed": args.seed, "k": args.k, "grid": GRID, "val_rungs": VAL_RUNGS,
              "max_prompt_tokens": args.max_prompt_tokens, "train": {}, "val": {}}

    # ---- 1. select base rows ------------------------------------------------
    seen: set = set()
    train_tables: list[tuple[str, str, pa.Table]] = []   # (domain, data_source, table)
    for domain, ds, pool, n in TRAIN_SOURCES:
        if n is not None:               # "all rows" entries (n=None) are not scaled
            n = overrides.get(ds, n)
        stats: dict = {"pool": pool}
        t = select_rows(pq.read_table(pool), ds, n, args.seed, gate, seen, stats)
        report["train"].setdefault(ds, []).append(stats)
        print(f"  train {ds:28s} {stats['selected']:5d} prompts  (pool {stats['pool_rows']}, dup {stats['duplicates_skipped']}, overlong {stats['overlong_dropped']})")
        train_tables.append((domain, ds, t))

    val_seen: set = set()
    val_tables: list[pa.Table] = []
    for pool, ds, n in VAL_SOURCES:
        stats = {"pool": pool}
        t = select_rows(pq.read_table(pool), ds, n, args.seed, gate, val_seen, stats)
        report["val"][os.path.basename(pool)] = stats
        print(f"  val   {os.path.basename(pool):40s} {stats['selected']:5d} prompts  (overlong {stats['overlong_dropped']})")
        val_tables.append(t)

    # ---- 2. one schema for everything -------------------------------------------
    all_tables = [t for _, _, t in train_tables] + val_tables
    unified, conflicts = _unified_extra_info_type([t.schema.field("extra_info").type for t in all_tables], drop_conflicts=True)
    if conflicts:
        print(f"  extra_info fields dropped for type conflicts: {sorted(conflicts)}")
    report["extra_info_conflicts_dropped"] = {k: v for k, v in conflicts.items()}
    train_tables = [(d, ds, _reconcile(t, unified)) for d, ds, t in train_tables]
    val_tables = [_reconcile(t, unified) for t in val_tables]
    base_schema = train_tables[0][2].schema
    ei_type = base_schema.field("extra_info").type
    ei_fields = [ei_type.field(i).name for i in range(ei_type.num_fields)] + ["lcpo_pid"]
    ei_type_out = pa.struct([ei_type.field(i) for i in range(ei_type.num_fields)] + [pa.field("lcpo_pid", pa.large_string())])
    out_schema = pa.schema(
        [f if f.name != "extra_info" else pa.field("extra_info", ei_type_out) for f in base_schema]
        + [pa.field("think_mode", pa.large_string()), pa.field("think_budget", pa.int64())]
    )
    report["extra_info_fields"] = ei_fields

    # ---- 3. train rows ---------------------------------------------------------
    def train_modes(pid: str, row: dict) -> list[tuple[str, int]]:
        pattern = slot_pattern(pid)
        n_budget = sum(1 for m in pattern if m == MODE_BUDGET)
        budgets = iter(draw_budgets(args.seed, row["data_source"], pid, n_budget))
        return [(m, next(budgets) if m == MODE_BUDGET else NO_BUDGET) for m in pattern]

    written = {}
    mode_counts = {MODE_BUDGET: 0, MODE_FREE: 0}
    budget_hist = {b: 0 for b in GRID}
    per_source_rows = {}
    for domain in ("coding", "medical", "general"):
        rows = []
        for d, ds, t in train_tables:
            if d != domain:
                continue
            built = materialise(t.to_pylist(), None, train_modes, ei_fields)
            per_source_rows[ds] = per_source_rows.get(ds, 0) + len(built)
            rows.extend(built)
        for r in rows:
            mode_counts[r["think_mode"]] += 1
            if r["think_mode"] == MODE_BUDGET:
                budget_hist[r["think_budget"]] += 1
        dst = os.path.join(args.out_dir, f"train_{domain}.parquet")
        pq.write_table(pa.Table.from_pylist(rows, schema=out_schema), dst)
        written[f"train_{domain}"] = {"path": dst, "rows": len(rows)}
        print(f"  wrote {dst}: {len(rows)} rows")

    # ---- 4. val ladder ---------------------------------------------------------
    val_rows = []
    for t in val_tables:
        val_rows.extend(t.to_pylist())
    variants = [(f"val_budget_{b:05d}", MODE_BUDGET, b) for b in VAL_RUNGS] + [("val_free", MODE_FREE, NO_BUDGET)]
    for name, mode, budget in variants:
        tag = variant_tag(mode, budget)
        rows = []
        for r in val_rows:
            rows.extend(materialise([r], f"{r['data_source']}@{tag}", lambda pid, row, m=mode, b=budget: [(m, b)], ei_fields))
        dst = os.path.join(args.out_dir, f"{name}.parquet")
        pq.write_table(pa.Table.from_pylist(rows, schema=out_schema), dst)
        written[name] = {"path": dst, "rows": len(rows), "mode": mode, "budget": budget}
        print(f"  wrote {dst}: {len(rows)} rows")

    # ---- 5. checks ------------------------------------------------------------
    schemas = {name: pq.read_schema(v["path"]) for name, v in written.items()}
    first = next(iter(schemas.values()))
    bad = [n for n, s in schemas.items() if not s.equals(first)]
    assert not bad, f"schema mismatch across outputs: {bad}"
    for name, v in written.items():
        seen_budgets: dict = {}
        for r in pq.read_table(v["path"]).to_pylist():
            last = next(m for m in reversed(r["prompt"]) if m["role"] != "assistant")
            assert parse_suffix_plain(last["content"]) == (r["think_mode"], r["think_budget"]), (name, r["think_mode"], r["think_budget"])
            if name.startswith("train") and r["think_mode"] == MODE_BUDGET:
                key = (r["data_source"], r["extra_info"]["lcpo_pid"])
                assert r["think_budget"] not in seen_budgets.setdefault(key, set()), f"repeated budget for {key}"
                seen_budgets[key].add(r["think_budget"])
    total = sum(mode_counts.values())
    frac_b = mode_counts[MODE_BUDGET] / total
    assert abs(frac_b - 0.70) < 0.005, f"budget share {frac_b:.4f} is not 70%"
    mean_h = sum(budget_hist.values()) / len(GRID)
    off = {b: c for b, c in budget_hist.items() if abs(c - mean_h) > 0.15 * mean_h}
    assert not off, f"budget histogram uneven (mean {mean_h:.0f}): {off}"

    report.update(files=written, rows_per_source=per_source_rows, mode_counts=mode_counts,
                  mode_fractions={m: round(c / total, 4) for m, c in mode_counts.items()},
                  budget_hist=budget_hist, train_rows=total)
    with open(os.path.join(args.out_dir, "_build_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  train rows {total}: budget {mode_counts[MODE_BUDGET]} ({frac_b:.1%}) free {mode_counts[MODE_FREE]}")
    print("  budgets: " + "  ".join(f"{b}:{c}" for b, c in budget_hist.items()))
    print("\n  TRAIN_FILES=[" + ",".join(written[f"train_{d}"]["path"] for d in ("coding", "medical", "general")) + "]")
    print("  VAL_FILES=[" + ",".join(written[n]["path"] for n, _, _ in variants) + "]")
    print(f"  saved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
