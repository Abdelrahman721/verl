"""Read an lm-eval budget sweep and report the score-vs-budget curve.

Consumes the `samples_*.jsonl` files that `--log_samples` writes, laid out by
`eval_budget_sweep.sh` as

    <root>/<variant>/<benchmark>/<model-path>/samples_<task>_<ts>.jsonl

where <variant> is `budget_01024`, `free`, or `nothink`.

Reports per benchmark and per variant:

    score         mean of the task's metric over the samples
    unclosed      fraction of generations with no </think> — ALWAYS report this
                  next to the score. Under the patched harness those responses
                  score 0, so a score drop at tight budgets can mean "ran out of
                  room" rather than "got it wrong", and the two are not the same
                  finding.
    think/answer  mean tokens before and after </think>, by character count / 4
                  unless a tokenizer is given with --model
    spearman      rank correlation between budget and score over the rungs
    inversions    pairs of rungs where a BIGGER budget scored worse by more than
                  the binomial standard error. Report this, not just spearman —
                  one large inversion hides easily inside a good correlation,
                  and it is the failure that actually matters.

Usage:
    python -m full_mix.lcpo.analyze_budget_curve --root <sweep results dir>
    python -m full_mix.lcpo.analyze_budget_curve --root ... --model <path>  # exact tokens
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

THINK_CLOSE = "</think>"

# lm-eval metric keys, most specific first. The first one present in a sample
# is the score for that task.
METRIC_KEYS = ("math_verify", "exact_match", "prompt_level_strict_acc",
               "prompt_level_strict", "acc", "score")


def sample_score(rec: dict) -> float | None:
    for k in METRIC_KEYS:
        v = rec.get(k)
        if isinstance(v, (int, float, bool)):
            return float(v)
    return None


def raw_text(rec: dict) -> str:
    resps = rec.get("resps") or []
    if resps and isinstance(resps[0], list) and resps[0]:
        return resps[0][0]
    filt = rec.get("filtered_resps") or []
    return filt[0] if filt else ""


def variant_sort_key(name: str) -> tuple:
    m = re.match(r"budget_(\d+)$", name)
    if m:
        return (0, int(m.group(1)))
    return (1 if name == "free" else 2, 0)


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="sweep results dir (contains one dir per variant)")
    ap.add_argument("--model", default=None, help="tokenizer path for exact token counts")
    ap.add_argument("--out", default=None, help="write the report as json here")
    args = ap.parse_args()

    tok = None
    if args.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)

    def count(text: str) -> int:
        if not text:
            return 0
        return len(tok(text, add_special_tokens=False)["input_ids"]) if tok else len(text) // 4

    # (benchmark, variant) -> accumulator
    acc: dict[tuple, dict] = defaultdict(
        lambda: {"n": 0, "score": 0.0, "unclosed": 0, "think": 0, "answer": 0, "scored": 0}
    )
    for path in glob.glob(os.path.join(args.root, "*", "*", "*", "samples_*.jsonl")):
        parts = path.split(os.sep)
        variant, benchmark = parts[-4], parts[-3]
        a = acc[(benchmark, variant)]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                a["n"] += 1
                s = sample_score(rec)
                if s is not None:
                    a["score"] += s
                    a["scored"] += 1
                text = raw_text(rec)
                if THINK_CLOSE in text:
                    head, tail = text.split(THINK_CLOSE, 1)
                    a["think"] += count(head)
                    a["answer"] += count(tail)
                else:
                    a["unclosed"] += 1
                    a["think"] += count(text)

    if not acc:
        raise SystemExit(f"no samples_*.jsonl found under {args.root}")

    benchmarks = sorted({b for b, _ in acc})
    report = {"root": args.root, "by_benchmark": {}}

    for bench in benchmarks:
        variants = sorted((v for b, v in acc if b == bench), key=variant_sort_key)
        print(f"\n=== {bench} ===")
        print(f"  {'variant':14s} {'n':>5s} {'score':>8s} {'unclosed':>9s} {'think':>8s} {'answer':>8s}")
        rows = {}
        for v in variants:
            a = acc[(bench, v)]
            n = max(1, a["n"])
            row = {
                "n": a["n"],
                "score": a["score"] / a["scored"] if a["scored"] else float("nan"),
                "unclosed": a["unclosed"] / n,
                "think_tokens": a["think"] / n,
                "answer_tokens": a["answer"] / n,
            }
            rows[v] = row
            print(f"  {v:14s} {a['n']:5d} {row['score']:8.4f} {row['unclosed']:9.1%} "
                  f"{row['think_tokens']:8.0f} {row['answer_tokens']:8.0f}")

        # Monotonicity over the budgeted rungs only; free/nothink are not on the ladder.
        rungs = [(int(re.match(r"budget_(\d+)$", v).group(1)), v)
                 for v in variants if re.match(r"budget_(\d+)$", v)]
        rungs.sort()
        summary = {"rows": rows}
        if len(rungs) >= 3:
            xs = [b for b, _ in rungs]
            ys = [rows[v]["score"] for _, v in rungs]
            rho = spearman(xs, ys)
            inversions = []
            for i in range(len(rungs)):
                for j in range(i + 1, len(rungs)):
                    n_i = max(1, rows[rungs[i][1]]["n"])
                    se = math.sqrt(max(ys[i] * (1 - ys[i]), 1e-9) / n_i)
                    if ys[i] > ys[j] + se:
                        inversions.append({"lo": xs[i], "hi": xs[j],
                                           "lo_score": ys[i], "hi_score": ys[j],
                                           "gap": ys[i] - ys[j]})
            summary["spearman"] = rho
            summary["inversions"] = inversions
            print(f"  spearman(budget, score) = {rho:+.3f}   significant inversions: {len(inversions)}")
            for inv in inversions:
                print(f"    budget {inv['lo']} scored {inv['gap']:+.4f} ABOVE budget {inv['hi']}")
        report["by_benchmark"][bench] = summary

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
