"""Measure how reliably a marker-SFT model obeys /think and /no_think.

Samples held-out prompts at the ROLLOUT sampling params (temperature 1.0,
top_p 1.0 by default) rather than greedily, because the number that matters is
adherence during GRPO rollouts, not adherence at temperature 0.

Prompts come from data/chat_ifeval_mix2 — the RL set, which is disjoint by
construction from data/chat_ifeval_remaining that the SFT data was generated
from. So these are prompts the SFT model never trained on, and they are exactly
the distribution RL will run on.

Truncation is reported SEPARATELY from non-adherence. A response cut off at
max_tokens has no closing </think> and would otherwise be scored malformed,
which silently inflates the "disobeyed" count and looks like a policy problem
when it is a budget problem.

Usage:
    python -m full_mix.sft_modes.probe_mode_adherence \
        --model /data/abdelrahman/qwen-sft/full_scale/out-qwen3-4b/dual-mode-sft-tagged
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SOURCES = {
    "chat": "chat_with_baseline_train.parquet",
    "ifeval": "ifeval_train.parquet",
}

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def classify(text: str, min_think_chars: int) -> dict:
    """Classify one raw generation into its observed mode.

    Returns keys:
      well_formed  opens <think>, closes it, non-empty answer after
      is_thinking  reasoning body >= min_think_chars
      is_instant   reasoning body is whitespace-only
      think_chars  length of the reasoning body
    """
    out = {"well_formed": False, "is_thinking": False, "is_instant": False,
           "think_chars": 0, "answer_chars": 0}
    if not text:
        return out
    s = text.lstrip()
    if not s.startswith(THINK_OPEN) or THINK_CLOSE not in s:
        return out
    body = s[len(THINK_OPEN):]
    reasoning, answer = body.split(THINK_CLOSE, 1)
    reasoning, answer = reasoning.strip(), answer.strip()
    if not answer:
        return out
    out["well_formed"] = True
    out["think_chars"] = len(reasoning)
    out["answer_chars"] = len(answer)
    out["is_instant"] = len(reasoning) == 0
    out["is_thinking"] = len(reasoning) >= min_think_chars
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default="/data/abdelrahman/verl/data/chat_ifeval_mix2")
    ap.add_argument("--n_prompts", type=int, default=200, help="per source")
    ap.add_argument("--n_samples", type=int, default=4, help="generations per (prompt, mode)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--min_think_chars", type=int, default=200)
    ap.add_argument("--marker_sep", default="\n\n")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--out", default="/data/abdelrahman/verl/data/sft_dual_mode/adherence_probe.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)

    # Build (source, mode, prompt) instances.
    instances = []
    for src, fname in SOURCES.items():
        rows = pq.read_table(os.path.join(args.data_dir, fname)).to_pylist()[: args.n_prompts]
        for row in rows:
            for mode, marker in (("thinking", "/think"), ("instant", "/no_think")):
                msgs = [dict(m) for m in row["prompt"]]
                msgs[-1]["content"] = msgs[-1]["content"].rstrip() + args.marker_sep + marker
                text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                instances.append({"source": src, "mode": mode, "text": text})

    print(f"prompts    : {args.n_prompts} per source x 2 modes = {len(instances)} instances")
    print(f"sampling   : n={args.n_samples} temp={args.temperature} top_p={args.top_p} "
          f"max_tokens={args.max_tokens}")

    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_tokens + 4096, enforce_eager=False)
    sp = SamplingParams(n=args.n_samples, temperature=args.temperature,
                        top_p=args.top_p, max_tokens=args.max_tokens)
    outs = llm.generate([i["text"] for i in instances], sp)

    agg = defaultdict(lambda: {"n": 0, "adherent": 0, "malformed": 0, "truncated": 0,
                               "wrong_mode": 0, "think_chars": [], "answer_chars": []})
    for inst, out in zip(instances, outs):
        key = (inst["source"], inst["mode"])
        for comp in out.outputs:
            a = agg[key]
            a["n"] += 1
            truncated = comp.finish_reason == "length"
            c = classify(comp.text, args.min_think_chars)
            if truncated:
                a["truncated"] += 1
            if not c["well_formed"]:
                a["malformed"] += 1
                continue
            a["think_chars"].append(c["think_chars"])
            a["answer_chars"].append(c["answer_chars"])
            ok = c["is_thinking"] if inst["mode"] == "thinking" else c["is_instant"]
            if ok:
                a["adherent"] += 1
            else:
                a["wrong_mode"] += 1

    report = {"model": args.model, "sampling": {"n": args.n_samples,
              "temperature": args.temperature, "top_p": args.top_p,
              "max_tokens": args.max_tokens}, "min_think_chars": args.min_think_chars,
              "by_source_mode": {}}
    print(f"\n{'source/mode':22s} {'n':>6s} {'adherent':>10s} {'wrong':>7s} "
          f"{'malformed':>10s} {'trunc':>7s} {'think_chars p50':>16s}")
    for (src, mode), a in sorted(agg.items()):
        p50 = int(statistics.median(a["think_chars"])) if a["think_chars"] else 0
        rate = a["adherent"] / a["n"] if a["n"] else 0.0
        print(f"{src + '/' + mode:22s} {a['n']:6d} {rate:9.1%} {a['wrong_mode']:7d} "
              f"{a['malformed']:10d} {a['truncated']:7d} {p50:16d}")
        report["by_source_mode"][f"{src}/{mode}"] = {
            "n": a["n"], "adherence_rate": rate, "wrong_mode": a["wrong_mode"],
            "malformed": a["malformed"], "truncated": a["truncated"],
            "think_chars_p50": p50,
            "answer_chars_p50": int(statistics.median(a["answer_chars"])) if a["answer_chars"] else 0,
        }

    tot_n = sum(a["n"] for a in agg.values())
    tot_ok = sum(a["adherent"] for a in agg.values())
    report["overall_adherence"] = tot_ok / tot_n if tot_n else 0.0
    print(f"\noverall adherence: {report['overall_adherence']:.1%}  ({tot_ok}/{tot_n})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
