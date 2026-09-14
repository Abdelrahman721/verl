"""Tests for the continuous length cost.

Run:  python3 qlcm/rewards/test_length_penalty.py
The replay section needs the stage-1 dumps; it skips cleanly without them.
"""
import os, sys, json, glob, statistics as st
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qlcm.rewards import length_penalty as lp

fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond: fails.append(name)

GRID = list(range(0, 30000, 100))

print(f"\n--- shape: linear at {lp.COST_PER_1K} per 1,000 words ---")
check("zero at zero words",     lp.penalty_for(0) == 0.0)
check("never positive",         all(lp.penalty_for(n) <= 0 for n in GRID))
check("exact value at 1k",      abs(lp.penalty_for(1000) + lp.COST_PER_1K) < 1e-12)
check("proportional",           abs(lp.penalty_for(4000) - 2 * lp.penalty_for(2000)) < 1e-12)
check("STRICTLY decreasing",    all(lp.penalty_for(n) > lp.penalty_for(n + 100) for n in GRID))
check("no dead zone anywhere",  all(abs(lp.penalty_for(n) - lp.penalty_for(n + 100)) > 1e-9 for n in GRID))
check("no saturation in the tail",
      lp.penalty_for(30000) < lp.penalty_for(20000) < lp.penalty_for(10000))
check("bounded by the generation cap",
      abs(lp.penalty_for(5900)) < 0.5, f"at 8,192 tokens ~= {lp.penalty_for(5900):.3f}")

print("\n--- the property that removes the need for a threshold ---")
# advantage = r_i - mean(r_group), so a constant shift of every length in a
# group must leave every advantage untouched. This is what makes the cost
# per-prompt adaptive with no reference length hard-coded anywhere.
group = [(900, 0.42), (1300, 0.55), (1800, 0.61), (2600, 0.48)]
def advantages(rows, shift=0):
    r = [s + lp.penalty_for(L + shift) for L, s in rows]
    m = sum(r) / len(r)
    return [x - m for x in r]
a0, a1 = advantages(group), advantages(group, shift=1500)
check("group-shift invariant", all(abs(x - y) < 1e-12 for x, y in zip(a0, a1)),
      "a whole group running longer incurs no net pressure")
check("still ranks shorter first within the group",
      advantages(group)[0] - advantages(group)[3] > 0,
      "the shortest sample keeps the largest length credit")

print("\n--- word counting ---")
check("empty is zero",              lp.count_words("") == 0)
check("no think block",             lp.count_words("a b c") == 3)
check("think + answer",             lp.count_words("<think>a b</think>c d") == 4)
check("delimiters not counted",     lp.count_words("<think>a</think>b") == 2)
check("unterminated think counts all", lp.count_words("<think>a b c") == 3)
check("multiline",                  lp.count_words("<think>a\nb</think>\nc") == 3)

print("\n--- apply() ---")
r = {"score": 0.8, "reward/reasoning_length": 3000, "reward/answer_length": 500}
lp.apply(r, "")
check("folds into score",   abs(r["score"] - (0.8 - 3.5 * lp.COST_PER_1K)) < 1e-12,
      f"0.8 -> {r['score']:.4f}")
check("records cost",       r[lp.PENALTY_KEY] < 0)
check("records word count", r[lp.WORDS_KEY] == 3500)

before = dict(r)
lp.apply(r, "")
check("idempotent (no double charge)", r["score"] == before["score"],
      f"{before['score']:.4f} -> {r['score']:.4f}")

r2 = {"score": 0.5, "reward/reasoning_length": 40, "reward/answer_length": 60}
lp.apply(r2, "")
check("short rows are charged too, just less", 0 > r2[lp.PENALTY_KEY] > -0.01,
      f"{r2[lp.PENALTY_KEY]:.5f} at 100 words")

r3 = {"score": 0.02, "reward/reasoning_length": 9000, "reward/answer_length": 0}
lp.apply(r3, "")
check("NOT floored at zero", r3["score"] < 0,
      f"{r3['score']:.3f} — flooring would recreate a dead zone")

r4 = {"score": 0.9}
lp.apply(r4, "<think>" + "w " * 3000 + "</think>" + "a " * 500)
check("falls back to raw counting", r4[lp.WORDS_KEY] == 3500, f"counted {r4[lp.WORDS_KEY]}")
check("non-dict passthrough",       lp.apply(None, "x") is None)

print("\n--- nesting: bench applies, then the mix pads and applies again ---")
import importlib.util
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mix = load("_mix_under_test", os.path.join(_root, "rewards", "compute_score_medical_mix.py"))

# ORDER THAT THE DISPATCHERS ACTUALLY USE: pad first, then apply. _pad injects
# the key set from UNION_DEFAULTS, so a presence-based guard would skip here
# and the mechanism would be inert on every row. This is the regression test
# for exactly that bug.
padded_first = mix._pad({"score": 0.8, "reward/reasoning_length": 3000,
                         "reward/answer_length": 500})
lp.apply(padded_first, "")
check("APPLIES after padding (pad -> apply)", padded_first[lp.PENALTY_KEY] < 0,
      f"score {padded_first['score']:.4f}, {padded_first[lp.WORDS_KEY]} words")
check("charge flag set", padded_first[lp.CHARGED_KEY] == 1.0)
lp.apply(padded_first, "")
check("and is still idempotent", padded_first["score"] == 0.8 - 3.5 * lp.COST_PER_1K)

# Reverse order (an inner scorer charges, then the mix pads and retries).
inner = {"score": 0.8, "reward/reasoning_length": 3000, "reward/answer_length": 500}
lp.apply(inner, "")
once = inner["score"]
padded = mix._pad(inner)
lp.apply(padded, "")
check("padding preserves the cost", padded[lp.PENALTY_KEY] == inner[lp.PENALTY_KEY])
check("mix does not double-charge (apply -> pad)", padded["score"] == once,
      f"{once:.4f} -> {padded['score']:.4f}")

short = mix._pad(lp.apply({"score": .5, "reward/reasoning_length": 50,   "reward/answer_length": 50}, ""))
long_ = mix._pad(lp.apply({"score": .5, "reward/reasoning_length": 5000, "reward/answer_length": 50}, ""))
check("padded rows share an identical key set", set(short) == set(long_),
      f"diff {set(short) ^ set(long_) or 'none'}")
check("new keys present in the union", {lp.PENALTY_KEY, lp.WORDS_KEY} <= set(short))
check("all reward values numeric (dump path casts via numpy)",
      all(isinstance(v, (int, float)) for k, v in short.items()
          if k.startswith("reward/") and k not in ("reward/gold", "reward/eval_mode")))

print("\n--- kill switch ---")
_saved = lp.ENABLED
lp.ENABLED = False
off = {"score": 0.8, "reward/reasoning_length": 9000, "reward/answer_length": 0}
lp.apply(off, "")
check("disabled leaves score untouched", off["score"] == 0.8)
check("disabled still emits the keys", off[lp.PENALTY_KEY] == 0.0 and off[lp.WORDS_KEY] == 9000)
lp.ENABLED = _saved

print("\n--- replay against the real stage-1 rollouts ---")
D = "/data/abdelrahman/verl/dumps/qlcm/medical_qa_rollouts"
files = glob.glob(os.path.join(D, "*.jsonl")) if os.path.isdir(D) else []
if not files:
    print("  SKIP  dumps not present")
else:
    g = defaultdict(list)
    for f in files:
        if int(os.path.basename(f).split(".")[0]) < 280: continue
        for line in open(f):
            line = line.strip()
            if not line: continue
            try: d = json.loads(line)
            except: continue
            if d.get("reward/reasoning_length", 0):
                g[d.get("input", "")].append(
                    (d["reward/reasoning_length"] + d["reward/answer_length"], d["score"]))
    grp = [v for v in g.values() if len(v) >= 8]
    allr = [x for v in grp for x in v]
    sstd = st.mean([st.pstdev([s for _, s in v]) for v in grp])
    lstd = st.mean([st.pstdev([L for L, _ in v]) for v in grp])
    ql, sv = [], []
    for v in grp:
        b0 = max(v, key=lambda x: x[1])
        b1 = max(v, key=lambda x: x[1] + lp.penalty_for(x[0]))
        ql.append(b0[1] - b1[1]); sv.append(b0[0] - b1[0])
    dead = sum(1 for L, _ in allr
               if abs(lp.penalty_for(L) - lp.penalty_for(L + 1)) < 1e-12) / len(allr)
    sig = lp.COST_PER_1K * lstd / 1000.0
    print(f"  n={len(allr):,} in {len(grp):,} groups   score std {sstd:.3f}   length std {lstd:,.0f} words")
    print(f"  length signal {sig:.4f} ({sig/sstd:.2f}x task)   judge cost {st.mean(ql):.4f}   "
          f"words saved {st.mean(sv):,.0f}   dead zone {100*dead:.1f}%")
    check("zero dead zone on real data", dead == 0.0,
          "the shipped band design left 67.2% flat")
    check("length stays subordinate to quality", 0.15 < sig / sstd < 0.40, f"{sig/sstd:.2f}x")
    check("cheaper per word than the band design",
          st.mean(ql) < 0.0094 and st.mean(sv) > 140,
          f"{st.mean(sv):,.0f} words for {st.mean(ql):.4f} vs 152 for 0.0094")

print("\n--- end to end through the real dispatcher (rule-scored row, no judge) ---")
IF_PARQUET = "/data/abdelrahman/verl/data/qlcm/curated/ifeval_train.parquet"
if not os.path.exists(IF_PARQUET):
    print("  SKIP  curated ifeval parquet not present")
else:
    import pyarrow.parquet as pq
    row = pq.read_table(IF_PARQUET).slice(0, 1).to_pylist()[0]
    ds, gt = row["data_source"], row["reward_model"]["ground_truth"]
    ei = row.get("extra_info") or {}
    def score_of(nwords):
        sol = "<think>" + "reason " * nwords + "</think>My answer."
        return mix.compute_score(data_source=ds, solution_str=sol,
                                 ground_truth=gt, extra_info=dict(ei))
    a, b = score_of(100), score_of(3000)
    print(f"  ~110 words  -> words={a[lp.WORDS_KEY]:>5}  cost={a[lp.PENALTY_KEY]:+.4f}  score={a['score']:+.4f}")
    print(f"  ~3010 words -> words={b[lp.WORDS_KEY]:>5}  cost={b[lp.PENALTY_KEY]:+.4f}  score={b['score']:+.4f}")
    check("dispatcher counts words", a[lp.WORDS_KEY] > 0 and b[lp.WORDS_KEY] > a[lp.WORDS_KEY])
    check("dispatcher charges the cost", b[lp.PENALTY_KEY] < a[lp.PENALTY_KEY] < 0)
    check("longer scores strictly lower", b["score"] < a["score"],
          f"{a['score']:+.4f} vs {b['score']:+.4f}")
    check("both rows share a key set", set(a) == set(b))

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
