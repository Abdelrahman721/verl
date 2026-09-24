# QLCM RL handoff — datasets + reward functions

GRPO training data and the reward functions that score it, lifted out of a verl
pipeline. Everything here is model-independent: parquets and Python, no
checkpoints, no trainer config, no curriculum machinery.

The datasets are the **portions we never trained on** — our runs were stopped
mid-epoch, so roughly a third of each set was never rolled out. Identity is the
exception and ships whole (new corpus, see below).

```
handoff/
├── README.md
├── example.env                every env var the reward code reads, with defaults
├── datasets/                  5 parquets, verl-shaped
└── reward_code/qlcm/          importable package, 2 reward entry points
```

---

## 1. Datasets → reward function

| file | rows | `data_source` | reward entry point |
|---|---:|---|---|
| `general_ifeval_unseen.parquet` | 10,752 | `local/dolci-ifeval-32b` | `qlcm/rewards/compute_score.py` |
| `general_chat_unseen.parquet` | 7,536 | `local/dolci-chat-32b` | `qlcm/rewards/compute_score.py` |
| `general_safety_unseen.parquet` | 4,281 | `local/safety-dpo-reference` | `qlcm/rewards/compute_score.py` |
| `identity_berry_train.parquet` | 1,581 | `local/avey-identity` | `qlcm/rewards/compute_score.py` |
| `stage2_unseen.parquet` | 10,552 | `medical_qa` (5,298), `medical_conv` (5,254) | `qlcm/rewards/qa_openrouter.py` |

Both entry points are named `compute_score`. `compute_score.py` routes internally
on `data_source`, so the four general/identity files can be trained together as
one mix. `stage2_unseen.parquet` uses a different entry point — see §4 before
mixing it with the others.

**Identity** is the "Avey Berry" corpus. It ships whole rather than as an unseen
remainder because the gold responses were rewritten: against the previous
identity corpus, 963 of the overlapping golds changed and 390 prompts are new.
The training target moved, so "already seen" doesn't carry over. It is an
ordinary family inside `compute_score.py`, scored by the same judge as chat and
safety — nothing about it is special-cased.

---

## 2. Verify before spending GPU time

```bash
cd reward_code
python -c "import sys; sys.path.insert(0,'.'); \
  import qlcm.rewards.compute_score, qlcm.rewards.qa_openrouter; print('ok')"
```

Then score one row with **no judge and no API key** — ifeval is rule-based:

```python
import sys; sys.path.insert(0, "reward_code")
import pyarrow.parquet as pq
from qlcm.rewards.compute_score import compute_score
r = pq.read_table("datasets/general_ifeval_unseen.parquet").slice(0,1).to_pylist()[0]
print(compute_score(data_source=r["data_source"],
                    solution_str="<think>x</think>hi",
                    ground_truth=r["reward_model"]["ground_truth"],
                    extra_info=r["extra_info"]))
# -> {'score': 0.0, 'prompt_uid': ..., 'data_source_code': 10}
```

Then set the env (§5) and score an identity row both ways — a correct answer
should return 1.0 and a "I'm ChatGPT" answer 0.0. If both come back 0.0 your
judge is unreachable; see gotcha 1.

---

## 3. Row schema

Every row in every file:

```python
{
  "data_source":  str,      # the dispatch key from the table above
  "prompt":       [{"role": "user", "content": str}],
  "ability":      str,      # informational; the reward does not read it
  "reward_model": {"style": "rule", "ground_truth": str},
  "extra_info":   {...},    # per-family; always has "index" and "split"
}
```

`reward_model.ground_truth` is **always a JSON-encoded string**, even when it
holds an object. The reward functions `json.loads` it. Keep it a string.

`extra_info` fields differ per family. Two that the reward actually reads:

- `user_prompt` — **required** for chat, safety and identity. The identity judge
  scores the rollout against the gold answer *for that prompt*; passing `""`
  still returns a number, but a materially worse one.
- `index` — the row's position in its original source file. `prompt_uid` is
  derived from it (`DATASET_ID * 10_000_000 + index`), so it must survive any
  filtering or shuffling you do. Slicing rows is safe; renumbering is not.

---

## 4. Wiring into verl

```bash
reward.custom_reward_function.path=/abs/path/reward_code/qlcm/rewards/compute_score.py
reward.custom_reward_function.name=compute_score
reward.reward_manager.name=dapo
```

`reward_code/` must be importable — either put it on `PYTHONPATH` or keep the
`qlcm/` tree intact next to the file you point at (the modules use absolute
`qlcm.*` imports, and `ifeval_reward.py` puts its own directory on `sys.path` to
reach the vendored `qlcm/ifeval/`).

### Both entry points accept singular OR batch kwargs

```python
compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
compute_score(data_sources=[...], solution_strs=[...], ground_truths=[...], extra_infos=[...])
```

Use the batch form. It partitions by family and runs the judge calls for the
whole batch concurrently; scoring row-by-row against a judge API is far slower.

### They return dicts, and the key sets DIFFER

`score` is the scalar reward; the rest is per-rollout logging.

```
compute_score.py    -> 3 keys   {score, prompt_uid, data_source_code}
qa_openrouter.py    -> 17 keys  {score, reward/accuracy, reward/clarity, ...}
```

verl's `DataProto.concat` asserts the **first** row's key set is a **superset**
of every later row's. Each entry point is internally consistent, but the two are
**not mutually compatible**. If you train the general/identity files and
`stage2_unseen` in one mixed run, you need a dispatcher that routes on
`data_source` and pads every result to the union of both key sets. Ours is not
included — writing it is your call. Training the two groups as separate runs
needs no such wrapper.

Related: the rollout-dump path casts reward values through numpy and fails on
string arrays, so every value must be int/float. That is why `data_source` is
echoed back as an integer `data_source_code` rather than a string.

---

## 5. Environment

`example.env` lists all 36 variables the package reads, with defaults, grouped
by judge. Two namespaces, no fallback between them:

- `QLCM_JUDGE_*` — chat, safety, identity (`compute_score.py`)
- `QA_JUDGE_*` — medical QA and conversations (`qa_openrouter.py`)

ifeval needs no judge at all. Required: an API base, a model, and a key for each
namespace you use.

---

## 6. Gotchas

**1. The judges fail closed, and it looks like a bad model.** An unreachable or
misauthenticated judge scores identity rows 0.0 and returns floor scores
elsewhere. Every row then carries the same constant reward, which under GRPO is
zero advantage — the run burns full generation cost and learns nothing while the
logs look healthy. Check for `[judge_client boot pid=...]` with `HAS_KEY=True` in
**every worker's** log, not just the launcher's.

**2. Env vars must reach the Ray actors.** Exports in a launch script configure
the driver only; reward workers read from the job `runtime_env` or the raylet.
Pass `ray job submit --runtime-env=...`, or push them as Hydra overrides
(`+ray_kwargs.ray_init.runtime_env.env_vars.KEY='value'`). Do **not** do both —
Ray rejects a driver runtime_env repeating any key the job runtime_env declares,
even with identical values, and the job dies at `ray.init()`.

**3. The parquet schemas are not mutually compatible.** Five distinct
`extra_info` struct types, and the general files use `string` where
`stage2_unseen` uses `large_string`. HuggingFace `datasets` concat-loads multiple
train files into one dataset and requires identical Arrow schemas. The four
general/identity files can be widened to a common union; mixing them with
`stage2_unseen` also needs `string` → `large_string`. This fails deep inside the
dataset loader, not at config time.

**4. `QLCM_IDENTITY_GATE` is off, and should stay off.** It applies an identity
check to *every non-identity row* — one extra judge call per rollout across all
domains. It was the dominant source of judge failures in our runs. The inner
switch `QLCM_GLOBAL_IDENTITY_CHECK_ENABLED` defaults to `True` in code, so
`example.env` sets it to `0` explicitly; enabling the outer gate without that
would silently start billing a judge call per sample.

**5. Judge failures score NEUTRAL, not 0.0.** Chat 0.5, safety
`QLCM_SAFETY_JUDGE_FAILURE_REWARD`, QA `QA_JUDGE_REFUSAL_REWARD`. A failed judge
produced no verdict; scoring 0.0 would assert "worst possible answer" on the
strength of an upstream 429. Genuine wrong-stance and wrong-identity verdicts
still score 0.0. Preserve this if you edit the scorers.

**6. A length cost is ON by default for stage 2.** `qa_openrouter.py` subtracts
`QLCM_LEN_COST_PER_1K_WORDS` (0.02) × total words, unbounded and unfloored, so
scores can go slightly negative. That is intended — it mirrors how verl's DAPO
manager applies its own overlong penalty. Set `QLCM_LEN_PENALTY_ENABLE=0` to
disable. The coefficient is model-specific; ours is an operating point, not a
recommendation.

**7. The judge output schemas are trimmed.** `QLCM_JUDGE_TRIM_QA` /
`QLCM_JUDGE_TRIM_CONV` drop the `*_justification` fields the reward never reads
(~29% cheaper, no measured score effect over paired repeats). The `*_category`
fields are kept deliberately — they precede each score in the schema and
scaffold it. Set either to `0` to restore the full schema.

---

## 7. Not included

- **Model checkpoints.** Nothing here depends on one.
- **The ~2/3 of each dataset we already trained on.** Ask if you want the full
  sets rather than the unused remainder.
- **A multi-family dispatcher** that pads across both entry points (§4).
- **Curriculum scripts.** `stage2_unseen` is a built parquet, not a recipe. Its
  difficulty curve came from *our* model's success rates; if your model differs
  substantially, that balance will not transfer.
- **Coding/SNOMED/ICD rewards** and the benchmark scorers (mcq, numeric, medec,
  text) — those belong to later stages and are not needed by any file here.
