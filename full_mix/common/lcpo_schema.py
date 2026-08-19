"""One extra_info schema shared by every parquet this project writes.

verl builds a training dataset by concatenating its train files and a validation
dataset by concatenating its val files. Within each set the extra_info structs
must line up, or `datasets.concatenate_datasets` either raises at launch or
silently coerces the columns.

The training sources and the eval sources disagree on their own:

    chat    baseline_model, baseline_response, custom_id, index, split, user_prompt
    ifeval  constraint, constraint_type, index, key, split
    math    index, original_dataset, split
    gsm8k   answer, index, question, split
    math500 index, level, split, subject
    ifbench index, key, split

So both builders homogenize to the union below. Keeping ONE list here — rather
than a copy in each builder — means train and val rows stay describable by the
same schema even though nothing forces it today.

`index` stays an int and is never rewritten: it carries row identity, and
`curriculum/encoding.py` derives prompt_uid from it, so it has to stay
comparable with every earlier mix built from these parquets.
"""

# Every field is a string. `index` is handled separately as an int.
EXTRA_INFO_STR_FIELDS = [
    "answer",
    "baseline_model",
    "baseline_response",
    "constraint",
    "constraint_type",
    "custom_id",
    "key",
    "level",
    "original_dataset",
    "question",
    "split",
    "subject",
    "user_prompt",
]


def homogenize_extra_info(ei: dict) -> dict:
    """Project one source's extra_info onto the shared schema.

    Missing fields become "" rather than None so the pyarrow column type stays a
    plain string. Values are stringified because `level` arrives as an int from
    math500 and as nothing elsewhere, and a column cannot be both.
    """
    ei = ei or {}
    out = {f: ("" if ei.get(f) is None else str(ei.get(f))) for f in EXTRA_INFO_STR_FIELDS}
    out["index"] = int(ei.get("index", 0))
    return out
