"""Preprocess the allenai/IF_multi_constraints_upto5 dataset to parquet format for verl."""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="[Deprecated] Use --local_save_dir instead.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="Local path to the raw dataset, if available.")
    parser.add_argument(
        "--local_save_dir", default="~/data/ifeval", help="Directory to save the preprocessed parquets."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.01,
        help="Fraction of train data to hold out as test split when the dataset has no test split."
    )

    args = parser.parse_args()

    data_source = "allenai/IF_multi_constraints_upto5"

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    if "test" in dataset:
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
    else:
        split = dataset["train"].train_test_split(test_size=args.test_ratio, seed=42)
        train_dataset = split["train"]
        test_dataset = split["test"]

    def make_map_fn(split_name):
        def process_fn(example, idx):
            prompt = example["messages"]

            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "instruction_following",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example["ground_truth"],
                },
                "extra_info": {
                    "split": split_name,
                    "index": idx,
                    "key": example.get("key", ""),
                    "constraint": example.get("constraint", ""),
                    "constraint_type": example.get("constraint_type", ""),
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Train split: {len(train_dataset)} examples -> {train_path}")
    print(f"Test split:  {len(test_dataset)} examples -> {test_path}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
