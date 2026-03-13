#!/usr/bin/env python3
"""Merge LoRA adapter weights into the base model and save the result."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(adapter_path: str, output_path: str, dtype: str = "bfloat16"):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    adapter_config = json.loads(
        (Path(adapter_path) / "adapter_config.json").read_text()
    )
    base_model = adapter_config["base_model_name_or_path"]

    print(f"Loading base model: {base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch_dtype)

    print("Merging weights ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path} ...")
    model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="./model",
        help="Path to the LoRA adapter directory (default: ./model)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./model_merged",
        help="Where to save the merged model (default: ./model_merged)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for loading the model (default: bfloat16)",
    )
    args = parser.parse_args()
    merge(args.adapter_path, args.output_path, args.dtype)
