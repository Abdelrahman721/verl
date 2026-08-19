#!/usr/bin/env python3
"""
Optimized evaluation script for the fine-tuned Qwen model using vLLM.

This script provides significant speedup through:
- vLLM's optimized attention mechanisms
- Efficient batching
- PagedAttention for memory efficiency
- Continuous batching for better throughput

Functions:
- evaluate_model_vllm: Single-label evaluation for ICD-10 code prediction
- evaluate_multi_model_vllm: Multi-label evaluation for ICD-10 code prediction
  Extracts all ICD codes after #### and calculates precision, recall, F1 at:
  * Instance-based level (averaged per instance)
  * Micro-averaged level (global across all labels)
  * Macro-averaged level (averaged per label)
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer  # CHANGED: needed to apply the chat template
from vllm import LLM, SamplingParams

os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen model with vLLM optimization")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="agadelmoula-avey/Qwen3-4B-Base",
        help="Path to the model checkpoint directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./train_set_sft_trajectories_a.parquet",
        help="Path to the evaluation dataset (parquet file)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for inference (default: 32, much larger than original)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,  # CHANGED: was 2048. Match verl MAX_RESPONSE_LEN; the model
                        # writes ~3k-token <think> traces before the answer, so 2048
                        # truncated the reasoning before the code list was emitted.
        help="Maximum tokens to generate (default: 16384, matches verl MAX_RESPONSE_LEN)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation_vllm_results.json",
        help="Output file for detailed results"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio (default: 0.9)"
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs for data parallelism (default: 2)"
    )
    parser.add_argument(
        "--dp-num-nodes",
        type=int,
        default=1,
        help="Number of nodes for data parallelism (default: 1)"
    )
    parser.add_argument(
        "--dp-node-rank",
        type=int,
        default=0,
        help="Rank of the node for data parallelism (default: 0)"
    )
    parser.add_argument("--dp-master-addr", type=str, default="",
        help="Address for data parallelism (default: '')"
    )
    parser.add_argument("--dp-master-port", type=int, default=0,
        help="Port for data parallelism (default: 0)"
    )
    parser.add_argument("--timeout", type=int, default=300,
        help="Timeout for data parallelism (default: 300)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multi"],
        default="single",
        help="Evaluation mode: 'single' for single-label, 'multi' for multi-label (default: single)"
    )
    return parser.parse_args()


def load_dataset(dataset_path: str, max_samples: int = None) -> pd.DataFrame:
    """Load the evaluation dataset."""
    print(f"Loading dataset from {dataset_path}...")

    df = pd.read_parquet(dataset_path)
    # shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    print(f"Dataset loaded. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    if max_samples is not None:
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples")

    return df


def format_input_for_model(input_text: str) -> str:
    """Format the input text for the model."""
    # The input column contains the conversation in the format expected by the model
    # We need to extract just the user message part for inference

    # Parse the input to extract the actual user query
    input_text = input_text
    if "user\n" in input_text and "assistant\n" in input_text:
        # Split by the assistant marker to get just the user part
        user_part = input_text.split("assistant\n")[0]
        # Remove the "user\n" prefix
        user_message = user_part.replace("user\n", "").strip()
    else:
        # Fallback: use the input as-is
        user_message = input_text.strip()

    # Format as a conversation for the model
    formatted_input = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    # print(formatted_input)
    # formatted_input = user_message

    return formatted_input


def extract_final_answer(text: str) -> str:
    import re
    """Extract the final answer from the generated text."""
    pattern = r'\b[A-Z][A-Z0-9]{2}(?:\.[A-Z0-9]{1,4})?\b'
    matches = re.findall(pattern, text)
    if len(matches) == 0:
        return "XXXXXX"
    return matches[-1]


def extract_multi_icd_codes(text: str) -> List[str]:
    """Extract all ICD-10 codes after #### marker."""
    import re
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    # Split by #### and take everything after it
    if '####' in text:
        text = text.split('####', 1)[1]

    # Extract all ICD-10 codes using the specified pattern
    icd_pattern = re.compile(r'\b[A-Z][A-Z0-9]{2}(?:\.[A-Z0-9]{1,4})?\b')
    matches = icd_pattern.findall(text)

    # Remove duplicates while preserving order
    seen = set()
    unique_matches = []
    for match in matches:
        if match.upper() not in seen:
            seen.add(match.upper())
            unique_matches.append(match.upper())

    return unique_matches


def calculate_multilabel_metrics(predictions: List[List[str]], targets: List[List[str]]) -> Dict[str, Any]:
    """
    Calculate precision, recall, and F1 scores for multilabel classification.

    Calculates:
    - Instance-based (average metrics per instance)
    - Micro-averaged (global metrics across all labels)
    - Macro-averaged (average metrics per label)
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")

    # Instance-based metrics
    instance_precisions = []
    instance_recalls = []
    instance_f1s = []

    # For micro-averaging
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # For macro-averaging: collect all unique labels
    all_labels = set()
    for target in targets:
        all_labels.update(target)
    for pred in predictions:
        all_labels.update(pred)

    # Track TP, FP, FN per label for macro-averaging
    label_stats = {label: {'tp': 0, 'fp': 0, 'fn': 0} for label in all_labels}

    # Calculate instance-based and collect stats for micro/macro
    for pred_list, target_list in zip(predictions, targets):
        pred_set = set(pred_list)
        target_set = set(target_list)

        # True positives, false positives, false negatives for this instance
        tp = len(pred_set & target_set)
        fp = len(pred_set - target_set)
        fn = len(target_set - pred_set)

        # Instance-based metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        instance_precisions.append(precision)
        instance_recalls.append(recall)
        instance_f1s.append(f1)

        # Accumulate for micro-averaging
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Update per-label statistics for macro-averaging
        for label in pred_set & target_set:
            label_stats[label]['tp'] += 1
        for label in pred_set - target_set:
            label_stats[label]['fp'] += 1
        for label in target_set - pred_set:
            label_stats[label]['fn'] += 1

    # Calculate final metrics
    # Instance-based (average across instances)
    instance_precision = sum(instance_precisions) / len(instance_precisions) if instance_precisions else 0.0
    instance_recall = sum(instance_recalls) / len(instance_recalls) if instance_recalls else 0.0
    instance_f1 = sum(instance_f1s) / len(instance_f1s) if instance_f1s else 0.0

    # Micro-averaged (global across all predictions)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Macro-averaged (average across labels)
    label_precisions = []
    label_recalls = []
    label_f1s = []

    for label, stats in label_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        label_precisions.append(precision)
        label_recalls.append(recall)
        label_f1s.append(f1)

    macro_precision = sum(label_precisions) / len(label_precisions) if label_precisions else 0.0
    macro_recall = sum(label_recalls) / len(label_recalls) if label_recalls else 0.0
    macro_f1 = sum(label_f1s) / len(label_f1s) if label_f1s else 0.0

    return {
        'instance_based': {
            'precision': instance_precision,
            'recall': instance_recall,
            'f1': instance_f1
        },
        'micro': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1
        },
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'counts': {
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'num_labels': len(all_labels),
            'num_samples': len(predictions)
        }
    }


def normalize_medical_code(code: str) -> str:
    """Normalize medical codes for comparison."""
    # Remove extra whitespace and convert to uppercase
    code = code.strip().upper()

    # Handle multiple codes separated by commas
    if ',' in code:
        codes = [c.strip() for c in code.split(',')]
        return codes[0]  # Take the first/primary code

    return code


def calculate_flexible_accuracy(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Calculate multiple accuracy metrics."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")

    exact_matches = 0
    normalized_matches = 0
    partial_matches = 0

    for pred, target in zip(predictions, targets):
        # Extract final answers
        pred_answer = extract_final_answer(pred)
        target_answer = target

        # Exact match (case-insensitive)
        if pred_answer.lower().strip() == target_answer.lower().strip():
            exact_matches += 1
            normalized_matches += 1
            partial_matches += 1
            continue

        # Normalized match (remove suffixes, handle multiple codes)
        pred_normalized = normalize_medical_code(pred_answer)
        target_normalized = normalize_medical_code(target_answer)

        if pred_normalized == target_normalized:
            normalized_matches += 1
            partial_matches += 1
            continue

        # Partial match (check if target code is contained in prediction)
        # Only count as partial match if both codes are non-empty
        if (pred_normalized and target_normalized and
            (target_normalized in pred_normalized or pred_normalized in target_normalized)):
            partial_matches += 1

    total = len(predictions)
    return {
        'exact_accuracy': exact_matches / total,
        'normalized_accuracy': normalized_matches / total,
        'partial_accuracy': partial_matches / total,
        'exact_matches': exact_matches,
        'normalized_matches': normalized_matches,
        'partial_matches': partial_matches
    }


def evaluate_multi_model_vllm(args):
    """
    Evaluation function for multilabel ICD-10 code prediction using vLLM.
    Extracts all ICD codes after #### and calculates precision, recall, F1 at multiple levels.
    """
    # Load dataset
    df = load_dataset(args.dataset_path, args.max_samples)

    # Initialize vLLM model
    print(f"Loading model with vLLM from {args.checkpoint_path}...")
    print(f"Using tensor parallel size: {args.tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")

    # vLLM model initialization with fixed tokenizer
    llm = LLM(
        model=args.checkpoint_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_tokens + 8192,  # CHANGED: was =args.max_tokens. Must fit
                                                # prompt (≤8192, verl MAX_PROMPT_LEN) + full
                                                # response, else prompts/generations truncate.
        enforce_eager=False,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=128,  # 48 GB memory
    )

    # Sampling parameters optimized for speed and accuracy
    sampling_params = SamplingParams(
        temperature=1.0,  # Low temperature for more deterministic outputs
        top_p=0.7,  # CHANGED: was 0.9. Match verl val_kwargs.top_p (VAL_TOP_P=0.7)
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>", "</answer>", "\nuser"],  # Stop tokens
        skip_special_tokens=False,
        ignore_eos=False
    )

    print("Model loaded successfully!")

    # Prepare inputs
    print("Preparing inputs...")
    inputs = []
    targets = []

    # CHANGED: build the SAME chat-templated prompt verl feeds the model (system +
    # user turn + assistant generation prompt) via the model's own tokenizer,
    # instead of passing the raw prompt[0]['content']. This is the main fix.
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    for _, row in df.iterrows():
        messages = [{"role": m["role"], "content": m["content"]} for m in row['prompt']]
        # verl's rollout prepends this default system message when the prompt has
        # none (verl/workers/rollout/schemas.py BASE_CHAT_HISTORY); match it so the
        # rendered prompt is byte-identical to what verl fed the model.
        if not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages
        formatted_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs.append(formatted_input)

        # Parse target ICD codes (assuming they're stored as a list or comma-separated string)
        try:
            target_codes = set(eval(row['icd']))
            targets.append(target_codes)
        except:
            target_codes = set(eval(row['reward_model']['ground_truth']))
            targets.append(target_codes)


    # Prepare for evaluation
    predictions = []
    detailed_results = []

    print(f"\nStarting vLLM multilabel evaluation on {len(df)} samples...")
    print(f"Batch size: {args.batch_size}")
    start_time = time.time()

    # Process in batches for optimal performance
    for i in tqdm(range(0, len(inputs), args.batch_size), desc="Processing batches"):
        batch_inputs = inputs[i:i + args.batch_size]
        batch_targets = targets[i:i + args.batch_size]

        try:
            # Generate responses for the batch
            outputs = llm.generate(batch_inputs, sampling_params)

            # Process outputs
            for j, output in enumerate(outputs):
                completion = output.outputs[0]
                generated_text = completion.text.strip()

                if completion.finish_reason == "length":
                    print(f"Warning: Sample {i+j} reached max length")

                # Extract all ICD codes after ####
                pred_codes = extract_multi_icd_codes(generated_text)
                predictions.append(pred_codes)

                target_codes = batch_targets[j]

                # Calculate instance-level metrics for this sample
                pred_set = set(pred_codes)
                target_set = set(target_codes)

                tp = len(pred_set & target_set)
                fp = len(pred_set - target_set)
                fn = len(target_set - pred_set)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                # Store detailed result
                detailed_results.append({
                    'index': i + j,
                    'input': df.iloc[i + j]['prompt'][0]['content'],
                    'generated_text': generated_text,
                    'expected_codes': str(target_codes),
                    'predicted_codes': str(pred_codes),
                    'true_positives': list(pred_set & target_set),
                    'false_positives': list(pred_set - target_set),
                    'false_negatives': list(target_set - pred_set),
                    'instance_precision': precision,
                    'instance_recall': recall,
                    'instance_f1': f1
                })

        except Exception as e:
            print(f"Error processing batch {i//args.batch_size + 1}: {e}")
            # Add empty predictions to maintain alignment
            for j in range(len(batch_inputs)):
                predictions.append([])
                detailed_results.append({
                    'index': i + j,
                    'input': df.iloc[i + j]['prompt'][0]['content'],
                    'generated_text': "",
                    'expected_codes': str(batch_targets[j]),
                    'predicted_codes': [],
                    'true_positives': [],
                    'false_positives': [],
                    'false_negatives': str(batch_targets[j]),
                    'instance_precision': 0.0,
                    'instance_recall': 0.0,
                    'instance_f1': 0.0,
                    'error': str(e)
                })

    # Calculate multilabel metrics
    metrics = calculate_multilabel_metrics(predictions, targets)

    # Calculate additional statistics
    total_samples = len(predictions)
    end_time = time.time()
    evaluation_time = end_time - start_time

    # Print results
    print(f"\n{'='*60}")
    print("vLLM MULTILABEL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Total unique labels: {metrics['counts']['num_labels']}")
    print(f"Total TP: {metrics['counts']['total_tp']}")
    print(f"Total FP: {metrics['counts']['total_fp']}")
    print(f"Total FN: {metrics['counts']['total_fn']}")
    print()

    print("INSTANCE-BASED METRICS (averaged per instance):")
    print(f"  Precision: {metrics['instance_based']['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['instance_based']['recall']*100:.2f}%")
    print(f"  F1 Score:  {metrics['instance_based']['f1']*100:.2f}%")
    print()

    print("MICRO-AVERAGED METRICS (global across all labels):")
    print(f"  Precision: {metrics['micro']['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['micro']['recall']*100:.2f}%")
    print(f"  F1 Score:  {metrics['micro']['f1']*100:.2f}%")
    print()

    print("MACRO-AVERAGED METRICS (averaged per label):")
    print(f"  Precision: {metrics['macro']['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['macro']['recall']*100:.2f}%")
    print(f"  F1 Score:  {metrics['macro']['f1']*100:.2f}%")
    print()

    print(f"Evaluation time: {evaluation_time:.2f} seconds")
    print(f"Average time per sample: {evaluation_time/total_samples:.3f} seconds")
    print(f"Throughput: {total_samples/evaluation_time:.2f} samples/second")

    # Save detailed results
    results_summary = {
        'checkpoint_path': args.checkpoint_path,
        'dataset_path': args.dataset_path,
        'total_samples': total_samples,
        'metrics': metrics,
        'evaluation_time_seconds': evaluation_time,
        'throughput_samples_per_second': total_samples / evaluation_time,
        'vllm_config': {
            'tensor_parallel_size': args.tensor_parallel_size,
            'gpu_memory_utilization': args.gpu_memory_utilization,
            'batch_size': args.batch_size,
            'max_tokens': args.max_tokens,
        },
        'args': vars(args),
        'detailed_results': detailed_results
    }

    output_path = Path(args.output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_path}")

    # Show a few examples
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*60}")

    for i, result in enumerate(detailed_results[:3]):  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print(f"Input: {result['input'][:200]}...")
        print(f"Expected codes: {result['expected_codes']}")
        print(f"Predicted codes: {result['predicted_codes']}")
        print(f"True Positives: {result['true_positives']}")
        print(f"False Positives: {result['false_positives']}")
        print(f"False Negatives: {result['false_negatives']}")
        print(f"Precision: {result['instance_precision']*100:.2f}%")
        print(f"Recall: {result['instance_recall']*100:.2f}%")
        print(f"F1: {result['instance_f1']*100:.2f}%")
        print("-" * 40)

    return results_summary


def evaluate_model_vllm(args):
    """Main evaluation function using vLLM."""
    # Load dataset
    df = load_dataset(args.dataset_path, args.max_samples)

    # Initialize vLLM model
    print(f"Loading model with vLLM from {args.checkpoint_path}...")
    print(f"Using tensor parallel size: {args.tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")

    # vLLM model initialization with fixed tokenizer
    llm = LLM(
        model=args.checkpoint_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_tokens + 8192,  # CHANGED: was =args.max_tokens (see multi).
        enforce_eager=False,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=128,  # 48 GB memory
        # Fix for tokenizer regex pattern issue
        # tokenizer_mode="auto",
        # Add this if the model supports it
        # fix_mistral_regex=True,
    )

    # Sampling parameters optimized for speed and accuracy
    sampling_params = SamplingParams(
        temperature=0.0,  # Low temperature for more deterministic outputs
        top_p=0.9,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>", "</answer>", "\nuser"],  # Stop tokens
        skip_special_tokens=False,
        ignore_eos=False
    )

    print("Model loaded successfully!")

    # Prepare inputs
    print("Preparing inputs...")
    inputs = []
    targets = []

    # CHANGED: chat-template the prompt (same fix as the multi path) instead of
    # feeding the raw prompt[0]['content'].
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    for _, row in df.iterrows():
        messages = [{"role": m["role"], "content": m["content"]} for m in row['prompt']]
        # verl's rollout prepends this default system message when the prompt has
        # none (verl/workers/rollout/schemas.py BASE_CHAT_HISTORY); match it so the
        # rendered prompt is byte-identical to what verl fed the model.
        if not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages
        formatted_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs.append(formatted_input)
        targets.append(row['icd'])

    # Prepare for evaluation
    predictions = []
    detailed_results = []

    print(f"\nStarting vLLM evaluation on {len(df)} samples...")
    print(f"Batch size: {args.batch_size}")
    start_time = time.time()

    # Process in batches for optimal performance
    for i in tqdm(range(0, len(inputs), args.batch_size), desc="Processing batches"):
        batch_inputs = inputs[i:i + args.batch_size]
        batch_targets = targets[i:i + args.batch_size]

        try:
            # Generate responses for the batch
            outputs = llm.generate(batch_inputs, sampling_params)

            # Process outputs
            for j, output in enumerate(outputs):
                # if the output finish reason is length print it
                completion = output.outputs[0]
                generated_text = completion.text.strip()
                if completion.finish_reason == "length":
                    print(completion)
                predictions.append(generated_text)

                # Calculate matches for this sample
                pred_answer = extract_final_answer(generated_text)
                target_answer = batch_targets[j]

                exact_match = pred_answer.lower().strip() == target_answer.lower().strip()
                normalized_match = normalize_medical_code(pred_answer) == normalize_medical_code(target_answer)
                pred_normalized = normalize_medical_code(pred_answer)
                target_normalized = normalize_medical_code(target_answer)
                partial_match = (pred_normalized and target_normalized and
                               (target_normalized in pred_normalized or pred_normalized in target_normalized))
                existence_match = target_answer in generated_text

                # Store detailed result
                detailed_results.append({
                    'index': i + j,
                    'input': df.iloc[i + j]['prompt'][0]['content'],
                    'expected_output': batch_targets[j],
                    'predicted_output': generated_text,
                    'expected_code': target_answer,
                    'predicted_code': pred_answer,
                    'exact_match': exact_match,
                    'normalized_match': normalized_match,
                    'partial_match': partial_match,
                    'existence_match': existence_match
                })

        except Exception as e:
            print(f"Error processing batch {i//args.batch_size + 1}: {e}")
            # Add empty predictions to maintain alignment
            for j in range(len(batch_inputs)):
                predictions.append("")
                detailed_results.append({
                    'index': i + j,
                    'input': df.iloc[i + j]['prompt'][0]['content'],
                    'expected_output': batch_targets[j],
                    'predicted_output': "",
                    'expected_code': extract_final_answer(batch_targets[j]),
                    'predicted_code': "",
                    'exact_match': False,
                    'normalized_match': False,
                    'partial_match': False,
                    'existence_match': False,
                    'error': str(e)
                })

    # Calculate accuracy metrics
    accuracy_metrics = calculate_flexible_accuracy(predictions, targets)
    # calculate existence accuracy
    existence_accuracy = sum([1 for pred, target in zip(predictions, targets) if target in pred]) / len(predictions)

    # Calculate additional metrics
    total_samples = len(predictions)
    end_time = time.time()
    evaluation_time = end_time - start_time

    # Print results
    print(f"\n{'='*60}")
    print("vLLM EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Exact matches: {accuracy_metrics['exact_matches']} ({accuracy_metrics['exact_accuracy']*100:.2f}%)")
    print(f"Normalized matches: {accuracy_metrics['normalized_matches']} ({accuracy_metrics['normalized_accuracy']*100:.2f}%)")
    print(f"Partial matches: {accuracy_metrics['partial_matches']} ({accuracy_metrics['partial_accuracy']*100:.2f}%)")
    print(f"Existence matches: {existence_accuracy} ({existence_accuracy*100:.2f}%)")

    print(f"Evaluation time: {evaluation_time:.2f} seconds")
    print(f"Average time per sample: {evaluation_time/total_samples:.3f} seconds")
    print(f"Throughput: {total_samples/evaluation_time:.2f} samples/second")

    # Save detailed results
    results_summary = {
        'checkpoint_path': args.checkpoint_path,
        'dataset_path': args.dataset_path,
        'total_samples': total_samples,
        'accuracy_metrics': accuracy_metrics,
        'evaluation_time_seconds': evaluation_time,
        'throughput_samples_per_second': total_samples / evaluation_time,
        'vllm_config': {
            'tensor_parallel_size': args.tensor_parallel_size,
            'gpu_memory_utilization': args.gpu_memory_utilization,
            'batch_size': args.batch_size,
            'max_tokens': args.max_tokens,
        },
        'args': vars(args),
        'detailed_results': detailed_results
    }

    output_path = Path(args.output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_path}")

    # Show a few examples
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*60}")

    for i, result in enumerate(detailed_results[:3]):  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print(f"Input: {result['input'][:200]}...")
        print(f"Expected: {result['expected_code']}")
        print(f"Predicted: {result['predicted_code']}")
        print(f"Exact match: {result['exact_match']}")
        print(f"Normalized match: {result['normalized_match']}")
        print(f"Partial match: {result['partial_match']}")
        print(f"Existence match: {result['existence_match']}")
        print("-" * 40)


def main():
    """Main entry point."""
    args = parse_args()

    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Check if checkpoint is a local path or HuggingFace model ID
    if not args.checkpoint_path.startswith("agadelmoula-avey/"):
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    print("Starting optimized model evaluation with vLLM...")
    print(f"Model: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mode: {args.mode}")

    if args.checkpoint_path == "agadelmoula-avey/Qwen3-4B-Base":
        print("Note: Using base model for vLLM performance demonstration")
        print("To use fine-tuned weights, fix the tokenizer configuration first")

    # Choose evaluation mode
    if args.mode == "multi":
        print("\nRunning multi-label evaluation...")
        evaluate_multi_model_vllm(args)
    else:
        print("\nRunning single-label evaluation...")
        evaluate_model_vllm(args)


if __name__ == "__main__":
    main()
