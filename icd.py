import os
import re
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

icd_pattern = re.compile(r'\b[A-Z][A-Z0-9]{2}(?:\.[A-Z0-9]{1,4})?\b')



def extract_answer(model_answer):
    if "<answer>" in model_answer:
        return model_answer.split("<answer>")[1]
    elif "</think>" in model_answer:
        return model_answer.split("</think>")[1]
    else:
        return model_answer




# =========================================================
# MULTI LABEL
# =========================================================

def compute_multi_label_scores(solution_str: str, ground_truth: list):
    """
    Multi-label classification using precision/recall/f1.
    """

    response = extract_answer(solution_str)

    pred_codes = set(icd_pattern.findall(response))
    ground_truth = str(ground_truth)
    gt_codes = set(icd_pattern.findall(ground_truth))

    if len(pred_codes) == 0:
        return {
            "reward/accuracy_score": 0.0,
            "reward/precision": 0.0,
            "reward/recall": 0.0,
            "reward/f1": 0.0,
            "score": 0.0,
        }

    tp = len(pred_codes & gt_codes)

    precision = tp / len(pred_codes) if len(pred_codes) > 0 else 0.0
    recall = tp / len(gt_codes) if len(gt_codes) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "reward/accuracy_score": f1,
        "reward/precision": precision,
        "reward/recall": recall,
        "reward/f1": f1,
        "score": f1,
    }

def compute_if_scores(solution_str, ground_truth):
    return {
        "reward/accuracy_score": 0,
        "reward/precision": 0,
        "reward/recall": 0,
        "reward/f1": 0,
        "score": 0,
    }


# =========================================================
# ENTRYPOINT
# =========================================================

def compute_score(data_source, solution_str, ground_truth, extra_info=None):

    if data_source != "if":
        return compute_multi_label_scores(solution_str, ground_truth)
    else:
        return compute_if_scores(solution_str, ground_truth)
