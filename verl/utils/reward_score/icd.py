# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    """Extract the ICD code from the solution string.
    
    The prompt asks models to output the answer after "####", similar to GSM8K format.
    ICD codes are typically in formats like:
    - E11.9 (with period)
    - E119 (without period)
    
    Args:
        solution_str: The full solution text from the model
        method: "strict" to extract from after "####", "flexible" to search anywhere
        
    Returns:
        str or None: The extracted ICD code, or None if not found
    """
    assert method in ["strict", "flexible"]
    
    # Optimization: Regular expression matching on very long strings can be slow.
    # For ICD problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
    
    if method == "strict":
        # Extract ICD code after "####" marker (like GSM8K format)
        # Pattern: #### followed by optional whitespace and then ICD code
        # ICD code format: Letter(s) followed by digits, optionally with periods
        icd_pattern = r'####\s*([A-Z]\d{2,3}(?:\.\d+)?)'
        matches = re.findall(icd_pattern, solution_str, re.IGNORECASE)
        if len(matches) == 0:
            return None
        else:
            # Take the last solution (most likely the final answer)
            final_answer = matches[-1].upper().strip()
            if len(final_answer) == 3:
                return final_answer + '.0'
            return final_answer
    elif method == "flexible":
        # Search for ICD codes anywhere in the solution
        # Pattern: Letter(s) followed by digits, optionally with periods
        icd_pattern = r'\b([A-Z]\d{2,3}(?:\.\d+)?)\b'
        matches = re.findall(icd_pattern, solution_str, re.IGNORECASE)
        if len(matches) == 0:
            return None
        else:
            # Return the last match (most likely the final answer)
            final_answer = matches[-1].upper().strip()
            if len(final_answer) == 0:
                return final_answer + '.0'
            return final_answer
    
    return None


def normalize_icd_code(code):
    """Normalize an ICD code for comparison.
    
    Args:
        code: ICD code string
        
    Returns:
        str: Normalized ICD code
    """
    if code is None:
        return None
    # Remove whitespace, convert to uppercase
    code = str(code).strip().upper()
    # Remove periods for comparison (E11.9 -> E119)
    code = code.replace('.', '')
    return code

def compute_length_score(solution_str, ground_truth):
    """Compute the length score for the solution string.
    
    Args:
        solution_str: the solution text from the model
        ground_truth: the ground truth ICD code. Can be a string (e.g., "E11.9") or list of strings
        
    Returns:
        float: The computed length score
    """
    total_length = len(solution_str.split())
    if total_length < 32:
        return -1.0
    elif total_length < 64:
        return -0.5
    elif total_length < 128:
        return 0.0
    elif total_length < 256:
        return -0.5
    else:
        return -1.0

def compute_accuracy_score(solution_str, ground_truth, method="strict"):
    """Compute the accuracy score for the solution string.
    
    Args:
        solution_str: the solution text from the model
        ground_truth: the ground truth ICD code. Can be a string (e.g., "E11.9") or list of strings
        
    Returns:
        float: The computed accuracy score
    """
    extracted_code = extract_solution(solution_str)
    if method == "strict":
        return int(extracted_code == ground_truth)
    total_length = max(len(extracted_code), len(ground_truth))
    matched_length = 0
    for c1, c2 in zip(extracted_code, ground_truth):
        if c1 == c2:
            matched_length += 1
        else:
            break
    return round(matched_length / total_length, 2)

import re

def compute_language_score(solution_str, ground_truth=None):
    """
    Compute the language score for the solution string.

    The score represents the proportion of English alphabetic characters
    among all alphabetic characters (ignoring spaces, numbers, punctuation).

    Args:
        solution_str (str): The solution text from the model.
        ground_truth: Unused (kept for compatibility).

    Returns:
        float: Score between 0.0 and 1.0
    """
    if not solution_str:
        return 0.0

    # Remove HTML, LaTeX, and math expressions
    clean_text = re.sub(r'<.*?>|\$.*?\$|\\\[.*?\\\]', '', solution_str, flags=re.DOTALL)

    # Keep only alphabetic characters (ignore spaces, digits, punctuation)
    letters = re.findall(r'[A-Za-z]', clean_text)
    all_alpha = re.findall(r'[^\W\d_]', clean_text, flags=re.UNICODE)

    total_alpha = len(all_alpha)

    if total_alpha == 0:
        return 0.0

    english_letters = len(letters)

    return round(english_letters / total_alpha, 2)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """The scoring function for ICD coding datasets.
    
    Reference: Similar to GSM8K reward function, adapted for ICD codes.
    
    Args:
        solution_str: the solution text from the model
        ground_truth: the ground truth ICD code. Can be a string (e.g., "E11.9") or list of strings
        method: the method to extract the solution, choices are 'strict' (extract from after "####") 
                and 'flexible' (search anywhere)
        
    Returns:
        float: The computed score
    """
    score_functions = {
        "reward/length_score": compute_length_score,
        "reward/accuracy_score": compute_accuracy_score,
        "reward/language_score": compute_language_score,
    }
    scores = {}
    for key, func in score_functions.items():
        scores[key] = func(solution_str, ground_truth)

    scores["score"] = sum(scores.values())
    scores["reward/accuracy_score_strict"] = compute_accuracy_score(solution_str, ground_truth, method="strict")
    return scores
