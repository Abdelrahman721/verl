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
    # Handle ground_truth - it might be a string or list
    # Based on the dataset creation script, ground_truth is stored as a string in reward_model.ground_truth
    if isinstance(ground_truth, list):
        ground_truth_list = ground_truth
    elif isinstance(ground_truth, str):
        ground_truth_list = [ground_truth]
    else:
        # If it's a dict, try to extract the target
        if isinstance(ground_truth, dict) and "target" in ground_truth:
            ground_truth_list = ground_truth["target"]
            if not isinstance(ground_truth_list, list):
                ground_truth_list = [ground_truth_list]
        else:
            ground_truth_list = [str(ground_truth)]
    
    # Normalize ground truth codes (remove periods for comparison)
    normalized_ground_truth = [normalize_icd_code(gt) for gt in ground_truth_list]
    
    # Extract ICD code from solution
    extracted_code = extract_solution(solution_str, method=method)
    if extracted_code is None:
        return -1.0
    
    normalized_extracted = normalize_icd_code(extracted_code)
    
    # Check for exact match (after normalization)
    if normalized_extracted in normalized_ground_truth:
        return 1.0
    else:
        return -1.0
