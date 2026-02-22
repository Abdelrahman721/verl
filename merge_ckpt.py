# merge_verl_ckpt.py
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

ckpt_dir = "/workspace/verl/checkpoints/MedQA-RL/llm_judge_grpo/global_step_180/actor"
hf_config_dir = os.path.join(ckpt_dir, "huggingface")
output_dir = "/workspace/verl/checkpoints/MedQA-RL/llm_judge_grpo/global_step_180/merged_hf"

device = "cuda:0"

def extract_local(t):
    if hasattr(t, '_local_tensor'):
        return t._local_tensor
    return t

config = AutoConfig.from_pretrained(hf_config_dir, trust_remote_code=True)
ref_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)
ref_sd = {k: v.shape for k, v in ref_model.state_dict().items()}
del ref_model

print("Loading shards to GPU...")
sd0 = torch.load(os.path.join(ckpt_dir, "model_world_size_2_