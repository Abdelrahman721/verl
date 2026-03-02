# merge_verl_ckpt.py
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ckpt_dir = "/workspace/models/checkpoint-1.65k/actor"
hf_config_dir = os.path.join(ckpt_dir, "huggingface")
output_dir = "/workspace/models/checkpoint-1.65k/merged_hf"

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
sd0 = torch.load(os.path.join(ckpt_dir, "model_world_size_2_rank_0.pt"), map_location=device, weights_only=False)
sd1 = torch.load(os.path.join(ckpt_dir, "model_world_size_2_rank_1.pt"), map_location=device, weights_only=False)

merged = {}
for key in sd0.keys():
    t0 = extract_local(sd0[key])
    t1 = extract_local(sd1[key])
    expected = ref_sd[key]

    if t0.shape == expected:
        merged[key] = t0.cpu()
    else:
        merged[key] = torch.cat([t0, t1], dim=0).cpu()

del sd0, sd1
torch.cuda.empty_cache()

print(f"Merged {len(merged)}/{len(ref_sd)} parameters. Saving...")
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.load_state_dict(merged, strict=True)
del merged

os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir, safe_serialization=True)
AutoTokenizer.from_pretrained(hf_config_dir, trust_remote_code=True).save_pretrained(output_dir)
print(f"Done! Saved to {output_dir}")