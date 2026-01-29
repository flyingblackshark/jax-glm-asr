
import argparse
import re
import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
from safetensors.torch import load_file
from flax import nnx

from configuration_glmasr import GlmAsrConfig
from modeling_glmasr_nnx import GlmAsrForConditionalGeneration
from transformers.utils.hub import cached_file

# Mapping from original PyTorch GLM-ASR (from HF Hub) to our Flax NNX implementation

MAPPING = {
    # PyTorch checkpoint key -> NNX model key
    # Source format: audio_encoder.whisper.* -> Target: audio_tower.*
    # Source format: model.* -> Target: language_model.model.*
    # Source format: lm_head.* -> Target: language_model.lm_head.*
    
    # =============
    # Audio Encoder (Whisper-like)
    # =============
    # Conv layers
    r"^audio_encoder\.whisper\.conv1\.weight$":         r"audio_tower.conv1.kernel",
    r"^audio_encoder\.whisper\.conv1\.bias$":           r"audio_tower.conv1.bias",
    r"^audio_encoder\.whisper\.conv2\.weight$":         r"audio_tower.conv2.kernel",
    r"^audio_encoder\.whisper\.conv2\.bias$":           r"audio_tower.conv2.bias",
    
    # Encoder layers - Attention
    r"^audio_encoder\.whisper\.layers\.(\d+)\.self_attn\.q_proj\.weight$":     r"audio_tower.layers[\1].self_attn.q_proj.kernel",
    r"^audio_encoder\.whisper\.layers\.(\d+)\.self_attn\.q_proj\.bias$":       r"audio_tower.layers[\1].self_attn.q_proj.bias",
    r"^audio_encoder\.whisper\.layers\.(\d+)\.self_attn\.k_proj\.weight$":     r"audio_tower.layers[\1].self_attn.k_proj.kernel",
    r"^audio_encoder\.whisper\.layers\.(\d+)\.self_attn\.v_proj\.weight$":     r"audio_tower.layers[\1].self_attn.v_proj.kernel",
    r"^audio_encoder\.whisper\.layers\.(\d+)\.self_attn\.v_proj\.bias$":       r"audio_tower.layers[\1].self_attn.v_proj.bias",
    r"^audio_encoder\.whisper\.layers\.(\d+)\.self_attn\.out_proj\.weight$":   r"audio_tower.layers[\1].self_attn.o_proj.kernel",
    r"^audio_encoder\.whisper\.layers\.(\d+)\.self_attn\.out_proj\.bias$":     r"audio_tower.layers[\1].self_attn.o_proj.bias",
    
    # Layer norms (Whisper uses self_attn_layer_norm -> our input_layernorm, final_layer_norm -> post_attention_layernorm)
    r"^audio_encoder\.whisper\.layers\.(\d+)\.self_attn_layer_norm\.weight$":  r"audio_tower.layers[\1].input_layernorm.scale",
    r"^audio_encoder\.whisper\.layers\.(\d+)\.self_attn_layer_norm\.bias$":    r"audio_tower.layers[\1].input_layernorm.bias",
    r"^audio_encoder\.whisper\.layers\.(\d+)\.final_layer_norm\.weight$":      r"audio_tower.layers[\1].post_attention_layernorm.scale",
    r"^audio_encoder\.whisper\.layers\.(\d+)\.final_layer_norm\.bias$":        r"audio_tower.layers[\1].post_attention_layernorm.bias",
    
    # MLP (Whisper uses fc1/fc2)
    r"^audio_encoder\.whisper\.layers\.(\d+)\.fc1\.weight$":    r"audio_tower.layers[\1].mlp.fc1.kernel",
    r"^audio_encoder\.whisper\.layers\.(\d+)\.fc1\.bias$":      r"audio_tower.layers[\1].mlp.fc1.bias",
    r"^audio_encoder\.whisper\.layers\.(\d+)\.fc2\.weight$":    r"audio_tower.layers[\1].mlp.fc2.kernel",
    r"^audio_encoder\.whisper\.layers\.(\d+)\.fc2\.bias$":      r"audio_tower.layers[\1].mlp.fc2.bias",
    
    # Audio encoder extras
    r"^audio_encoder\.layer_norm\.weight$":  r"audio_tower.norm.scale",
    r"^audio_encoder\.layer_norm\.bias$":    r"audio_tower.norm.bias",
    
    # Multi-modal projector uses audio_encoder.adapting (5120->4096->2048)
    r"^audio_encoder\.adapting\.0\.weight$":  r"multi_modal_projector.linear_1.kernel",
    r"^audio_encoder\.adapting\.0\.bias$":    r"multi_modal_projector.linear_1.bias",
    r"^audio_encoder\.adapting\.2\.weight$":  r"multi_modal_projector.linear_2.kernel",
    r"^audio_encoder\.adapting\.2\.bias$":    r"multi_modal_projector.linear_2.bias",

    
    # =============
    # Language Model (Llama-like)
    # =============
    r"^model\.embed_tokens\.weight$":        r"language_model.model.embed_tokens.embedding",
    r"^model\.norm\.weight$":                r"language_model.model.norm.weight",
    r"^lm_head\.weight$":                    r"language_model.lm_head.kernel",
    
    # Layers - Attention
    r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$":  r"language_model.model.layers[\1].self_attn.q_proj.kernel",
    r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$":  r"language_model.model.layers[\1].self_attn.k_proj.kernel",
    r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$":  r"language_model.model.layers[\1].self_attn.v_proj.kernel",
    r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$":  r"language_model.model.layers[\1].self_attn.o_proj.kernel",
    
    # Layer norms
    r"^model\.layers\.(\d+)\.input_layernorm\.weight$":          r"language_model.model.layers[\1].input_layernorm.weight",
    r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$": r"language_model.model.layers[\1].post_attention_layernorm.weight",
    
    # MLP (Llama uses gate_proj/up_proj/down_proj)
    r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$":  r"language_model.model.layers[\1].mlp.gate_proj.kernel",
    r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$":    r"language_model.model.layers[\1].mlp.up_proj.kernel",
    r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$":  r"language_model.model.layers[\1].mlp.down_proj.kernel",
}


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        if re.match(pattern, key):
            if replacement is None:
                return None
            return re.sub(pattern, replacement, key)
    return None

def permute_rope_torch(tensor, config):
    # Re-implement the permutation logic in torch/numpy for weight adjustment
    if tensor.dim() == 2:
        dim1, dim2 = tensor.shape
    else:
        dim1 = tensor.shape[0]

    n_heads = config.audio_config.num_attention_heads
    head_dim = config.audio_config.head_dim
    rope_dim = dim1 // 2

    rope_indices = torch.arange(rope_dim)
    rope_indices = rope_indices.view(n_heads, rope_dim // n_heads // 2, 2)
    rope_indices = rope_indices.transpose(1, 2)
    rope_indices = rope_indices.reshape(n_heads, -1)

    non_rope_start = head_dim // 2
    non_rope_indices = torch.arange(non_rope_start, head_dim, dtype=torch.long)
    non_rope_indices = non_rope_indices.expand(n_heads, -1)

    head_offsets = torch.arange(n_heads, dtype=torch.long)[:, None] * (head_dim // 2)
    non_rope_indices = non_rope_indices + head_offsets.expand(n_heads, head_dim // 2)

    combined_indices = torch.cat([rope_indices, non_rope_indices], dim=1)
    global_head_offsets = torch.arange(n_heads, dtype=torch.long)[:, None] * (head_dim // 2)
    combined_indices = combined_indices + global_head_offsets.expand(n_heads, head_dim)

    permutation_indices = combined_indices.reshape(-1)
    tensor = tensor[permutation_indices]

    return tensor

def key_str_to_nnx_tuple_path(key_str: str) -> tuple:
    parts = []
    for part in key_str.split("."):
        if "[" in part and "]" in part:
            name, idx = part.split("[")
            idx = int(idx[:-1])
            parts.append(name)
            parts.append(idx)
        else:
            parts.append(part)
    return tuple(parts)

def convert_pytorch_state_dict_to_nnx_state_dict(state_dict, config, *, verbose: bool = False):
    new_state = {}
    for k, v in state_dict.items():
        new_key_str = convert_key(k, MAPPING)
        if new_key_str is None:
            continue

        # RoPE permutation for Audio Tower (apply before transpose)
        if "audio_encoder.whisper" in k and ("q_proj" in k or "k_proj" in k):
            v = permute_rope_torch(v, config)

        # Linear weights: PyTorch (out, in) -> Flax (in, out)
        if "kernel" in new_key_str and v.dim() == 2:
            v = v.t()

        # Conv weights: PyTorch (out, in, k) -> Flax (k, in, out)
        if "conv" in new_key_str and "kernel" in new_key_str and v.dim() == 3:
            v = v.permute(2, 1, 0)

        if v.dtype == torch.bfloat16:
            v = v.float()

        key_tuple = key_str_to_nnx_tuple_path(new_key_str)
        new_state[key_tuple] = jnp.array(v.detach().cpu().numpy())

        if verbose and len(new_state) <= 5:
            print(f"{k} -> {new_key_str} ({tuple(new_state[key_tuple].shape)})")

    return new_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="zai-org/GLM-ASR-Nano-2512")
    parser.add_argument("--revision", type=str, default="91967eab799804ab256a3819a085b92378906eb2")
    parser.add_argument("--local_path", type=str, default=None, help="Local directory containing model.safetensors and config.json")
    args = parser.parse_args()

    # Determine paths
    model_path = None
    config_path = None
    
    if args.local_path:
        if os.path.exists(os.path.join(args.local_path, "model.safetensors")):
            model_path = os.path.join(args.local_path, "model.safetensors")
        elif os.path.isfile(args.local_path): # Maybe the user passed the file directly
            model_path = args.local_path
            
        if os.path.exists(os.path.join(args.local_path, "config.json")):
            config_path = os.path.join(args.local_path, "config.json")
        elif os.path.isfile(os.path.join(os.path.dirname(args.local_path), "config.json")):
             # Try side-by-side if they passed the model file
             config_path = os.path.join(os.path.dirname(args.local_path), "config.json")

    # Fallback to Hub if local not found or not provided
    if model_path is None:
        print(f"Downloading from {args.repo_id}...")
        model_path = cached_file(args.repo_id, "model.safetensors", revision=args.revision)
    else:
        print(f"Loading weights from local path: {model_path}")

    print(f"Loaded {model_path}")
    state_dict = load_file(model_path)

    # Initialize NNX Model
    print("Initializing NNX Model...")
    
    if config_path:
        print(f"Loading config from {config_path}")
        import json
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = GlmAsrConfig(**config_dict)
    else:
        print(f"Downloading config from {args.repo_id}...")
        config = GlmAsrConfig.from_pretrained(args.repo_id, revision=args.revision)
    
    rngs = nnx.Rngs(0)
    model = GlmAsrForConditionalGeneration(config, rngs=rngs)
    
    # We want to traverse `model` and fill values from `state_dict`
    # Flax NNX graph is mutable. We can create a flat state, update it, and split.
    
    # Create target flat state
    # We need to manually build the mapping since keys differ significantly.
    
    graph_state = nnx.state(model)
    flat_state = graph_state.flat_state()
    
    print("Converting weights...")
    new_state = convert_pytorch_state_dict_to_nnx_state_dict(state_dict, config, verbose=True)
        
    print(f"Converted {len(new_state)} weights")
    
    print("Saving Flax model...")
    import pickle
    with open("model_flax.pkl", "wb") as f:
        pickle.dump(new_state, f)
    
    print("Done. Verification recommended.")


if __name__ == "__main__":
    main()
