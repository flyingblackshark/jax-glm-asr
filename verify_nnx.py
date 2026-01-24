
import os
import torch
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import json

from modeling_glmasr_nnx import GlmAsrConfig, LlamaAttention, LlamaMLP, LlamaRMSNorm, GlmAsrEncoder
# Import PyTorch classes
# We need to hack the import because `torch_glmasr_model.py` depends on relative imports if run as script or issues.
# But since it is in the root, we can import it if dependencies (transformers) are met.
# It imports `configuration_glmasr` from local.
# We might need to mock some relative imports if they fail.
# It uses `...modeling_outputs`, `...cache_utils` etc from transformers. 
# This might be tricky if we don't have the full transformers source structure or if it expects to be part of it.
# However, `torch_glmasr_model.py` seems specific.
# Let's try importing it.

# To simplify, we might implement a minimal dummy Config class for PyTorch if needed,
# or assume the environment has transformers.

try:
    from torch_glmasr_model import GlmAsrAttention as PtAttention, GlmAsrMLP as PtMLP, GlmAsrEncoder as PtEncoder
except (ImportError, ValueError) as e:
    print(f"Comparison script warning: Failed to import PyTorch modules directly: {e}")
    print("Will try to patch sys.path or mock.")
    # Assuming the user file is standalone enough or we fix imports.
    import sys
    sys.path.append('.')
    from torch_glmasr_model import GlmAsrAttention as PtAttention, GlmAsrMLP as PtMLP, GlmAsrEncoder as PtEncoder

def to_jax(tensor):
    return jnp.array(tensor.detach().cpu().numpy())

def to_torch(array):
    return torch.tensor(np.array(array))

def check_close(a, b, atol=1e-2, name=""):
    diff = np.abs(a - b)
    max_diff = np.max(diff)
    print(f"[{name}] Max Diff: {max_diff:.6f}")
    if max_diff > atol:
        print(f"!! FAIL: {name} mismatch !!")
        return False
    print(f"PASS: {name} within tolerance {atol}")
    return True


def verify_attention():
    print("\n--- Verifying Question: Attention ---")
    # Config
    with open("weights_and_config/config.json", "r") as f:
        config_dict = json.load(f)
    config = GlmAsrConfig(**config_dict)
    
    rngs = nnx.Rngs(0)
    
    # Verify GlmAsrAttention (Audio Encoder Attention)
    pt_attn = PtAttention(config.audio_config, layer_idx=0)
    pt_attn.eval()
    from modeling_glmasr_nnx import GlmAsrAttention as FlaxGlmAsrAttention
    flax_attn = FlaxGlmAsrAttention(config.audio_config, layer_idx=0, rngs=rngs)

    
    # Copy Weights
    # PyTorch: q_proj.weight, q_proj.bias ...
    # Flax: q_proj.kernel, q_proj.bias ...
    
    # Mapping
    # pt.q_proj.weight (out, in) -> flax.q_proj.kernel (in, out)
    state = nnx.state(flax_attn)
    flat = state.flat_state()
    
    # Set weights
    def set_linear(pt_lin, flax_lin):
        # Weight: (out, in) -> (in, out)
        w = pt_lin.weight.detach().numpy().T
        flax_lin.kernel.value = jnp.array(w)
        if pt_lin.bias is not None:
            b = pt_lin.bias.detach().numpy()
            flax_lin.bias.value = jnp.array(b)
            
    set_linear(pt_attn.q_proj, flax_attn.q_proj)
    set_linear(pt_attn.k_proj, flax_attn.k_proj)
    set_linear(pt_attn.v_proj, flax_attn.v_proj)
    set_linear(pt_attn.o_proj, flax_attn.o_proj)
    
    # Inputs
    B, S, H = 1, 16, config.audio_config.hidden_size
    hidden_states_pt = torch.randn(B, S, H)
    hidden_states_flax = to_jax(hidden_states_pt)
    
    # Run
    # PyTorch forward: hidden_states, position_embeddings=...
    # We need dummy position embeddings (cos, sin)
    head_dim = config.audio_config.hidden_size // config.audio_config.num_attention_heads
    cos_pt = torch.randn(B, S, head_dim) # partial rotary? rope dim usually head_dim/2 or similar?
    sin_pt = torch.randn(B, S, head_dim)
    # PyTorch `apply_rotary_pos_emb` expects inputs aligned with this shape or unsqueezed.
    # In `torch_glmasr_model.py`: `cos, sin = position_embeddings`
    
    with torch.no_grad():
        out_pt, _ = pt_attn(hidden_states_pt, position_embeddings=(cos_pt, sin_pt))
        
    out_flax, _ = flax_attn(hidden_states_flax, position_embeddings=(to_jax(cos_pt), to_jax(sin_pt)), deterministic=True)
    
    check_close(out_pt.numpy(), np.array(out_flax), name="Audio Attention Output")

def verify_mlp():
    print("\n--- Verifying MLP ---")
    with open("weights_and_config/config.json", "r") as f:
        config_dict = json.load(f)
    config = GlmAsrConfig(**config_dict)
    
    pt_mlp = PtMLP(config.audio_config)
    rngs = nnx.Rngs(0)
    from modeling_glmasr_nnx import GlmAsrMLP as FlaxGlmAsrMLP
    flax_mlp = FlaxGlmAsrMLP(config.audio_config, rngs=rngs)
    
    # Copy
    # pt.fc1 -> flax.fc1
    def set_linear(pt_lin, flax_lin):
        w = pt_lin.weight.detach().numpy().T
        flax_lin.kernel.value = jnp.array(w)
        if pt_lin.bias is not None:
            flax_lin.bias.value = jnp.array(pt_lin.bias.detach().numpy())
            
    set_linear(pt_mlp.fc1, flax_mlp.fc1)
    set_linear(pt_mlp.fc2, flax_mlp.fc2)
    
    # Input
    x_pt = torch.randn(1, 16, config.audio_config.hidden_size)
    x_flax = to_jax(x_pt)
    
    with torch.no_grad():
        out_pt = pt_mlp(x_pt)
        
    out_flax = flax_mlp(x_flax)
    
    check_close(out_pt.numpy(), np.array(out_flax), name="Audio MLP Output")

if __name__ == "__main__":
    verify_mlp()
    verify_attention()
