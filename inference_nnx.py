
import os
import pickle
import json
import argparse
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import numpy as np
import librosa
import torch

from configuration_glmasr import GlmAsrConfig
from modeling_glmasr_nnx import GlmAsrForConditionalGeneration
from processing_glmasr import GlmAsrProcessor
from transformers import WhisperFeatureExtractor, AutoTokenizer

def load_audio(audio_path, sampling_rate=16000):
    speech, _ = librosa.load(audio_path, sr=sampling_rate)
    return speech

def create_sharding_rules(state):
    # Retrieve flat state to identify keys
    rules = []
    # Helper to create rule
    # (regex, PartitionSpec)
    # NNX keys loop: (path_tuple, value)
    # We will iterate the state keys and build a matching ruleset or apply directly.
    # But defining rules is easier for generalization.
    
    # Axis 'tp' = 8
    
    # Llama Attention
    # q_proj.kernel (in, out=heads*dim) -> shard 'out' -> (None, 'tp')
    # o_proj.kernel (in=heads*dim, out) -> shard 'in' -> ('tp', None) 
    # MLP
    # gate_proj (in, out=intermediate) -> (None, 'tp')
    # down_proj (in=intermediate, out) -> ('tp', None)
    
    # We need to map keys.
    # Key structure in NNX State is a flat dict of (tuple path) -> value.
    # e.g. ('language_model', 'model', 'layers', 0, 'self_attn', 'q_proj', 'kernel', 'value')
    
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="test_audio/test.mp3")
    parser.add_argument("--weights_path", type=str, default="model_flax.pkl")
    parser.add_argument("--config_path", type=str, default="weights_and_config/config.json")
    parser.add_argument("--tokenizer_path", type=str, default="weights_and_config")
    args = parser.parse_args()

    # Mesh Setup
    devices = jax.devices()
    num_devices = len(devices)
    print(f"Devices detected: {num_devices}")
    
    if num_devices < 8:
        print("Warning: Requested 8 TPUs, but fewer found. Using 1 device or whatever is available, sharding might fail if not 8 divisibility.")
        # We will proceed anyway for demo/verification logic; on simulated mesh it works if we use mesh(devices[:1]) etc.
        # But real 'tp' usually assumes matching topology.
        # Just use all devices.
    
    mesh = Mesh(devices, axis_names=('tp',))
    print(f"Mesh: {mesh}")

    # 1. Config & Model
    with open(args.config_path, "r") as f:
        config_dict = json.load(f)
    config = GlmAsrConfig(**config_dict)
    
    rngs = nnx.Rngs(0)
    model = GlmAsrForConditionalGeneration(config, rngs=rngs)
    
    # 2. Processor
    feature_extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, padding_value=0.0)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    chat_template_path = os.path.join(args.tokenizer_path, "chat_template.jinja")
    chat_template = None
    if os.path.exists(chat_template_path):
        with open(chat_template_path, "r") as f:
            chat_template = f.read()
    processor = GlmAsrProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer, chat_template=chat_template)

    # 3. Load Weights (Host)
    print(f"Loading weights from {args.weights_path}...")
    with open(args.weights_path, "rb") as f:
        state_dict = pickle.load(f) # CPU weights
    
    # 4. Partitioning & Sharding
    # We need to construct the sharded state.
    # NNX Graph API allow us to update.
    
    print("Sharding model weights...")
    

    # Define specs for each leaf in state
    flat_state = nnx.state(model).flat_state()
    sharded_state = {}
    
    print("Sample keys from FlatState:", list(dict(flat_state).keys())[:5])
    print("Sample keys from Loaded Weights:", list(state_dict.keys())[:5])
    
    for key, value_placeholder in dict(flat_state).items():
        # key is a tuple
        key_str = ".".join(str(k) for k in key)
        spec = PartitionSpec() # Default: replicated (None, None) means fully replicated if not specified? 
        # Actually None usually means replicated on that axis? 
        # NamedSharding(mesh, PartitionSpec(None)) -> replicated.
        
        # Simple heuristic for Tensor Parallelism
        # Check specific layers in Language Model
        if "language_model" in key_str:
            # Attention
            if "self_attn" in key_str:
                if "q_proj.kernel" in key_str or "k_proj.kernel" in key_str or "v_proj.kernel" in key_str:
                    # Column Parallel: (In, Out). Output dimension is sharded.
                    # Flax Linear is (In, Out).
                    # Shard axis 1.
                    spec = PartitionSpec(None, 'tp')
                elif "o_proj.kernel" in key_str:
                    # Row Parallel: (In, Out). Input dimension is sharded.
                    spec = PartitionSpec('tp', None)
                
                # KV Cache variables
                # k_cache: (batch, num_kv_heads, max_len, head_dim)
                # Shard num_kv_heads (axis 1)
                elif "k_cache" in key_str or "v_cache" in key_str:
                    spec = PartitionSpec(None, 'tp', None, None)
                    
            # MLP
            if "mlp" in key_str:
                if "gate_proj.kernel" in key_str or "up_proj.kernel" in key_str:
                    # Column Parallel
                    spec = PartitionSpec(None, 'tp')
                elif "down_proj.kernel" in key_str:
                    # Row Parallel
                    spec = PartitionSpec('tp', None)
            
            # Embeddings ? 
            # Usually vocab is huge. Shard vocab (axis 0).
            # embed_tokens.embedding: (vocab, hidden)
            # lm_head.kernel: (hidden, vocab)
            # Parallelizing vocab is good for memory.
            if "embed_tokens.embedding" in key_str:
                spec = PartitionSpec('tp', None)
            if "lm_head.kernel" in key_str:
                spec = PartitionSpec(None, 'tp')

        # Create Sharding
        sharding = NamedSharding(mesh, spec)
        
        # Determine value to put
        # We need the value from state_dict (converted weights) if loading, or current value if init.
        # The key tuple might match if state_dict uses tuples. 
        # My converter saved keys as tuples.
        
        # Check if key is in loaded weights
        # Note: converter saved with 'value' at end? Yes.
        val = state_dict.get(key)
        if val is None:
            # If not in weights (e.g. cache initialized but empty), use current init value
            val = value_placeholder.value if hasattr(value_placeholder, 'value') else value_placeholder
            # Variable.value is the array.
            
        # Device Put
        # This distributes the array across devices according to spec
        sharded_val = jax.device_put(val, sharding)
        sharded_state[key] = sharded_val

    # Update model with sharded weights
    # sharded_state is a dict of (path_tuple) -> Array
    # We must convert to State object for update to recognize paths
    sharded_graph_state = nnx.State.from_flat_path(sharded_state)
    nnx.update(model, sharded_graph_state)
    print("Model sharded successfully.")

    # 5. Inputs
    audio = load_audio(args.audio_path)
    user_prompt = "Please transcribe this audio into text"
    text_input = f"<|user|>\n{user_prompt}<|begin_of_audio|><|pad|><|end_of_audio|><|assistant|>"
    full_inputs = processor(text=text_input, audio=audio, sampling_rate=16000, return_tensors="pt")
    
    input_ids = jnp.array(full_inputs["input_ids"].numpy())
    input_features = jnp.array(full_inputs["input_features"].numpy())
    input_features_mask = jnp.array(full_inputs["input_features_mask"].numpy())
    
    batch_size = input_ids.shape[0]
    max_length = 2048
    
    # Helper to enforce input sharding (replicated)
    replicated = NamedSharding(mesh, PartitionSpec())
    input_ids = jax.device_put(input_ids, replicated)
    input_features = jax.device_put(input_features, replicated)
    input_features_mask = jax.device_put(input_features_mask, replicated)

    # 6. Init Cache (Sharded)
    # We can call init_cache, but it initializes on default device (0). 
    # We want it sharded.
    # Easier to init then update, OR override init_cache logic.
    # Since we already ran the sharding loop over 'k_cache' keys in `flat_state`,
    # if `init_cache` was called BEFORE sharding loop, it would be sharded.
    # BUT `init_cache` creates new Variables. 
    # Let's call init_cache FIRST, then do the sharding loop which sees the cache keys.
    # (Moved init_cache call up before sharding loop)
    pass # Adjusting code order below

    # 7. JIT Functions with Annotation
    # nnx.jit allows configuring sharding?
    # Or just standard jax.jit with `in_shardings`, `out_shardings`.
    # Since inputs are `NamedSharded` arrays, `jax.jit` should respect them (auto-sharding propagation).
    # We mainly need to ensure operations don't force gathering.
    # Using `nnx.jit` wraps `jax.jit`.
    
    @nnx.jit
    def prefill(model, input_ids, input_features, input_features_mask):
        cache_index = jnp.array(0, dtype=jnp.int32)
        logits = model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            deterministic=True,
            cache_index=cache_index
        )
        return logits[:, -1, :], cache_index + input_ids.shape[1]

    @nnx.jit
    def decode(model, input_ids, cache_index):
        logits = model(
            input_ids=input_ids,
            deterministic=True,
            cache_index=cache_index
        )
        return logits[:, -1, :], cache_index + 1

    # Actual Execution Flow
    # Init Cache
    model.init_cache(batch_size, max_length)
    
    # Rerun sharding loop to catch cache variables
    # (Copy-paste the loop logic or encapsulate function)
    # See `create_sharded_state` logic roughly implemented above.
    
    # Redoing sharding loop properly
    flat_state = nnx.state(model).flat_state() # Now contains cache
    final_sharded_state = {}
    for key, leaf in dict(flat_state).items():
        key_str = ".".join(str(k) for k in key)
        spec = PartitionSpec() 
        
        if "language_model" in key_str:
            if "self_attn" in key_str:
                if "q_proj.kernel" in key_str or "k_proj.kernel" in key_str or "v_proj.kernel" in key_str:
                    spec = PartitionSpec(None, 'tp')
                elif "o_proj.kernel" in key_str:
                    spec = PartitionSpec('tp', None)
                elif "k_cache" in key_str or "v_cache" in key_str:
                    # Cache is (batch, head, seq, dim)
                    # Heads (4) < Devices (8), so cannot shard head.
                    # Shard sequence length (2048).
                    spec = PartitionSpec(None, None, 'tp', None)
                    
            if "mlp" in key_str:
                if "down_proj.kernel" in key_str:
                    spec = PartitionSpec('tp', None)
                elif "gate_proj.kernel" in key_str or "up_proj.kernel" in key_str:
                    spec = PartitionSpec(None, 'tp')
            
            if "embed_tokens.embedding" in key_str:
                spec = PartitionSpec('tp', None)
            if "lm_head.kernel" in key_str:
                spec = PartitionSpec(None, 'tp')

        sharding = NamedSharding(mesh, spec)
        
        # Get data: prefer loaded weights, else leaf value
        if key in state_dict:
            val = state_dict[key]
        else:
            # leaf is likely a nnx.Param or Variable wrapper from flat_state? 
            # No, flat_state(model) returns the values (arrays).
            val = leaf
            
        final_sharded_state[key] = jax.device_put(val, sharding)

    final_sharded_graph_state = nnx.State.from_flat_path(final_sharded_state)
    nnx.update(model, final_sharded_graph_state)
    
    print("Running prefill...")
    logits, cache_index = prefill(model, input_ids, input_features, input_features_mask)
    
    print("Generating...")
    generated_ids = []
    eos_token_ids = config.text_config.eos_token_id
    if isinstance(eos_token_ids, int): eos_token_ids = [eos_token_ids]
    token_id = jnp.argmax(logits, axis=-1)[:, None]
    
    for i in range(256):
        # We need token_id to be replicated/sharded correctly.
        # It's small (batch, 1). Replicated ok.
        token_id = jax.device_put(token_id, replicated)
        
        scalar = token_id[0, 0].item()
        generated_ids.append(scalar)
        if scalar in eos_token_ids: break
        
        logits, cache_index = decode(model, token_id, cache_index)
        token_id = jnp.argmax(logits, axis=-1)[:, None]
        if i % 10 == 0: print(f"\rStep {i}", end="", flush=True)

    print("\nDecoding...")
    print(processor.batch_decode([generated_ids], skip_special_tokens=True)[0])

if __name__ == "__main__":
    main()
