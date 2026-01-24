
import os
import pickle
import json
import argparse
import yaml
import re
import jax
import jax.numpy as jnp
from flax import nnx
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
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

def get_spec_for_path(path):
    # path is a tuple of strings or ints
    path_str = ".".join(map(str, path))
    
    # 1. Embeddings & Head
    if "embed_tokens.embedding" in path_str:
        return PartitionSpec('vocab', 'embed')
    if "lm_head.kernel" in path_str:
        return PartitionSpec('embed', 'vocab')
        
    # 2. Attention
    # q, k, v: input(embed) -> output(heads)
    if "q_proj.kernel" in path_str or "k_proj.kernel" in path_str or "v_proj.kernel" in path_str:
        return PartitionSpec('embed', 'heads')
    if "o_proj.kernel" in path_str:
        return PartitionSpec('heads', 'embed')
        
    # Bias for Attention
    if "q_proj.bias" in path_str or "k_proj.bias" in path_str or "v_proj.bias" in path_str:
        return PartitionSpec('heads')
    if "o_proj.bias" in path_str:
        return PartitionSpec('embed')

    # 3. MLP (Audio Encoder & LLM)
    # fc1 / gate / up -> input(embed) -> output(mlp)
    if "fc1.kernel" in path_str or "gate_proj.kernel" in path_str or "up_proj.kernel" in path_str or "linear_1.kernel" in path_str:
        return PartitionSpec('embed', 'mlp')
    
    # fc2 / down -> input(mlp) -> output(embed)
    if "fc2.kernel" in path_str or "down_proj.kernel" in path_str or "linear_2.kernel" in path_str:
        return PartitionSpec('mlp', 'embed')

    # Bias for MLP
    if "fc1.bias" in path_str or "gate_proj.bias" in path_str or "up_proj.bias" in path_str or "linear_1.bias" in path_str:
        return PartitionSpec('mlp')
    if "fc2.bias" in path_str or "down_proj.bias" in path_str or "linear_2.bias" in path_str:
        return PartitionSpec('embed')

    # Default None (Replicated)
    return PartitionSpec()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="test_audio/test.mp3")
    parser.add_argument("--weights_path", type=str, default="model_flax.pkl")
    parser.add_argument("--config_path", type=str, default="weights_and_config/config.json")
    parser.add_argument("--sharding_config", type=str, default="sharding_config.yml")
    parser.add_argument("--tokenizer_path", type=str, default="weights_and_config")
    args = parser.parse_args()

    # 1. Config & Rules
    print(f"Loading sharding config from {args.sharding_config}...")
    with open(args.sharding_config, "r") as f:
        sharding_cfg = yaml.safe_load(f)
    logical_axis_rules = sharding_cfg.get('logical_axis_rules', [])
    print(f"Logical Axis Rules: {logical_axis_rules}")

    devices = jax.devices()
    num_devices = len(devices)
    devices = mesh_utils.create_device_mesh((1, num_devices))
    mesh = Mesh(devices, axis_names=('fsdp', 'tensor'))
    print(f"Mesh: {mesh}")

    with open(args.config_path, "r") as f:
        config_dict = json.load(f)
    config = GlmAsrConfig(**config_dict)
    
    batch_size = 1
    max_length = 2048

    # 2. Abstract Model
    print("Creating abstract model...")
    def create_model_fn():
        rngs = nnx.Rngs(0)
        model = GlmAsrForConditionalGeneration(config, rngs=rngs)
        model.init_cache(batch_size, max_length)
        return model

    # No mesh context needed for abstract model if no with_partitioning used
    abstract_model = nnx.eval_shape(create_model_fn)
        
    state = nnx.state(abstract_model)
    graphdef = nnx.graphdef(abstract_model)
    
    # 3. Apply Manual Sharding Specs
    print("Applying logical sharding specs...")
    flat_abstract = dict(nnx.to_flat_state(state))
    flat_sharding_specs = {}
    
    for path, val in flat_abstract.items():
        spec = get_spec_for_path(path)
        flat_sharding_specs[path] = spec
        
    # 4. Resolve to Mesh Sharding
    # We use tree_map logic manually or reuse logical_to_mesh_sharding if it accepts flat dict?
    # logical_to_mesh_sharding takes PyTree. flat_sharding_specs is a PyTree (dict).
    sharded_specs = nn.logical_to_mesh_sharding(flat_sharding_specs, mesh, logical_axis_rules)
    
    # 5. Load & Distribute
    print(f"Loading weights from {args.weights_path}...")
    with open(args.weights_path, "rb") as f:
        state_dict = pickle.load(f)

    print("Distributing weights...")
    flat_state = {}
    replicated = NamedSharding(mesh, PartitionSpec())
    
    for path, abstract_val in flat_abstract.items():
        sharding = sharded_specs[path]
        if sharding is None: sharding = replicated
        
        if path in state_dict:
            val = state_dict[path]
        else:
            if hasattr(abstract_val, 'shape'):
                val = jnp.zeros(abstract_val.shape, dtype=abstract_val.dtype)
            else:
                val = 0
                
        flat_state[path] = jax.device_put(val, sharding)

    # 6. Reconstruct
    sharded_state = nnx.State.from_flat_path(flat_state)
    model = nnx.merge(graphdef, sharded_state)
    print("Model ready.")

    # 7. Inference Setup
    feature_extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, padding_value=0.0)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    chat_template = None
    if os.path.exists(os.path.join(args.tokenizer_path, "chat_template.jinja")):
        with open(os.path.join(args.tokenizer_path, "chat_template.jinja"), "r") as f:
            chat_template = f.read()
    processor = GlmAsrProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer, chat_template=chat_template)

    audio = load_audio(args.audio_path)
    user_prompt = "Please transcribe this audio into text"
    text_input = f"<|user|>\n{user_prompt}<|begin_of_audio|><|pad|><|end_of_audio|><|assistant|>"
    full_inputs = processor(text=text_input, audio=audio, sampling_rate=16000, return_tensors="pt")
    
    input_ids = jax.device_put(jnp.array(full_inputs["input_ids"].numpy()), replicated)
    input_features = jax.device_put(jnp.array(full_inputs["input_features"].numpy()), replicated)
    input_features_mask = jax.device_put(jnp.array(full_inputs["input_features_mask"].numpy()), replicated)

    @nnx.jit
    def prefill(model, input_ids, input_features, input_features_mask):
        cache_index = jnp.array(0, dtype=jnp.int32)
        logits = model(input_ids=input_ids, input_features=input_features, 
                      input_features_mask=input_features_mask, deterministic=True, cache_index=cache_index)
        return logits[:, -1, :], cache_index + input_ids.shape[1]

    @nnx.jit
    def decode(model, input_ids, cache_index):
        logits = model(input_ids=input_ids, deterministic=True, cache_index=cache_index)
        return logits[:, -1, :], cache_index + 1

    print("Generating...")
    logits, cache_index = prefill(model, input_ids, input_features, input_features_mask)
    generated_ids = []
    eos = config.text_config.eos_token_id
    if isinstance(eos, int): eos = [eos]
    
    token_id = jnp.argmax(logits, axis=-1)[:, None]
    
    for i in range(256):
        token_id = jax.device_put(token_id, replicated)
        scalar = token_id[0,0].item()
        generated_ids.append(scalar)
        if scalar in eos: break
        logits, cache_index = decode(model, token_id, cache_index)
        token_id = jnp.argmax(logits, axis=-1)[:, None]
        if i % 10 == 0: print(f"\rStep {i}", end="", flush=True)

    print("\nResult:")
    print(processor.batch_decode([generated_ids], skip_special_tokens=True)[0])

if __name__ == "__main__":
    main()
