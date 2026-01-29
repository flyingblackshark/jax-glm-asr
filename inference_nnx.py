
import os
import json
import argparse
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import numpy as np
import librosa
try:
    import torch
except ImportError:
    torch = None

from configuration_glmasr import GlmAsrConfig
from modeling_glmasr_nnx import GlmAsrForConditionalGeneration
from processing_glmasr import GlmAsrProcessor
from transformers import WhisperFeatureExtractor, AutoTokenizer, PreTrainedTokenizerFast
from transformers.utils.hub import cached_file

from convert_glmasr_weights_to_nnx import convert_pytorch_state_dict_to_nnx_state_dict

def load_audio(audio_path, sampling_rate=16000):
    speech, _ = librosa.load(audio_path, sr=sampling_rate)
    return speech

def load_tokenizer(tokenizer_source, revision):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_source, revision=revision, trust_remote_code=True)
    except ValueError as e:
        # GLM-ASR tokenizer_config.json may set tokenizer_class="TokenizersBackend" which older transformers
        # doesn't resolve via AutoTokenizer. Fallback to a generic fast tokenizer from tokenizer.json.
        if "TokenizersBackend" not in str(e):
            raise

    tokenizer_file = cached_file(tokenizer_source, "tokenizer.json", revision=revision)
    tokenizer_config_file = cached_file(tokenizer_source, "tokenizer_config.json", revision=revision)
    with open(tokenizer_config_file, "r") as f:
        tok_cfg = json.load(f)

    init_kwargs = {}
    for k in ("bos_token", "eos_token", "unk_token", "pad_token", "mask_token"):
        if k in tok_cfg:
            init_kwargs[k] = tok_cfg[k]
    if "padding_side" in tok_cfg:
        init_kwargs["padding_side"] = tok_cfg["padding_side"]
    if "model_max_length" in tok_cfg:
        init_kwargs["model_max_length"] = tok_cfg["model_max_length"]
    if "clean_up_tokenization_spaces" in tok_cfg:
        init_kwargs["clean_up_tokenization_spaces"] = tok_cfg["clean_up_tokenization_spaces"]
    if "extra_special_tokens" in tok_cfg:
        init_kwargs["additional_special_tokens"] = tok_cfg["extra_special_tokens"]

    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, **init_kwargs)

def load_feature_extractor(model_id, revision):
    try:
        return WhisperFeatureExtractor.from_pretrained(model_id, revision=revision)
    except Exception:
        pass

    try:
        processor_config_file = cached_file(model_id, "processor_config.json", revision=revision)
        with open(processor_config_file, "r") as f:
            processor_cfg = json.load(f)
        fe_cfg = dict(processor_cfg.get("feature_extractor", {}))
        fe_cfg.pop("feature_extractor_type", None)
        if fe_cfg:
            return WhisperFeatureExtractor(**fe_cfg)
    except Exception:
        pass

    return WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, padding_value=0.0)

def get_tp_spec(key_tuple):
    key_str = ".".join(str(k) for k in key_tuple)
    spec = PartitionSpec() # Default Replicated
    
    # 1. Attention
    # q, k, v head_dim projection -> Shard Output (heads) -> Column Parallel
    if "q_proj.kernel" in key_str or "k_proj.kernel" in key_str or "v_proj.kernel" in key_str:
        spec = PartitionSpec(None, 'tp')
    # o output projection -> Shard Input (heads) -> Row Parallel
    elif "o_proj.kernel" in key_str:
        spec = PartitionSpec('tp', None)
    
    # Bias (if present, usually matches output dim)
    if "q_proj.bias" in key_str or "k_proj.bias" in key_str or "v_proj.bias" in key_str:
        spec = PartitionSpec('tp')
    
    # 2. KV Cache
    # (batch, heads, len, head_dim)
    # Heads (4) < Devices (8). Shard length (axis 2).
    if "k_cache" in key_str or "v_cache" in key_str:
        spec = PartitionSpec(None, None, 'tp', None)
    
    # 3. MLP
    # gate/up -> Shard Output (intermediate) -> Column Parallel
    if "gate_proj.kernel" in key_str or "up_proj.kernel" in key_str or "linear_1.kernel" in key_str:
        spec = PartitionSpec(None, 'tp')
    # down -> Shard Input (intermediate) -> Row Parallel
    elif "down_proj.kernel" in key_str or "linear_2.kernel" in key_str:
        spec = PartitionSpec('tp', None)
        
    # Bias
    if "gate_proj.bias" in key_str or "up_proj.bias" in key_str or "linear_1.bias" in key_str:
        spec = PartitionSpec('tp')

    # 4. Embeddings / Head
    # Shard Vocab. 
    # embed: (vocab, hidden) -> Shard axis 0
    if "embed_tokens.embedding" in key_str:
        spec = PartitionSpec('tp', None)
    # head: (hidden, vocab) -> Shard axis 1
    if "lm_head.kernel" in key_str:
        spec = PartitionSpec(None, 'tp')
        
    return spec

def shard_model(model, mesh, state_dict=None):
    # Flatten state
    # We use nnx.state(model).flat_state() which returns values
    flat_state = nnx.state(model).flat_state()
    sharded_flat = {}
    
    print(f"Sharding {len(flat_state)} leaves...")
    
    for key, leaf in dict(flat_state).items():
        spec = get_tp_spec(key)
        sharding = NamedSharding(mesh, spec)
        
        # Determine value
        val = leaf
        if state_dict and key in state_dict:
            val = state_dict[key]
        
        # Device Put
        sharded_flat[key] = jax.device_put(val, sharding)
        
    # Update model
    nnx.update(model, nnx.State.from_flat_path(sharded_flat))
    print("Model sharding update complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="test_audio/test3.mp3")
    parser.add_argument("--weights_path", type=str, default=None, help="Optional local .safetensors weights path. If omitted, download via HF cache.")
    parser.add_argument("--config_path", type=str, default=None, help="Optional local config.json path. If omitted, download via HF cache.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional local tokenizer directory. If omitted, load from HF (uses HF cache).")
    parser.add_argument("--model_id", type=str, default="zai-org/GLM-ASR-Nano-2512")
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()

    # Mesh
    devices = jax.devices()
    num_devices = len(devices)
    print(f"Devices available: {num_devices}")
    
    # Ensure divisible by 1 (always true) or user desired TP size
    # Assuming full device mesh if > 1
    mesh = Mesh(devices, axis_names=('tp',))
    print(f"Mesh: {mesh}")

    # Config
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            config_dict = json.load(f)
        config = GlmAsrConfig(**config_dict)
    else:
        config = GlmAsrConfig.from_pretrained(args.model_id, revision=args.revision)
    
    # Model Init (on CPU/default first)
    rngs = nnx.Rngs(0)
    model = GlmAsrForConditionalGeneration(config, rngs=rngs)
    
    # Init Cache (so it exists for sharding)
    model.init_cache(1, 2048)
    
    # Load weights for sharding (local .safetensors or download via HF cache, then convert)
    state_dict = None
    if args.weights_path and os.path.exists(args.weights_path):
        if args.weights_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            print(f"Loading safetensors weights from {args.weights_path}...")
            pt_state_dict = load_file(args.weights_path)
            print("Converting weights to NNX format...")
            state_dict = convert_pytorch_state_dict_to_nnx_state_dict(pt_state_dict, config, verbose=True)
        else:
            raise ValueError(f"Unsupported --weights_path format: {args.weights_path}")
    else:
        from safetensors.torch import load_file

        print(f"Downloading weights via HF cache from {args.model_id}...")
        model_path = cached_file(args.model_id, "model.safetensors", revision=args.revision)
        pt_state_dict = load_file(model_path)
        print("Converting weights to NNX format...")
        state_dict = convert_pytorch_state_dict_to_nnx_state_dict(pt_state_dict, config, verbose=True)
        
    # Shard Everything
    shard_model(model, mesh, state_dict)
    
    # Processor & Inference
    feature_extractor = load_feature_extractor(args.model_id, args.revision)

    tokenizer_source = args.tokenizer_path if args.tokenizer_path and os.path.exists(args.tokenizer_path) else args.model_id
    tokenizer = load_tokenizer(tokenizer_source, args.revision)

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        try:
            chat_template_file = cached_file(tokenizer_source, "chat_template.jinja", revision=args.revision)
            with open(chat_template_file, "r") as f:
                chat_template = f.read()
        except Exception:
            chat_template = None

    processor = GlmAsrProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer, chat_template=chat_template)

    audio = load_audio(args.audio_path)
    user_prompt = "Please transcribe this audio into text"
    if chat_template:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": ""},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=chat_template,
            )
        except Exception:
            text_input = f"<|user|>\n{user_prompt}<|begin_of_audio|><|pad|><|end_of_audio|><|assistant|>"
    else:
        text_input = f"<|user|>\n{user_prompt}<|begin_of_audio|><|pad|><|end_of_audio|><|assistant|>"
    full_inputs = processor(text=text_input, audio=audio, sampling_rate=16000, return_tensors="pt")
    
    input_ids = jnp.array(full_inputs["input_ids"].numpy())
    input_features = jnp.array(full_inputs["input_features"].numpy())
    input_features_mask = jnp.array(full_inputs["input_features_mask"].numpy())
    
    replicated = NamedSharding(mesh, PartitionSpec())
    input_ids = jax.device_put(input_ids, replicated)
    input_features = jax.device_put(input_features, replicated)
    input_features_mask = jax.device_put(input_features_mask, replicated)

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
    
    eos_token_ids = config.text_config.eos_token_id
    if isinstance(eos_token_ids, int): eos_token_ids = [eos_token_ids]
    
    token_id = jnp.argmax(logits, axis=-1)[:, None]
    
    for i in range(256):
        token_id = jax.device_put(token_id, replicated)
        scalar = token_id[0,0].item()
        generated_ids.append(scalar)
        if scalar in eos_token_ids: break
        
        logits, cache_index = decode(model, token_id, cache_index)
        token_id = jnp.argmax(logits, axis=-1)[:, None]
        if i % 10 == 0: print(f"\rStep {i}", end="", flush=True)

    print("\nResult:")
    print(processor.batch_decode([generated_ids], skip_special_tokens=True, strip_prefix=True)[0])

if __name__ == "__main__":
    main()
