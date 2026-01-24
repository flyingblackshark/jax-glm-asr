
import os
import pickle
import json
import argparse
import jax
import jax.numpy as jnp
from flax import nnx
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="test_audio/test.mp3")
    parser.add_argument("--weights_path", type=str, default="model_flax.pkl")
    parser.add_argument("--config_path", type=str, default="weights_and_config/config.json")
    parser.add_argument("--tokenizer_path", type=str, default="weights_and_config")
    args = parser.parse_args()

    # 1. Config
    with open(args.config_path, "r") as f:
        config_dict = json.load(f)
    config = GlmAsrConfig(**config_dict)
    
    # 2. Model
    rngs = nnx.Rngs(0)
    model = GlmAsrForConditionalGeneration(config, rngs=rngs)
    
    # 3. Weights
    print(f"Loading weights from {args.weights_path}...")
    with open(args.weights_path, "rb") as f:
        state_dict = pickle.load(f)
    # Convert flat path dict to nnx.State  
    flax_state = nnx.State.from_flat_path(state_dict)
    nnx.update(model, flax_state)

    
    # 4. Processor
    feature_extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, padding_value=0.0)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    # Load chat template if exists
    chat_template_path = os.path.join(args.tokenizer_path, "chat_template.jinja")
    chat_template = None
    if os.path.exists(chat_template_path):
        with open(chat_template_path, "r") as f:
            chat_template = f.read()
            
    processor = GlmAsrProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer, chat_template=chat_template)

    # 5. Load inputs
    audio = load_audio(args.audio_path)
    user_prompt = "Please transcribe this audio into text"
    text_input = f"<|user|>\n{user_prompt}<|begin_of_audio|><|pad|><|end_of_audio|><|assistant|>"
    
    full_inputs = processor(text=text_input, audio=audio, sampling_rate=16000, return_tensors="pt")
    input_ids_np = full_inputs["input_ids"].numpy()
    input_features_np = full_inputs["input_features"].numpy()
    input_features_mask_np = full_inputs["input_features_mask"].numpy()
    
    # 6. Setup Static Optimizations
    max_new_tokens = 256
    max_length = 2048 # Fixed buffer size for KV cache
    
    input_ids = jnp.array(input_ids_np)
    input_features = jnp.array(input_features_np)
    input_features_mask = jnp.array(input_features_mask_np)
    
    batch_size = input_ids.shape[0]
    
    print("Initializing KV Cache...")
    model.init_cache(batch_size, max_length)
    
    # Define JIT functions
    
    # PREFILL: Process the prompt + audio. 
    # Returns the next cache index and the logits for the last token (to predict first new token).
    @nnx.jit
    def prefill(model, input_ids, input_features, input_features_mask):
        # We need to compute where to write in cache.
        # input_ids shape (batch, seq)
        # We write from 0 to seq.
        # cache_index is usually the start index.
        # For prefill, we write the whole block.
        # Note: Our `LlamaAttention.__call__` uses `dynamic_update_slice` at `cache_index`.
        # If we pass `cache_index=0`, it updates [0:seq].
        
        cache_index = jnp.array(0, dtype=jnp.int32)
        logits = model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            deterministic=True,
            cache_index=cache_index
        )
        
        # Next index is current length
        next_cache_index = cache_index + input_ids.shape[1]
        
        # Return last logits
        return logits[:, -1, :], next_cache_index

    # DECODE: Generate one token
    @nnx.jit
    def decode(model, input_ids, cache_index):
        # input_ids: (batch, 1)
        logits = model(
            input_ids=input_ids,
            deterministic=True,
            cache_index=cache_index
        )
        return logits[:, -1, :], cache_index + 1

    print("Running prefill...")
    logits, cache_index = prefill(model, input_ids, input_features, input_features_mask)
    
    # Greedy generation
    print("Generating...")
    generated_ids = []
    
    eos_token_ids = config.text_config.eos_token_id
    if isinstance(eos_token_ids, int): eos_token_ids = [eos_token_ids]
    
    token_id = jnp.argmax(logits, axis=-1)[:, None] # (batch, 1)
    
    for i in range(max_new_tokens):
        # Store token
        token_scalar = token_id[0, 0].item()
        generated_ids.append(token_scalar)
        
        if token_scalar in eos_token_ids:
            break
            
        logits, cache_index = decode(model, token_id, cache_index)
        token_id = jnp.argmax(logits, axis=-1)[:, None]
        
        # Hacky progress
        if i % 10 == 0:
            print(f"\rStep {i}", end="", flush=True)

    print("\nDecoding...")
    text = processor.batch_decode([generated_ids], skip_special_tokens=True)[0]
    print(text)

if __name__ == "__main__":
    main()
