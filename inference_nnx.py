
import os
import pickle
import json
import argparse
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import librosa
import torch # Used for processor which relies on torch tensors potentially

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

    # 1. Load Configuration
    print(f"Loading config from {args.config_path}...")
    with open(args.config_path, "r") as f:
        config_dict = json.load(f)
    config = GlmAsrConfig(**config_dict)

    # 2. Add some inference-specific configs if missing
    # (e.g. forced_decoder_ids not strictly needed if we implement manual loop, but check EOS)
    eos_token_id = config.text_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_ids = [eos_token_id]
    else:
        eos_token_ids = eos_token_id
    
    # 3. Initialize Model
    print("Initializing model...")
    rngs = nnx.Rngs(0)
    model = GlmAsrForConditionalGeneration(config, rngs=rngs)

    # 4. Load Weights
    print(f"Loading weights from {args.weights_path}...")
    with open(args.weights_path, "rb") as f:
        state_dict = pickle.load(f)
    
    # Update model state
    nnx.update(model, state_dict)
    print("Model weights loaded.")

    # 5. Initialize Processor
    print("Initializing processor...")
    feature_extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, padding_value=0.0)
    # Tokenizer: load from local path
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    # Load chat template
    chat_template_path = os.path.join(args.tokenizer_path, "chat_template.jinja")
    chat_template = None
    if os.path.exists(chat_template_path):
        with open(chat_template_path, "r") as f:
            chat_template = f.read()
            
    processor = GlmAsrProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer, chat_template=chat_template)

    # 6. Load and Process Audio
    print(f"Loading audio from {args.audio_path}...")
    audio = load_audio(args.audio_path)
    
    # Prepare input
    # The prompt usually needs to be formatted.
    # GLM-ASR uses specific chat template or prompt structure.
    # Default prompt in processor: "Please transcribe this audio into text"
    # We can use apply_transcription_request from processor if available?
    # processing_glmasr.py has apply_transcription_request.
    

    # inputs is BatchFeature containing 'input_ids', 'input_features', 'input_features_mask'
    # If inputs doesn't contain audio features (because apply_chat_template output might not merge them if processor logic is split), 
    # we need to get them.
    # The `apply_transcription_request` in `processing_glmasr.py` seems to rely on `self.apply_chat_template`.
    # But `apply_chat_template` in `ProcessorMixin` usually just tokenizes text unless overridden or passed audio.
    # `GlmAsrProcessor`'s `__call__` handles audio. `apply_chat_template` usually just handles text formatting.
    # Let's verify `GlmAsrProcessor` implementation...
    # It seems `apply_transcription_request` calls `apply_chat_template`. 
    # If `apply_chat_template` doesn't call `self.__call__` with audio, we get text only.
    
    # We will manually process audio and merge.
    
    # 1. Get input_ids from request
    # prompt_ids = jnp.array(to_numpy(inputs["input_ids"]))
    
    # 2. Get audio features
    # audio_inputs = feature_extractor(audio, sampling_rate=16000, return_attention_mask=True)
    # input_features = jnp.array(to_numpy(audio_inputs["input_features"]))
    # input_features_mask = jnp.array(to_numpy(audio_inputs["attention_mask"])) # usually named attention_mask in FE output
    
    # 3. Expand input_ids for audio placeholders
    # The processor usually does this expansion. If we got input_ids from `apply_transcription_request`, 
    # did it expand <sound>? 
    # If `apply_chat_template` was used without audio-aware logic, it might just produce "<|user|>\n<|begin_of_audio|><|pad|><|end_of_audio|>..."
    # We need to expand tokens if they are not expanded.
    # The config `audio_token_id` is 59260.
    
    # Let's inspect prompt_ids length.
    # print("Prompt shape:", prompt_ids.shape)
    
    # We can try to use `processor(text=..., audio=...)` which we know calls `__call__` and presumably handles everything.
    # We need to construct the text part manually if we use `processor` directly.
    # But let's stick to what we have.
    
    # Re-call processor properly?
    # inputs = processor(text=text_prompts, audio=audio)
    
    # Constructing prompt text manually to use processor.__call__
    # Format: "<|system|>\n<|user|>\n<|begin_of_audio|><|pad|><|end_of_audio|>Please transcribe..."
    # Actually, let's just use `processor` with the raw text if possible.
    # Or, rely on manual expansion logic implicitly? 
    # Wait, `processing_glmasr.py` has `__call__` which does: `expand audio tokens in text`.
    # So if we call `processor(text=..., audio=...)`, it should work.
    
    # We need the conversation formatted text.
    # We can use tokenizer.apply_chat_template(conversations, tokenize=False) to get the string.
    
    # conversations = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "Please transcribe this audio into text"},
    #             {"type": "audio", "audio": audio} # Processor expects audio object or path?
    #         ]
    #     }
    # ]
    # We need to handle audio separately for the text string?
    # apply_chat_template usually puts placeholders.
    
    # Let's try calling processor directly with formatted text.
    # Using a simple trick: 
    # text = "<|user|>\nPlease transcribe this audio into text<|begin_of_audio|><|pad|><|end_of_audio|><|assistant|>"
    # Use processor(text=text, audio=audio)
    
    # Actually, let's look at `processing_glmasr.py` again.
    # `apply_transcription_request` returns `self.apply_chat_template(...)`.
    # If `apply_chat_template` doesn't handle audio, we are lost.
    
    # Fallback: manual full call
    user_prompt = "Please transcribe this audio into text"
    # Note: verify template logic.
    # Using a simplified text for now to get it running.
    text_input = f"<|user|>\n{user_prompt}<|begin_of_audio|><|pad|><|end_of_audio|><|assistant|>"
    
    full_inputs = processor(text=text_input, audio=audio, sampling_rate=16000, return_tensors="pt")
    input_ids = jnp.array(full_inputs["input_ids"].numpy())
    input_features = jnp.array(full_inputs["input_features"].numpy())
    input_features_mask = jnp.array(full_inputs["input_features_mask"].numpy())
    
    print("Input shapes:", input_ids.shape, input_features.shape)

    # 7. Generation Loop (Greedy)
    print("Starting generation...")
    max_new_tokens = 256
    generated_ids = []
    
    curr_input_ids = input_ids
    
    for i in range(max_new_tokens):
        # Forward pass
        # Note: inefficient, recomputing full context each time
        logits = model(
            input_ids=curr_input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            deterministic=True,
            rngs=rngs
        )
        
        # Get next token (argmax)
        next_token_logits = logits[:, -1, :]
        next_token_id = jnp.argmax(next_token_logits, axis=-1)
        
        token_scalar = next_token_id[0].item() # Assuming batch size 1
        generated_ids.append(token_scalar)
        
        # Check EOS
        if token_scalar in eos_token_ids:
            print("EOS reached.")
            break
            
        # Update input_ids
        curr_input_ids = jnp.concatenate([curr_input_ids, next_token_id[:, None]], axis=1)
        
        # Print progress
        print(f"\rGenerated {i+1} tokens...", end="", flush=True)

    print("\nDecoding...")
    text = processor.batch_decode([generated_ids], skip_special_tokens=True)[0]
    print("-" * 40)
    print("Transcription:")
    print(text)
    print("-" * 40)

if __name__ == "__main__":
    main()
