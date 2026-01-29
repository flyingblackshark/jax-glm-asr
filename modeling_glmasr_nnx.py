
import math
from typing import Optional, Tuple, Union, List

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig, FlaxPreTrainedModel

from configuration_glmasr import GlmAsrConfig, GlmAsrEncoderConfig


class GlmAsrRotaryEmbedding(nnx.Module):
    def __init__(self, config: GlmAsrConfig, rngs: nnx.Rngs = None):
        self.config = config
        self.max_position_embeddings = config.max_position_embeddings
        rope_parameters = config.rope_parameters or {}
        self.rope_theta = rope_parameters.get("rope_theta", 10000.0)
        self.partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        self.rope_type = rope_parameters.get("rope_type", "default")
        
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.dim = int(head_dim * self.partial_rotary_factor)
        
        # Calculate inv_freq
        # Note: 'default' RoPE logic. If 'llama3' or others needed, would need condition.
        inv_freq = 1.0 / (
            self.rope_theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
        )
        self.inv_freq = nnx.Cache(inv_freq)
        self.attention_scaling = 1.0 


    def __call__(self, x: jax.Array, position_ids: jax.Array) -> Tuple[jax.Array, jax.Array]:
        inv_freq = self.inv_freq.value
        
        # Expand inv_freq: (1, dim/2, 1) to broadcast
        inv_freq_expanded = inv_freq[None, :, None]
        
        # position_ids: (batch, seq_len) -> (batch, 1, seq_len)
        position_ids_expanded = position_ids[:, None, :].astype(jnp.float32)
        
        # Matmul: (batch, dim/2, 1) * (batch, 1, seq_len) -> (batch, dim/2, seq_len)
        # Transpose to (batch, seq_len, dim/2)
        freqs = jnp.matmul(inv_freq_expanded, position_ids_expanded).transpose(0, 2, 1)
        
        # Concat along last dim to get (batch, seq_len, dim) where pairs are identical
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        
        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling
        
        return cos, sin

def rotate_half(x: jax.Array) -> jax.Array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb(q: jax.Array, k: jax.Array, cos: jax.Array, sin: jax.Array, unsqueeze_dim: int = 1) -> Tuple[jax.Array, jax.Array]:
    """
    Applies Rotary Position Embedding to the query and key tensors.
    """
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    
    rotary_dim = cos.shape[-1]
    
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    
    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    
    # Concatenate back to full shape
    q_embed = jnp.concatenate([q_embed, q_pass], axis=-1)
    k_embed = jnp.concatenate([k_embed, k_pass], axis=-1)
    
    return q_embed, k_embed

class GlmAsrAttention(nnx.Module):
    def __init__(self, config: GlmAsrConfig, layer_idx: int, rngs: nnx.Rngs):
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads or config.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        
        self.q_proj = nnx.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, use_bias=True, rngs=rngs)
        self.k_proj = nnx.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, use_bias=True, rngs=rngs)
        self.o_proj = nnx.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, use_bias=True, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        position_embeddings: Optional[Tuple[jax.Array, jax.Array]] = None,
        deterministic: bool = True,
        rngs: Optional[nnx.Rngs] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).reshape(hidden_shape).transpose((0, 2, 1, 3)) # (batch, head, seq, dim)
        key_states = self.k_proj(hidden_states).reshape(hidden_shape).transpose((0, 2, 1, 3))
        value_states = self.v_proj(hidden_states).reshape(hidden_shape).transpose((0, 2, 1, 3))

        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Expand cos/sin for broadcasting: (batch, 1, seq, dim) matches (batch, head, seq, dim)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        # Repeat KV for GQA
        if self.num_key_value_groups > 1:
            key_states = jnp.repeat(key_states, self.num_key_value_groups, axis=1)
            value_states = jnp.repeat(value_states, self.num_key_value_groups, axis=1)

        # Attention: (batch, head, seq, seq)
        attn_weights = jnp.matmul(query_states, key_states.transpose((0, 1, 3, 2))) * self.scaling
        
        # Softmax
        attn_weights = nnx.softmax(attn_weights, axis=-1)
        
        if not deterministic and rngs:
             attn_weights = nnx.dropout(attn_weights, self.attention_dropout, rngs=rngs)

        attn_output = jnp.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose((0, 2, 1, 3)).reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights

class GlmAsrMLP(nnx.Module):
    def __init__(self, config: GlmAsrConfig, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.fc2 = nnx.Linear(config.intermediate_size, config.hidden_size, rngs=rngs)
        self.hidden_act = config.hidden_act

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = self.fc1(hidden_states)
        if self.hidden_act == "gelu":
            hidden_states = jax.nn.gelu(hidden_states, approximate=False)
        else:
            hidden_states = jax.nn.relu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class GlmAsrEncoderLayer(nnx.Module):
    def __init__(self, config: GlmAsrConfig, layer_idx: int, rngs: nnx.Rngs):
        self.hidden_size = config.hidden_size
        self.self_attn = GlmAsrAttention(config, layer_idx, rngs=rngs)
        self.mlp = GlmAsrMLP(config, rngs=rngs)
        self.input_layernorm = nnx.LayerNorm(config.hidden_size, rngs=rngs)
        self.post_attention_layernorm = nnx.LayerNorm(config.hidden_size, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        position_embeddings: Optional[Tuple[jax.Array, jax.Array]] = None,
        deterministic: bool = True,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jax.Array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            deterministic=deterministic,
            rngs=rngs
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class GlmAsrEncoder(nnx.Module):
    def __init__(self, config: GlmAsrEncoderConfig, rngs: nnx.Rngs):
        self.config = config
        self.conv1 = nnx.Conv(config.num_mel_bins, config.hidden_size, kernel_size=(3,), padding=(1,), rngs=rngs)
        self.conv2 = nnx.Conv(config.hidden_size, config.hidden_size, kernel_size=(3,), strides=(2,), padding=(1,), rngs=rngs)
        
        self.layers = nnx.List([
            GlmAsrEncoderLayer(config, layer_idx, rngs=rngs)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = nnx.LayerNorm(config.hidden_size, rngs=rngs)
        self.rotary_emb = GlmAsrRotaryEmbedding(config=config)

    def __call__(self, input_features: jax.Array, deterministic: bool = True, rngs: Optional[nnx.Rngs] = None) -> jax.Array:
        # Input: (batch, n_mels, length)
        # Transpose for Conv: (batch, length, n_mels)
        hidden_states = input_features.transpose((0, 2, 1))
        
        hidden_states = self.conv1(hidden_states)
        hidden_states = nnx.gelu(hidden_states)
        
        hidden_states = self.conv2(hidden_states)
        hidden_states = nnx.gelu(hidden_states)
        
        # After Conv, shape is (batch, new_length, hidden_size).
        
        # Rotary embeddings
        position_embeddings = self.rotary_emb(
            hidden_states, position_ids=jnp.arange(hidden_states.shape[1])[None, :]
        )

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings, deterministic=deterministic, rngs=rngs)

        hidden_states = self.norm(hidden_states)
        return hidden_states

class GlmAsrMultiModalProjector(nnx.Module):
    def __init__(self, config: GlmAsrConfig, rngs: nnx.Rngs):
        self.linear_1 = nnx.Linear(config.audio_config.intermediate_size, config.text_config.hidden_size * 2, rngs=rngs)
        self.act = nnx.gelu if config.projector_hidden_act == "gelu" else nnx.relu
        self.linear_2 = nnx.Linear(config.text_config.hidden_size * 2, config.text_config.hidden_size, rngs=rngs)

    def __call__(self, audio_features: jax.Array) -> jax.Array:
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class GlmAsrForConditionalGeneration(nnx.Module):
    def __init__(self, config: GlmAsrConfig, rngs: nnx.Rngs):
        self.config = config
        self.audio_tower = GlmAsrEncoder(config.audio_config, rngs=rngs)
        self.multi_modal_projector = GlmAsrMultiModalProjector(config, rngs=rngs)
        
        self.language_model = LlamaForCausalLM(config.text_config, rngs=rngs)

    def get_audio_features(self, input_features: jax.Array, input_features_mask: jax.Array) -> Tuple[jax.Array, jax.Array]:
        audio_outputs = self.audio_tower(input_features)
        
        # Reshape: (batch, seq, intermediate_size)
        # matches PyTorch: audio_hidden_states.reshape(input_features.shape[0], -1, self.config.audio_config.intermediate_size)
        b = audio_outputs.shape[0]
        inter = self.config.audio_config.intermediate_size
        audio_hidden_states = audio_outputs.reshape(b, -1, inter)
        
        audio_embeds = self.multi_modal_projector(audio_hidden_states)
        
        # Masking logic
        audio_lengths = input_features_mask.sum(-1)
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
             audio_lengths = (audio_lengths + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        
        merge_factor = 4
        post_lengths = (audio_lengths - merge_factor) // merge_factor + 1
        
        # In JAX/Flax we usually avoid dynamic boolean indexing if possible for JIT.
        # But for 'get_audio_features', we can probably return the full sequence and let the caller handle masking,
        # or apply the mask to zero out invalid positions.
        # The PyTorch code flattens/selects valid ones: `audio_embeds[valid_mask]`.
        # I will return the full sequence and the lengths.
        
        return audio_embeds, post_lengths

    def __call__(
        self,
        input_ids: jax.Array,
        input_features: Optional[jax.Array] = None,
        input_features_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
        rngs: Optional[nnx.Rngs] = None,
    ):
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        if input_features is not None:
            audio_embeds, post_lengths = self.get_audio_features(input_features, input_features_mask)
            
            # Combine embeddings
            batch, seq_len = input_ids.shape
            
            # Identify indices where audio tokens are located
            audio_token_id = self.config.audio_token_id
            is_audio_token = (input_ids == audio_token_id)
            
            # Check if we have enough audio embeddings for the tokens in each sequence
            # This logic assumes the structure is well-formed where contiguous <sound> tokens match audio length
            # For simplicity in this static graph port, we will use a loop or assume pre-calculated match.
            # However, JAX handles this better with scatter/masks.
            
            # We need to flatten and scatter.
            # audio_embeds: (batch, n_audio_tokens_total_or_max, hidden)
            # We must broadcast audio_embeds into inputs_embeds at is_audio_token positions
            
            # Note: The PyTorch code uses masked_scatter which fills slots sequentially from source.
            # JAX's .at[mask].set(values) requires values to match the shape of the mask updates.
            # The 'audio_embeds' returned by get_audio_features might have padding if batching.
            # We need to ensure the number of TRUE in is_audio_token matches valid audio_embeds.
            
            # Simplified Logic:
            # If we assume 1:1 mapping of audio tokens in input_ids to audio_embeds sequence:
            # We can use jnp.where
            
            # Expand to match dimensions if necessary or flatten
            # If `audio_embeds` is (batch, audio_seq, hidden) and we want to place it into `input_ids` audio slots.
            # But the audio slots might be at different positions.
            
            # Let's try to assume strict alignment for this port as doing dynamic ragging in JAX is hard.
            # We select non-audio embeddings from inputs_embeds and audio embeddings from audio_embeds
            
            num_audio_tokens = is_audio_token.sum(axis=1)
            # In a real scenario, we'd need sophisticated padding handling.
            # For now, we will perform a safe scatter if sizes match, or just return inputs_embeds if complex.
            
            # Implementation for one sequence (vmap can handle batch):
            def combine_single(ids, text_emb, audio_emb, valid_audio_len):
                 # ids: (seq,), text_emb: (seq, dim), audio_emb: (audio_seq, dim)
                 mask = (ids == audio_token_id)
                 # We take the first `valid_audio_len` embeddings from audio_emb
                 # And place them into the slots where mask is True.
                 
                 # To do this cleanly: create a range array to index audio_emb
                 audio_indices = jnp.cumsum(mask) - 1
                 # Select only where mask is True
                 valid_audio_indices = jnp.where(mask, audio_indices, 0)
                 
                 replacement = audio_emb[valid_audio_indices]
                 
                 return jnp.where(mask[:, None], replacement, text_emb)

            # Vmap over batch
            # We need valid_audio_len from post_lengths
            inputs_embeds = jax.vmap(combine_single)(input_ids, inputs_embeds, audio_embeds, post_lengths)

    def init_cache(self, batch_size, max_length, dtype=jnp.bfloat16):
        # Propagate to all layers
        for layer in self.language_model.model.layers:
            layer.self_attn.init_cache(batch_size, max_length, dtype)

    def __call__(
        self,
        input_ids: jax.Array,
        input_features: Optional[jax.Array] = None,
        input_features_mask: Optional[jax.Array] = None,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
        rngs: Optional[nnx.Rngs] = None,
        cache_index: Optional[jax.Array] = None,
    ):
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        if input_features is not None:
            audio_embeds, post_lengths = self.get_audio_features(input_features, input_features_mask)
            
            # Combine embeddings logic ...
            # Assume combine_single is defined in scope above or replicated
            batch, seq_len = input_ids.shape
            audio_token_id = self.config.audio_token_id
            
            def combine_single(ids, text_emb, audio_emb, valid_audio_len):
                 mask = (ids == audio_token_id)
                 audio_indices = jnp.cumsum(mask) - 1
                 valid_audio_indices = jnp.where(mask, audio_indices, 0)
                 replacement = audio_emb[valid_audio_indices]
                 return jnp.where(mask[:, None], replacement, text_emb)

            inputs_embeds = jax.vmap(combine_single)(input_ids, inputs_embeds, audio_embeds, post_lengths)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_index=cache_index,
        )


class LlamaRMSNorm(nnx.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, rngs: nnx.Rngs = None):
        self.variance_epsilon = eps
        self.weight = nnx.Param(jnp.ones((hidden_size,), dtype=jnp.float32))

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.variance_epsilon)
        return (self.weight.value * hidden_states).astype(input_dtype)

class LlamaRotaryEmbedding(nnx.Module):
    def __init__(self, config: PretrainedConfig, rngs: nnx.Rngs = None):
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        rope_parameters = config.rope_parameters or {}
        self.rope_theta = rope_parameters.get("rope_theta", 10000.0)
        
        inv_freq = 1.0 / (
            self.rope_theta ** (jnp.arange(0, self.head_dim, 2, dtype=jnp.float32) / self.head_dim)
        )
        self.inv_freq = nnx.Cache(inv_freq)

    def __call__(self, x: jax.Array, position_ids: jax.Array) -> Tuple[jax.Array, jax.Array]:
        inv_freq = self.inv_freq.value
        inv_freq_expanded = inv_freq[None, :, None]
        position_ids_expanded = position_ids[:, None, :].astype(jnp.float32)
        freqs = jnp.matmul(inv_freq_expanded, position_ids_expanded).transpose(0, 2, 1)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        return cos, sin

class LlamaAttention(nnx.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int, rngs: nnx.Rngs):
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        
        attention_bias = getattr(config, "attention_bias", False)
        
        self.q_proj = nnx.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, use_bias=attention_bias, rngs=rngs)
        self.k_proj = nnx.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, use_bias=attention_bias, rngs=rngs)
        self.v_proj = nnx.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, use_bias=attention_bias, rngs=rngs)

        self.o_proj = nnx.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, use_bias=attention_bias, rngs=rngs)

    def init_cache(self, batch_size: int, max_length: int, dtype=jnp.bfloat16):
        cache_shape = (batch_size, self.num_key_value_heads, max_length, self.head_dim)
        self.k_cache = nnx.Variable(jnp.zeros(cache_shape, dtype=dtype))
        self.v_cache = nnx.Variable(jnp.zeros(cache_shape, dtype=dtype))

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        position_embeddings: Optional[Tuple[jax.Array, jax.Array]] = None,
        deterministic: bool = True,
        rngs: Optional[nnx.Rngs] = None,
        cache_index: Optional[jax.Array] = None,
    ) -> jax.Array:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_proj(hidden_states).reshape(hidden_shape).transpose((0, 2, 1, 3))
        key_states = self.k_proj(hidden_states).reshape(hidden_shape).transpose((0, 2, 1, 3))
        value_states = self.v_proj(hidden_states).reshape(hidden_shape).transpose((0, 2, 1, 3))
        
        if position_embeddings is not None:
             cos, sin = position_embeddings
             query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        # KV Cache Update
        if hasattr(self, "k_cache") and cache_index is not None:
            # cache_index: scalar or (batch,) 
            # If we are filling a sequence (prefill), we might use lax.dynamic_update_slice
            # If decoding one token, same.
            
            # Simple case: cache_index is start index
            idx = cache_index
            
            # update cache
            key_states = key_states.astype(self.k_cache.value.dtype)
            value_states = value_states.astype(self.v_cache.value.dtype)
            
            self.k_cache.value = jax.lax.dynamic_update_slice(self.k_cache.value, key_states, (0, 0, idx, 0))
            self.v_cache.value = jax.lax.dynamic_update_slice(self.v_cache.value, value_states, (0, 0, idx, 0))
            
            # Use cached values for attention
            key_states = self.k_cache.value
            value_states = self.v_cache.value
            
        # GQA repeat
        if self.num_key_value_groups > 1:
            key_states = jnp.repeat(key_states, self.num_key_value_groups, axis=1)
            value_states = jnp.repeat(value_states, self.num_key_value_groups, axis=1)
            
        attn_weights = jnp.matmul(query_states, key_states.transpose((0, 1, 3, 2))) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nnx.softmax(attn_weights, axis=-1)
        
        attn_output = jnp.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose((0, 2, 1, 3)).reshape(*input_shape, -1)
        return self.o_proj(attn_output)


class LlamaMLP(nnx.Module):
    def __init__(self, config: PretrainedConfig, rngs: nnx.Rngs):
        mlp_bias = getattr(config, "mlp_bias", False)
        self.gate_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=mlp_bias, rngs=rngs)
        self.up_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=mlp_bias, rngs=rngs)
        self.down_proj = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=mlp_bias, rngs=rngs)

        self.act_fn = nnx.silu

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nnx.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int, rngs: nnx.Rngs):
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config, layer_idx, rngs=rngs)
        self.mlp = LlamaMLP(config, rngs=rngs)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, rngs=rngs)


    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        position_embeddings: Optional[Tuple[jax.Array, jax.Array]] = None,
        deterministic: bool = True,
        rngs: Optional[nnx.Rngs] = None,
        cache_index: Optional[jax.Array] = None,
    ) -> jax.Array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
             hidden_states, attention_mask, position_embeddings, deterministic, rngs, cache_index
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states



class LlamaModel(nnx.Module):
    def __init__(self, config: PretrainedConfig, rngs: nnx.Rngs):
        self.config = config
        self.embed_tokens = nnx.Embed(config.vocab_size, config.hidden_size, rngs=rngs)
        self.layers = nnx.List([
            LlamaDecoderLayer(config, i, rngs=rngs) for i in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, rngs=rngs)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def __call__(
        self,
        input_ids: Optional[jax.Array] = None,
        inputs_embeds: Optional[jax.Array] = None,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
        rngs: Optional[nnx.Rngs] = None,
        cache_index: Optional[jax.Array] = None,
    ) -> jax.Array:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        hidden_states = inputs_embeds
        # seq_len logic: if caching, inputs_embeds is current chunk (1 or n), but full seq len matters for RoPE/Mask.
        # But RoPE traditionally depends on position indices passed.
        # We need to compute position_ids accurately.
        
        # If cache_index is provided:
        # cache_index tells us where 'hidden_states' starts in the global sequence.
        # So position_ids = cache_index + arange(len)
        
        seq_len = hidden_states.shape[1]
        
        if cache_index is not None:
            # Assume single batch index or same for all
            # (batch, seq)
            position_ids = cache_index + jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        else:
            position_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
            
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Causal mask (simple implementation)
        # If caching, attention_mask should cover the full cache size if we are attending to it.
        # Usually provided by caller if complex.

        # Causal mask (simple implementation)
        # If caching, attention_mask should cover the full cache size (including current tokens)
        # The cache holds up to max_len.
        
        # We need a mask of shape (1, 1, seq_len, total_key_len)
        # However, our KV cache is fixed size `max_length`.
        # When using cache, we usually mask out future tokens in the cache buffer as well?
        # Actually, `dynamic_update_slice` only writes to valid part. The rest is zeros/garbage.
        # We typically need the mask to allow attention only to `[:cache_index + seq_len]`.
        
        # But for the "causal" part within `seq_len`:
        # i can attend to j if j <= i + cache_index (in global coords).
        # Query positions: cache_index, cache_index+1, ...
        # Key positions: 0, 1, 2, ...
        
        # For simplicity in this static shape constrained prefill/decode:
        # We can pass specific mask or compute it.
        # Key length is `max_length` if we use the cache directly.
        
        # Let's assume the callers (Attention) use `self.k_cache` which has size `max_length`.
        # So we need a mask (1, 1, seq_len, max_length).
        
        if attention_mask is None:
            if cache_index is not None:
                # We need to construct a mask that allows attention to:
                # 1. Previous tokens in cache (0 to cache_index-1)
                # 2. Current tokens in causal manner (cache_index to cache_index + seq_len - 1)
                
                # Global indices of queries:
                # q_idx = cache_index + arange(seq_len)
                # Global indices of keys:
                # k_idx = arange(max_length)
                
                # Mask: q_idx >= k_idx
                # (seq_len, 1) >= (1, max_len)
                
                max_len = self.layers[0].self_attn.k_cache.value.shape[2] # Access cache shape from a layer
                
                q_idx = position_ids # (batch, seq)
                k_idx = jnp.arange(max_len, dtype=jnp.int32)[None, :]
                
                # Causal mask: attend if q >= k. Also k must be valid (< cache_index + seq_len)?
                # Usually we assume the cache is filled sequentially.
                # Garbage data is at k_idx >= current_total_len.
                current_total_len = cache_index + seq_len
                
                mask_causal = (q_idx[..., None] >= k_idx)
                mask_valid = (k_idx < current_total_len)
                
                mask = mask_causal & mask_valid
                mask = mask[:, None, :, :] # (batch, 1, seq, max_len)
                
                attention_mask = jnp.where(mask, 0, -1e9)
            elif seq_len > 1:
                mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len)))
                attention_mask = jnp.where(mask == 1, 0, -1e9)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask, position_embeddings, deterministic, rngs, cache_index
            )
            
        return self.norm(hidden_states)

        
    def get_input_embeddings(self):
        return self.embed_tokens

class LlamaForCausalLM(nnx.Module):
    def __init__(self, config: PretrainedConfig, rngs: nnx.Rngs):
        self.model = LlamaModel(config, rngs=rngs)
        self.lm_head = nnx.Linear(config.hidden_size, config.vocab_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        input_ids: Optional[jax.Array] = None,
        inputs_embeds: Optional[jax.Array] = None,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
        rngs: Optional[nnx.Rngs] = None,
        cache_index: Optional[jax.Array] = None,
    ) -> jax.Array:
        hidden_states = self.model(input_ids, inputs_embeds, attention_mask, deterministic, rngs, cache_index)
        logits = self.lm_head(hidden_states)
        return logits
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
