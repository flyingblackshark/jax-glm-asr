import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

try:
    import torch  # noqa: F401
except ImportError as e:
    raise RuntimeError(
        "PyTorch is required for GLM-ASR preprocessing (`GlmAsrProcessor` uses `return_tensors='pt'`)."
    ) from e

from configuration_glmasr import GlmAsrConfig
from convert_glmasr_weights_to_nnx import convert_pytorch_state_dict_to_nnx_state_dict
from inference_nnx import load_audio, load_feature_extractor, load_tokenizer, shard_model
from modeling_glmasr_nnx import GlmAsrForConditionalGeneration
from processing_glmasr import GlmAsrProcessor
from transformers.utils.hub import cached_file


DEFAULT_AUDIO_EXTS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".webm",
    ".mp4",
    ".mkv",
}


def _build_key_valid_base(token_attention_mask: jax.Array, cache_len: int) -> jax.Array:
    """
    Build a per-sample key validity mask over the whole KV cache length.
    - Prompt padding tokens are invalid keys (0 in tokenizer attention_mask).
    - Future (generated) positions are valid keys once filled.
    """

    prompt_len = token_attention_mask.shape[1]
    if prompt_len > cache_len:
        raise ValueError(f"Prompt length {prompt_len} exceeds cache length {cache_len}. Increase --cache_len.")

    prompt_valid = token_attention_mask.astype(bool)
    if prompt_len == cache_len:
        return prompt_valid

    future_valid = jnp.ones((token_attention_mask.shape[0], cache_len - prompt_len), dtype=bool)
    return jnp.concatenate([prompt_valid, future_valid], axis=1)


@nnx.jit
def prefill(model, input_ids, input_features, input_features_mask, key_valid_base):
    cache_index = jnp.array(0, dtype=jnp.int32)
    prompt_len = input_ids.shape[1]
    cache_len = key_valid_base.shape[1]

    q = jnp.arange(prompt_len, dtype=jnp.int32)[:, None]  # (prompt_len, 1)
    k = jnp.arange(cache_len, dtype=jnp.int32)[None, :]  # (1, cache_len)
    causal = q >= k  # (prompt_len, cache_len)
    valid_k = k < prompt_len  # (1, cache_len)
    allow = causal[None, :, :] & valid_k[None, :, :] & key_valid_base[:, None, :]  # (batch, prompt_len, cache_len)
    attention_mask = jnp.where(allow, 0.0, -1e9)[:, None, :, :]  # (batch, 1, prompt_len, cache_len)

    logits = model(
        input_ids=input_ids,
        input_features=input_features,
        input_features_mask=input_features_mask,
        attention_mask=attention_mask,
        deterministic=True,
        cache_index=cache_index,
    )
    return logits[:, -1, :], cache_index + prompt_len


@nnx.jit
def decode(model, input_ids, cache_index, key_valid_base):
    cache_len = key_valid_base.shape[1]
    k = jnp.arange(cache_len, dtype=jnp.int32)[None, :]  # (1, cache_len)
    valid_k = k < (cache_index + 1)  # (1, cache_len)
    allow = valid_k & key_valid_base  # (batch, cache_len)
    attention_mask = jnp.where(allow, 0.0, -1e9)[:, None, None, :]  # (batch, 1, 1, cache_len)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        deterministic=True,
        cache_index=cache_index,
    )
    return logits[:, -1, :], cache_index + 1


def iter_audio_files(input_path: str, exts: set[str]) -> list[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in exts else []
    if not p.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    audio_files: list[str] = []
    for root, dirs, files in os.walk(p):
        dirs.sort()
        for name in sorted(files):
            file_path = Path(root) / name
            if file_path.suffix.lower() in exts:
                audio_files.append(str(file_path))
    return audio_files


def load_config(args) -> GlmAsrConfig:
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            config_dict = json.load(f)
        return GlmAsrConfig(**config_dict)
    return GlmAsrConfig.from_pretrained(args.model_id, revision=args.revision)


def load_nnx_state_dict(args, config: GlmAsrConfig):
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
    return state_dict


def build_processor(args) -> GlmAsrProcessor:
    feature_extractor = load_feature_extractor(args.model_id, args.revision)

    tokenizer_source = (
        args.tokenizer_path if args.tokenizer_path and os.path.exists(args.tokenizer_path) else args.model_id
    )
    tokenizer = load_tokenizer(tokenizer_source, args.revision)

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        try:
            chat_template_file = cached_file(tokenizer_source, "chat_template.jinja", revision=args.revision)
            with open(chat_template_file, "r") as f:
                chat_template = f.read()
        except Exception:
            chat_template = None

    return GlmAsrProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer, chat_template=chat_template)


def transcribe_batch(
    *,
    model: GlmAsrForConditionalGeneration,
    processor: GlmAsrProcessor,
    config: GlmAsrConfig,
    mesh: Mesh,
    audio_paths: list[str],
    prompt: str,
    max_new_tokens: int,
    strip_prefix: bool,
    pad_to: int,
    cache_len: int,
):
    if not audio_paths:
        return []

    batch_size = len(audio_paths)
    if batch_size > pad_to:
        raise ValueError(f"Batch size {batch_size} exceeds pad_to {pad_to}.")

    padded_paths = list(audio_paths)
    if batch_size < pad_to:
        padded_paths.extend([audio_paths[-1]] * (pad_to - batch_size))

    audios = [load_audio(p) for p in padded_paths]
    if processor.chat_template:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": ""},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_inputs = [
                processor.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=processor.chat_template,
                )
                for _ in range(len(padded_paths))
            ]
        except Exception:
            text_inputs = [
                f"<|user|>\n{prompt}<|begin_of_audio|><|pad|><|end_of_audio|><|assistant|>"
                for _ in range(len(padded_paths))
            ]
    else:
        text_inputs = [
            f"<|user|>\n{prompt}<|begin_of_audio|><|pad|><|end_of_audio|><|assistant|>"
            for _ in range(len(padded_paths))
        ]
    full_inputs = processor(text=text_inputs, audio=audios, sampling_rate=16000, return_tensors="pt")

    if full_inputs["input_ids"].shape[0] != pad_to:
        raise RuntimeError("Unexpected input_ids batch shape from processor.")
    if full_inputs["input_features"].shape[0] != pad_to:
        raise RuntimeError(
            "Audio chunking produced a mismatched batch. Please try shorter audio or run with --batch_size 1."
        )

    token_attention_mask = jnp.array(full_inputs["attention_mask"].numpy())
    input_ids = jnp.array(full_inputs["input_ids"].numpy())
    input_features = jnp.array(full_inputs["input_features"].numpy())
    input_features_mask = jnp.array(full_inputs["input_features_mask"].numpy())

    prompt_len = int(input_ids.shape[1])
    if prompt_len + max_new_tokens > cache_len:
        raise ValueError(
            f"cache_len={cache_len} is too small for prompt_len={prompt_len} + max_new_tokens={max_new_tokens}. "
            "Increase --cache_len."
        )

    key_valid_base = _build_key_valid_base(token_attention_mask, cache_len)

    replicated = NamedSharding(mesh, PartitionSpec())
    input_ids = jax.device_put(input_ids, replicated)
    input_features = jax.device_put(input_features, replicated)
    input_features_mask = jax.device_put(input_features_mask, replicated)
    key_valid_base = jax.device_put(key_valid_base, replicated)

    logits, cache_index = prefill(model, input_ids, input_features, input_features_mask, key_valid_base)

    eos_token_ids = config.text_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos_ids_host = np.asarray(eos_token_ids, dtype=np.int32)
    eos_fallback = int(eos_ids_host[0])

    generated_ids: list[list[int]] = [[] for _ in range(pad_to)]
    finished = np.zeros((pad_to,), dtype=bool)

    token_id = jnp.argmax(logits, axis=-1)  # (batch,)
    for _ in range(max_new_tokens):
        token_host = np.asarray(token_id)

        for i in range(pad_to):
            if finished[i]:
                continue
            tid = int(token_host[i])
            generated_ids[i].append(tid)
            if (eos_ids_host == tid).any():
                finished[i] = True

        if bool(finished.all()):
            break

        token_in = token_host.astype(np.int32, copy=True)
        token_in[finished] = eos_fallback
        token_in = jax.device_put(jnp.array(token_in)[:, None], replicated)

        logits, cache_index = decode(model, token_in, cache_index, key_valid_base)
        token_id = jnp.argmax(logits, axis=-1)

    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True, strip_prefix=strip_prefix)
    return decoded[:batch_size]


def main():
    parser = argparse.ArgumentParser(description="Batch ASR inference (recursive) and export CSV.")
    parser.add_argument("--input_path", type=str, required=True, help="Audio file or directory to scan recursively.")
    parser.add_argument("--output_csv", type=str, default="transcripts.csv", help="Output CSV path.")

    parser.add_argument("--weights_path", type=str, default=None, help="Optional local .safetensors weights path. If omitted, download via HF cache.")
    parser.add_argument("--config_path", type=str, default=None, help="Optional local config.json path. If omitted, download via HF cache.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional local tokenizer directory. If omitted, load from HF (uses HF cache).")
    parser.add_argument("--model_id", type=str, default="zai-org/GLM-ASR-Nano-2512")
    parser.add_argument("--revision", type=str, default=None)

    parser.add_argument("--prompt", type=str, default="Please transcribe this audio into text")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of audios to decode simultaneously.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--cache_len", type=int, default=2048, help="KV cache length for generation.")
    parser.add_argument(
        "--no_strip_prefix",
        action="store_true",
        help="Do not strip assistant prefix/quotes from decoded text.",
    )
    parser.add_argument("--basename", action="store_true", help="Write only basename to CSV (default: relative path).")
    parser.add_argument(
        "--exts",
        type=str,
        default=",".join(sorted(DEFAULT_AUDIO_EXTS)),
        help="Comma-separated audio extensions to include (e.g. .wav,.mp3).",
    )
    args = parser.parse_args()

    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    if not exts:
        raise ValueError("--exts is empty.")

    audio_files = iter_audio_files(args.input_path, exts)
    if not audio_files:
        print("No audio files found.", file=sys.stderr)
        return 2

    input_root = Path(args.input_path).resolve()
    if input_root.is_file():
        input_root = input_root.parent

    out_path = Path(args.output_csv)
    if out_path.parent and str(out_path.parent) != ".":
        out_path.parent.mkdir(parents=True, exist_ok=True)

    devices = jax.devices()
    print(f"Devices available: {len(devices)}")
    mesh = Mesh(devices, axis_names=("tp",))

    config = load_config(args)

    rngs = nnx.Rngs(0)
    model = GlmAsrForConditionalGeneration(config, rngs=rngs)
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    model.init_cache(args.batch_size, args.cache_len)

    state_dict = load_nnx_state_dict(args, config)
    shard_model(model, mesh, state_dict)

    processor = build_processor(args)

    print(f"Found {len(audio_files)} audio files. Running inference...")

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["文件名", "转译内容"])

        total = len(audio_files)
        batch_size = args.batch_size

        for start in range(0, total, batch_size):
            batch_paths = audio_files[start : start + batch_size]
            idx_end = start + len(batch_paths)

            try:
                transcripts = transcribe_batch(
                    model=model,
                    processor=processor,
                    config=config,
                    mesh=mesh,
                    audio_paths=batch_paths,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    strip_prefix=not args.no_strip_prefix,
                    pad_to=batch_size,
                    cache_len=args.cache_len,
                )
            except Exception as e:
                print(f"[{start+1}/{total}] BATCH ERROR: {e}", file=sys.stderr)
                transcripts = []
                for p in batch_paths:
                    try:
                        transcripts.append(
                            transcribe_batch(
                                model=model,
                                processor=processor,
                                config=config,
                                mesh=mesh,
                                audio_paths=[p],
                                prompt=args.prompt,
                                max_new_tokens=args.max_new_tokens,
                                strip_prefix=not args.no_strip_prefix,
                                pad_to=batch_size,
                                cache_len=args.cache_len,
                            )[0]
                        )
                    except Exception as e2:
                        print(f"  ERROR: {p}: {e2}", file=sys.stderr)
                        transcripts.append("")

            for audio_path, transcript in zip(batch_paths, transcripts, strict=True):
                name = os.path.basename(audio_path) if args.basename else os.path.relpath(audio_path, input_root)
                writer.writerow([name, transcript])

            if idx_end % (10 * batch_size) == 0 or idx_end == total:
                last_name = (
                    os.path.basename(batch_paths[-1])
                    if args.basename
                    else os.path.relpath(batch_paths[-1], input_root)
                )
                print(f"[{idx_end}/{total}] {last_name}")

    print(f"Done. Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
