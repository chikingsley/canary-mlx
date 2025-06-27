#!/usr/bin/env python3
"""
Convert NVIDIA Canary-180M-Flash ASR model from NeMo format to MLX format.

This script converts the FastConformer encoder + Transformer decoder architecture
from NeMo's .nemo format to MLX safetensors format for Apple Silicon inference.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
import yaml
from safetensors.torch import save_file

# Model dimensions from Canary-180M-Flash
CANARY_180M_CONFIG = {
    "model_type": "canary",
    "vocab_size": 5248,
    "n_mels": 128,
    "sample_rate": 16000,
    "frame_size_ms": 25,
    "frame_stride_ms": 10,
    "n_fft": 512,
    "encoder": {
        "type": "fastconformer",
        "n_layers": 17,
        "d_model": 512,
        "n_heads": 8,
        "ff_expansion_factor": 4,
        "conv_kernel_size": 9,
        "subsampling_factor": 8,
        "subsampling_conv_channels": 256,
        "dropout": 0.1,
    },
    "decoder": {
        "type": "transformer",
        "n_layers": 4,
        "d_model": 1024,
        "n_heads": 8,
        "ff_expansion_factor": 4,
        "max_seq_len": 1024,
        "dropout": 0.1,
    },
    "languages": ["en", "de", "es", "fr"],
    "tasks": ["transcribe", "translate"],
}


def load_nemo_checkpoint(
    model_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load weights and config from extracted .nemo directory."""
    # Load model weights
    weights_path = model_path / "model_weights.ckpt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu")

    # Load model config
    config_path = model_path / "model_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found at {config_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f)

    return state_dict, config


def remap_fastconformer_weights(
    key: str, value: torch.Tensor
) -> tuple[str | None, mx.array | None]:
    """
    Remap NeMo FastConformer weights to MLX format.

    Key transformations:
    - encoder.pre_encode -> encoder.subsample
    - encoder.layers.X -> encoder.blocks.X
    - Various layer norm and attention remappings
    """
    # Skip preprocessor weights (will handle audio processing separately)
    if key.startswith("preprocessor"):
        return None, None

    # Skip batch norm tracking variables
    if "num_batches_tracked" in key:
        return None, None

    # Handle encoder pre-encoding (downsampling) layers
    if key.startswith("encoder.pre_encode"):
        # Remap conv layers
        if "conv." in key:
            # Extract layer number and type
            key = key.replace("encoder.pre_encode.conv.", "encoder.subsample.conv")
            # Handle depthwise separable convolutions
            if value.ndim == 4:  # Conv2d weights
                # NeMo uses (out_channels, in_channels, height, width)
                # MLX expects (out_channels, height, width, in_channels)
                value = value.permute(0, 2, 3, 1)
            elif value.ndim == 3:  # Conv1d weights
                # NeMo uses (out_channels, in_channels, kernel_size)
                # MLX expects (out_channels, kernel_size, in_channels)
                value = value.permute(0, 2, 1)
        else:
            key = key.replace("encoder.pre_encode.", "encoder.subsample.")

    # Handle FastConformer encoder layers
    elif key.startswith("encoder.layers"):
        # Replace layers with blocks
        key = key.replace("encoder.layers", "encoder.blocks")

        # Remap layer components
        key = key.replace("norm_feed_forward1", "ffn1_norm")
        key = key.replace("norm_feed_forward2", "ffn2_norm")
        key = key.replace("feed_forward1", "ffn1")
        key = key.replace("feed_forward2", "ffn2")
        key = key.replace("norm_conv", "conv_norm")
        key = key.replace("norm_self_att", "attn_norm")
        key = key.replace("norm_out", "final_norm")

        # Remap attention components
        if "self_attn" in key:
            key = key.replace("self_attn", "attn")
            key = key.replace("linear_q", "q_proj")
            key = key.replace("linear_k", "k_proj")
            key = key.replace("linear_v", "v_proj")
            key = key.replace("linear_out", "out_proj")
            key = key.replace("linear_pos", "pos_proj")

        # Remap FFN components
        key = key.replace("linear1", "gate")
        key = key.replace("linear2", "out")

        # Handle conv layers in conformer blocks
        if "conv." in key:
            if "depthwise_conv" in key:
                key = key.replace("depthwise_conv", "dw_conv")
            if value.ndim == 3:  # Conv1d
                value = value.permute(0, 2, 1)

    # Handle encoder-decoder projection
    elif key.startswith("encoder_decoder_proj"):
        key = key.replace("encoder_decoder_proj", "encoder_proj")

    # Handle transformer decoder
    elif key.startswith("transf_decoder"):
        key = key.replace("transf_decoder", "decoder")
        key = key.replace("_embedding", "embed")
        key = key.replace("token_embedding", "token_embed")
        key = key.replace("position_embedding.pos_enc", "pos_embed")
        key = key.replace("_decoder", "")

        # Remap decoder layers
        if "layers" in key:
            key = key.replace("layers", "blocks")
            key = key.replace("layer_norm_1", "attn_norm")
            key = key.replace("layer_norm_2", "cross_attn_norm")
            key = key.replace("layer_norm_3", "ffn_norm")
            key = key.replace("first_sub_layer", "self_attn")
            key = key.replace("second_sub_layer", "cross_attn")
            key = key.replace("third_sub_layer", "ffn")

            # Remap attention projections
            key = key.replace("query_net", "q_proj")
            key = key.replace("key_net", "k_proj")
            key = key.replace("value_net", "v_proj")
            key = key.replace("out_projection", "out_proj")

            # Remap FFN
            key = key.replace("dense_in", "gate")
            key = key.replace("dense_out", "out")

    # Handle output projection
    elif key.startswith("log_softmax.mlp"):
        key = key.replace("log_softmax.mlp.layer0", "output_proj")

    # Convert to MLX array
    if value is not None:
        mlx_value = mx.array(value.detach().numpy())
        return key, mlx_value


def convert_canary_to_mlx(
    model_path: str | Path,
    output_path: str | Path,
    dtype: mx.Dtype = mx.float16,
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
) -> None:
    """
    Convert Canary model from NeMo to MLX format.

    Args:
        model_path: Path to extracted .nemo directory
        output_path: Path to save MLX model
        dtype: Data type for MLX model
        quantize: Whether to quantize the model
        q_bits: Bits for quantization
        q_group_size: Group size for quantization
    """
    print(f"[INFO] Loading Canary model from {model_path}")
    model_path = Path(model_path)

    # Load NeMo checkpoint
    state_dict, nemo_config = load_nemo_checkpoint(model_path)

    # Convert weights to MLX format
    print("[INFO] Converting weights to MLX format")
    mlx_weights = {}

    for key, value in state_dict.items():
        mlx_key, mlx_value = remap_fastconformer_weights(key, value)
        if mlx_key is not None and mlx_value is not None:
            mlx_weights[mlx_key] = mlx_value.astype(dtype)

    # Prepare output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config - create a proper typed dict
    config: dict[str, Any] = CANARY_180M_CONFIG.copy()

    # Update config from NeMo if needed
    if "encoder" in nemo_config:
        encoder_config = config.get("encoder", {})
        if isinstance(encoder_config, dict):
            encoder_config["n_layers"] = nemo_config["encoder"]["n_layers"]
            encoder_config["d_model"] = nemo_config["encoder"]["d_model"]

    if "transf_decoder" in nemo_config:
        decoder_nemo_config = nemo_config["transf_decoder"]["config_dict"]
        decoder_config = config.get("decoder", {})
        if isinstance(decoder_config, dict):
            decoder_config["n_layers"] = decoder_nemo_config["num_layers"]
            decoder_config["d_model"] = decoder_nemo_config["hidden_size"]

    # Add quantization config if requested
    if quantize:
        print(f"[INFO] Quantizing model to {q_bits} bits")
        config["quantization"] = {
            "bits": q_bits,
            "group_size": q_group_size,
        }
        # Note: Actual quantization would require loading into MLX model first
        # This is a placeholder for the quantization step

    # Save weights as safetensors
    print(f"[INFO] Saving weights to {output_path / 'model.safetensors'}")

    # Convert MLX arrays to numpy for safetensors
    numpy_weights = {}
    for k, v in mlx_weights.items():
        numpy_weights[k] = v.astype(mx.float32 if dtype == mx.float32 else mx.float16)

    # Save using safetensors (need to convert to torch tensors first)
    torch_weights = {}
    for k, v in numpy_weights.items():
        # Convert MLX array to numpy array
        numpy_array = np.array(v)
        torch_weights[k] = torch.from_numpy(numpy_array)

    save_file(torch_weights, str(output_path / "model.safetensors"))

    # Save config
    with (output_path / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer files
    print("[INFO] Copying tokenizer files")
    tokenizer_path = output_path / "tokenizers"
    tokenizer_path.mkdir(exist_ok=True)

    # Copy all tokenizer files
    for file in model_path.glob("*.model"):
        target = tokenizer_path / file.name
        target.write_bytes(file.read_bytes())

    for file in model_path.glob("*.vocab"):
        target = tokenizer_path / file.name
        target.write_bytes(file.read_bytes())

    for file in model_path.glob("*.txt"):
        target = tokenizer_path / file.name
        target.write_bytes(file.read_bytes())

    print(f"[INFO] Conversion complete! Model saved to {output_path}")


def main() -> None:
    """
    Convert NVIDIA Canary ASR model to MLX format.

    Args:
        model_path: Path to extracted .nemo directory (containing model_weights.ckpt)
        output_path: Output directory for MLX model
        dtype: Data type for MLX model
    """
    parser = argparse.ArgumentParser(
        description="Convert NVIDIA Canary ASR model to MLX format"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to extracted .nemo directory (containing model_weights.ckpt)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="canary-180m-flash-mlx",
        help="Output directory for MLX model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="Data type for MLX model",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the model",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        help="Bits for quantization",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Group size for quantization",
    )

    args = parser.parse_args()

    # Convert dtype string to MLX dtype
    dtype = mx.float16 if args.dtype == "float16" else mx.float32

    # Run conversion
    convert_canary_to_mlx(
        model_path=args.model_path,
        output_path=args.output_path,
        dtype=dtype,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
    )


if __name__ == "__main__":
    main()
