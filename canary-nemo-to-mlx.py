# canary-nemo-to-mlx.py
#
# This script converts NVIDIA NeMo Canary-1B-Flash model weights from a.ckpt file
# to the.safetensors format for use with the MLX framework on Apple Silicon.
#
# It adapts the successful conversion logic from the Parakeet-to-MLX script for the
# shared FastConformer encoder and introduces new logic for Canary's Transformer decoder,
# referencing conventions from other MLX model ports like Llama.

import torch
from safetensors.torch import save_file
import os
import tarfile
from collections import OrderedDict

# --- Configuration ---
NEMO_FILE_NAME = "canary-1b-flash.nemo"
EXTRACT_DIR = "canary_nemo_extracted"
INPUT_CKPT_NAME = "model_weights.ckpt"
OUTPUT_SAFETENSORS_NAME = "model.safetensors"

def convert_weights():
"""
Main function to perform the weight conversion.
"""
# --- Step 1: Extract the.nemo archive ---
if not os.path.exists(NEMO_FILE_NAME):
print(f"Error: {NEMO_FILE_NAME} not found.")
print("Please download the model from Hugging Face: hf.co/nvidia/canary-1b-flash")
return

    print(f"Extracting '{NEMO_FILE_NAME}'...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with tarfile.open(NEMO_FILE_NAME, 'r') as tar:
        tar.extractall(path=EXTRACT_DIR)
    print(f"Extraction complete. Files are in '{EXTRACT_DIR}/'")

    input_ckpt_path = os.path.join(EXTRACT_DIR, INPUT_CKPT_NAME)
    if not os.path.exists(input_ckpt_path):
        raise FileNotFoundError(f"Could not find '{INPUT_CKPT_NAME}' in the extracted archive.")

    # --- Step 2: Load the NeMo checkpoint ---
    print(f"Loading NeMo checkpoint from '{input_ckpt_path}'...")
    # Load onto CPU to avoid GPU memory issues and ensure portability
    checkpoint = torch.load(input_ckpt_path, map_location="cpu")

    # The actual weights are in the 'state_dict'
    if 'state_dict' not in checkpoint:
        raise KeyError("'state_dict' not found in the checkpoint.")

    nemo_state_dict = checkpoint['state_dict']
    print(f"Loaded {len(nemo_state_dict)} tensors from the checkpoint.")

    # --- Step 3: Convert the state_dict to MLX format ---
    mlx_state_dict = OrderedDict()

    print("Converting tensor keys and shapes for MLX compatibility...")
    for key, value in nemo_state_dict.items():
        # Skip unnecessary keys from the training framework
        if key.startswith("preprocessor") or 'num_batches_tracked' in key:
            continue

        new_key = key

        # --- Encoder and general key mappings ---
        # The FastConformer encoder structure is similar to Parakeet
        new_key = new_key.replace("encoder.layers", "encoder.blocks")
        new_key = new_key.replace("self_attn", "attention")
        new_key = new_key.replace("linear_q", "wq")
        new_key = new_key.replace("linear_k", "wk")
        new_key = new_key.replace("linear_v", "wv")
        new_key = new_key.replace("linear_out", "wo")
        new_key = new_key.replace("feed_forward.layer1", "feed_forward.w1")
        new_key = new_key.replace("feed_forward.layer2", "feed_forward.w2")

        # --- Decoder-specific key mappings (Transformer logic) ---
        new_key = new_key.replace("decoder.decoder_layers", "decoder.layers")
        new_key = new_key.replace("multi_head_attn", "attention")
        new_key = new_key.replace("q_proj", "wq")
        new_key = new_key.replace("k_proj", "wk")
        new_key = new_key.replace("v_proj", "wv")
        new_key = new_key.replace("out_proj", "wo")
        new_key = new_key.replace("decoder.final_norm", "decoder.norm")
        new_key = new_key.replace("log_softmax", "output") # Final classification layer

        # LayerNorm mappings
        new_key = new_key.replace("norm1", "attention_norm")
        new_key = new_key.replace("norm2", "ffn_norm")
        new_key = new_key.replace("norm_feed_forward", "ffn_norm") # Some conformer blocks use this name
        new_key = new_key.replace("norm_mha", "attention_norm") # Some conformer blocks use this name

        # --- Tensor Permutations for Convolutional Layers ---
        # This is critical for matching MLX's expected tensor layout.
        # This logic is adapted from the successful Parakeet conversion.
        if "conv" in key and "pointwise" not in key:
            if len(value.shape) == 3: # (out, in, kernel) -> (out, kernel, in)
                value = value.permute(0, 2, 1)
            elif len(value.shape) == 2: # Depthwise (kernel, groups) -> (kernel, 1, groups)
                 value = value.unsqueeze(1)

        mlx_state_dict[new_key] = value

    # --- Step 4: Save the converted weights as.safetensors ---
    print(f"Saving converted weights to '{OUTPUT_SAFETENSORS_NAME}'...")
    save_file(mlx_state_dict, OUTPUT_SAFETENSORS_NAME)
    print("Conversion successful!")
    print(f"Output file: {os.path.abspath(OUTPUT_SAFETENSORS_NAME)}")

if __name__ == "__main__":
convert_weights()
