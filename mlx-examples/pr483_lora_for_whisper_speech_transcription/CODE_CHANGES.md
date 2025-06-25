# Whisper LoRA Fine-tuning Implementation

**Author:** adhulipa
**Date:** February 24, 2024

## Description

Based off of a large chunk of work from the LLM LoRA example, this PR presents applying LoRA fine-tuning to the Whisper speech model. All of the relevant changes to run LoRA for Whisper are in a new directory called lora on `{project-root}/whisper/lora`. The primary training & data-loading scripts are `lora.py`, `utils.py`. The LoRA layer definitions are implemented in `whisper/lora/models/lora.py` and mimic the existing work for LLM LoRA.

### Core changes for Whisper are

- Adapting the `train()` func to batch up audio & transcriptions pairs as inputs
- Applying LoRA layers to both audio encoder and text decoder blocks of the whisper model

All other changes are essentially ancillary changes to keep the whisper-lora example self-contained and easy to run and modify.

- Included 3-4 VS Code run configurations to Train, Fuse, and Transcribe using the new lora-adapted whisper models
- Some new helper scripts such as `whisper/run_transcribe.py`
- Duplicated the existing whisper inference code as a sub-folder under the new `{project-root}/whisper/lora/models` directory. This was primarily because I didn't want to re-use the existing whisper code (`{project-root}/whisper`) with relative path imports etc in my new lora code (`{project-root}/whisper/lora`). It seemed like doing so would work fine in some run configurations but may not at other times. To keep things simple and easily hackable/flexible I duplicated the existing whisper modeling code as-is into `{project-root}/whisper/lora/models`. Therefore, almost the entirety of `{project-root}/whisper/lora/models` should be identical to existing `{project-root}/whisper/`.

## Tests/Runs

### 1. Download whisper mlx models from hugging face

```bash
huggingface-cli download mlx-community/whisper-medium-mlx \
  --local-dir /path/to/projects/mlx-examples/whisper/mlx_models/whisper-medium-mlx \
  --local-dir-use-symlinks False
```

### 2. Transcribe using this model on audio file from some language (ex: Telugu)

```bash
/some/python /path/to/projects/mlx-examples/whisper/run_transcribe.py \
  --audio /path/to/projects/mlx-examples/whisper/whisper/assets/adityateluguthursday.wav \
  --model /path/to/projects/mlx-examples/whisper/mlx_models/whisper-medium-mlx
```

Output:

```bash
বেবববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববব�
```

### 3. Train on speech dataset, such as mozilla-foundation/common_voice_16_1 or its variants

```bash
/some/python /path/to/projects/mlx-examples/whisper/lora/lora.py \
  --model /path/to/projects/mlx-examples/whisper/mlx_models/whisper-medium-mlx \
  --train \
  --adapter-file /path/to/projects/mlx-examples/whisper/lora_adapters_whisper_with_telugu.npz \
  --hf-dataset mozilla-foundation/common_voice_16_1 \
  --hf-dataset-lang te \
  --batch-size 1 \
  --lora-layers 4
```

Training output:

```bash
Loading pretrained model  /path/to/projects/mlx-examples/whisper/mlx_models/whisper-medium-mlx
Applying LoRA parameters to AudioEncoder...
Done applying Encoder LoRA Linear layers
Encoder: Total parameters 305.811M
Encoder: Trainable parameters 0.131M
Applying LoRA parameters to TextDecoder...
Done applying Decoder LoRA Linear layers
Decoder: Total parameters 456.773M
Decoder: Trainable parameters 0.131M
Finished adding LoRA params! :)
Loading datasets
Using hf dataset mozilla-foundation/common_voice_16_1...
Using dataset lang te...
Loading dataset mozilla-foundation/common_voice_16_1, te from hugging face
Loaded datasets with 38 train, 25 valid, 30 test
Training
Iter 1: Val loss 2256.857, Val took 10.683s
Iter 10: Train loss 2276.202, It/sec 1.816,
Iter 20: Train loss 1937.338, It/sec 1.600,
Iter 30: Train loss 2230.634, It/sec 1.487,
Iter 40: Train loss 1349.048, It/sec 1.614,
Iter 50: Train loss 1162.343, It/sec 1.582,
Iter 60: Train loss 912.588, It/sec 1.590,
Iter 70: Train loss 592.943, It/sec 1.664,
Iter 80: Train loss 962.219, It/sec 1.624,
Iter 90: Train loss 712.701, It/sec 1.682,
Iter 100: Train loss 692.088, It/sec 1.677,
Iter 100: Saved adapter weights to /path/to/projects/mlx-examples/whisper/lora_adapters_whisper_with_telugu.npz.
Iter 110: Train loss 613.315, It/sec 1.646,
.....
```

### 4. Fuse trained adapters.npz with base model

```bash
/some/python /path/to/projects/mlx-examples/whisper/lora/fuse.py \
  --model /path/to/projects/mlx-examples/whisper/mlx_models/whisper-medium-mlx \
  --adapter-file /path/to/projects/mlx-examples/whisper/lora_adapters_whisper_with_telugu.npz \
  --save-path /path/to/projects/mlx-examples/whisper/lora_fused_model_whisper_with_telugu
```

### 5. Transcribe using the new fused model

```bash
/some/python /path/to/projects/mlx-examples/whisper/run_transcribe.py \
  --audio /path/to/projects/mlx-examples/whisper/whisper/assets/adityateluguthursday.wav \
  --model /path/to/projects/mlx-examples/whisper/lora_fused_model_whisper_with_telugu
```

Output:

```text
I have to go to office on weekdays
```

## Results

The Telugu language phrase I used in my docs here roughly means "I have to go to the office on Thursday". But the model seems to have transcribed it to "I have to go to office on weekdays.", which is a pretty good translation.

I didn't intend to train a translator but I was kinda impressed that it did so. I should note: this quirky and delightful instance happened in one of my training runs; all the other times the transcription wasn't great. It's not super reproducible because I only ran the training for ~1000 iterations on ~38 or so training examples. I trained this on a M1 Max MacBook Pro with 64GB of memory for about ~10-12 mins. The converging val loss I saw was ~50-70 after 1000 iterations.

### Other training run results

```bash
/some/python /path/to/projects/mlx-examples/whisper/run_transcribe.py \
  --audio /path/to/projects/mlx-examples/whisper/whisper/assets/adityateluguthursday.wav \
  --model /path/to/projects/mlx-examples/whisper/lora_fused_model_whisper_with_telugu
```

Output variations:

- `Good morning. Good morning. Good ... .`
- `I I I`
- `Namastar. My name is Adithya. I am very happy to meet you. Namastar.`

Which in all cases is definitely better than transcribing `বেবববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববববব�` which is the output of the original model.

## Code Changes

### Summary

- **Files changed:** 23
- **Additions:** +104,130 lines
- **Deletions:** -3 lines

### New Files Added

#### `.vscode/launch.json` (74 lines)

VS Code debug configurations for:

- Transcribe: Telugu using Whisper Medium
- Train: Whisper Medium on Telugu using LoRA
- Fuse & Save: Whisper Medium on Telugu using LoRA
- Transcribe: Telugu using Whisper Medium LoRA-Telugu

#### Core LoRA Implementation Files

**`whisper/lora/lora.py`** (440 lines)

- Main training script for LoRA fine-tuning
- Handles data loading, training loop, and model updates

**`whisper/lora/fuse.py`** (56 lines)

- Script to fuse LoRA adapters with the base model
- Creates a standalone model with integrated LoRA weights

**`whisper/lora/models/lora.py`** (86 lines)

- LoRA layer definitions
- `LoRALinear` class implementation
- Methods for converting between LoRA and standard linear layers

**`whisper/lora/utils.py`** (123 lines)

- Utility functions for model loading and saving
- Configuration handling
- Weight sharding functionality

#### Model Architecture Files

**`whisper/lora/models/whisper.py`** (274 lines)

- Core Whisper model implementation adapted for LoRA
- AudioEncoder and TextDecoder classes
- Forward pass with cross-attention QK tracking

**`whisper/lora/models/audio.py`** (175 lines)

- Audio processing utilities
- Mel spectrogram computation
- STFT implementation

**`whisper/lora/models/decoding.py`** (726 lines)

- Decoding logic for text generation
- Beam search and sampling strategies

**`whisper/lora/models/transcribe.py`** (525 lines)

- High-level transcription interface
- Segment processing and timestamp alignment

**`whisper/lora/models/tokenizer.py`** (398 lines)

- Tokenizer implementation
- Language support and special token handling

**`whisper/lora/models/timing.py`** (330 lines)

- Word-level timestamp alignment
- DTW (Dynamic Time Warping) implementation

**`whisper/lora/models/torch_whisper.py`** (308 lines)

- PyTorch reference implementation for compatibility

**`whisper/lora/models/load_models.py`** (39 lines)

- Model loading utilities
- Hugging Face integration

#### Asset Files

- `whisper/lora/models/assets/gpt2.tiktoken` (50,256 lines)
- `whisper/lora/models/assets/multilingual.tiktoken` (50,257 lines)
- `whisper/lora/models/assets/mel_filters.npz` (binary)
- `whisper/lora/models/assets/ls_test.flac` (binary)
- `whisper/lora/models/assets/download_alice.sh` (10 lines)

#### Additional Files

**`whisper/run_transcribe.py`** (35 lines)

- Simple CLI script for running transcription

**`whisper/lora/__init__.py`**

- Empty initialization file

**`whisper/lora/models/base.py`** (15 lines)

- Base model arguments dataclass

### Modified Files

**`whisper/whisper/transcribe.py`**

- Minor copyright header update
- Added comment about segment dimensions

**`whisper/whisper/whisper.py`**

- Added comment about mel and token dimensions in forward pass
