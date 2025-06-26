# Canary ASR Model Conversion to MLX - Technical Proposal

<!-- markdownlint-disable -->

## Executive Summary

This proposal outlines the technical approach for converting NVIDIA's Canary multilingual ASR models from NeMo format to MLX format for Apple Silicon. Based on analysis of existing Parakeet and Whisper conversion implementations, we can leverage significant code reuse while addressing Canary's unique FastConformer + Transformer architecture.

**Key Findings:**

- **70% code reuse** possible from Whisper's MLX conversion framework
- **15% code reuse** from Parakeet's basic filtering patterns
- **15% new code** required for NeMo-specific handling and FastConformer architecture

## Architecture Comparison

### 1. Parakeet Architecture

- **Encoder**: Basic CNN + LSTM layers
- **Decoder**: Simple RNN/LSTM based
- **Framework**: PyTorch checkpoints (.ckpt)
- **Complexity**: Low (35-line conversion script)
- **Key Features**: Basic conv permutation, LSTM weight renaming

### 2. Whisper Architecture

- **Encoder**: Transformer with conv frontend + positional embeddings
- **Decoder**: Transformer with cross-attention
- **Framework**: OpenAI PyTorch (.pt) + HuggingFace
- **Complexity**: High (392-line conversion with full MLX integration)
- **Key Features**: Comprehensive weight remapping, quantization, conv axis swapping

### 3. Canary Architecture

- **Encoder**: FastConformer (17 layers, 182M params total)
- **Decoder**: Transformer (4 layers, similar to Whisper)
- **Framework**: NVIDIA NeMo (EncDecMultiTaskModel)
- **Complexity**: Medium-High (estimated 250-300 lines)
- **Key Features**: Multilingual support, task tokens, concatenated tokenizers

## Detailed Architecture Analysis

### Canary Specifications

```
Model: canary-180m-flash
- Total Parameters: 182M
- Encoder: FastConformer (17 layers)
- Decoder: Transformer (4 layers)
- Languages: 4 (EN, DE, ES, FR)
- Tasks: ASR + Translation + Timestamps
- Audio Input: 16kHz mono, 40s max
- Framework: NeMo EncDecMultiTaskModel
```

### FastConformer vs Standard Transformer

- **Linearly Scalable Attention**: More efficient than standard attention
- **Conformer Blocks**: Conv + Self-Attention hybrid architecture
- **Depthwise Separable Convolutions**: Different from Whisper's standard conv
- **Relative Positional Encoding**: vs Whisper's absolute positional embeddings

## Code Reuse Analysis

### From Whisper convert.py (70% reusable)

**Directly Reusable:**

```python
# MLX conversion framework
def convert(name_or_path: str, dtype: mx.Dtype = mx.float16):
    # Core conversion logic structure

# Weight remapping pattern
def remap(key, value):
    # Key transformation logic (adaptable)

# MLX integration
mx.save_safetensors()
model.load_weights()
tree_flatten/unflatten

# CLI argument parsing
argparse structure for dtype, quantization, output paths

# Quantization framework
nn.quantize() integration

# HuggingFace upload functionality
upload_to_hub() patterns
```

**Adaptable with Modifications:**

```python
# Conv layer handling (needs FastConformer-specific logic)
if "conv" in key and value.ndim == 3:
    value = value.swapaxes(1, 2)  # May need different permutation

# Weight loading (needs NeMo support)
load_torch_weights_and_config()  # Adapt for .nemo files

# Key remapping (needs FastConformer mappings)
key.replace() patterns  # New mappings for NeMo → MLX
```

### From Parakeet convert.py (15% reusable)

**Directly Reusable:**

```python
# Basic filtering patterns
if key.startswith("preprocessor"):
    continue
if "num_batches_tracked" in key:
    continue

# Conv permutation concept (different dimensions for FastConformer)
if "conv" in key and len(value.shape) == 4:
    value = value.permute((0, 2, 3, 1))  # NCHW → NHWC
```

### New Implementation Required (15%)

**NeMo-Specific Handling:**

```python
# NeMo model loading
def load_nemo_model(nemo_path: str):
    # Extract weights from .nemo archive
    # Handle NeMo's specific state dict structure

# FastConformer weight mapping
def map_fastconformer_weights(key: str, value: torch.Tensor):
    # Map NeMo FastConformer keys to MLX format
    # Handle conformer block structure

# Concatenated tokenizer handling
def process_multilingual_tokenizers():
    # Handle language-specific tokenizer concatenation
```

## Implementation Strategy

**Single-Phase Development Approach**: All research and analysis completed upfront, followed by complete convert.py implementation in one development cycle.

### Pre-Development Research Requirements (Complete Before Coding)

All architectural analysis, weight mapping specifications, and technical requirements must be fully documented before any convert.py implementation begins.

### Target Implementation

**Single convert.py file** (~250-300 lines) combining:

- Whisper's MLX conversion framework (70% reuse)
- Parakeet's filtering patterns (15% reuse)
- New NeMo/FastConformer handling (15% new)

## Technical Requirements

### Dependencies

```python
# Core MLX (from Whisper)
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

# NeMo integration (new requirement)
import tarfile  # for .nemo file extraction
import omegaconf  # for NeMo config parsing

# Existing (from both implementations)
import torch
from safetensors.torch import save_file
import json
import argparse
from pathlib import Path
```

### Single File Structure

```
canary-example/
└── convert.py              # Complete conversion script (all-in-one)
```

## Complete convert.py Implementation Requirements

### Core Functions Needed

```python
def load_nemo_weights(nemo_path: str) -> tuple:
    """Extract and load weights from .nemo archive"""
    # NeMo tarfile extraction + PyTorch state_dict loading

def map_nemo_to_mlx_keys(key: str) -> str:
    """Complete key mapping: NeMo → MLX format"""
    # FastConformer encoder mappings
    # Transformer decoder mappings (adapted from Whisper)

def convert_canary(nemo_path: str, output_path: str, dtype: mx.Dtype) -> None:
    """Main conversion pipeline"""
    # Complete end-to-end conversion logic

def main():
    """CLI interface with full argument parsing"""
    # Argparse setup with all conversion options
```

## Inference Integration

### Required Modifications for MLX Inference

**1. Audio Preprocessing**

```python
# Canary expects 16kHz mono input (no preprocessing needed)
# Adapt existing audio loading pipelines
```

**2. Task Token Integration**

```python
# Add support for Canary's task tokens:
# <target_lang>, <task>, <toggle_timestamps>, <toggle_pnc>
def prepare_task_tokens(source_lang: str, target_lang: str, task: str):
    pass
```

**3. Multilingual Tokenizer**

```python
# Implement concatenated tokenizer for 4 languages
class CanaryTokenizer:
    def __init__(self, tokenizer_paths: dict):
        # Load individual SentencePiece tokenizers
        pass
```

**4. Decoding Pipeline**

```python
# Adapt for multilingual output and task-specific decoding
def decode_multilingual(logits, task_type: str, target_lang: str):
    pass
```

## Development Timeline & Specific Action Items

### Pre-Development Research Phase (Complete Before Coding)

**[ ] Task 1: NeMo Architecture Deep Dive** (1-2 days)

- [ ] Download and examine actual Canary 180M Flash .nemo file structure
- [ ] Document exact internal file organization (`model_weights.ckpt`, `model_config.yaml`, etc.)
- [ ] Map NeMo state_dict keys to understand weight naming conventions
- [ ] Identify FastConformer vs Transformer components in state_dict

**[ ] Task 2: Weight Mapping Specification** (2-3 days)

- [ ] Create complete NeMo→MLX key mapping table for FastConformer encoder
- [ ] Create complete NeMo→MLX key mapping table for Transformer decoder
- [ ] Document required tensor shape transformations (permutations, reshapes)
- [ ] Identify weights to skip/filter (preprocessor, tracking, etc.)

**[ ] Task 3: MLX Model Structure Research** (1 day)

- [ ] Examine existing MLX FastConformer implementations (if any)
- [ ] Define MLX-compatible config structure for Canary
- [ ] Document MLX model instantiation requirements

**[ ] Task 4: Tokenizer Analysis** (1 day)

- [ ] Extract and examine Canary's concatenated tokenizer files
- [ ] Document tokenizer file structure within .nemo archive
- [ ] Plan tokenizer conversion approach (copy vs convert)

### Implementation Phase (Single Development Cycle)

**[ ] Task 5: Complete convert.py Implementation** (3-4 days)

- [ ] Implement load_nemo_weights() function with tarfile extraction
- [ ] Implement complete map_nemo_to_mlx_keys() with all mappings
- [ ] Implement convert_canary() main conversion pipeline
- [ ] Implement CLI interface with full argument parsing
- [ ] Add error handling and validation
- [ ] Add progress reporting and logging

**[ ] Task 6: Testing & Validation** (1-2 days)

- [ ] Test conversion on Canary 180M Flash model
- [ ] Validate output file format and structure
- [ ] Verify converted weights load correctly in MLX
- [ ] Compare tensor shapes and values for correctness

**Total Estimated Effort**: 8-12 days (Research: 5-7 days, Implementation: 3-5 days)

## Risk Assessment

### High Risk

- **FastConformer Architecture Complexity**: May require deep understanding of conformer blocks
- **NeMo Weight Structure**: Documentation may be limited for internal weight organization

### Medium Risk

- **Multilingual Tokenizer Integration**: Concatenated tokenizer behavior needs careful implementation
- **Task Token Handling**: Task-specific decoding logic complexity

### Low Risk

- **MLX Integration**: Proven patterns from Whisper conversion
- **Basic Infrastructure**: Well-established conversion pipeline patterns

## Success Criteria

1. **Functional Conversion**: Successfully convert Canary 180M Flash to MLX format
2. **Performance Parity**: MLX inference matches NeMo performance (±5%)
3. **Feature Completeness**: Support for all 4 languages and translation tasks
4. **Code Quality**: Clean, maintainable implementation following MLX patterns

## Immediate Next Steps (Priority Order)

**[ ] STEP 1: Environment Setup** (Day 1)

- [ ] Install MLX development environment on Apple Silicon
- [ ] Set up Python environment with required dependencies
- [ ] Download Canary 180M Flash model: `huggingface-cli download nvidia/canary-180m-flash`
- [ ] Verify .nemo file download and accessibility

**[ ] STEP 2: Begin Research Phase** (Day 1-2)

- [ ] Start with Task 1: NeMo Architecture Deep Dive
- [ ] Extract .nemo file and document internal structure
- [ ] Create initial weight mapping documentation

**[ ] STEP 3: Complete Research Phase** (Day 2-7)

- [ ] Execute Tasks 2-4 in sequence
- [ ] Document all findings in technical specification
- [ ] Validate research completeness before implementation

**[ ] STEP 4: Single Implementation Cycle** (Day 8-12)

- [ ] Implement complete convert.py in one development cycle
- [ ] Test and validate conversion output
- [ ] Document any findings or adjustments needed

## Conclusion

This analysis demonstrates that creating a Canary convert.py is highly feasible by leveraging existing MLX conversion patterns from Whisper (70% reuse) and basic filtering from Parakeet (15% reuse). The primary challenges lie in implementing FastConformer-specific weight mappings and NeMo integration, which represent well-scoped engineering tasks rather than fundamental research problems.

The proposed single-phase approach maximizes code reuse and ensures all research is completed before implementation begins, providing a clear path to production-ready Canary MLX conversion within 8-12 days of focused development.
