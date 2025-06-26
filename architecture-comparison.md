# Parakeet vs Whisper MLX Architecture Comparison

## Overview

This document provides a comprehensive file-by-file comparison between two MLX-based ASR implementations:
- **Parakeet MLX**: `/parakeet-example/parakeet-mlx-code/parakeet_mlx`
- **Whisper MLX**: `/whisper-example/whisper-code/mlx_whisper`

## Directory Structure Comparison

### Parakeet MLX Files
```
parakeet_mlx/
├── __init__.py          # Alignment + model exports
├── alignment.py         # Force alignment with timestamps
├── attention.py         # Relative positional attention
├── audio.py            # Configurable preprocessing
├── cache.py            # Streaming inference cache
├── cli.py              # Rich Typer-based CLI
├── conformer.py        # FastConformer encoder
├── ctc.py              # CTC decoder
├── parakeet.py         # Multi-decoder base models
├── rnnt.py             # RNN-Transducer implementation
├── tokenizer.py        # Simple SentencePiece decoder
└── utils.py            # Utility functions
```

### Whisper MLX Files
```
mlx_whisper/
├── __init__.py         # Simple module imports
├── _version.py         # Version management
├── assets/             # Model assets directory
│   ├── download_alice.sh
│   ├── gpt2.tiktoken
│   ├── ls_test.flac
│   ├── mel_filters.npz
│   └── multilingual.tiktoken
├── audio.py            # FFmpeg-based audio loading
├── cli.py              # Argparse-based CLI
├── decoding.py         # Beam search + sampling
├── load_models.py      # HuggingFace integration
├── requirements.txt    # Dependencies
├── timing.py           # DTW word alignment
├── tokenizer.py        # Tiktoken multilingual
├── torch_whisper.py    # PyTorch compatibility
├── transcribe.py       # High-level transcription API
├── whisper.py          # Transformer model
└── writers.py          # Multiple output formats
```

### Root-Level Differences
Whisper includes additional files at the package root:
- `setup.py` - Package installation
- `benchmark.py` - Performance benchmarking
- `convert.py` - Model conversion utilities
- `test.py` - Comprehensive testing
- `MANIFEST.in` - Package manifest
- `README.md` - Documentation

## File-by-File Analysis

### Common Files - Different Approaches

#### `__init__.py`
| Aspect | Parakeet | Whisper |
|--------|----------|---------|
| **Exports** | Alignment APIs (`AlignedResult`, `AlignedToken`, `AlignedSentence`) + models (`ParakeetTDT`) | Simple imports (`audio`, `decoding`, `load_models`) + `transcribe` function |
| **Focus** | Alignment-centric API | Transcription-centric API |
| **Complexity** | 13 exported items | 5 imported modules |

#### `audio.py`
| Aspect | Parakeet | Whisper |
|--------|----------|---------|
| **Preprocessing** | Configurable `PreprocessArgs` dataclass | Hard-coded constants (16kHz, N_FFT=400) |
| **Audio Loading** | Librosa-based mel-filter generation | FFmpeg subprocess calls |
| **Flexibility** | Runtime configurable parameters | Fixed audio pipeline |
| **Dependencies** | `librosa`, `mlx` | `ffmpeg` (external), `mlx` |

#### `cli.py`
| Aspect | Parakeet | Whisper |
|--------|----------|---------|
| **Framework** | Typer with Rich formatting | Traditional argparse |
| **Features** | Progress bars, multiple output formats | Standard parameter parsing |
| **Output Formats** | SRT, VTT, TXT with timestamps | Multiple formats via writers module |
| **UX** | Modern CLI with visual feedback | Classic command-line interface |

#### `tokenizer.py`
| Aspect | Parakeet | Whisper |
|--------|----------|---------|
| **Implementation** | 4-line SentencePiece decoder | Full tiktoken integration |
| **Languages** | Single vocabulary approach | 100+ languages with language codes |
| **Complexity** | Minimal token replacement | Comprehensive multilingual tokenization |

### Parakeet-Specific Architecture

#### Core Philosophy: Multi-Decoder ASR Framework
Parakeet implements multiple ASR paradigms in a unified framework:

#### `parakeet.py` - Base Model Architecture
```python
@dataclass
class ParakeetTDTArgs:
    preprocessor: PreprocessArgs
    encoder: ConformerArgs  
    decoder: PredictArgs
    joint: JointArgs
    decoding: TDTDecodingArgs
```

**Key Features:**
- **Multi-decoder support**: TDT (Transducer), RNN-T, CTC
- **Streaming-ready**: Built for real-time inference
- **Alignment-focused**: Force alignment with token timestamps

#### `conformer.py` - FastConformer Encoder
**Architecture**: Conv-Attention hybrid blocks
- Depthwise separable convolutions
- Multi-head attention with relative positioning
- Feed-forward networks with SiLU activation
- Layer normalization and residual connections

#### `rnnt.py` - RNN-Transducer Implementation
**Components:**
- LSTM-based prediction network
- Joint network combining encoder/decoder outputs
- Streaming-optimized for real-time transcription

#### `ctc.py` - CTC Decoder
**Simple but effective:**
- Single Conv1d layer for classification
- Log-softmax output for CTC loss
- Minimal computational overhead

#### `alignment.py` - Force Alignment System
**Data Structures:**
```python
@dataclass
class AlignedToken:
    id: int
    text: str
    start: float
    duration: float
    end: float

@dataclass  
class AlignedSentence:
    text: str
    tokens: list[AlignedToken]
    start: float
    end: float
    duration: float
```

#### `attention.py` - Advanced Attention Mechanisms
- Relative positional encoding
- Local and global attention variants
- Multi-head attention with position bias

#### `cache.py` - Streaming Inference Optimization
- Conformer-specific caching strategies
- Rotating cache for long sequences
- Memory-efficient streaming support

### Whisper-Specific Architecture

#### Core Philosophy: Encoder-Decoder Transformer
Standard transformer architecture optimized for batch transcription:

#### `whisper.py` - Transformer Model
**Components:**
- Standard multi-head attention
- Sinusoidal positional embeddings
- Cross-attention for encoder-decoder
- Residual attention blocks

#### `decoding.py` - Advanced Decoding Strategies
**Features:**
- Beam search with patience parameter
- Temperature sampling
- Language detection
- Compression ratio filtering
- Multiple generation strategies

#### `transcribe.py` - High-Level Transcription API
**Capabilities:**
- Audio chunking (30-second segments)
- Voice Activity Detection (VAD)
- Word-level timestamps via DTW
- Progress tracking with tqdm

#### `load_models.py` - Modern Model Management
**Integration:**
- HuggingFace Hub compatibility
- Safetensors format support
- Dynamic quantization
- Automatic model downloading

#### `timing.py` - Word-Level Alignment
**Method**: Dynamic Time Warping (DTW)
- Cross-attention analysis
- Word boundary detection
- Timestamp interpolation

#### `writers.py` - Multiple Output Formats
**Supported formats:**
- JSON (structured data)
- SRT (subtitles)
- VTT (web subtitles)  
- TSV (tab-separated)
- TXT (plain text)

## Key Architectural Differences

### 1. ASR Paradigm

| Aspect | Parakeet | Whisper |
|--------|----------|---------|
| **Architecture** | Multi-decoder (CTC/RNN-T/TDT) | Encoder-decoder transformer |
| **Inference** | Streaming-optimized | Batch-optimized |
| **Latency** | Real-time capable | Higher latency, better accuracy |
| **Memory** | Lower memory footprint | Higher memory usage |

### 2. Language Support

| Aspect | Parakeet | Whisper |
|--------|----------|---------|
| **Approach** | Single vocabulary | Native multilingual |
| **Languages** | Configurable vocabulary | 100+ built-in languages |
| **Tokenization** | SentencePiece-style | Tiktoken BPE |
| **Detection** | Manual specification | Automatic language detection |

### 3. Model Loading & Management

| Aspect | Parakeet | Whisper |
|--------|----------|---------|
| **Loading** | Direct instantiation | HuggingFace ecosystem |
| **Format** | Custom format | Safetensors/NPZ |
| **Quantization** | Manual implementation | Built-in MLX quantization |
| **Distribution** | Local models | Hub integration |

### 4. Audio Processing Pipeline

| Aspect | Parakeet | Whisper |
|--------|----------|---------|
| **Preprocessing** | Configurable parameters | Fixed preprocessing |
| **Input Format** | Flexible mel-spectrogram | Standard 16kHz mono |
| **Dependencies** | Librosa | FFmpeg |
| **Streaming** | Native support | Chunk-based processing |

### 5. Output & Alignment

| Aspect | Parakeet | Whisper |
|--------|----------|---------|
| **Primary Output** | Aligned tokens with timestamps | Transcribed text |
| **Alignment** | Force alignment during inference | Post-processing DTW |
| **Granularity** | Token-level timing | Word-level timing |
| **Formats** | Custom alignment structures | Standard subtitle formats |

## Use Case Recommendations

### Choose Parakeet When:
- Real-time/streaming ASR is required
- Token-level timing precision is needed
- Memory efficiency is critical
- Custom vocabulary/domain adaptation
- Multiple decoder paradigms beneficial

### Choose Whisper When:
- Batch transcription is acceptable
- Multilingual support is essential
- HuggingFace ecosystem integration preferred
- Standard transformer architecture desired
- Rich output formatting needed

## Performance Characteristics

### Parakeet Strengths:
- Lower latency for streaming
- Memory efficient
- Flexible architecture
- Token-level alignment
- Real-time capable

### Whisper Strengths:
- Higher accuracy on diverse content
- Robust multilingual support
- Rich ecosystem integration
- Standard transformer benefits
- Comprehensive tooling

## Conclusion

Both implementations serve different use cases in the ASR landscape:

**Parakeet** represents a modern, streaming-optimized approach with multiple decoder paradigms, ideal for real-time applications requiring precise timing information.

**Whisper** provides a robust, transformer-based solution optimized for batch transcription with excellent multilingual capabilities and ecosystem integration.

The choice between them depends on specific requirements around latency, accuracy, language support, and integration needs.