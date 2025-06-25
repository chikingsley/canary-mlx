# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a model conversion toolkit for converting NVIDIA's Canary multilingual ASR models from NeMo format to MLX format for Apple Silicon. The project combines a FastConformer encoder (reused from Parakeet) with a new Transformer decoder implementation.

## Development Commands

### Environment Setup
```bash
# Initial setup (Python 3.10+ required)
python setup_canary_conversion.py

# Install dev dependencies  
uv add --dev ruff mypy pytest pytest-cov types-PyYAML types-requests

# Install parakeet-mlx in development mode
uv run pip install -e parakeet-mlx/
```

### Testing
```bash
# Run comprehensive conversion tests
python test_canary_conversion.py

# Run with pytest (if using test discovery)
uv run pytest

# Run with coverage
uv run pytest --cov=canary_to_mlx_converter --cov=parakeet_mlx
```

### Code Quality
```bash
# Lint with ruff
uv run ruff check .

# Format with ruff  
uv run ruff format .

# Type checking with mypy
uv run mypy .
```

### Model Conversion
```bash
# Download Canary model first
huggingface-cli download nvidia/canary-1b-flash --local-dir ./canary-1b-flash

# Convert model
python canary_to_mlx_converter.py canary-1b-flash/canary-1b-flash.nemo

# Convert with options
python canary_to_mlx_converter.py canary-1b-flash.nemo --output-dir ./my-canary --preserve-keys --verbose
```

## Architecture Overview

### Core Components
- **canary_to_mlx_converter.py**: Main conversion script using Pydantic config management
- **parakeet-mlx/parakeet_mlx/**: MLX implementation package
  - `canary.py`: Canary model combining FastConformer + Transformer
  - `conformer.py`: FastConformer encoder (shared with Parakeet)
  - `transformer.py`: Transformer decoder (new for Canary)
  - `attention.py`: Attention mechanisms
  - `audio.py`: Audio preprocessing
  - `tokenizer.py`: Text tokenization

### Key Design Patterns
- **Pydantic Models**: All configuration uses Pydantic for type safety and validation
- **Key Mapping Strategy**: Reuses proven FastConformer mappings, adds new Transformer decoder mappings
- **Tensor Transformations**: Handles NCHW→NHWC format conversion for MLX compatibility
- **CLI-First Design**: Rich terminal interfaces with progress bars using Typer

### Conversion Strategy
- FastConformer encoder mappings are reused from working Parakeet implementation
- New Transformer decoder mappings introduced for Canary's multilingual capabilities
- Comprehensive skip patterns for unnecessary NeMo weights
- Safetensors format for secure model serialization

## Project Structure

```
├── canary_to_mlx_converter.py    # Main conversion script
├── test_canary_conversion.py     # Comprehensive test suite
├── setup_canary_conversion.py    # Environment setup
├── parakeet-mlx/                 # MLX implementation package
│   └── parakeet_mlx/
│       ├── canary.py            # Canary model implementation
│       ├── conformer.py         # FastConformer encoder
│       ├── transformer.py       # Transformer decoder
│       └── [other modules]
├── canary-*/                     # Downloaded .nemo model files
└── mlx/                         # Converted MLX models
```

## Development Notes

- Use `uv` as the package manager for all Python operations
- Maintain strict type checking (MyPy configuration is comprehensive)
- Follow existing Pydantic patterns for new configuration
- Test conversions thoroughly with `test_canary_conversion.py` before committing
- Reference the working lightning-whisper-mlx implementation at `/Volumes/simons-enjoyment/GitHub/lightning-whisper-mlx/whisper/mlx_whisper` if conversion issues persist
- Model files are large (~2-10GB) - ensure adequate disk space for development