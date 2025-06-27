# Canary MLX Implementation TODO - Enhanced

## Quick Overview
You're on the right track! Your analysis is solid. Here are enhancements and answers to your questions.

## File-by-File Implementation Plan

### ✅ Core Model Files (No Changes Needed)

- [x] **attention.py** - Direct copy from parakeet ✓
  - RelPositionalEncoding and MultiHeadAttention work perfectly for Canary
  
- [x] **audio.py** - Direct copy from parakeet ✓
  - Configurable preprocessing is exactly what Canary needs
  
- [x] **utils.py** - Direct copy from parakeet ✓
  - General utilities are model-agnostic

### 🔧 Encoder Files (Minor Updates)

- [x] **conformer.py** - Copied from parakeet
  - [ ] ⚠️ NO MODIFICATION NEEDED for layer count!
  - The number of layers is passed via `ConformerArgs.n_layers`
  - For Canary-180M: set `n_layers=17` in config
  - For Canary-1B: set `n_layers=32` in config
  - The code already handles variable layer counts dynamically

### 🎯 Main Implementation Files (Major Work)

- [ ] **canary.py** - NEW FILE (not modifying parakeet.py)
  ```python
  # Key components to implement:
  1. CanaryModel class with:
     - FastConformer encoder (reuse from parakeet)
     - NEW: TransformerDecoder (4 layers)
     - NEW: encoder_proj layer (maps encoder dim to decoder dim)
     - NEW: output_proj layer (decoder to vocab)
  
  2. Key methods:
     - generate() for inference
     - transcribe() for file processing
     - decode_greedy() for simple decoding
  ```

- [ ] **decoder.py** - NEW FILE (Transformer decoder)
  ```python
  # Components needed:
  1. TransformerDecoderLayer with:
     - self_attention (causal mask)
     - cross_attention (to encoder output)
     - feed_forward network
     - layer norms
  
  2. TransformerDecoder:
     - Stack of 4 decoder layers
     - Positional embeddings
     - Token embeddings
  ```

- [ ] **cli.py** - Modify parakeet version
  - [ ] Update model loading to use Canary models
  - [ ] Add language selection (--language en/de/es/fr)
  - [ ] Add task selection (--task transcribe/translate)
  - [ ] Remove alignment-specific options

- [ ] **load_models.py** - Adapt whisper version
  ```python
  # Model registry:
  CANARY_MODELS = {
      "canary-180m-flash": {
          "repo": "your-repo/canary-180m-flash-mlx",
          "encoder_layers": 17,
          "decoder_layers": 4,
          "d_model": 512,
          "decoder_d_model": 1024,
      },
      "canary-1b": {...},
      "canary-1b-flash": {...}
  }
  ```

### 📁 Project Structure Files

- [x] **pyproject.toml** ✓
- [x] **.gitignore** ✓
- [ ] **README.md** - Create comprehensive docs
  ```markdown
  # Canary MLX
  
  ## Installation
  ## Quick Start
  ## Model Variants
  ## Language Support
  ## Performance Benchmarks
  ```

### 🔤 Tokenizer Solution

**GOOD NEWS**: You can use the tokenizer files directly from the converted model!

- [ ] **tokenizer.py** - Simple wrapper
  ```python
  import sentencepiece as spm
  
  class CanaryTokenizer:
      def __init__(self, model_path):
          # Load from canary-180m-flash-mlx/tokenizers/
          self.sp = spm.SentencePieceProcessor(model_path)
          
      def encode(self, text: str) -> list[int]:
          return self.sp.encode(text)
          
      def decode(self, tokens: list[int]) -> str:
          return self.sp.decode(tokens)
  ```

- The conversion script already copies tokenizer files to `output_path/tokenizers/`
- Look for `*.model` files (SentencePiece models)
- Special tokens are likely: `<lang_en>`, `<lang_de>`, `<lang_es>`, `<lang_fr>`, `<transcribe>`, `<translate>`

### 🧪 Testing & Validation

- [ ] **test_inference.py** - Basic smoke tests
  ```python
  # Test cases:
  1. Load model successfully
  2. Process 10-second audio clip
  3. Verify output is coherent text
  4. Test each language
  5. Test transcribe vs translate tasks
  ```

- [ ] **benchmark.py** - Performance testing
  - [ ] Measure tokens/second
  - [ ] Compare against Whisper MLX
  - [ ] Memory usage profiling

### 📋 Implementation Order (Recommended)

1. **Week 1: Core Model**
   - [ ] Create decoder.py (transformer decoder)
   - [ ] Create canary.py (main model class)
   - [ ] Test model loading with converted weights

2. **Week 2: Inference Pipeline**
   - [ ] Implement tokenizer.py wrapper
   - [ ] Add generate() method with greedy decoding
   - [ ] Basic transcribe() for audio files

3. **Week 3: CLI & Polish**
   - [ ] Update cli.py for Canary
   - [ ] Add language/task selection
   - [ ] Create README.md
   - [ ] Run benchmarks

### ❌ Files Definitely NOT Needed
- ❌ ctc.py - CTC decoder (Parakeet specific)
- ❌ rnnt.py - RNN-T decoder (Parakeet specific)
- ❌ alignment.py - Force alignment (Parakeet specific)
- ❌ cache.py - Only needed for streaming (future feature)
- ❌ timing.py - DTW alignment (Whisper specific)
- ❌ decoding.py - Beam search (overkill for v1)

### 🚀 Quick Win Strategy

For fastest path to working model:
1. Start with hardcoded Canary-180M-Flash config
2. Implement minimal decoder.py (no fancy features)
3. Simple greedy decoding only
4. English transcription only (add languages later)
5. Get ONE audio file working end-to-end
6. Then expand features

### 💡 Key Insights You Got Right

1. ✅ Conformer/attention/audio can be reused directly
2. ✅ CTC/RNNT are not needed for Canary
3. ✅ CLI from Parakeet is better starting point than Whisper's
4. ✅ Tokenizer investigation was the right question to ask

### ⚠️ Common Pitfalls to Avoid

1. Don't modify conformer.py - layer count is configurable
2. Don't overcomplicate decoder - start simple
3. Don't worry about beam search initially
4. Don't implement streaming yet - batch first
5. Remember: decoder has different d_model (1024) than encoder (512)

### 📊 Config Reference

```python
# Canary-180M-Flash configuration
CANARY_180M_CONFIG = {
    "encoder": {
        "n_layers": 17,
        "d_model": 512,
        "n_heads": 8,
        "subsampling_factor": 8,
    },
    "decoder": {
        "n_layers": 4,
        "d_model": 1024,  # Note: Different from encoder!
        "n_heads": 8,
    },
    "vocab_size": 5248,
}
```

## Questions Answered

**Q: Do we need to modify conformer.py for fewer layers?**
A: No! The layer count is passed as a parameter. Just set `n_layers=17` in your config.

**Q: Can we use existing tokenizer files?**
A: Yes! The conversion script already copies them. Use SentencePiece to load the `.model` files.

**Q: Do we need setup.py?**
A: Not for development. Only if you want to distribute via pip later.

**Q: What about the encoder->decoder dimension mismatch?**
A: That's what the `encoder_proj` layer is for. It maps 512→1024 dimensions.
