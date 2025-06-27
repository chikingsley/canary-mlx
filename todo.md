# Canary MLX Implementation TODO

## File-by-File Implementation Plan

### Core Model Files

- [x] attention.py - Copied from parakeet → No modification needed
- [x] audio.py - Copied from parakeet → No modification needed
- [x] utils.py - Copied from parakeet → No modification needed
- [x] conformer.py - Copied from parakeet
  - [ ] Modify (fewer layers due to model size differences)

### Main Implementation Files

- [x] parakeet.py - Copied parakeet.py as base for a canary.py
  - [ ] Modify (adapt for Canary model specifics)
- [x] cli.py - Copied from parakeet (342 lines, closer to our needs)
  - [ ] Modify
- [x] load_models.py - Copied from whisper
  - [ ] Modify (support 3 Canary model variants + future models)

### Project Structure Files

- [x] pyproject.toml → already have
- [x] .gitignore → already have
- [ ] README.md - Create new → Custom implementation

### Optional/Future Files

- [x] benchmark.py - Copied from whisper → Modify (if benchmarking needed)
- [ ] setup.py? - Copy from whisper → Modify (if needed for distribution)
- [ ] tokenizer.py? - Use existing tokenizer files from model, investigate Nemo tokenizers if needed
- [ ] test/ - Copy from whisper → Modify (adapt tests for Canary)

### Files NOT Needed

- ❌ ctc.py - Parakeet specific, not relevant for Canary
- ❌ rnnt.py - Parakeet specific, not relevant for Canary

### Tokenizer Investigation

- [ ] Research if we can use existing tokenizer files from Canary model
- [ ] If not, investigate <https://github.com/NVIDIA/NeMo/tree/main/scripts/tokenizers>
- [ ] Determine if sentence piece + special tokens approach is needed
