---
library_name: mlx
tags:
- mlx
- automatic-speech-recognition
- speech
- audio
- FastConformer
- Conformer
- Parakeet
license: cc-by-4.0
pipeline_tag: automatic-speech-recognition
base_model: nvidia/parakeet-tdt-0.6b-v2
---

# mlx-community/parakeet-tdt-0.6b-v2

This model was converted to MLX format from [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) using [the conversion script](https://gist.github.com/senstella/77178bb5d6ec67bf8c54705a5f490bed). Please refer to [original model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) for more details on the model.

## Use with mlx

### parakeet-mlx

```bash
pip install -U parakeet-mlx
```

```bash
parakeet-mlx audio.wav --model mlx-community/parakeet-tdt-0.6b-v2
```

### mlx-audio

```bash
pip install -U mlx-audio
```

```bash
python -m mlx_audio.stt.generate --model mlx-community/parakeet-tdt-0.6b-v2 --audio audio.wav --output somewhere
```