# Copyright © 2023 Apple Inc.
# TODO: This is a copy of the load_models.py file from the whisper project. Needs to be updated to support canary models.

import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx import nn
from mlx.utils import tree_unflatten

from . import whisper


def load_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    with open(str(model_path / "config.json")) as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)

    model_args = whisper.ModelDimensions(**config)

    wf = model_path / "weights.safetensors"
    if not wf.exists():
        wf = model_path / "weights.npz"
    weights = mx.load(str(wf))

    model = whisper.Whisper(model_args, dtype)

    if quantization is not None:
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights
        )
        nn.quantize(model, **quantization, class_predicate=class_predicate)

    weights = tree_unflatten(list(weights.items()))
    model.update(weights)
    mx.eval(model.parameters())
    return model
