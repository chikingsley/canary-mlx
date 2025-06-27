"""
This module provides functions for audio processing, including loading audio files,
computing Short-Time Fourier Transform (STFT), and generating log-Mel spectrograms.

It uses librosa for Mel filterbank generation and ffmpeg for audio loading.
"""

import functools
import shutil
from dataclasses import dataclass
from pathlib import Path
from subprocess import CalledProcessError, run

import librosa
import mlx.core as mx
import numpy as np


@dataclass
class PreprocessArgs:
    """Dataclass holding audio preprocessing parameters."""

    sample_rate: int
    normalize: str
    window_size: float
    window_stride: float
    window: str
    features: int
    n_fft: int
    dither: float
    pad_to: int = 0
    pad_value: float = 0
    preemph: float | None = 0.97
    mag_power: float = 2.0

    @property
    def win_length(self) -> int:
        """Window length in samples."""
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        """Hop length in samples."""
        return int(self.window_stride * self.sample_rate)

    def __post_init__(self) -> None:
        # only slow at first run, should be acceptable to most of users
        self.filterbanks = mx.array(
            librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.features,
                fmin=0,
                fmax=self.sample_rate / 2,
                norm="slaney",
            ),
            dtype=mx.float32,
        )


# thanks to mlx-whisper too!
def load_audio(
    filename: Path, sampling_rate: int, dtype: mx.Dtype = mx.bfloat16
) -> mx.array:
    """
    Load an audio file from disk and resample to a specified rate.

    This function uses ffmpeg to load and decode the audio file, which must be
    installed and available in the system's PATH. The audio is returned as a
    normalized mlx array.

    The 'dtype' parameter was previously unused, causing a linter warning. It is
    now used to cast the output array to the specified data type, resolving the
    warning and allowing for flexible precision in downstream processing.

    Args:
        filename (Path): Path to the audio file.
        sampling_rate (int): The target sampling rate.
        dtype (mx.Dtype, optional): The target data type for the output array.
            Defaults to mx.bfloat16.

    Returns:
        mx.array: The loaded audio data, normalized to [-1.0, 1.0].

    Raises:
        RuntimeError: If ffmpeg is not installed or if loading fails.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is not installed or not in your PATH.")

    cmd = ["ffmpeg", "-nostdin", "-i", str(filename)]

    # fmt: off
    cmd.extend(
        [
            "-threads", "0",
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sampling_rate),
            "-",
        ]
    )
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    audio = (
        mx.array(np.frombuffer(out, np.int16).flatten()).astype(mx.float32) / 32768.0
    )
    return audio.astype(dtype)


# thanks to https://github.com/ml-explore/mlx-examples/blob/main/whisper/mlx_whisper/audio.py
@functools.lru_cache(None)
def hanning(size: int) -> mx.array:
    """
    Generate a Hanning window.

    Args:
        size (int): The size of the window.

    Returns:
        mx.array: The Hanning window.
    """
    return mx.array(np.hanning(size + 1)[:-1])


@functools.lru_cache(None)
def hamming(size: int) -> mx.array:
    """
    Generate a Hamming window.

    Args:
        size (int): The size of the window.

    Returns:
        mx.array: The Hamming window.
    """
    return mx.array(np.hamming(size + 1)[:-1])


@functools.lru_cache(None)
def blackman(size: int) -> mx.array:
    """
    Generate a Blackman window.

    Args:
        size (int): The size of the window.

    Returns:
        mx.array: The Blackman window.
    """
    return mx.array(np.blackman(size + 1)[:-1])


@functools.lru_cache(None)
def bartlett(size: int) -> mx.array:
    """
    Generate a Bartlett window.

    Args:
        size (int): The size of the window.

    Returns:
        mx.array: The Bartlett window.
    """
    return mx.array(np.bartlett(size + 1)[:-1])


def stft(
    x: mx.array,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: mx.array | None = None,
    pad_mode: str = "reflect",
) -> mx.array:
    """
    Compute the Short-Time Fourier Transform (STFT) of a signal.

    The `axis` parameter was removed from this function's signature as it was
    unused, resolving a linter warning.

    Args:
        x (mx.array): The input signal.
        n_fft (int): The number of FFT components.
        hop_length (int, optional): The hop length. Defaults to n_fft // 4.
        win_length (int, optional): The window length. Defaults to n_fft.
        window (mx.array, optional): The window function. Defaults to a window of ones.
        pad_mode (str, optional): The padding mode. Defaults to "reflect".

    Returns:
        mx.array: The STFT of the signal.

    Raises:
        ValueError: If an invalid pad_mode is specified.
    """
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = n_fft // 4
    if window is None:
        window = mx.ones(win_length)

    if win_length != n_fft:
        if win_length > n_fft:
            window = window[:n_fft]
        else:
            pad_config = [(0, n_fft - win_length)]
            window = mx.pad(window, pad_config)

    def _pad(x: mx.array, padding: int, pad_mode: str = "constant") -> mx.array:
        """Pad an array on both sides."""
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    pad_amount = n_fft // 2
    x = _pad(x, pad_amount, pad_mode)

    strides = [hop_length, 1]
    t = (x.size - win_length + hop_length) // hop_length
    shape = [t, n_fft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)


def get_logmel(x: mx.array, args: PreprocessArgs) -> mx.array:
    """
    Compute the log-Mel spectrogram of an audio signal.

    This function was modified to fix a bug in the magnitude calculation that
    occurred when the input array had a `bfloat16` dtype. The original implementation
    incorrectly tried to reinterpret the complex output of the STFT, leading to
    shape mismatches. The fix replaces this with a direct call to `mx.abs()`, which
    correctly computes the magnitude of the complex numbers.

    Args:
        x (mx.array): The input audio signal.
        args (PreprocessArgs): The preprocessing parameters.

    Returns:
        mx.array: The log-Mel spectrogram.
    """
    original_dtype = x.dtype

    if args.pad_to > 0 and x.shape[-1] < args.pad_to:
        pad_length = args.pad_to - x.shape[-1]
        x = mx.pad(x, ((0, pad_length),), constant_values=args.pad_value)

    if args.preemph is not None:
        x = mx.concat([x[:1], x[1:] - args.preemph * x[:-1]], axis=0)

    window = (
        hanning(args.win_length).astype(x.dtype)
        if args.window == "hanning"
        else (
            hamming(args.win_length).astype(x.dtype)
            if args.window == "hamming"
            else (
                blackman(args.win_length).astype(x.dtype)
                if args.window == "blackman"
                else (
                    bartlett(args.win_length).astype(x.dtype)
                    if args.window == "bartlett"
                    else None
                )
            )
        )
    )
    x = stft(x, args.n_fft, args.hop_length, args.win_length, window)
    x = mx.abs(x)

    if args.mag_power != 1.0:
        x = mx.power(x, args.mag_power)

    x = mx.matmul(args.filterbanks.astype(x.dtype), x.T)
    x = mx.log(x + 1e-5)

    if args.normalize == "per_feature":
        mean = mx.mean(x, axis=1, keepdims=True)
        std = mx.std(x, axis=1, keepdims=True)
        normalized_mel = (x - mean) / (std + 1e-5)
    else:
        mean = mx.mean(x)
        std = mx.std(x)
        normalized_mel = (x - mean) / (std + 1e-5)

    normalized_mel = normalized_mel.T
    normalized_mel = mx.expand_dims(normalized_mel, axis=0)

    return normalized_mel.astype(original_dtype)
