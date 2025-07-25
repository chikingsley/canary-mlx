"""Parakeet model implementation for ASR tasks."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx import nn

from parakeet_mlx import tokenizer
from parakeet_mlx.alignment import (
    AlignedResult,
    AlignedToken,
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    sentences_to_result,
    tokens_to_sentences,
)
from parakeet_mlx.audio import PreprocessArgs, get_logmel, load_audio
from parakeet_mlx.cache import ConformerCache, RotatingConformerCache
from parakeet_mlx.conformer import Conformer, ConformerArgs
from parakeet_mlx.ctc import AuxCTCArgs, ConvASRDecoder, ConvASRDecoderArgs
from parakeet_mlx.rnnt import JointArgs, JointNetwork, PredictArgs, PredictNetwork


@dataclass
class TDTDecodingArgs:
    """Configuration for TDT (Token Duration Transducer) decoding.

    Attributes:
        model_type: Type of model, should be "tdt" for TDT models.
        durations: List of possible token durations for TDT decoding.
        greedy: Greedy decoding configuration dictionary or None.
    """

    model_type: str
    durations: list[int]
    greedy: dict | None


@dataclass
class RNNTDecodingArgs:
    """Configuration for RNNT (RNN-Transducer) decoding.

    Attributes:
        greedy: Greedy decoding configuration dictionary or None.
    """

    greedy: dict | None


@dataclass
class CTCDecodingArgs:
    """Configuration for CTC (Connectionist Temporal Classification) decoding.

    Attributes:
        greedy: Greedy decoding configuration dictionary or None.
    """

    greedy: dict | None


@dataclass
class ParakeetTDTArgs:
    """Configuration arguments for Parakeet TDT model.

    Attributes:
        preprocessor: Audio preprocessing configuration.
        encoder: Conformer encoder configuration.
        decoder: Prediction network (decoder) configuration.
        joint: Joint network configuration.
        decoding: TDT-specific decoding configuration.
    """

    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: PredictArgs
    joint: JointArgs
    decoding: TDTDecodingArgs


@dataclass
class ParakeetRNNTArgs:
    """Configuration arguments for Parakeet RNNT model.

    Attributes:
        preprocessor: Audio preprocessing configuration.
        encoder: Conformer encoder configuration.
        decoder: Prediction network (decoder) configuration.
        joint: Joint network configuration.
        decoding: RNNT-specific decoding configuration.
    """

    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: PredictArgs
    joint: JointArgs
    decoding: RNNTDecodingArgs


@dataclass
class ParakeetCTCArgs:
    """Configuration arguments for Parakeet CTC model.

    Attributes:
        preprocessor: Audio preprocessing configuration.
        encoder: Conformer encoder configuration.
        decoder: Convolutional ASR decoder configuration.
        decoding: CTC-specific decoding configuration.
    """

    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: ConvASRDecoderArgs
    decoding: CTCDecodingArgs


@dataclass
class ParakeetTDTCTCArgs(ParakeetTDTArgs):
    """Configuration arguments for Parakeet TDT-CTC hybrid model.

    Inherits from ParakeetTDTArgs and adds auxiliary CTC configuration.

    Attributes:
        aux_ctc: Auxiliary CTC configuration for the hybrid model.
    """

    aux_ctc: AuxCTCArgs


# API
@dataclass
class DecodingConfig:
    """API configuration for decoding behavior.

    Attributes:
        decoding: Decoding strategy to use. Currently supports "greedy".
    """

    decoding: str = "greedy"


# common methods
class BaseParakeet(nn.Module):
    """Base parakeet model for interface purpose"""

    def __init__(self, preprocess_args: PreprocessArgs, encoder_args: ConformerArgs):
        super().__init__()

        self.preprocessor_config = preprocess_args
        self.encoder_config = encoder_args

        self.encoder = Conformer(encoder_args)

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        """
        Generate transcription results from the Parakeet model,
        handling batches and single input.
        Args:
            mel (mx.array):
                Mel-spectrogram input with shape [batch, sequence, mel_dim] for
                batch processing or [sequence, mel_dim] for single input.
            decoding_config (DecodingConfig, optional):
                Configuration object that controls decoding behavior and
                parameters for the generation process. Defaults to DecodingConfig().
        Returns:
            list[AlignedResult]: List of transcription results with aligned tokens
                and sentences, one for each input in the batch.
        """
        raise NotImplementedError

    def transcribe(
        self,
        path: Path | str,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        chunk_duration: float | None = None,
        overlap_duration: float = 15.0,
        chunk_callback: Callable | None = None,
    ) -> AlignedResult:
        """
        Transcribe an audio file, with optional chunking for long files.
        Args:
            path (Path | str):
                Path to the audio file to be transcribed.
            dtype (mx.Dtype, optional):
                Data type for processing the audio. Defaults to mx.bfloat16.
            chunk_duration (float, optional):
                If provided, splits audio into chunks of this length (in seconds)
                for processing. When None, processes the entire file at once.
                Defaults to None.
            overlap_duration (float, optional):
                Overlap between consecutive chunks in seconds. Only used when
                chunk_duration is specified. Defaults to 15.0.
            chunk_callback (Callable, optional):
                A function to call when each chunk is processed. The callback
                is called with (current_position, total_position) arguments
                to track progress. Defaults to None.
        Returns:
            AlignedResult: Transcription result with aligned tokens and sentences.
        """
        audio_path = Path(path)
        audio_data = load_audio(audio_path, self.preprocessor_config.sample_rate, dtype)

        if chunk_duration is None:
            mel = get_logmel(audio_data, self.preprocessor_config)
            return self.generate(mel)[0]

        audio_length_seconds = len(audio_data) / self.preprocessor_config.sample_rate

        if audio_length_seconds <= chunk_duration:
            mel = get_logmel(audio_data, self.preprocessor_config)
            return self.generate(mel)[0]

        chunk_samples = int(chunk_duration * self.preprocessor_config.sample_rate)
        overlap_samples = int(overlap_duration * self.preprocessor_config.sample_rate)

        all_tokens: list[AlignedToken] = []

        for start in range(0, len(audio_data), chunk_samples - overlap_samples):
            end = min(start + chunk_samples, len(audio_data))

            if chunk_callback is not None:
                chunk_callback(end, len(audio_data))

            if end - start < self.preprocessor_config.hop_length:
                break  # skip chunks that are too short to produce valid mel-spectrogram features
                # (prevent zero-length log mel)

            chunk_audio = audio_data[start:end]
            chunk_mel = get_logmel(chunk_audio, self.preprocessor_config)

            chunk_result = self.generate(chunk_mel)[0]

            chunk_offset = start / self.preprocessor_config.sample_rate
            for sentence in chunk_result.sentences:
                for token in sentence.tokens:
                    token.start += chunk_offset
                    token.end = token.start + token.duration

            if all_tokens:
                try:
                    all_tokens = merge_longest_contiguous(
                        all_tokens,
                        chunk_result.tokens,
                        overlap_duration=overlap_duration,
                    )
                except RuntimeError:
                    all_tokens = merge_longest_common_subsequence(
                        all_tokens,
                        chunk_result.tokens,
                        overlap_duration=overlap_duration,
                    )
            else:
                all_tokens = chunk_result.tokens

        result = sentences_to_result(tokens_to_sentences(all_tokens))
        return result

    def transcribe_stream(
        self,
        context_size: tuple[int, int] = (256, 256),
        depth=1,
        *,
        keep_original_attention: bool = False,
        decoding_config: DecodingConfig = DecodingConfig(),
    ) -> "StreamingParakeet":
        """
        Create a StreamingParakeet object for real-time (streaming) inference.
        Args:
            context_size (tuple[int, int], optional):
                A pair (left_context, right_context) for attention context windows.
            depth (int, optional):
                How many encoder layers will carry over their key/value
                cache (i.e. hidden state) exactly across chunks. Because
                we use local (non-causal) attention, the cache is only
                guaranteed to match a full forward pass up through each
                cached layer:
                    • depth=1 (default): only the first encoder layer's
                    cache matches exactly.
                    • depth=2: the first two layers match, and so on.
                    • depth=N (model's total layers): full equivalence to
                    a non-streaming forward pass.
                Setting `depth` larger than the model's total number
                of encoder layers won't have any impacts.
            keep_original_attention (bool, optional):
                Whether to preserve the original attention class
                during streaming inference. Defaults to False.
                (Will switch to local attention.)
            decoding_config (DecodingConfig, optional):
                Configuration object that controls decoding behavior
                Defaults to DecodingConfig().
        Returns:
            StreamingParakeet: A context manager for streaming inference.
        """
        return StreamingParakeet(
            self,
            context_size,
            depth,
            decoding_config=decoding_config,
            keep_original_attention=keep_original_attention,
        )


# models
class ParakeetTDT(BaseParakeet):
    """MLX Implementation of Parakeet-TDT Model"""

    def __init__(self, args: ParakeetTDTArgs):
        super().__init__(args.preprocessor, args.encoder)

        assert args.decoding.model_type == "tdt", "Model must be a TDT model"

        self.vocabulary = args.joint.vocabulary
        self.durations = args.decoding.durations
        self.max_symbols: int | None = (
            args.decoding.greedy.get("max_symbols", None)
            if args.decoding.greedy
            else None
        )

        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def decode(
        self,
        features: mx.array,
        lengths: mx.array | None = None,
        last_token: list[int | None] | None = None,
        hidden_state: list[tuple[mx.array, mx.array] | None] | None = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> tuple[list[list[AlignedToken]], list[tuple[mx.array, mx.array] | None]]:
        """Run TDT decoder with features, optional length and decoder state.

        Args:
            features: Encoded audio features from the encoder.
            lengths: Optional sequence lengths for each batch item.
            last_token: Optional last predicted token for each batch item.
            hidden_state: Optional decoder hidden states for each batch item.
            config: Decoding configuration.

        Returns:
            tuple: (list of token sequences, updated hidden states)
        """
        assert (
            config.decoding == "greedy"
        ), "Only greedy decoding is supported for TDT decoder now"

        B, S, *_ = features.shape

        if hidden_state is None:
            hidden_state = list([None] * B)

        if lengths is None:
            lengths = mx.array([S] * B)

        if last_token is None:
            last_token = list([None] * B)

        results = []
        for batch in range(B):
            hypothesis = []

            feature = features[batch : batch + 1]
            length = int(lengths[batch])

            step = 0
            new_symbols = 0

            while step < length:
                # decoder pass
                decoder_out, (hidden, cell) = self.decoder(
                    (
                        mx.array([[last_token[batch]]])
                        if last_token[batch] is not None
                        else None
                    ),
                    hidden_state[batch],
                )
                decoder_out = decoder_out.astype(feature.dtype)
                decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )

                # joint pass
                joint_out = self.joint(feature[:, step : step + 1], decoder_out)

                # sampling
                pred_token = int(
                    mx.argmax(joint_out[0, 0, :, : len(self.vocabulary) + 1])
                )
                decision = int(
                    mx.argmax(joint_out[0, 0, :, len(self.vocabulary) + 1 :])
                )

                # tdt decoding rule
                if pred_token != len(self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=step
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            duration=self.durations[decision]
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            text=tokenizer.decode([pred_token], self.vocabulary),
                        )
                    )
                    last_token[batch] = pred_token
                    hidden_state[batch] = decoder_hidden

                step += self.durations[int(decision)]

                # prevent stucking rule
                new_symbols += 1

                if self.durations[int(decision)] != 0:
                    new_symbols = 0
                elif self.max_symbols is not None and self.max_symbols <= new_symbols:
                    step += 1
                    new_symbols = 0

            results.append(hypothesis)

        return results, hidden_state

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)

        result, _ = self.decode(features, lengths, config=decoding_config)

        return [
            sentences_to_result(tokens_to_sentences(hypothesis))
            for hypothesis in result
        ]


class ParakeetRNNT(BaseParakeet):
    """MLX Implementation of Parakeet-RNNT Model"""

    def __init__(self, args: ParakeetRNNTArgs):
        super().__init__(args.preprocessor, args.encoder)

        self.vocabulary = args.joint.vocabulary
        self.max_symbols: int | None = (
            args.decoding.greedy.get("max_symbols", None)
            if args.decoding.greedy
            else None
        )

        self.decoder = PredictNetwork(args.decoder)
        self.joint = JointNetwork(args.joint)

    def decode(
        self,
        features: mx.array,
        lengths: mx.array | None = None,
        last_token: list[int | None] | None = None,
        hidden_state: list[tuple[mx.array, mx.array] | None] | None = None,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> tuple[list[list[AlignedToken]], list[tuple[mx.array, mx.array] | None]]:
        """Run RNNT decoder with features, optional length and decoder state.

        Args:
            features: Encoded audio features from the encoder.
            lengths: Optional sequence lengths for each batch item.
            last_token: Optional last predicted token for each batch item.
            hidden_state: Optional decoder hidden states for each batch item.
            config: Decoding configuration.

        Returns:
            tuple: (list of token sequences, updated hidden states)
        """
        assert (
            config.decoding == "greedy"
        ), "Only greedy decoding is supported for RNNT decoder now"

        B, S, *_ = features.shape

        if hidden_state is None:
            hidden_state = list([None] * B)

        if lengths is None:
            lengths = mx.array([S] * B)

        if last_token is None:
            last_token = list([None] * B)

        results = []
        for batch in range(B):
            hypothesis = []

            feature = features[batch : batch + 1]
            length = int(lengths[batch])

            step = 0
            new_symbols = 0

            while step < length:
                # decoder pass
                decoder_out, (hidden, cell) = self.decoder(
                    (
                        mx.array([[last_token[batch]]])
                        if last_token[batch] is not None
                        else None
                    ),
                    hidden_state[batch],
                )
                decoder_out = decoder_out.astype(feature.dtype)
                decoder_hidden = (
                    hidden.astype(feature.dtype),
                    cell.astype(feature.dtype),
                )

                # joint pass
                joint_out = self.joint(feature[:, step : step + 1], decoder_out)

                # sampling
                pred_token = int(mx.argmax(joint_out[0, 0]))

                # rnnt decoding rule
                if pred_token != len(self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=step
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            duration=1
                            * self.encoder_config.subsampling_factor
                            / self.preprocessor_config.sample_rate
                            * self.preprocessor_config.hop_length,  # hop
                            text=tokenizer.decode([pred_token], self.vocabulary),
                        )
                    )
                    last_token[batch] = pred_token
                    hidden_state[batch] = decoder_hidden

                    # prevent stucking
                    new_symbols += 1
                    if self.max_symbols is not None and self.max_symbols <= new_symbols:
                        step += 1
                        new_symbols = 0
                else:
                    step += 1
                    new_symbols = 0

            results.append(hypothesis)

        return results, hidden_state

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)

        result, _ = self.decode(features, lengths, config=decoding_config)

        return [
            sentences_to_result(tokens_to_sentences(hypothesis))
            for hypothesis in result
        ]


class ParakeetCTC(BaseParakeet):
    """MLX Implementation of Parakeet-CTC Model"""

    def __init__(self, args: ParakeetCTCArgs):
        super().__init__(args.preprocessor, args.encoder)

        self.vocabulary = args.decoder.vocabulary

        self.decoder = ConvASRDecoder(args.decoder)

    def decode(
        self,
        features: mx.array,
        lengths: mx.array,
        *,
        config: DecodingConfig = DecodingConfig(),
    ) -> list[list[AlignedToken]]:
        """Run CTC decoder with features and lengths.

        Args:
            features: Encoded audio features from the encoder.
            lengths: Sequence lengths for each batch item.
            config: Decoding configuration
            - (Currently implemented) Greedy/best path
            - (Not implemented) Beam search and lexicon based decoding

        Returns:
            list[list[AlignedToken]]: List of decoded token sequences.
        """
        assert (
            config.decoding == "greedy"
        ), "Only greedy decoding is supported for CTC decoder now"

        B, S, *_ = features.shape

        logits = self.decoder(features)
        mx.eval(logits, lengths)

        results = []
        for batch in range(B):
            length = int(lengths[batch])
            predictions = logits[batch, :length]
            best_tokens = mx.argmax(predictions, axis=1)

            hypothesis = []
            token_boundaries: list[tuple[int, int | None]] = []
            prev_token = -1

            for t, token_id in enumerate(best_tokens):
                token_idx = int(token_id)

                if token_idx == len(self.vocabulary):
                    continue

                if token_idx == prev_token:
                    continue

                if prev_token != -1:
                    token_start_time = (
                        token_boundaries[-1][0]
                        * self.encoder_config.subsampling_factor
                        / self.preprocessor_config.sample_rate
                        * self.preprocessor_config.hop_length
                    )

                    token_end_time = (
                        t
                        * self.encoder_config.subsampling_factor
                        / self.preprocessor_config.sample_rate
                        * self.preprocessor_config.hop_length
                    )

                    token_duration = token_end_time - token_start_time

                    hypothesis.append(
                        AlignedToken(
                            prev_token,
                            start=token_start_time,
                            duration=token_duration,
                            text=tokenizer.decode([prev_token], self.vocabulary),
                        )
                    )

                token_boundaries.append((t, None))
                prev_token = token_idx

            if prev_token != -1:
                last_non_blank = length - 1
                for t in range(length - 1, token_boundaries[-1][0], -1):
                    if int(best_tokens[t]) != len(self.vocabulary):
                        last_non_blank = t
                        break

                token_start_time = (
                    token_boundaries[-1][0]
                    * self.encoder_config.subsampling_factor
                    / self.preprocessor_config.sample_rate
                    * self.preprocessor_config.hop_length
                )

                token_end_time = (
                    (last_non_blank + 1)
                    * self.encoder_config.subsampling_factor
                    / self.preprocessor_config.sample_rate
                    * self.preprocessor_config.hop_length
                )

                token_duration = token_end_time - token_start_time

                hypothesis.append(
                    AlignedToken(
                        prev_token,
                        start=token_start_time,
                        duration=token_duration,
                        text=tokenizer.decode([prev_token], self.vocabulary),
                    )
                )

            results.append(hypothesis)

        return results

    def generate(
        self, mel: mx.array, *, decoding_config: DecodingConfig = DecodingConfig()
    ) -> list[AlignedResult]:
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)

        features, lengths = self.encoder(mel)

        result = self.decode(features, lengths, config=decoding_config)

        return [
            sentences_to_result(tokens_to_sentences(hypothesis))
            for hypothesis in result
        ]


class ParakeetTDTCTC(ParakeetTDT):
    """MLX Implementation of Parakeet-TDT-CTC Model

    Has ConvASRDecoder decoder in `.ctc_decoder` but `.generate` uses TDT decoder
    all the times (Please open an issue if you need CTC decoder use-case!)
    """

    def __init__(self, args: ParakeetTDTCTCArgs):
        super().__init__(args)

        self.ctc_decoder = ConvASRDecoder(args.aux_ctc.decoder)


# streaming
class StreamingParakeet:
    """Context manager for real-time streaming ASR inference.

    This class enables real-time transcription by maintaining internal buffers
    and caches to process audio chunks incrementally. It supports local attention
    mechanisms for efficient streaming and maintains decoder state across chunks.

    Attributes:
        model: The base Parakeet model to use for inference.
        cache: List of Conformer layer caches for efficient streaming.
        audio_buffer: Buffer for raw audio samples.
        mel_buffer: Buffer for mel-spectrogram features.
        decoder_hidden: Hidden state from the decoder (for RNNT/TDT models).
        last_token: Last predicted token (for RNNT/TDT models).
        finalized_tokens: Tokens that have been finalized and won't change.
        draft_tokens: Preliminary tokens that may change with more audio.
        context_size: Left and right context sizes for local attention.
        depth: Number of encoder layers to cache for exact equivalence.
        decoding_config: Configuration for decoding behavior.
        keep_original_attention: Whether to preserve original attention mechanism.
    """

    model: "BaseParakeet"
    cache: list[ConformerCache]

    audio_buffer: mx.array
    mel_buffer: mx.array | None
    decoder_hidden: tuple[mx.array, mx.array] | None = None
    last_token: int | None = None

    finalized_tokens: list[AlignedToken]
    draft_tokens: list[AlignedToken]

    context_size: tuple[int, int]
    depth: int
    decoding_config: DecodingConfig
    keep_original_attention: bool = False

    def __init__(
        self,
        model: "BaseParakeet",
        context_size: tuple[int, int],
        depth: int = 1,
        *,
        keep_original_attention: bool = False,
        decoding_config: DecodingConfig = DecodingConfig(),
    ) -> None:
        """Initialize StreamingParakeet for real-time inference.

        Args:
            model: The Parakeet model to use for streaming inference.
            context_size: Tuple of (left_context, right_context) sizes.
            depth: Number of encoder layers to cache for exact equivalence.
            keep_original_attention: Whether to preserve original attention.
            decoding_config: Configuration for decoding behavior.
        """
        self.context_size = context_size
        self.depth = depth
        self.decoding_config = decoding_config
        self.keep_original_attention = keep_original_attention

        self.model = model
        self.cache = [
            RotatingConformerCache(self.keep_size, cache_drop_size=self.drop_size)
            for _ in range(len(model.encoder.layers))
        ]

        self.audio_buffer = mx.array([])
        self.mel_buffer = None
        self.finalized_tokens = []
        self.draft_tokens = []

    def __enter__(self):
        """Enter the streaming context and configure attention model.

        Returns:
            self: The StreamingParakeet instance.
        """
        if not self.keep_original_attention:
            self.model.encoder.set_attention_model(
                "rel_pos_local_attn", self.context_size
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the streaming context and clean up resources.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        if not self.keep_original_attention:
            self.model.encoder.set_attention_model(
                "rel_pos"
            )  # hard-coded; might cache if there's new variant than rel_pos
        del self.audio_buffer
        del self.cache

        mx.clear_cache()

    @property
    def keep_size(self):
        """Number of encoded feature frames to keep in KV cache.

        Returns:
            int: The left context size, indicating frames to retain.
        """
        return self.context_size[0]

    @property
    def drop_size(self):
        """Number of encoded feature frames to drop from cache.

        Returns:
            int: Right context size multiplied by depth.
        """
        return self.context_size[1] * self.depth

    @property
    def result(self) -> AlignedResult:
        """Current transcription result including finalized and draft tokens.

        Returns:
            AlignedResult: Complete transcription with aligned tokens and sentences.
        """
        return sentences_to_result(
            tokens_to_sentences(self.finalized_tokens + self.draft_tokens)
        )

    def add_audio(self, audio: mx.array) -> None:
        """Add audio chunk for streaming transcription.

        Processes the audio chunk incrementally, maintaining internal buffers
        and updating transcription results. The method handles mel-spectrogram
        conversion, encoder processing with cache management, and decoding.

        Args:
            audio: 1D audio array to be processed and transcribed.

        Note:
            The audio array must be 1-dimensional and match the model's
            expected sample rate.
        """

        self.audio_buffer = mx.concat(
            [
                self.audio_buffer,
                audio,
            ],
            axis=0,
        )
        mel = get_logmel(
            self.audio_buffer[
                : (
                    len(self.audio_buffer)
                    // self.model.preprocessor_config.hop_length
                    * self.model.preprocessor_config.hop_length
                )
            ],
            self.model.preprocessor_config,
        )

        if self.mel_buffer is None:  # init
            self.mel_buffer = mel
        else:
            self.mel_buffer = mx.concat([self.mel_buffer, mel], axis=1)

        self.audio_buffer = self.audio_buffer[
            (mel.shape[1] * self.model.preprocessor_config.hop_length) :
        ]

        features, lengths = self.model.encoder(
            self.mel_buffer[
                :,
                : (
                    self.mel_buffer.shape[1]
                    // self.model.encoder_config.subsampling_factor
                    * self.model.encoder_config.subsampling_factor
                ),
            ],
            cache=self.cache,
        )
        mx.eval(features, lengths)
        length = int(lengths[0])

        # cache will automatically dropped in cache level
        leftover = self.mel_buffer.shape[1] - (
            length * self.model.encoder_config.subsampling_factor
        )
        self.mel_buffer = self.mel_buffer[
            :,
            -(
                self.drop_size * self.model.encoder_config.subsampling_factor + leftover
            ) :,
        ]

        # we decode in two phase
        # first phase: finalized region decode
        # second phase: draft region decode (will be dropped)
        finalized_length = max(0, length - self.drop_size)

        if isinstance(self.model, ParakeetTDT | ParakeetRNNT):
            finalized_tokens, finalized_state = self.model.decode(
                features,
                mx.array([finalized_length]),
                [self.last_token],
                [self.decoder_hidden],
                config=self.decoding_config,
            )

            self.decoder_hidden = finalized_state[0]
            self.last_token = (
                finalized_tokens[0][-1].id if len(finalized_tokens[0]) > 0 else None
            )

            draft_tokens, _ = self.model.decode(
                features[:, finalized_length:],
                mx.array(
                    [
                        features[:, finalized_length:].shape[1]
                    ]  # i believe in lazy evaluation
                ),
                [self.last_token],
                [self.decoder_hidden],
                config=self.decoding_config,
            )

            self.finalized_tokens.extend(finalized_tokens[0])
            self.draft_tokens = draft_tokens[0]
        elif isinstance(self.model, ParakeetCTC):
            finalized_tokens = self.model.decode(
                features, mx.array([finalized_length]), config=self.decoding_config
            )

            draft_tokens = self.model.decode(
                features[:, finalized_length:],
                mx.array(
                    [
                        features[:, finalized_length:].shape[1]
                    ]  # i believe in lazy evaluation
                ),
                config=self.decoding_config,
            )

            self.finalized_tokens.extend(finalized_tokens[0])
            self.draft_tokens = draft_tokens[0]
        else:
            raise NotImplementedError("This model does not support real-time decoding")
