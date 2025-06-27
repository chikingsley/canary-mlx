"""MLX implementation of CTC for ASR."""

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn


@dataclass
class ConvASRDecoderArgs:
    """Configuration for the ConvASRDecoder.

    Args:
        feat_in (int): The number of input features.
        num_classes (int): The number of output classes. If <= 0, it is inferred
            from the vocabulary size.
        vocabulary (list[str]): The vocabulary of the model.
    """

    feat_in: int
    num_classes: int
    vocabulary: list[str]


@dataclass
class AuxCTCArgs:
    """Configuration for the auxiliary CTC component.

    Args:
        decoder (ConvASRDecoderArgs): The configuration for the CTC decoder.
    """

    decoder: ConvASRDecoderArgs


class ConvASRDecoder(nn.Module):
    """A convolutional ASR decoder for CTC-based models.

    This module takes encoder features and produces log probabilities over the
    vocabulary.
    """

    def __init__(self, args: ConvASRDecoderArgs):
        super().__init__()

        args.num_classes = (
            len(args.vocabulary) if args.num_classes <= 0 else args.num_classes
        ) + 1

        self.decoder_layers = [
            nn.Conv1d(args.feat_in, args.num_classes, kernel_size=1, bias=True)
        ]

        self.temperature = 1.0  # change manually if desired

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass of the decoder.

        Args:
            x (mx.array): The input tensor from the encoder, with shape
                (batch, features, time).

        Returns:
            mx.array: The output log probabilities, with shape
                (batch, classes, time).
        """
        return nn.log_softmax(self.decoder_layers[0](x) / self.temperature, axis=1)
