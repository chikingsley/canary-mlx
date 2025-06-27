"""
This file defines the Conformer model architecture, a sequence-to-sequence model
that uses a combination of self-attention and convolution layers. It is commonly
used in automatic speech recognition.

The implementation includes the main Conformer block, feed-forward networks,
convolution modules, and subsampling layers.
"""

import math
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
from mlx import nn
from mlx.nn.utils import tree_flatten

from parakeet_mlx.attention import (
    LocalRelPositionalEncoding,
    MultiHeadAttention,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadLocalAttention,
)


@dataclass
class ConformerArgs:
    """Configuration arguments for the Conformer model."""

    feat_in: int  # mel-log
    n_layers: int
    d_model: int
    n_heads: int
    ff_expansion_factor: int
    subsampling_factor: int
    self_attention_model: str
    subsampling: str
    conv_kernel_size: int
    subsampling_conv_channels: int
    pos_emb_max_len: int
    causal_downsampling: bool = False
    use_bias: bool = True
    xscaling: bool = False
    pos_bias_u: mx.array | None = None
    pos_bias_v: mx.array | None = None
    subsampling_conv_chunking_factor: int = 1
    att_context_size: list[int] | None = None


class FeedForward(nn.Module):
    """A standard two-layer feed-forward network with SiLU activation.

    Args:
        d_model (int): The input and output dimension.
        d_ff (int): The hidden dimension of the feed-forward network.
        use_bias (bool): Whether to include a bias term in the linear layers.
    """

    def __init__(self, d_model: int, d_ff: int, use_bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Applies the feed-forward network to the input tensor."""
        return self.linear2(nn.silu(self.linear1(x)))


class Convolution(nn.Module):
    """A conformer convolution module.

    This module consists of a pointwise convolution, a depthwise convolution,
    batch normalization, SiLU activation, and another pointwise convolution.

    Args:
        args (ConformerArgs): The conformer model configuration.
    """

    def __init__(self, args: ConformerArgs):
        assert (args.conv_kernel_size - 1) % 2 == 0
        super().__init__()

        self.padding = (args.conv_kernel_size - 1) // 2

        self.pointwise_conv1 = nn.Conv1d(
            args.d_model,
            args.d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=args.use_bias,
        )
        self.depthwise_conv = nn.Conv1d(
            args.d_model,
            args.d_model,
            kernel_size=args.conv_kernel_size,
            stride=1,
            padding=0,
            groups=args.d_model,
            bias=args.use_bias,
        )
        self.batch_norm = nn.BatchNorm(args.d_model)
        self.pointwise_conv2 = nn.Conv1d(
            args.d_model,
            args.d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=args.use_bias,
        )

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        """Applies the convolution module to the input tensor.
        x = x.swapaxes(1, 2) - removed

        Args:
            x (mx.array): The input tensor.
            cache: An optional cache for incremental decoding.

        Returns:
            mx.array: The output of the convolution module.
        """

        x = self.pointwise_conv1(x)
        x = nn.glu(
            x, axis=2
        )  # TODO: Investigate using other activation functions for the GLU gate.

        # Applied cache for streaming convolution, which is crucial for real-time processing.
        if cache is not None:
            x = cache.update_and_fetch_conv(x, padding=self.padding)
        else:
            x = mx.pad(x, ((0, 0), (self.padding, self.padding), (0, 0)))
        x = self.depthwise_conv(x)

        x = self.batch_norm(x)
        x = nn.silu(x)
        x = self.pointwise_conv2(x)

        return x


class ConformerBlock(nn.Module):
    """A single block of the Conformer model.

    Each block consists of two feed-forward modules, a self-attention module,
    and a convolution module, with layer normalization and residual connections.

    Args:
        args (ConformerArgs): The conformer model configuration.
    """

    def __init__(self, args: ConformerArgs):
        super().__init__()
        ff_hidden_dim = args.d_model * args.ff_expansion_factor

        self.args = args

        self.norm_feed_forward1 = nn.LayerNorm(args.d_model)
        self.feed_forward1 = FeedForward(args.d_model, ff_hidden_dim, args.use_bias)

        self.norm_self_att = nn.LayerNorm(args.d_model)
        self.self_attn = (
            RelPositionMultiHeadAttention(
                args.n_heads,
                args.d_model,
                bias=args.use_bias,
                pos_bias_u=args.pos_bias_u,
                pos_bias_v=args.pos_bias_v,
            )
            if args.self_attention_model == "rel_pos"
            else (
                RelPositionMultiHeadLocalAttention(
                    args.n_heads,
                    args.d_model,
                    bias=args.use_bias,
                    pos_bias_u=args.pos_bias_u,
                    pos_bias_v=args.pos_bias_v,
                    context_size=(
                        (args.att_context_size[0], args.att_context_size[1])
                        if args.att_context_size is not None
                        else (-1, -1)
                    ),
                )
                if args.self_attention_model == "rel_pos_local_attn"
                else MultiHeadAttention(
                    args.n_heads,
                    args.d_model,
                    bias=True,
                )
            )
        )

        self.norm_conv = nn.LayerNorm(args.d_model)
        self.conv = Convolution(args)

        self.norm_feed_forward2 = nn.LayerNorm(args.d_model)
        self.feed_forward2 = FeedForward(args.d_model, ff_hidden_dim, args.use_bias)

        self.norm_out = nn.LayerNorm(args.d_model)

    def set_attention_model(
        self,
        name: Literal["rel_pos", "rel_pos_local_attn", "normal"],
        context_size: tuple[int, int] | None = (256, 256),
    ):
        """Dynamically sets the attention mechanism for the block.

        This allows switching between different attention types (e.g., global,
        local) at runtime, transferring the weights from the old model.

        Args:
            name: The type of attention model to use.
            context_size: The context window size for local attention.
        """
        new_attn = (
            RelPositionMultiHeadAttention(
                self.args.n_heads,
                self.args.d_model,
                bias=self.args.use_bias,
                pos_bias_u=self.args.pos_bias_u,
                pos_bias_v=self.args.pos_bias_v,
            )
            if name == "rel_pos"
            else (
                RelPositionMultiHeadLocalAttention(
                    self.args.n_heads,
                    self.args.d_model,
                    bias=self.args.use_bias,
                    pos_bias_u=self.args.pos_bias_u,
                    pos_bias_v=self.args.pos_bias_v,
                    context_size=context_size if context_size is not None else (-1, -1),
                )
                if name == "rel_pos_local_attn"
                else MultiHeadAttention(
                    self.args.n_heads,
                    self.args.d_model,
                    bias=True,
                )
            )
        )

        new_attn.load_weights(tree_flatten(self.self_attn.parameters()))

        self.self_attn = new_attn

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        """Forward pass for the Conformer block.

        Args:
            x (mx.array): The input tensor.
            pos_emb (mx.array | None): The positional embedding tensor.
            mask (mx.array | None): An optional mask for the attention scores.
            cache: An optional cache for incremental decoding.

        Returns:
            mx.array: The output of the conformer block.
        """
        x += 0.5 * self.feed_forward1(self.norm_feed_forward1(x))

        x_norm = self.norm_self_att(x)
        x += self.self_attn(
            x_norm, x_norm, x_norm, mask=mask, pos_emb=pos_emb, cache=cache
        )

        x += self.conv(self.norm_conv(x), cache=cache)
        x += 0.5 * self.feed_forward2(self.norm_feed_forward2(x))

        return self.norm_out(x)


class DwStridingSubsampling(nn.Module):
    """A depthwise striding convolutional subsampling layer.

    This layer reduces the sequence length of the input by a given factor
    using a series of 2D convolutions.

    Args:
        args (ConformerArgs): The conformer model configuration.
    """

    def __init__(self, args: ConformerArgs):
        super().__init__()

        assert (
            args.subsampling_factor > 0
            and (args.subsampling_factor & (args.subsampling_factor - 1)) == 0
        )
        self.subsampling_conv_chunking_factor = args.subsampling_conv_chunking_factor
        self._conv_channels = args.subsampling_conv_channels
        self._sampling_num = int(math.log(args.subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = 3
        self._padding = (self._kernel_size - 1) // 2

        in_channels = 1
        final_freq_dim = args.feat_in
        for _ in range(self._sampling_num):
            final_freq_dim = (
                math.floor(
                    (final_freq_dim + 2 * self._padding - self._kernel_size)
                    / self._stride
                )
                + 1
            )
            if final_freq_dim < 1:
                raise ValueError("Non-positive final frequency dimension!")

        self.conv = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self._conv_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=self._padding,
            ),
            nn.ReLU(),
        ]
        in_channels = self._conv_channels

        for _ in range(self._sampling_num - 1):
            self.conv.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                    groups=in_channels,
                )
            )
            self.conv.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self._conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )
            self.conv.append(nn.ReLU())

        self.out = nn.Linear(self._conv_channels * final_freq_dim, args.d_model)

    def conv_forward(self, x: mx.array) -> mx.array:
        """Applies the stack of convolutional layers."""
        x = x.transpose((0, 2, 3, 1))
        for layer in self.conv:
            x = layer(x)
        return x.transpose((0, 3, 1, 2))

    def conv_split_by_batch(self, x: mx.array) -> tuple[mx.array, bool]:
        """Splits the batch for convolution to manage memory usage.

        Args:
            x (mx.array): The input tensor.

        Returns:
            A tuple containing the processed tensor and a boolean indicating
            if the split was successful.
        """
        b = x.shape[0]
        if b == 1:
            return x, False

        cf: int
        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
        else:
            x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(x.size / x_ceil, 2))
            cf = 2**p

        new_batch_size = b // cf
        if new_batch_size == 0:
            return x, False

        return (
            mx.concat(
                [self.conv_forward(chunk) for chunk in mx.split(x, new_batch_size, 0)]
            ),
            True,
        )

    def __call__(self, x: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        """Forward pass for the subsampling layer.

        Args:
            x (mx.array): The input tensor (features).
            lengths (mx.array): The lengths of the input sequences.

        Returns:
            A tuple containing the subsampled output tensor and the new lengths.
        """
        for _ in range(self._sampling_num):
            lengths = (
                mx.floor(
                    (lengths + 2 * self._padding - self._kernel_size) / self._stride
                )
                + 1.0
            )
        lengths = lengths.astype(mx.int32)

        x = mx.expand_dims(x, axis=1)

        if self.subsampling_conv_chunking_factor != -1:
            if self.subsampling_conv_chunking_factor == 1:
                x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
                need_to_split = x.size > x_ceil
            else:
                need_to_split = True

            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                if not success:
                    # TODO: Add channel splitting
                    x = self.conv_forward(x)  # try anyways
            else:
                x = self.conv_forward(x)
        else:
            x = self.conv_forward(x)

        x = x.swapaxes(1, 2).reshape(x.shape[0], x.shape[2], -1)
        x = self.out(x)
        return x, lengths


class Conformer(nn.Module):
    """The main Conformer model.

    This class assembles the full Conformer architecture, including the
    subsampling layer, positional encoding, and a stack of Conformer blocks.

    Args:
        args (ConformerArgs): The conformer model configuration.
    """

    pre_encode: nn.Module
    pos_enc: RelPositionalEncoding | LocalRelPositionalEncoding | None
    layers: list[ConformerBlock]
    args: ConformerArgs

    def __init__(self, args: ConformerArgs):
        super().__init__()

        self.args = args

        if args.self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=args.d_model,
                max_len=args.pos_emb_max_len,
                scale_input=args.xscaling,
            )
        elif args.self_attention_model == "rel_pos_local_attn":
            self.pos_enc = LocalRelPositionalEncoding(
                d_model=args.d_model,
                max_len=args.pos_emb_max_len,
                scale_input=args.xscaling,
                context_size=(
                    (args.att_context_size[0], args.att_context_size[1])
                    if args.att_context_size is not None
                    else (-1, -1)
                ),
            )
        else:
            self.pos_enc = None

        if args.subsampling_factor > 1:
            if args.subsampling == "dw_striding" and args.causal_downsampling is False:
                self.pre_encode = DwStridingSubsampling(args)
            else:
                self.pre_encode = nn.Identity()
                raise NotImplementedError(
                    "Other subsampling haven't been implemented yet!"
                )
        else:
            self.pre_encode = nn.Linear(args.feat_in, args.d_model)

        self.layers = [ConformerBlock(args) for _ in range(args.n_layers)]

    def set_attention_model(
        self,
        name: Literal["rel_pos", "rel_pos_local_attn", "normal"],
        context_size: tuple[int, int] | None = (256, 256),
    ):
        """Dynamically sets the attention mechanism for the entire model.

        This propagates the change to the positional encoding and all conformer blocks.

        Args:
            name: The type of attention model to use.
            context_size: The context window size for local attention.
        """
        if name == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=self.args.d_model,
                max_len=self.args.pos_emb_max_len,
                scale_input=self.args.xscaling,
            )
        elif name == "rel_pos_local_attn":
            self.pos_enc = LocalRelPositionalEncoding(
                d_model=self.args.d_model,
                max_len=self.args.pos_emb_max_len,
                scale_input=self.args.xscaling,
                context_size=context_size if context_size else (-1, -1),
            )
        else:
            self.pos_enc = None

        for layer in self.layers:
            layer.set_attention_model(name, context_size)

    def __call__(
        self, x: mx.array, lengths: mx.array | None = None, cache=None
    ) -> tuple[mx.array, mx.array]:
        """The main forward pass for the Conformer model.

        Args:
            x (mx.array): The input feature tensor.
            lengths (mx.array | None): The lengths of the input sequences.
            cache: An optional cache for incremental decoding.

        Returns:
            A tuple containing the output tensor and the output lengths.

        Raises:
            NotImplementedError: If an unsupported pre-encoding layer is configured.
        """
        if lengths is None:
            lengths = mx.full(
                (x.shape[0],),
                x.shape[-2],
                dtype=mx.int64,
            )

        if isinstance(self.pre_encode, DwStridingSubsampling):
            x, out_lengths = self.pre_encode(x, lengths)
        elif isinstance(self.pre_encode, nn.Linear):
            x = self.pre_encode(x)
            out_lengths = lengths
        else:
            raise NotImplementedError("Non-implemented pre-encoding layer type!")

        if cache is None:
            cache = [None] * len(self.layers)

        pos_emb = None
        if self.pos_enc is not None:
            x, pos_emb = self.pos_enc(
                x,
                offset=cache[0].offset if cache[0] is not None else 0,  # type: ignore
            )

        for layer, c in zip(self.layers, cache, strict=False):
            x = layer(x, pos_emb=pos_emb, cache=c)

        return x, out_lengths
