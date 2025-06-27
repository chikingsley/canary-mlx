"""Recurrent Neural Network Transducer (RNN-T) components for the Parakeet model."""

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn


@dataclass
class PredictNetworkArgs:
    """
    Args:
        pred_hidden (int): _description_
        pred_rnn_layers (int): _description_
        rnn_hidden_size (int | None, optional): _description_. Defaults to None.
    """

    pred_hidden: int
    pred_rnn_layers: int
    rnn_hidden_size: int | None = None


@dataclass
class JointNetworkArgs:
    """
    Args:
        joint_hidden (int): _description_
        activation (str): _description_
        encoder_hidden (int): _description_
        pred_hidden (int): _description_
    """

    joint_hidden: int
    activation: str
    encoder_hidden: int
    pred_hidden: int


@dataclass
class PredictArgs:
    """
    Args:
        blank_as_pad (bool): _description_
        vocab_size (int): _description_
        prednet (PredictNetworkArgs): _description_
    """

    blank_as_pad: bool
    vocab_size: int
    prednet: PredictNetworkArgs


@dataclass
class JointArgs:
    """
    Args:
        num_classes (int): _description_
        vocabulary (list[str]): _description_
        jointnet (JointNetworkArgs): _description_
        num_extra_outputs (int, optional): _description_. Defaults to 0.
    """

    num_classes: int
    vocabulary: list[str]
    jointnet: JointNetworkArgs
    num_extra_outputs: int = 0


class LSTM(nn.Module):
    """
    Args:
        input_size (int): _description_
        hidden_size (int): _description_
        num_layers (int, optional): _description_. Defaults to 1.
        bias (bool, optional): _description_. Defaults to True.
        batch_first (bool, optional): _description_. Defaults to True.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.lstm = [
            nn.LSTM(input_size if i == 0 else hidden_size, hidden_size, bias=bias)
            for i in range(num_layers)
        ]

    def __call__(
        self, x: mx.array, h_c: tuple[mx.array, mx.array] | None = None
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        if self.batch_first:
            x = mx.transpose(x, (1, 0, 2))

        h: list[mx.array | None]
        c: list[mx.array | None]
        if h_c is None:
            h = [None] * self.num_layers
            c = [None] * self.num_layers
        else:
            h_arr, c_arr = h_c
            h = [h_arr[i] for i in range(h_arr.shape[0])]
            c = [c_arr[i] for i in range(c_arr.shape[0])]

        outputs = x
        next_h = []
        next_c = []

        for i in range(self.num_layers):
            layer = self.lstm[i]

            all_h_steps, all_c_steps = layer(outputs, hidden=h[i], cell=c[i])
            outputs = all_h_steps
            next_h.append(all_h_steps[-1])
            next_c.append(all_c_steps[-1])

        if self.batch_first:
            outputs = mx.transpose(outputs, (1, 0, 2))

        final_h = mx.stack(next_h, axis=0)
        final_c = mx.stack(next_c, axis=0)

        return outputs, (final_h, final_c)


class PredictNetwork(nn.Module):
    """
    Args:
        args (PredictArgs): _description_
    """

    def __init__(self, args: PredictArgs):
        super().__init__()

        self.pred_hidden = args.prednet.pred_hidden

        self.prediction = {
            "embed": nn.Embedding(
                args.vocab_size if not args.blank_as_pad else args.vocab_size + 1,
                args.prednet.pred_hidden,
            ),
            "dec_rnn": LSTM(
                args.prednet.pred_hidden,
                (
                    args.prednet.rnn_hidden_size
                    if args.prednet.rnn_hidden_size
                    else args.prednet.pred_hidden
                ),
                args.prednet.pred_rnn_layers,
            ),
        }

    def __call__(
        self, y: mx.array | None, h_c: tuple[mx.array, mx.array] | None = None
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        if y is not None:
            embedded_y = self.prediction["embed"](y)
        else:
            batch = 1 if h_c is None else h_c[0].shape[1]
            embedded_y = mx.zeros((batch, 1, self.pred_hidden))
        return self.prediction["dec_rnn"](embedded_y, h_c)


class JointNetwork(nn.Module):
    """
    Args:
        args (JointArgs): _description_
    """

    def __init__(self, args: JointArgs):
        super().__init__()
        self._num_classes = args.num_classes + 1 + args.num_extra_outputs

        if args.jointnet.activation not in ["relu", "sigmoid", "tanh"]:
            raise ValueError(
                "Unsupported activation for joint step - please pass one of "
                "[relu, sigmoid, tanh]"
            )

        activation_str = args.jointnet.activation.lower()

        activation: nn.Module
        if activation_str == "relu":
            activation = nn.ReLU()
        elif activation_str == "sigmoid":
            activation = nn.Sigmoid()
        else:
            activation = nn.Tanh()

        self.pred = nn.Linear(args.jointnet.pred_hidden, args.jointnet.joint_hidden)
        self.enc = nn.Linear(args.jointnet.encoder_hidden, args.jointnet.joint_hidden)
        self.joint_net = [
            activation,
            nn.Identity(),
            nn.Linear(args.jointnet.joint_hidden, self._num_classes),
        ]

    def __call__(self, enc: mx.array, pred: mx.array) -> mx.array:
        enc = self.enc(enc)
        pred = self.pred(pred)

        x = mx.expand_dims(enc, 2) + mx.expand_dims(pred, 1)

        for layer in self.joint_net:
            x = layer(x)

        return x
