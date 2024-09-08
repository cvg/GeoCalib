"""Various modules used in the decoder of the model.

Adapted from https://github.com/jinlinyi/PerspectiveFields
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """DropBlock, DropPath

    PyTorch implementations of DropBlock and DropPath (Stochastic Depth) regularization layers.

    Papers:
    DropBlock: A regularization method for convolutional networks (https://arxiv.org/abs/1810.12890)

    Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)

    Code:
    DropBlock impl inspired by two Tensorflow impl:
    - https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L74
    - https://github.com/clovaai/assembled-cnn/blob/master/nets/blocks.py

    Hacked together by / Copyright 2020 Ross Wightman
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    """Replacement for mmcv.cnn.ConvModule to avoid mmcv dependency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        use_norm: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activate(x)


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, unit2only=False, upsample=True):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()
        self.upsample = upsample

        if not unit2only:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass."""
        output = xs[0]

        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        if self.upsample:
            output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)

        return output


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False
        )

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        return self.conv1(self.relu1(self.norm1(concated_features)))

    def any_requires_grad(self, inp):
        return any(tensor.requires_grad for tensor in inp)

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, inp):
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *inp)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, inp) -> Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, inp):  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, inp):  # noqa: F811
        prev_features = [inp] if isinstance(inp, Tensor) else inp
        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers,
        num_input_features,
        bn_size,
        growth_rate,
        drop_rate,
        memory_efficient=False,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            num_input_features, num_output_features, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
