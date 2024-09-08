"""Basic ResNet encoder for image feature extraction.

https://pytorch.org/hub/pytorch_vision_resnet/
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

from siclib.models.base_model import BaseModel

# mypy: ignore-errors


def remove_conv_stride(conv):
    """Remove the stride from a convolutional layer."""
    conv_new = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        bias=conv.bias is not None,
        stride=1,
        padding=conv.padding,
    )
    conv_new.weight = conv.weight
    conv_new.bias = conv.bias
    return conv_new


class ResNet(BaseModel):
    """ResNet encoder for image features extraction."""

    default_conf = {
        "encoder": "resnet18",
        "pretrained": True,
        "input_dim": 3,
        "remove_stride_from_first_conv": True,
        "num_downsample": None,  # how many downsample bloc
        "pixel_mean": [0.485, 0.456, 0.406],
        "pixel_std": [0.229, 0.224, 0.225],
    }

    required_data_keys = ["image"]

    def build_encoder(self, conf):
        """Build the encoder from the configuration."""
        if conf.pretrained:
            assert conf.input_dim == 3

        Encoder = getattr(torchvision.models, conf.encoder)

        layers = ["layer1", "layer2", "layer3", "layer4"]
        kw = {"replace_stride_with_dilation": [False, False, False]}

        if conf.num_downsample is not None:
            layers = layers[: conf.num_downsample]

        encoder = Encoder(weights="DEFAULT" if conf.pretrained else None, **kw)
        encoder = create_feature_extractor(encoder, return_nodes=layers)

        if conf.remove_stride_from_first_conv:
            encoder.conv1 = remove_conv_stride(encoder.conv1)

        return encoder, layers

    def _init(self, conf):
        self.register_buffer("pixel_mean", torch.tensor(conf.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(conf.pixel_std).view(-1, 1, 1), False)

        self.encoder, self.layers = self.build_encoder(conf)

    def _forward(self, data):
        image = data["image"]
        image = (image - self.pixel_mean) / self.pixel_std
        skip_features = list(self.encoder(image).values())

        # print(f"skip_features: {[f.shape for f in skip_features]}")
        return {"features": skip_features}

    def loss(self, pred, data):
        """Compute the loss."""
        raise NotImplementedError
