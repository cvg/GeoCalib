"""Simple VGG encoder for image features extraction."""

import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

from siclib.models.base_model import BaseModel

# mypy: ignore-errors


class VGG(BaseModel):
    """VGG encoder for image features extraction."""

    default_conf = {
        "encoder": "vgg13",
        "pretrained": True,
        "input_dim": 3,
        "num_downsample": None,  # how many downsample blocs to use
        "pixel_mean": [0.485, 0.456, 0.406],
        "pixel_std": [0.229, 0.224, 0.225],
    }

    required_data_keys = ["image"]

    def build_encoder(self, conf):
        """Build the encoder from the configuration."""
        if conf.pretrained:
            assert conf.input_dim == 3

        Encoder = getattr(torchvision.models, conf.encoder)

        kw = {}
        if conf.encoder == "vgg13":
            layers = [
                "features.3",
                "features.8",
                "features.13",
                "features.18",
                "features.23",
            ]
        elif conf.encoder == "vgg16":
            layers = [
                "features.3",
                "features.8",
                "features.15",
                "features.22",
                "features.29",
            ]
        else:
            raise NotImplementedError(f"Encoder not implemented: {conf.encoder}")

        if conf.num_downsample is not None:
            layers = layers[: conf.num_downsample]

        encoder = Encoder(weights="DEFAULT" if conf.pretrained else None, **kw)
        encoder = create_feature_extractor(encoder, return_nodes=layers)

        return encoder, layers

    def _init(self, conf):
        self.register_buffer("pixel_mean", torch.tensor(conf.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(conf.pixel_std).view(-1, 1, 1), False)

        self.encoder, self.layers = self.build_encoder(conf)

    def _forward(self, data):
        image = data["image"]
        image = (image - self.pixel_mean) / self.pixel_std
        skip_features = self.encoder(image).values()
        return {"features": skip_features}

    def loss(self, pred, data):
        """Compute the loss."""
        raise NotImplementedError
