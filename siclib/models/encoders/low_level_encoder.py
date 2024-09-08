import logging

import torch.nn as nn

from siclib.models.base_model import BaseModel
from siclib.models.utils.modules import ConvModule

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


class LowLevelEncoder(BaseModel):
    default_conf = {
        "feat_dim": 64,
        "in_channel": 3,
        "keep_resolution": True,
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        logger.debug(f"Initializing LowLevelEncoder with {conf}")

        if self.conf.keep_resolution:
            self.conv1 = ConvModule(conf.in_channel, conf.feat_dim, kernel_size=3, padding=1)
            self.conv2 = ConvModule(conf.feat_dim, conf.feat_dim, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(
                conf.in_channel, conf.feat_dim, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(conf.feat_dim)
            self.relu = nn.ReLU(inplace=True)

    def _forward(self, data):
        x = data["image"]

        assert (
            x.shape[-1] % 32 == 0 and x.shape[-2] % 32 == 0
        ), "Image size must be multiple of 32 if not using single image input."

        if self.conf.keep_resolution:
            c1 = self.conv1(x)
            c2 = self.conv2(c1)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            c2 = self.relu(x)

        return {"features": c2}

    def loss(self, pred, data):
        raise NotImplementedError
