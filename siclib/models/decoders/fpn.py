import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from siclib.models import BaseModel
from siclib.models.utils.modules import ConvModule, FeatureFusionBlock

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


class DecoderBlock(nn.Module):
    def __init__(
        self,
        previous,
        out,
        ksize=3,
        num_convs=1,
        norm_str="BatchNorm2d",
        padding="zeros",
        fusion="sum",
    ):
        super().__init__()

        self.fusion = fusion

        if self.fusion == "sum":
            self.fusion_layers = nn.Identity()
        elif self.fusion == "glu":
            self.fusion_layers = nn.Sequential(
                nn.Conv2d(2 * out, 2 * out, 1, padding=0, bias=True),
                nn.GLU(dim=1),
            )
        elif self.fusion == "ff":
            self.fusion_layers = FeatureFusionBlock(out, upsample=False)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")

        if norm_str is not None:
            norm = getattr(nn, norm_str, None)

        layers = []
        for i in range(num_convs):
            conv = nn.Conv2d(
                previous if i == 0 else out,
                out,
                kernel_size=ksize,
                padding=ksize // 2,
                bias=norm_str is None,
                padding_mode=padding,
            )
            layers.append(conv)
            if norm_str is not None:
                layers.append(norm(out))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, previous, skip):
        _, _, hp, wp = previous.shape
        _, _, hs, ws = skip.shape
        scale = 2 ** np.round(np.log2(np.array([hs / hp, ws / wp])))

        upsampled = nn.functional.interpolate(
            previous, scale_factor=scale.tolist(), mode="bilinear", align_corners=False
        )
        # If the shape of the input map `skip` is not a multiple of 2,
        # it will not match the shape of the upsampled map `upsampled`.
        # If the downsampling uses ceil_mode=False, we need to crop `skip`.
        # If it uses ceil_mode=True (not supported here), we should pad it.
        _, _, hu, wu = upsampled.shape
        _, _, hs, ws = skip.shape
        if (hu <= hs) and (wu <= ws):
            skip = skip[:, :, :hu, :wu]
        elif (hu >= hs) and (wu >= ws):
            skip = nn.functional.pad(skip, [0, wu - ws, 0, hu - hs])
        else:
            raise ValueError(f"Inconsistent skip vs upsampled shapes: {(hs, ws)}, {(hu, wu)}")

        skip = skip.clone()
        feats_skip = self.layers(skip)
        if self.fusion == "sum":
            return self.fusion_layers(feats_skip + upsampled)
        elif self.fusion == "glu":
            x = torch.cat([feats_skip, upsampled], dim=1)
            return self.fusion_layers(x)
        elif self.fusion == "ff":
            return self.fusion_layers(feats_skip, upsampled)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")


class FPN(BaseModel):
    default_conf = {
        "predict_uncertainty": True,
        "in_channels_list": [64, 128, 256, 512],
        "out_channels": 64,
        "num_convs": 1,
        "norm": None,
        "padding": "zeros",
        "fusion": "sum",
        "with_low_level": True,
    }

    required_data_keys = ["hl"]

    def _init(self, conf):
        self.in_channels_list = conf.in_channels_list
        self.out_channels = conf.out_channels

        self.num_convs = conf.num_convs
        self.norm = conf.norm
        self.padding = conf.padding

        self.fusion = conf.fusion

        self.first = nn.Conv2d(
            self.in_channels_list[-1], self.out_channels, 1, padding=0, bias=True
        )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    c,
                    self.out_channels,
                    ksize=1,
                    num_convs=self.num_convs,
                    norm_str=self.norm,
                    padding=self.padding,
                    fusion=self.fusion,
                )
                for c in self.in_channels_list[::-1][1:]
            ]
        )
        self.out = nn.Sequential(
            ConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            ConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

        self.predict_uncertainty = conf.predict_uncertainty
        if self.predict_uncertainty:
            self.linear_pred_uncertainty = nn.Sequential(
                ConvModule(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.Conv2d(in_channels=self.out_channels, out_channels=1, kernel_size=1),
            )

        self.with_ll = conf.with_low_level
        if self.with_ll:
            self.out_conv = ConvModule(self.out_channels, self.out_channels, 3, padding=1)
            self.ll_fusion = FeatureFusionBlock(self.out_channels, upsample=False)

    def _forward(self, features):
        layers = features["hl"]
        feats = None

        for idx, x in enumerate(reversed(layers)):
            feats = self.first(x) if feats is None else self.blocks[idx - 1](feats, x)

        feats = self.out(feats)
        feats = F.interpolate(feats, scale_factor=2, mode="bilinear", align_corners=False)
        feats = self.out_conv(feats)

        if self.with_ll:
            assert "ll" in features, "Low-level features are required for this model"
            feats_ll = features["ll"].clone()
            feats = self.ll_fusion(feats, feats_ll)

        uncertainty = (
            self.linear_pred_uncertainty(feats).squeeze(1) if self.predict_uncertainty else None
        )
        return feats, uncertainty

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
