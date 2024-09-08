"""Light HamHead Decoder.

Adapted from:
https://github.com/Visual-Attention-Network/SegNeXt/blob/main/mmseg/models/decode_heads/ham_head.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from siclib.models import BaseModel
from siclib.models.utils.modules import ConvModule, FeatureFusionBlock

# flake8: noqa
# mypy: ignore-errors


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial = True

        self.S = 1
        self.D = 512
        self.R = 64

        self.train_steps = 6
        self.eval_steps = 7

        self.inv_t = 100
        self.eta = 0.9

        self.rand_init = True

    def _build_bases(self, B, S, D, R, device="cpu"):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, "bases"):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer("bases", bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W) if self.spatial else x.transpose(1, 2).view(B, C, H, W)
        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self):
        super().__init__()

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device="cpu"):
        bases = torch.rand((B * S, D, R)).to(device)
        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    def __init__(self, ham_channels=512, norm_cfg=None, **kwargs):
        super().__init__()

        self.ham_in = ConvModule(ham_channels, ham_channels, 1)

        self.ham = NMF2D()

        self.ham_out = ConvModule(ham_channels, ham_channels, 1)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=False)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=False)

        return ham


class LightHamHead(BaseModel):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.

    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.
    """

    default_conf = {
        "predict_uncertainty": True,
        "out_channels": 64,
        "in_channels": [64, 128, 320, 512],
        "in_index": [0, 1, 2, 3],
        "ham_channels": 512,
        "with_low_level": True,
    }

    def _init(self, conf):
        self.in_index = conf.in_index
        self.in_channels = conf.in_channels
        self.out_channels = conf.out_channels
        self.ham_channels = conf.ham_channels
        self.align_corners = False
        self.predict_uncertainty = conf.predict_uncertainty

        self.squeeze = ConvModule(sum(self.in_channels), self.ham_channels, 1)

        self.hamburger = Hamburger(self.ham_channels)

        self.align = ConvModule(self.ham_channels, self.out_channels, 1)

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
            self.out_conv = ConvModule(
                self.out_channels, self.out_channels, 3, padding=1, bias=False
            )
            self.ll_fusion = FeatureFusionBlock(self.out_channels, upsample=False)

    def _forward(self, features):
        """Forward function."""
        # inputs = self._transform_inputs(inputs)
        inputs = [features["hl"][i] for i in self.in_index]

        inputs = [
            F.interpolate(
                level, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
            for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        feats = self.align(x)

        if self.with_ll:
            assert "ll" in features, "Low-level features are required for this model"
            feats = F.interpolate(feats, scale_factor=2, mode="bilinear", align_corners=False)
            feats = self.out_conv(feats)
            feats = F.interpolate(feats, scale_factor=2, mode="bilinear", align_corners=False)
            feats_ll = features["ll"].clone()
            feats = self.ll_fusion(feats, feats_ll)

        uncertainty = (
            self.linear_pred_uncertainty(feats).squeeze(1) if self.predict_uncertainty else None
        )

        return feats, uncertainty

    def loss(self, pred, data):
        raise NotImplementedError
