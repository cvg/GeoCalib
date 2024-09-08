"""Implementation of MSCAN from SegNeXt: Rethinking Convolutional Attention Design for Semantic 
Segmentation (NeurIPS 2022)

based on: https://github.com/Visual-Attention-Network/SegNeXt
"""

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple

from siclib.models import BaseModel
from siclib.models.utils.modules import DropPath, DWConv

# flake8: noqa
# mypy: ignore-errors


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        """Initialize the MLP."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward pass."""
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(
                out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        )
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


class MSCAN(BaseModel):
    default_conf = {
        "in_channels": 3,
        "embed_dims": [64, 128, 320, 512],
        "mlp_ratios": [8, 8, 4, 4],
        "drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "depths": [3, 3, 12, 3],
        "num_stages": 4,
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        self.depths = conf.depths
        self.num_stages = conf.num_stages

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, conf.drop_path_rate, sum(conf.depths))]
        cur = 0

        for i in range(conf.num_stages):
            if i == 0:
                patch_embed = StemConv(3, conf.embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_chans=conf.in_chans if i == 0 else conf.embed_dims[i - 1],
                    embed_dim=conf.embed_dims[i],
                )

            block = nn.ModuleList(
                [
                    Block(
                        dim=conf.embed_dims[i],
                        mlp_ratio=conf.mlp_ratios[i],
                        drop=conf.drop_rate,
                        drop_path=dpr[cur + j],
                    )
                    for j in range(conf.depths[i])
                ]
            )
            norm = nn.LayerNorm(conf.embed_dims[i])
            cur += conf.depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def _forward(self, data):
        img = data["image"]
        # rgb -> bgr and from [0, 1] to [0, 255]
        x = img[:, [2, 1, 0], :, :] * 255.0

        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return {"features": outs}

    def loss(self, pred, data):
        """Compute the loss."""
        raise NotImplementedError
