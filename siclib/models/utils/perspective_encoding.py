"""Perspective field utilities.

Adapted from https://github.com/jinlinyi/PerspectiveFields
"""

import torch

from siclib.utils.conversions import deg2rad, rad2deg


def encode_up_bin(vector_field: torch.Tensor, num_bin: int) -> torch.Tensor:
    """Encode vector field into classification bins.

    Args:
        vector_field (torch.Tensor): gravity field of shape (2, h, w), with channel 0 cos(theta) and
        1 sin(theta)
        num_bin (int): number of classification bins

    Returns:
        torch.Tensor: encoded bin indices of shape (1, h, w)
    """
    angle = (
        torch.atan2(vector_field[1, :, :], vector_field[0, :, :]) / torch.pi * 180 + 180
    ) % 360  # [0,360)
    angle_bin = torch.round(torch.div(angle, (360 / (num_bin - 1)))).long()
    angle_bin[angle_bin == num_bin - 1] = 0
    invalid = (vector_field == 0).sum(0) == vector_field.size(0)
    angle_bin[invalid] = num_bin - 1
    return deg2rad(angle_bin.type(torch.LongTensor))


def decode_up_bin(angle_bin: torch.Tensor, num_bin: int) -> torch.Tensor:
    """Decode classification bins into vector field.

    Args:
        angle_bin (torch.Tensor): bin indices of shape (1, h, w)
        num_bin (int): number of classification bins

    Returns:
        torch.Tensor: decoded vector field of shape (2, h, w)
    """
    angle = (angle_bin * (360 / (num_bin - 1)) - 180) / 180 * torch.pi
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    vector_field = torch.stack((cos, sin), dim=1)
    invalid = angle_bin == num_bin - 1
    invalid = invalid.unsqueeze(1).repeat(1, 2, 1, 1)
    vector_field[invalid] = 0
    return vector_field


def encode_bin_latitude(latimap: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Encode latitude map into classification bins.

    Args:
        latimap (torch.Tensor): latitude map of shape (h, w) with values in [-90, 90]
        num_classes (int): number of classes

    Returns:
        torch.Tensor: encoded latitude bin indices
    """
    boundaries = torch.arange(-90, 90, 180 / num_classes)[1:]
    binmap = torch.bucketize(rad2deg(latimap), boundaries)
    return binmap.type(torch.LongTensor)


def decode_bin_latitude(binmap: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Decode classification bins to latitude map.

    Args:
        binmap (torch.Tensor): encoded classification bins
        num_classes (int): number of classes

    Returns:
        torch.Tensor: latitude map of shape (h, w)
    """
    bin_size = 180 / num_classes
    bin_centers = torch.arange(-90, 90, bin_size) + bin_size / 2
    bin_centers = bin_centers.to(binmap.device)
    latimap = bin_centers[binmap]

    return deg2rad(latimap)
