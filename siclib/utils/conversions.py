"""Utility functions for conversions between different representations."""

from typing import Optional

import torch


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3).

    Args:
        (torch.Tensor): Vector of size (..., 3).

    Returns:
        (torch.Tensor): Skew-symmetric matrix of size (..., 3, 3).
    """
    z = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            z,
            -v[..., 2],
            v[..., 1],
            v[..., 2],
            z,
            -v[..., 0],
            -v[..., 1],
            v[..., 0],
            z,
        ],
        dim=-1,
    ).reshape(v.shape[:-1] + (3, 3))


def rad2rotmat(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Convert (batched) roll, pitch, yaw angles (in radians) to rotation matrix.

    Args:
        roll (torch.Tensor): Roll angle in radians.
        pitch (torch.Tensor): Pitch angle in radians.
        yaw (torch.Tensor, optional): Yaw angle in radians. Defaults to None.

    Returns:
        torch.Tensor: Rotation matrix of shape (..., 3, 3).
    """
    if yaw is None:
        yaw = roll.new_zeros(roll.shape)

    Rx = pitch.new_zeros(pitch.shape + (3, 3))
    Rx[..., 0, 0] = 1
    Rx[..., 1, 1] = torch.cos(pitch)
    Rx[..., 1, 2] = torch.sin(pitch)
    Rx[..., 2, 1] = -torch.sin(pitch)
    Rx[..., 2, 2] = torch.cos(pitch)

    Ry = yaw.new_zeros(yaw.shape + (3, 3))
    Ry[..., 0, 0] = torch.cos(yaw)
    Ry[..., 0, 2] = -torch.sin(yaw)
    Ry[..., 1, 1] = 1
    Ry[..., 2, 0] = torch.sin(yaw)
    Ry[..., 2, 2] = torch.cos(yaw)

    Rz = roll.new_zeros(roll.shape + (3, 3))
    Rz[..., 0, 0] = torch.cos(roll)
    Rz[..., 0, 1] = torch.sin(roll)
    Rz[..., 1, 0] = -torch.sin(roll)
    Rz[..., 1, 1] = torch.cos(roll)
    Rz[..., 2, 2] = 1

    return Rz @ Rx @ Ry


def fov2focal(fov: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Compute focal length from (vertical/horizontal) field of view.

    Args:
        fov (torch.Tensor): Field of view in radians.
        size (torch.Tensor): Image height / width in pixels.

    Returns:
        torch.Tensor: Focal length in pixels.
    """
    return size / 2 / torch.tan(fov / 2)


def focal2fov(focal: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Compute (vertical/horizontal) field of view from focal length.

    Args:
        focal (torch.Tensor): Focal length in pixels.
        size (torch.Tensor): Image height / width in pixels.

    Returns:
        torch.Tensor: Field of view in radians.
    """
    return 2 * torch.arctan(size / (2 * focal))


def pitch2rho(pitch: torch.Tensor, f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Compute the distance from principal point to the horizon.

    Args:
        pitch (torch.Tensor): Pitch angle in radians.
        f (torch.Tensor): Focal length in pixels.
        h (torch.Tensor): Image height in pixels.

    Returns:
        torch.Tensor: Relative distance to the horizon.
    """
    return torch.tan(pitch) * f / h


def rho2pitch(rho: torch.Tensor, f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Compute the pitch angle from the distance to the horizon.

    Args:
        rho (torch.Tensor): Relative distance to the horizon.
        f (torch.Tensor): Focal length in pixels.
        h (torch.Tensor): Image height in pixels.

    Returns:
        torch.Tensor: Pitch angle in radians.
    """
    return torch.atan(rho * h / f)


def rad2deg(rad: torch.Tensor) -> torch.Tensor:
    """Convert radians to degrees.

    Args:
        rad (torch.Tensor): Angle in radians.

    Returns:
        torch.Tensor: Angle in degrees.
    """
    return rad / torch.pi * 180


def deg2rad(deg: torch.Tensor) -> torch.Tensor:
    """Convert degrees to radians.

    Args:
        deg (torch.Tensor): Angle in degrees.

    Returns:
        torch.Tensor: Angle in radians.
    """
    return deg / 180 * torch.pi
