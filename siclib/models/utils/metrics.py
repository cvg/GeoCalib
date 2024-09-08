"""Various metrics for evaluating predictions."""

import logging

import torch
from torch.nn import functional as F

from siclib.geometry.base_camera import BaseCamera
from siclib.geometry.gravity import Gravity
from siclib.utils.conversions import rad2deg

logger = logging.getLogger(__name__)


def pitch_error(pred_gravity: Gravity, target_gravity: Gravity) -> torch.Tensor:
    """Computes the pitch error between two gravities.

    Args:
        pred_gravity (Gravity): Predicted camera.
        target_gravity (Gravity): Ground truth camera.

    Returns:
        torch.Tensor: Pitch error in degrees.
    """
    return rad2deg(torch.abs(pred_gravity.pitch - target_gravity.pitch))


def roll_error(pred_gravity: Gravity, target_gravity: Gravity) -> torch.Tensor:
    """Computes the roll error between two gravities.

    Args:
        pred_gravity (Gravity): Predicted Gravity.
        target_gravity (Gravity): Ground truth Gravity.

    Returns:
        torch.Tensor: Roll error in degrees.
    """
    return rad2deg(torch.abs(pred_gravity.roll - target_gravity.roll))


def gravity_error(pred_gravity: Gravity, target_gravity: Gravity) -> torch.Tensor:
    """Computes the gravity error between two gravities.

    Args:
        pred_gravity (Gravity): Predicted Gravity.
        target_gravity (Gravity): Ground truth Gravity.

    Returns:
        torch.Tensor: Gravity error in degrees.
    """
    assert (
        pred_gravity.vec3d.shape == target_gravity.vec3d.shape
    ), f"{pred_gravity.vec3d.shape} != {target_gravity.vec3d.shape}"
    assert pred_gravity.vec3d.ndim == 2, f"{pred_gravity.vec3d.ndim} != 2"
    assert pred_gravity.vec3d.shape[1] == 3, f"{pred_gravity.vec3d.shape[1]} != 3"

    cossim = F.cosine_similarity(pred_gravity.vec3d, target_gravity.vec3d, dim=-1).clamp(-1, 1)
    return rad2deg(torch.acos(cossim))


def vfov_error(pred_cam: BaseCamera, target_cam: BaseCamera) -> torch.Tensor:
    """Computes the vertical field of view error between two cameras.

    Args:
        pred_cam (Camera): Predicted camera.
        target_cam (Camera): Ground truth camera.

    Returns:
        torch.Tensor: Vertical field of view error in degrees.
    """
    return rad2deg(torch.abs(pred_cam.vfov - target_cam.vfov))


def dist_error(pred_cam: BaseCamera, target_cam: BaseCamera) -> torch.Tensor:
    """Computes the distortion parameter error between two cameras.

    Returns zero if the cameras do not have distortion parameters.

    Args:
        pred_cam (Camera): Predicted camera.
        target_cam (Camera): Ground truth camera.

    Returns:
        torch.Tensor: distortion error.
    """
    if hasattr(pred_cam, "dist") and hasattr(target_cam, "dist"):
        return torch.abs(pred_cam.dist[..., 0] - target_cam.dist[..., 0])

    logger.debug(
        f"Predicted / target camera doesn't have distortion parameters: {pred_cam}/{target_cam}"
    )
    return pred_cam.new_zeros(pred_cam.f.shape[0])


def latitude_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes the latitude error between two tensors.

    Args:
        predictions (torch.Tensor): Predicted latitude field of shape (B, 1, H, W).
        targets (torch.Tensor): Ground truth latitude field of shape (B, 1, H, W).

    Returns:
        torch.Tensor: Latitude error in degrees of shape (B, H, W).
    """
    return rad2deg(torch.abs(predictions - targets)).squeeze(1)


def up_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes the up error between two tensors.

    Args:
        predictions (torch.Tensor): Predicted up field of shape (B, 2, H, W).
        targets (torch.Tensor): Ground truth up field of shape (B, 2, H, W).

    Returns:
        torch.Tensor: Up error in degrees of shape (B, H, W).
    """
    assert predictions.shape == targets.shape, f"{predictions.shape} != {targets.shape}"
    assert predictions.ndim == 4, f"{predictions.ndim} != 4"
    assert predictions.shape[1] == 2, f"{predictions.shape[1]} != 2"

    angle = F.cosine_similarity(predictions, targets, dim=1).clamp(-1, 1)
    return rad2deg(torch.acos(angle))
