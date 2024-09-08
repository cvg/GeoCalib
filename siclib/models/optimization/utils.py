import logging
from typing import Dict

import torch

from siclib.geometry.base_camera import BaseCamera
from siclib.geometry.gravity import Gravity
from siclib.utils.conversions import deg2rad, focal2fov

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


def get_initial_estimation(
    data: Dict[str, torch.Tensor], camera_model: BaseCamera, trivial_init: bool = True
) -> BaseCamera:
    """Get initial camera for optimization using heuristics."""
    return (
        get_trivial_estimation(data, camera_model)
        if trivial_init
        else get_heuristic_estimation(data, camera_model)
    )


def get_heuristic_estimation(data: Dict[str, torch.Tensor], camera_model: BaseCamera) -> BaseCamera:
    """Get initial camera for optimization using heuristics.

    Initial camera is initialized with the following heuristics:
    - roll is the angle of the up vector at the center of the image
    - pitch is the value at the center of the latitude map
    - vfov is the difference between the central top and bottom of the latitude map
    - distortions are set to zero

    Use the prior values if available.

    Args:
        data (Dict[str, torch.Tensor]): Input data dictionary.
        camera_model (BaseCamera): Camera model to use.

    Returns:
        BaseCamera: Initial camera for optimization.
    """
    up_ref = data["up_field"].detach()
    latitude_ref = data["latitude_field"].detach()

    h, w = up_ref.shape[-2:]
    batch_h, batch_w = (
        up_ref.new_ones((up_ref.shape[0],)) * h,
        up_ref.new_ones((up_ref.shape[0],)) * w,
    )

    # init roll is angle of the up vector at the center of the image
    init_r = -torch.atan2(
        up_ref[:, 0, int(h / 2), int(w / 2)], -up_ref[:, 1, int(h / 2), int(w / 2)]
    )
    init_r = init_r.clamp(min=-deg2rad(45), max=deg2rad(45))

    # init pitch is the value at the center of the latitude map
    init_p = latitude_ref[:, 0, int(h / 2), int(w / 2)]
    init_p = init_p.clamp(min=-deg2rad(45), max=deg2rad(45))

    # init vfov is the difference between the central top and bottom of the latitude map
    init_vfov = latitude_ref[:, 0, 0, int(w / 2)] - latitude_ref[:, 0, -1, int(w / 2)]
    init_vfov = torch.abs(init_vfov)
    init_vfov = init_vfov.clamp(min=deg2rad(20), max=deg2rad(120))

    focal = data.get("prior_focal")
    init_vfov = init_vfov if focal is None else focal2fov(focal, h)

    params = {"width": batch_w, "height": batch_h, "vfov": init_vfov}
    params |= {"scales": data["scales"]} if "scales" in data else {}
    params |= {"k1": data["prior_k1"]} if "prior_k1" in data else {}
    camera = camera_model.from_dict(params)
    camera = camera.float().to(data["up_field"].device)

    gravity = Gravity.from_rp(init_r, init_p).float().to(data["up_field"].device)
    if "prior_gravity" in data:
        gravity = data["prior_gravity"].float().to(up_ref.device)

    return camera, gravity


def get_trivial_estimation(data: Dict[str, torch.Tensor], camera_model: BaseCamera) -> BaseCamera:
    """Get initial camera for optimization with roll=0, pitch=0, vfov=0.7 * max(h, w).

    Args:
        data (Dict[str, torch.Tensor]): Input data dictionary.
        camera_model (BaseCamera): Camera model to use.

    Returns:
        BaseCamera: Initial camera for optimization.
    """
    """Get initial camera for optimization with roll=0, pitch=0, vfov=0.7 * max(h, w)."""
    ref = data.get("up_field", data["latitude_field"])
    ref = ref.detach()

    h, w = ref.shape[-2:]
    batch_h, batch_w = (
        ref.new_ones((ref.shape[0],)) * h,
        ref.new_ones((ref.shape[0],)) * w,
    )

    init_r = ref.new_zeros((ref.shape[0],))
    init_p = ref.new_zeros((ref.shape[0],))

    focal = data.get("prior_focal", 0.7 * torch.max(batch_h, batch_w))
    init_vfov = init_vfov if focal is None else focal2fov(focal, h)

    params = {"width": batch_w, "height": batch_h, "vfov": init_vfov}
    params |= {"scales": data["scales"]} if "scales" in data else {}
    params |= {"k1": data["prior_k1"]} if "prior_k1" in data else {}
    camera = camera_model.from_dict(params)
    camera = camera.float().to(ref.device)

    gravity = Gravity.from_rp(init_r, init_p).float().to(ref.device)

    if "prior_gravity" in data:
        gravity = data["prior_gravity"].float().to(ref.device)

    return camera, gravity


def early_stop(new_cost: torch.Tensor, prev_cost: torch.Tensor, atol: float, rtol: float) -> bool:
    """Early stopping criterion based on cost convergence."""
    return torch.allclose(new_cost, prev_cost, atol=atol, rtol=rtol)


def update_lambda(
    lamb: torch.Tensor,
    prev_cost: torch.Tensor,
    new_cost: torch.Tensor,
    lambda_min: float = 1e-6,
    lambda_max: float = 1e2,
) -> torch.Tensor:
    """Update damping factor for Levenberg-Marquardt optimization."""
    new_lamb = lamb.new_zeros(lamb.shape)
    new_lamb = lamb * torch.where(new_cost > prev_cost, 10, 0.1)
    lamb = torch.clamp(new_lamb, lambda_min, lambda_max)
    return lamb


def optimizer_step(
    G: torch.Tensor, H: torch.Tensor, lambda_: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """One optimization step with Gauss-Newton or Levenberg-Marquardt.

    Args:
        G (torch.Tensor): Batched gradient tensor of size (..., N).
        H (torch.Tensor): Batched hessian tensor of size (..., N, N).
        lambda_ (torch.Tensor): Damping factor for LM (use GN if lambda_=0) with shape (B,).
        eps (float, optional): Epsilon for damping. Defaults to 1e-6.

    Returns:
        torch.Tensor: Batched update tensor of size (..., N).
    """
    diag = H.diagonal(dim1=-2, dim2=-1)
    diag = diag * lambda_.unsqueeze(-1)  # (B, 3)

    H = H + diag.clamp(min=eps).diag_embed()

    H_, G_ = H.cpu(), G.cpu()
    try:
        U = torch.linalg.cholesky(H_)
    except RuntimeError:
        logger.warning("Cholesky decomposition failed. Stopping.")
        delta = H.new_zeros((H.shape[0], H.shape[-1]))  # (B, 3)
    else:
        delta = torch.cholesky_solve(G_[..., None], U)[..., 0]

    return delta.to(H.device)
