"""Jacobians for optimization."""

import torch

# flake8: noqa: E741


@torch.jit.script
def J_vecnorm(vec: torch.Tensor) -> torch.Tensor:
    """Compute the jacobian of vec / norm2(vec).

    Args:
        vec (torch.Tensor): [..., D] tensor.

    Returns:
        torch.Tensor: [..., D, D] Jacobian.
    """
    D = vec.shape[-1]
    norm_x = torch.norm(vec, dim=-1, keepdim=True).unsqueeze(-1)  # (..., 1, 1)

    if (norm_x == 0).any():
        norm_x = norm_x + 1e-6

    xxT = torch.einsum("...i,...j->...ij", vec, vec)  # (..., D, D)
    identity = torch.eye(D, device=vec.device, dtype=vec.dtype)  # (D, D)

    return identity / norm_x - (xxT / norm_x**3)  # (..., D, D)


@torch.jit.script
def J_focal2fov(focal: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Compute the jacobian of the focal2fov function."""
    return -4 * h / (4 * focal**2 + h**2)


@torch.jit.script
def J_up_projection(uv: torch.Tensor, abc: torch.Tensor, wrt: str = "uv") -> torch.Tensor:
    """Compute the jacobian of the up-vector projection.

    Args:
        uv (torch.Tensor): Normalized image coordinates of shape (..., 2).
        abc (torch.Tensor): Gravity vector of shape (..., 3).
        wrt (str, optional): Parameter to differentiate with respect to. Defaults to "uv".

    Raises:
        ValueError: If the wrt parameter is unknown.

    Returns:
        torch.Tensor: Jacobian with respect to the parameter.
    """
    if wrt == "uv":
        c = abc[..., 2][..., None, None, None]
        return -c * torch.eye(2, device=uv.device, dtype=uv.dtype).expand(uv.shape[:-1] + (2, 2))

    elif wrt == "abc":
        J = uv.new_zeros(uv.shape[:-1] + (2, 3))
        J[..., 0, 0] = 1
        J[..., 1, 1] = 1
        J[..., 0, 2] = -uv[..., 0]
        J[..., 1, 2] = -uv[..., 1]
        return J

    else:
        raise ValueError(f"Unknown wrt: {wrt}")
