"""Implementation of the pinhole, simple radial, and simple divisional camera models."""

from typing import Tuple

import torch

from siclib.geometry.base_camera import BaseCamera
from siclib.utils.tensor import autocast

# flake8: noqa: E741

# mypy: ignore-errors


class Pinhole(BaseCamera):
    """Implementation of the pinhole camera model.

    Use this model for undistorted images.
    """

    @classmethod
    def name(cls) -> str:
        return "pinhole"

    def distort(self, p2d: torch.Tensor, return_scale: bool = False) -> Tuple[torch.Tensor]:
        """Distort normalized 2D coordinates."""
        if return_scale:
            return p2d.new_ones(p2d.shape[:-1] + (1,))

        return p2d, p2d.new_ones((p2d.shape[0], 1)).bool()

    def J_distort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the distortion function."""
        if wrt == "pts":
            return torch.eye(2, device=p2d.device, dtype=p2d.dtype).expand(p2d.shape[:-1] + (2, 2))

        raise ValueError(f"Unknown wrt: {wrt}")

    def undistort(self, pts: torch.Tensor) -> Tuple[torch.Tensor]:
        """Undistort normalized 2D coordinates."""
        return pts, pts.new_ones((pts.shape[0], 1)).bool()

    def J_undistort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the undistortion function."""
        if wrt == "pts":
            return torch.eye(2, device=p2d.device, dtype=p2d.dtype).expand(p2d.shape[:-1] + (2, 2))

        raise ValueError(f"Unknown wrt: {wrt}")

    def J_up_projection_offset(self, p2d: torch.Tensor, wrt: str = "uv") -> torch.Tensor:
        """Jacobian of the up-projection offset."""
        if wrt == "uv":
            return torch.zeros(p2d.shape[:-1] + (2, 2), device=p2d.device, dtype=p2d.dtype)

        raise ValueError(f"Unknown wrt: {wrt}")


class SimpleRadial(BaseCamera):
    """Implementation of the simple radial camera model.

    Use this model for weakly distorted images.

    The distortion model is 1 + k1 * r^2 where r^2 = x^2 + y^2.
    The undistortion model is 1 - k1 * r^2 estimated as in
    "An Exact Formula for Calculating Inverse Radial Lens Distortions" by Pierre Drap.
    """

    @classmethod
    def name(cls) -> str:
        return "simple_radial"

    @property
    def dist(self) -> torch.Tensor:
        """Distortion parameters, with shape (..., 1)."""
        return self._data[..., 6:]

    @classmethod
    def num_dist_params(cls) -> int:
        """Number of distortion parameters."""
        return 1

    @property
    def k1(self) -> torch.Tensor:
        """Distortion parameters, with shape (...)."""
        return self._data[..., 6]

    def update_dist(self, delta: torch.Tensor, dist_range: Tuple[float, float] = (-0.7, 0.7)):
        """Update the self parameters after changing the k1 distortion parameter."""
        delta_dist = self.new_ones(self.dist.shape) * delta
        dist = (self.dist + delta_dist).clamp(*dist_range)
        data = torch.cat([self.size, self.f, self.c, dist], -1)
        return self.__class__(data)

    @autocast
    def check_valid(self, p2d: torch.Tensor) -> torch.Tensor:
        """Check if the distorted points are valid."""
        return p2d.new_ones(p2d.shape[:-1]).bool()

    def distort(self, p2d: torch.Tensor, return_scale: bool = False) -> Tuple[torch.Tensor]:
        """Distort normalized 2D coordinates and check for validity of the distortion model."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        radial = 1 + self.k1[..., None, None] * r2

        if return_scale:
            return radial, None

        return p2d * radial, self.check_valid(p2d)

    def J_distort(self, p2d: torch.Tensor, wrt: str = "pts"):
        """Jacobian of the distortion function."""
        if wrt == "scale2dist":  # (..., 1)
            return torch.sum(p2d**2, -1, keepdim=True)
        elif wrt == "scale2pts":  # (..., 2)
            return 2 * self.k1[..., None, None] * p2d
        else:
            return super().J_distort(p2d, wrt)

    @autocast
    def undistort(self, p2d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Undistort normalized 2D coordinates and check for validity of the distortion model."""
        b1 = -self.k1[..., None, None]
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        radial = 1 + b1 * r2
        return p2d * radial, self.check_valid(p2d)

    @autocast
    def J_undistort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the undistortion function."""
        b1 = -self.k1[..., None, None]
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        if wrt == "dist":
            return (-r2 * p2d)[..., None]  # (..., 2, K)
        elif wrt == "pts":
            radial = 1 + b1 * r2
            radial_diag = torch.diag_embed(radial.expand(radial.shape[:-1] + (2,)))
            ppT = torch.einsum("...i,...j->...ij", p2d, p2d)  # (..., 2, 2)
            return (2 * b1[..., None] * ppT) + radial_diag
        else:
            return super().J_undistort(p2d, wrt)

    def J_up_projection_offset(self, p2d: torch.Tensor, wrt: str = "uv") -> torch.Tensor:
        """Jacobian of the up-projection offset."""
        if wrt == "uv":  # (..., 2, 2)
            return torch.diag_embed((2 * self.k1[..., None, None]).expand(p2d.shape[:-1] + (2,)))
        elif wrt == "dist":
            return (2 * p2d)[..., None]  # (..., 2)
        else:
            return super().J_up_projection_offset(p2d, wrt)


class Radial(BaseCamera):
    """Implementation of the radial camera model.

    Use this model for distorted images.

    The distortion model is 1 + k1 * r^2 + k2 * r^4 where r^2 = x^2 + y^2.
    The undistortion model is 1 - k1 * r^2 + (3 * k1^2 - k2) * r^4 estimated as in
    "An Exact Formula for Calculating Inverse Radial Lens Distortions" by Pierre Drap.
    """

    @classmethod
    def name(cls) -> str:
        return "radial"

    @property
    def dist(self) -> torch.Tensor:
        """
        Distortion parameters, with shape (..., 2).
        Camera parameters with shape (..., {w, h, fx, fy, cx, cy, k1, k2}).
        """
        return self._data[..., 6:8]

    @classmethod
    def num_dist_params(cls) -> int:
        """Number of distortion parameters."""
        return 2

    @property
    def k1(self) -> torch.Tensor:
        """Distortion parameters, with shape (...)."""
        return self._data[..., 6]

    @property
    def k2(self) -> torch.Tensor:
        """Distortion parameters, with shape (...)."""
        return self._data[..., 7]

    def update_dist(self, delta: torch.Tensor, dist_range: Tuple[float, float] = (-0.7, 0.7)):
        """Update the self parameters after changing the k1, k2 distortion parameter."""
        delta_dist = self.new_ones(self.dist.shape) * delta
        dist = (self.dist + delta_dist).clamp(*dist_range)
        data = torch.cat([self.size, self.f, self.c, dist], -1)
        return self.__class__(data)

    @autocast
    def check_valid(self, p2d: torch.Tensor) -> torch.Tensor:
        """Check if the distorted points are valid."""
        return p2d.new_ones(p2d.shape[:-1]).bool()

    def distort(self, p2d: torch.Tensor, return_scale: bool = False) -> Tuple[torch.Tensor]:
        """Distort normalized 2D coordinates and check for validity of the distortion model."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        r4 = r2**2
        radial = 1 + self.k1[..., None, None] * r2 + self.k2[..., None, None] * r4

        if return_scale:
            return radial, None

        return p2d * radial, self.check_valid(p2d)

    def J_distort(self, p2d: torch.Tensor, wrt: str = "pts"):
        """Jacobian of the distortion function."""
        if wrt == "scale2dist":  # (..., 2)
            r2 = torch.sum(p2d**2, -1, keepdim=True)  # (..., 1)
            r4 = r2**2  # (..., 1)
            return torch.cat([r2, r4], -1)  # (..., 2)
        elif wrt == "scale2pts":  # (..., 2)
            r2 = torch.sum(p2d**2, -1, keepdim=True)
            k1, k2 = self.k1[..., None, None], self.k2[..., None, None]
            return (2 * k1 + 4 * k2 * r2) * p2d  # (..., 2)
        else:
            return super().J_distort(p2d, wrt)

    @autocast
    def undistort(self, p2d: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Undistort normalized 2D coordinates and check for validity of the distortion model.
        d = 1 - k1 * r^2 + (3 * k1^2 - k2) * r^4
        """
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        k1, k2 = self.k1[..., None, None], self.k2[..., None, None]
        b1, b2 = -k1, 3 * k1**2 - k2
        radial = 1 + b1 * r2 + b2 * r2**2
        return p2d * radial, self.check_valid(p2d)

    @autocast
    def J_undistort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the undistortion function."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        r4 = r2**2

        k1, k2 = self.k1[..., None, None], self.k2[..., None, None]

        if wrt == "dist":
            J_k1 = (6 * r4 * k1 - r2) * p2d
            J_k2 = -r4 * p2d
            return torch.stack([J_k1, J_k2], -1)
        elif wrt == "pts":  # (..., 2, K)
            b1, b2 = -k1, 3 * k1**2 - k2

            uv_uvT = torch.einsum("...i,...j->...ij", p2d, p2d)  # (..., 2, 2)
            J = (4 * r2 * b2 + 2 * b1)[..., None] * uv_uvT

            I = torch.eye(2, device=p2d.device, dtype=p2d.dtype).expand(p2d.shape[:-1] + (2, 2))
            return J + (1 + b1 * r2 + b2 * r4)[..., None] * I

        else:
            return super().J_undistort(p2d, wrt)

    def J_up_projection_offset(self, p2d: torch.Tensor, wrt: str = "uv") -> torch.Tensor:
        """Jacobian of the up-projection offset.
        up-projection offset := 2*(k1 + 2k2*r^2) [u v] (Eq 11 in paper)
        """
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        k1, k2 = self.k1[..., None, None], self.k2[..., None, None]

        if wrt == "uv":
            uv_uvT = torch.einsum("...i,...j->...ij", p2d, p2d)  # (..., 2, 2)
            I = torch.eye(2, device=p2d.device, dtype=p2d.dtype).expand(p2d.shape[:-1] + (2, 2))
            return 8 * k2[..., None] * uv_uvT + (2 * k1 + 4 * k2 * r2)[..., None] * I
        elif wrt == "dist":
            return torch.stack([2 * p2d, 4 * r2 * p2d], -1)  # (..., 2, K)

        return super().J_up_projection_offset(p2d, wrt)


class SimpleDivisional(BaseCamera):
    """Implementation of the simple divisional camera model.

    Use this model for strongly distorted images.

    The distortion model is (1 - sqrt(1 - 4 * k1 * r^2)) / (2 * k1 * r^2) where r^2 = x^2 + y^2.
    The undistortion model is 1 / (1 + k1 * r^2).
    """

    @classmethod
    def name(cls) -> str:
        return "simple_divisional"

    @property
    def dist(self) -> torch.Tensor:
        """Distortion parameters, with shape (..., 1)."""
        return self._data[..., 6:]

    @classmethod
    def num_dist_params(cls) -> int:
        """Number of distortion parameters."""
        return 1

    @property
    def k1(self) -> torch.Tensor:
        """Distortion parameters, with shape (...)."""
        return self._data[..., 6]

    def update_dist(self, delta: torch.Tensor, dist_range: Tuple[float, float] = (-3.0, 3.0)):
        """Update the self parameters after changing the k1 distortion parameter."""
        delta_dist = self.new_ones(self.dist.shape) * delta
        dist = (self.dist + delta_dist).clamp(*dist_range)
        data = torch.cat([self.size, self.f, self.c, dist], -1)
        return self.__class__(data)

    @autocast
    def check_valid(self, p2d: torch.Tensor) -> torch.Tensor:
        """Check if the distorted points are valid."""
        return p2d.new_ones(p2d.shape[:-1]).bool()

    def distort(self, p2d: torch.Tensor, return_scale: bool = False) -> Tuple[torch.Tensor]:
        """Distort normalized 2D coordinates and check for validity of the distortion model."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        radial = 1 - torch.sqrt((1 - 4 * self.k1[..., None, None] * r2).clamp(min=0))
        denom = 2 * self.k1[..., None, None] * r2

        ones = radial.new_ones(radial.shape)
        radial = torch.where(denom == 0, ones, radial / denom.masked_fill(denom == 0, 1e6))

        if return_scale:
            return radial, None

        return p2d * radial, self.check_valid(p2d)

    def J_distort(self, p2d: torch.Tensor, wrt: str = "pts"):
        """Jacobian of the distortion function."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        t0 = torch.sqrt((1 - 4 * self.k1[..., None, None] * r2).clamp(min=1e-6))
        if wrt == "scale2pts":  # (B, N, 2)
            d1 = t0 * 2 * r2
            d2 = self.k1[..., None, None] * r2**2
            denom = d1 * d2
            return p2d * (4 * d2 - (1 - t0) * d1) / denom.masked_fill(denom == 0, 1e6)

        elif wrt == "scale2dist":
            d1 = 2 * self.k1[..., None, None] * t0
            d2 = 2 * r2 * self.k1[..., None, None] ** 2
            denom = d1 * d2
            return (2 * d2 - (1 - t0) * d1) / denom.masked_fill(denom == 0, 1e6)

        else:
            return super().J_distort(p2d, wrt)

    @autocast
    def undistort(self, p2d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Undistort normalized 2D coordinates and check for validity of the distortion model."""
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        denom = 1 + self.k1[..., None, None] * r2
        radial = 1 / denom.masked_fill(denom == 0, 1e6)
        return p2d * radial, self.check_valid(p2d)

    def J_undistort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the undistortion function."""
        # return super().J_undistort(p2d, wrt)
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        k1 = self.k1[..., None, None]
        if wrt == "dist":
            denom = (1 + k1 * r2) ** 2
            return (-r2 / denom.masked_fill(denom == 0, 1e6) * p2d)[..., None]  # (..., 2, K)
        elif wrt == "pts":
            t0 = 1 + k1 * r2
            t0 = t0.masked_fill(t0 == 0, 1e6)
            ppT = torch.einsum("...i,...j->...ij", p2d, p2d)  # (..., 2, 2)
            J = torch.diag_embed((1 / t0).expand(p2d.shape[:-1] + (2,)))
            return J - 2 * k1[..., None] * ppT / t0[..., None] ** 2  # (..., N, 2, 2)

        else:
            return super().J_undistort(p2d, wrt)

    def J_up_projection_offset(self, p2d: torch.Tensor, wrt: str = "uv") -> torch.Tensor:
        """Jacobian of the up-projection offset.

        func(uv, dist) = 4 / (2 * norm2(uv)^2 * (1-4*k1*norm2(uv)^2)^0.5) * uv
        - (1-(1-4*k1*norm2(uv)^2)^0.5) / (k1 * norm2(uv)^4) * uv
        """
        k1 = self.k1[..., None, None]
        r2 = torch.sum(p2d**2, -1, keepdim=True)
        t0 = (1 - 4 * k1 * r2).clamp(min=1e-6)
        t1 = torch.sqrt(t0)
        if wrt == "dist":
            denom = 4 * t0 ** (3 / 2)
            denom = denom.masked_fill(denom == 0, 1e6)
            J = 16 / denom

            denom = r2 * t1 * k1
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J - 2 / denom

            denom = (r2 * k1) ** 2
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J + (1 - t1) / denom

            return (J * p2d)[..., None]  # (..., 2, K)
        elif wrt == "uv":
            # ! unstable (gradient checker might fail), rewrite to use single division (by denom)
            ppT = torch.einsum("...i,...j->...ij", p2d, p2d)  # (..., 2, 2)

            denom = 2 * r2 * t1
            denom = denom.masked_fill(denom == 0, 1e6)
            J = torch.diag_embed((4 / denom).expand(p2d.shape[:-1] + (2,)))

            denom = 4 * t1 * r2**2
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J - 16 / denom[..., None] * ppT

            denom = 4 * r2 * t0 ** (3 / 2)
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J + (32 * k1[..., None]) / denom[..., None] * ppT

            denom = r2**2 * t1
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J - 4 / denom[..., None] * ppT

            denom = k1 * r2**3
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J + (4 * (1 - t1) / denom)[..., None] * ppT

            denom = k1 * r2**2
            denom = denom.masked_fill(denom == 0, 1e6)
            J = J - torch.diag_embed(((1 - t1) / denom).expand(p2d.shape[:-1] + (2,)))

            return J
        else:
            return super().J_up_projection_offset(p2d, wrt)


camera_models = {
    "pinhole": Pinhole,
    "radial": Radial,
    "simple_radial": SimpleRadial,
    "simple_divisional": SimpleDivisional,
}
