# Adapted from PixLoc, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/cvg/pixloc
# Released under the Apache License 2.0

"""Convenience classes a for camera models.

Based on PyTorch tensors: differentiable, batched, with GPU support.
"""

from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
from torch.func import jacfwd, vmap
from torch.nn import functional as F

from siclib.geometry.gravity import Gravity
from siclib.utils.conversions import deg2rad, focal2fov, fov2focal, rad2rotmat
from siclib.utils.tensor import TensorWrapper, autocast

# mypy: ignore-errors


class BaseCamera(TensorWrapper):
    """Camera tensor class."""

    eps = 1e-3

    @autocast
    def __init__(self, data: torch.Tensor):
        """Camera parameters with shape (..., {w, h, fx, fy, cx, cy, *dist}).

        Tensor convention: (..., {w, h, fx, fy, cx, cy, pitch, roll, *dist}) where
        - w, h: image size in pixels
        - fx, fy: focal lengths in pixels
        - cx, cy: principal points in normalized image coordinates
        - dist: distortion parameters

        Args:
            data (torch.Tensor): Camera parameters with shape (..., {6, 7, 8}).
        """
        # w, h, fx, fy, cx, cy, dist
        assert data.shape[-1] in {6, 7, 8}, data.shape

        pad = data.new_zeros(data.shape[:-1] + (8 - data.shape[-1],))
        data = torch.cat([data, pad], -1) if data.shape[-1] != 8 else data
        super().__init__(data)

    @classmethod
    def from_dict(cls, param_dict: Dict[str, torch.Tensor]) -> "BaseCamera":
        """Create a Camera object from a dictionary of parameters.

        Args:
            param_dict (Dict[str, torch.Tensor]): Dictionary of parameters.

        Returns:
            Camera: Camera object.
        """
        for key, value in param_dict.items():
            if not isinstance(value, torch.Tensor):
                param_dict[key] = torch.tensor(value)

        h, w = param_dict["height"], param_dict["width"]
        cx, cy = param_dict.get("cx", w / 2), param_dict.get("cy", h / 2)

        vfov = param_dict.get("vfov")

        f = param_dict.get("f") if vfov is None else fov2focal(vfov, h)
        if "dist" in param_dict:
            k1 = param_dict["dist"][..., (0,)]
            k2 = (
                param_dict["dist"][..., (1,)]
                if param_dict["dist"].shape[-1] == 2
                else torch.zeros_like(k1)
            )
        elif "k1_hat" in param_dict:
            k1 = param_dict["k1_hat"] * (f / h) ** 2

            k2 = param_dict.get("k2", torch.zeros_like(k1))
        else:
            k1 = param_dict.get("k1", torch.zeros_like(f))
            k2 = param_dict.get("k2", torch.zeros_like(f))

        fx, fy = f, f
        if "scales" in param_dict:
            scales = param_dict["scales"]
            fx = fx * scales[..., 0] / scales[..., 1]

        params = torch.stack([w, h, fx, fy, cx, cy, k1, k2], dim=-1)
        return cls(params)

    def pinhole(self):
        """Return the pinhole camera model."""
        return self.__class__(self._data[..., :6])

    @property
    def size(self) -> torch.Tensor:
        """Size (width height) of the images, with shape (..., 2)."""
        return self._data[..., :2]

    @property
    def f(self) -> torch.Tensor:
        """Focal lengths (fx, fy) with shape (..., 2)."""
        return self._data[..., 2:4]

    @property
    def vfov(self) -> torch.Tensor:
        """Vertical field of view in radians."""
        return focal2fov(self.f[..., 1], self.size[..., 1])

    @property
    def hfov(self) -> torch.Tensor:
        """Horizontal field of view in radians."""
        return focal2fov(self.f[..., 0], self.size[..., 0])

    @property
    def c(self) -> torch.Tensor:
        """Principal points (cx, cy) with shape (..., 2)."""
        return self._data[..., 4:6]

    @property
    def K(self) -> torch.Tensor:
        """Returns the self intrinsic matrix with shape (..., 3, 3)."""
        shape = self.shape + (3, 3)
        K = self._data.new_zeros(shape)
        K[..., 0, 0] = self.f[..., 0]
        K[..., 1, 1] = self.f[..., 1]
        K[..., 0, 2] = self.c[..., 0]
        K[..., 1, 2] = self.c[..., 1]
        K[..., 2, 2] = 1
        return K

    def update_focal(self, delta: torch.Tensor, as_log: bool = False):
        """Update the self parameters after changing the focal length."""
        f = torch.exp(torch.log(self.f) + delta) if as_log else self.f + delta

        # clamp focal length to a reasonable range for stability during training
        min_f = fov2focal(self.new_ones(self.shape[0]) * deg2rad(150), self.size[..., 1])
        max_f = fov2focal(self.new_ones(self.shape[0]) * deg2rad(5), self.size[..., 1])
        min_f = min_f.unsqueeze(-1).expand(-1, 2)
        max_f = max_f.unsqueeze(-1).expand(-1, 2)
        f = f.clamp(min=min_f, max=max_f)

        # make sure focal ration stays the same (avoid inplace operations)
        fx = f[..., 1] * self.f[..., 0] / self.f[..., 1]
        f = torch.stack([fx, f[..., 1]], -1)

        dist = self.dist if hasattr(self, "dist") else self.new_zeros(self.f.shape)
        return self.__class__(torch.cat([self.size, f, self.c, dist], -1))

    def scale(self, scales: Union[float, int, Tuple[Union[float, int]]]):
        """Update the self parameters after resizing an image."""
        scales = (scales, scales) if isinstance(scales, (int, float)) else scales
        s = scales if isinstance(scales, torch.Tensor) else self.new_tensor(scales)

        dist = self.dist if hasattr(self, "dist") else self.new_zeros(self.f.shape)
        return self.__class__(torch.cat([self.size * s, self.f * s, self.c * s, dist], -1))

    def crop(self, pad: Tuple[float]):
        """Update the self parameters after cropping an image."""
        pad = pad if isinstance(pad, torch.Tensor) else self.new_tensor(pad)
        size = self.size + pad.to(self.size)
        c = self.c + pad.to(self.c) / 2

        dist = self.dist if hasattr(self, "dist") else self.new_zeros(self.f.shape)
        return self.__class__(torch.cat([size, self.f, c, dist], -1))

    def undo_scale_crop(self, data: Dict[str, torch.Tensor]):
        """Undo transforms done during scaling and cropping."""
        camera = self.crop(-data["crop_pad"]) if "crop_pad" in data else self
        return camera.scale(1.0 / data["scales"])

    @autocast
    def in_image(self, p2d: torch.Tensor):
        """Check if 2D points are within the image boundaries."""
        assert p2d.shape[-1] == 2
        size = self.size.unsqueeze(-2)
        return torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)

    @autocast
    def project(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Project 3D points into the self plane and check for visibility."""
        z = p3d[..., -1]
        valid = z > self.eps
        z = z.clamp(min=self.eps)
        p2d = p3d[..., :-1] / z.unsqueeze(-1)
        return p2d, valid

    def J_project(self, p3d: torch.Tensor):
        """Jacobian of the projection function."""
        x, y, z = p3d[..., 0], p3d[..., 1], p3d[..., 2]
        zero = torch.zeros_like(z)
        z = z.clamp(min=self.eps)
        J = torch.stack([1 / z, zero, -x / z**2, zero, 1 / z, -y / z**2], dim=-1)
        J = J.reshape(p3d.shape[:-1] + (2, 3))
        return J  # N x 2 x 3

    @abstractmethod
    def distort(self, pts: torch.Tensor, return_scale: bool = False) -> Tuple[torch.Tensor]:
        """Distort normalized 2D coordinates and check for validity of the distortion model."""
        raise NotImplementedError("distort() must be implemented.")

    def J_distort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the distortion function."""
        if wrt == "scale2pts":  # (..., 2)
            J = [
                vmap(jacfwd(lambda x: self[idx].distort(x, return_scale=True)[0]))(p2d[idx])[None]
                for idx in range(p2d.shape[0])
            ]

            return torch.cat(J, dim=0).squeeze(-3, -2)

        elif wrt == "scale2dist":  # (..., 1)
            J = []
            for idx in range(p2d.shape[0]):  # loop to batch pts dimension

                def func(x):
                    params = torch.cat([self._data[idx, :6], x[None]], -1)
                    return self.__class__(params).distort(p2d[idx], return_scale=True)[0]

                J.append(vmap(jacfwd(func))(self[idx].dist))

            return torch.cat(J, dim=0)

        else:
            raise NotImplementedError(f"Jacobian not implemented for wrt={wrt}")

    @abstractmethod
    def undistort(self, pts: torch.Tensor) -> Tuple[torch.Tensor]:
        """Undistort normalized 2D coordinates and check for validity of the distortion model."""
        raise NotImplementedError("undistort() must be implemented.")

    def J_undistort(self, p2d: torch.Tensor, wrt: str = "pts") -> torch.Tensor:
        """Jacobian of the undistortion function."""
        if wrt == "pts":  # (..., 2, 2)
            J = [
                vmap(jacfwd(lambda x: self[idx].undistort(x)[0]))(p2d[idx])[None]
                for idx in range(p2d.shape[0])
            ]

            return torch.cat(J, dim=0).squeeze(-3)

        elif wrt == "dist":  # (..., 1)
            J = []
            for batch_idx in range(p2d.shape[0]):  # loop to batch pts dimension

                def func(x):
                    params = torch.cat([self._data[batch_idx, :6], x[None]], -1)
                    return self.__class__(params).undistort(p2d[batch_idx])[0]

                J.append(vmap(jacfwd(func))(self[batch_idx].dist))

            return torch.cat(J, dim=0)
        else:
            raise NotImplementedError(f"Jacobian not implemented for wrt={wrt}")

    @autocast
    def up_projection_offset(self, p2d: torch.Tensor) -> torch.Tensor:
        """Compute the offset for the up-projection."""
        return self.J_distort(p2d, wrt="scale2pts")  # (B, N, 2)

    def J_up_projection_offset(self, p2d: torch.Tensor, wrt: str = "uv") -> torch.Tensor:
        """Jacobian of the distortion offset for up-projection."""
        if wrt == "uv":  # (B, N, 2, 2)
            J = [
                vmap(jacfwd(lambda x: self[idx].up_projection_offset(x)[0, 0]))(p2d[idx])[None]
                for idx in range(p2d.shape[0])
            ]

            return torch.cat(J, dim=0)

        elif wrt == "dist":  # (B, N, 2)
            J = []
            for batch_idx in range(p2d.shape[0]):  # loop to batch pts dimension

                def func(x):
                    params = torch.cat([self._data[batch_idx, :6], x[None]], -1)[None]
                    return self.__class__(params).up_projection_offset(p2d[batch_idx][None])

                J.append(vmap(jacfwd(func))(self[batch_idx].dist))

            return torch.cat(J, dim=0).squeeze(1)
        else:
            raise NotImplementedError(f"Jacobian not implemented for wrt={wrt}")

    @autocast
    def denormalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return p2d * self.f.unsqueeze(-2) + self.c.unsqueeze(-2)

    def J_denormalize(self):
        """Jacobian of the denormalization function."""
        return torch.diag_embed(self.f)  # ..., 2 x 2

    @autocast
    def normalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert pixel coordinates into normalized 2D coordinates."""
        return (p2d - self.c.unsqueeze(-2)) / (self.f.unsqueeze(-2))

    def J_normalize(self, p2d: torch.Tensor, wrt: str = "f"):
        """Jacobian of the normalization function."""
        # ... x N x 2 x 2
        if wrt == "f":
            J_f = -(p2d - self.c.unsqueeze(-2)) / ((self.f.unsqueeze(-2)) ** 2)
            return torch.diag_embed(J_f)
        elif wrt == "pts":
            J_pts = 1 / self.f
            return torch.diag_embed(J_pts)
        else:
            raise NotImplementedError(f"Jacobian not implemented for wrt={wrt}")

    def pixel_coordinates(self) -> torch.Tensor:
        """Pixel coordinates in self frame.

        Returns:
            torch.Tensor: Pixel coordinates as a tensor of shape (B, h * w, 2).
        """
        w, h = self.size[0].unbind(-1)
        h, w = h.round().to(int), w.round().to(int)

        # create grid
        x = torch.arange(0, w, dtype=self.dtype, device=self.device)
        y = torch.arange(0, h, dtype=self.dtype, device=self.device)
        x, y = torch.meshgrid(x, y, indexing="xy")
        xy = torch.stack((x, y), dim=-1).reshape(-1, 2)  # shape (h * w, 2)

        # add batch dimension (normalize() would broadcast but we make it explicit)
        B = self.shape[0]
        xy = xy.unsqueeze(0).expand(B, -1, -1)  # if B > 0 else xy

        return xy.to(self.device).to(self.dtype)

    def normalized_image_coordinates(self) -> torch.Tensor:
        """Normalized image coordinates in self frame.

        Returns:
            torch.Tensor: Normalized image coordinates as a tensor of shape (B, h * w, 3).
        """
        xy = self.pixel_coordinates()
        uv1, _ = self.image2world(xy)

        B = self.shape[0]
        uv1 = uv1.reshape(B, -1, 3)
        return uv1.to(self.device).to(self.dtype)

    @autocast
    def pixel_bearing_many(self, p3d: torch.Tensor) -> torch.Tensor:
        """Get the bearing vectors of pixel coordinates.

        Args:
            p2d (torch.Tensor): Pixel coordinates as a tensor of shape (..., 3).

        Returns:
            torch.Tensor: Bearing vectors as a tensor of shape (..., 3).
        """
        return F.normalize(p3d, dim=-1)

    @autocast
    def world2image(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Transform 3D points into 2D pixel coordinates."""
        p2d, visible = self.project(p3d)
        p2d, mask = self.distort(p2d)
        p2d = self.denormalize(p2d)
        valid = visible & mask & self.in_image(p2d)
        return p2d, valid

    @autocast
    def J_world2image(self, p3d: torch.Tensor):
        """Jacobian of the world2image function."""
        p2d_proj, valid = self.project(p3d)

        J_dnorm = self.J_denormalize()
        J_dist = self.J_distort(p2d_proj)
        J_proj = self.J_project(p3d)

        J = torch.einsum("...ij,...jk,...kl->...il", J_dnorm, J_dist, J_proj)
        return J, valid

    @autocast
    def image2world(self, p2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform point in the image plane to 3D world coordinates."""
        p2d = self.normalize(p2d)
        p2d, valid = self.undistort(p2d)
        ones = p2d.new_ones(p2d.shape[:-1] + (1,))
        p3d = torch.cat([p2d, ones], -1)
        return p3d, valid

    @autocast
    def J_image2world(self, p2d: torch.Tensor, wrt: str = "f") -> Tuple[torch.Tensor, torch.Tensor]:
        """Jacobian of the image2world function."""
        if wrt == "dist":
            p2d_norm = self.normalize(p2d)
            return self.J_undistort(p2d_norm, wrt)
        elif wrt == "f":
            J_norm2f = self.J_normalize(p2d, wrt)
            p2d_norm = self.normalize(p2d)
            J_dist2norm = self.J_undistort(p2d_norm, "pts")

            return torch.einsum("...ij,...jk->...ik", J_dist2norm, J_norm2f)
        else:
            raise ValueError(f"Unknown wrt: {wrt}")

    @autocast
    def undistort_image(self, img: torch.Tensor) -> torch.Tensor:
        """Undistort an image using the distortion model."""
        assert self.shape[0] == 1, "Batch size must be 1."
        W, H = self.size.unbind(-1)
        H, W = H.int().item(), W.int().item()

        x, y = torch.arange(0, W), torch.arange(0, H)
        x, y = torch.meshgrid(x, y, indexing="xy")
        coords = torch.stack((x, y), dim=-1).reshape(-1, 2)

        p3d, _ = self.pinhole().image2world(coords.to(self.device).to(self.dtype))
        p2d, _ = self.world2image(p3d)

        mapx, mapy = p2d[..., 0].reshape((1, H, W)), p2d[..., 1].reshape((1, H, W))
        grid = torch.stack((mapx, mapy), dim=-1)
        grid = 2.0 * grid / torch.tensor([W - 1, H - 1]).to(grid) - 1
        return F.grid_sample(img, grid, align_corners=True)

    def get_img_from_pano(
        self,
        pano_img: torch.Tensor,
        gravity: Gravity,
        yaws: torch.Tensor = 0.0,
        resize_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Render an image from a panorama.

        Args:
            pano_img (torch.Tensor): Panorama image of shape (3, H, W) in [0, 1].
            gravity (Gravity): Gravity direction of the camera.
            yaws (torch.Tensor | list, optional): Yaw angle in radians. Defaults to 0.0.
            resize_factor (torch.Tensor, optional): Resize the panorama to be a multiple of the
            field of view. Defaults to 1.

        Returns:
            torch.Tensor: Image rendered from the panorama.
        """
        B = self.shape[0]
        if B > 0:
            assert self.size[..., 0].unique().shape[0] == 1, "All images must have the same width."
            assert self.size[..., 1].unique().shape[0] == 1, "All images must have the same height."

        w, h = self.size[0].unbind(-1)
        h, w = h.round().to(int), w.round().to(int)

        if isinstance(yaws, (int, float)):
            yaws = [yaws]
        if isinstance(resize_factor, (int, float)):
            resize_factor = [resize_factor]

        yaws = (
            yaws.to(self.dtype).to(self.device)
            if isinstance(yaws, torch.Tensor)
            else self.new_tensor(yaws)
        )

        if isinstance(resize_factor, torch.Tensor):
            resize_factor = resize_factor.to(self.dtype).to(self.device)
        elif resize_factor is not None:
            resize_factor = self.new_tensor(resize_factor)

        assert isinstance(pano_img, torch.Tensor), "Panorama image must be a torch.Tensor."
        pano_img = pano_img if pano_img.dim() == 4 else pano_img.unsqueeze(0)  # B x 3 x H x W

        pano_imgs = []
        for i, yaw in enumerate(yaws):
            if resize_factor is not None:
                # resize the panorama such that the fov of the panorama has the same height as the
                # image
                vfov = self.vfov[i] if B != 0 else self.vfov
                scale = torch.pi / float(vfov) * float(h) / pano_img.shape[-2] * resize_factor[i]
                pano_shape = (int(pano_img.shape[-2] * scale), int(pano_img.shape[-1] * scale))

                mode = "bicubic" if scale >= 1 else "area"
                resized_pano = F.interpolate(pano_img, size=pano_shape, mode=mode)
                # clamp as bicubic interpolation can over- or under-shoot
                resized_pano = resized_pano.clamp(pano_img.min(), pano_img.max())
            else:
                # make sure to copy: resized_pano = pano_img
                resized_pano = pano_img
                pano_shape = pano_img.shape[-2:][::-1]

            pano_imgs.append((resized_pano, pano_shape))

        xy = self.pixel_coordinates()
        uv1, valid = self.image2world(xy)
        bearings = self.pixel_bearing_many(uv1)

        # rotate bearings
        R_yaw = rad2rotmat(self.new_zeros(yaw.shape), self.new_zeros(yaw.shape), yaws)
        rotated_bearings = bearings @ gravity.R @ R_yaw

        # spherical coordinates
        lon = torch.atan2(rotated_bearings[..., 0], rotated_bearings[..., 2])
        lat = torch.atan2(
            rotated_bearings[..., 1], torch.norm(rotated_bearings[..., [0, 2]], dim=-1)
        )

        images = []
        for idx, (resized_pano, pano_shape) in enumerate(pano_imgs):
            min_lon, max_lon = -torch.pi, torch.pi
            min_lat, max_lat = -torch.pi / 2.0, torch.pi / 2.0
            min_x, max_x = 0, pano_shape[0] - 1.0
            min_y, max_y = 0, pano_shape[1] - 1.0

            # map Spherical Coordinates to Panoramic Coordinates
            nx = (lon[idx] - min_lon) / (max_lon - min_lon) * (max_x - min_x) + min_x
            ny = (lat[idx] - min_lat) / (max_lat - min_lat) * (max_y - min_y) + min_y

            # reshape and cast to numpy for remap
            mapx = nx.reshape((1, h, w))
            mapy = ny.reshape((1, h, w))

            grid = torch.stack((mapx, mapy), dim=-1)  # Add batch dimension
            # Normalize to [-1, 1]
            grid = 2.0 * grid / torch.tensor([pano_shape[-2] - 1, pano_shape[-1] - 1]).to(grid) - 1
            # Apply grid sample
            image = F.grid_sample(resized_pano, grid, align_corners=True)
            images.append(image)

        return torch.concatenate(images, 0) if B > 0 else images[0]

    def __repr__(self):
        """Print the Camera object."""
        return f"{self.__class__.__name__} {self.shape} {self.dtype} {self.device}"
