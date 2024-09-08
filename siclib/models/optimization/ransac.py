from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from siclib.geometry.camera import Pinhole
from siclib.geometry.gravity import Gravity
from siclib.geometry.perspective_fields import get_latitude_field, get_up_field
from siclib.models.base_model import BaseModel
from siclib.models.utils.metrics import (
    latitude_error,
    pitch_error,
    roll_error,
    up_error,
    vfov_error,
)
from siclib.utils.conversions import skew_symmetric

# flake8: noqa
# mypy: ignore-errors


def get_up_lines(up, xy):
    up_lines = torch.cat([up, torch.zeros_like(up[..., :1])], dim=-1)

    xy1 = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)

    xy2 = xy1 + up_lines

    return torch.einsum("...ij,...j->...i", skew_symmetric(xy1), xy2)


def calculate_vvp(line1, line2):
    return torch.einsum("...ij,...j->...i", skew_symmetric(line1), line2)


def calculate_vvps(xs, ys, up):
    xy_grav = torch.stack([xs[..., :2], ys[..., :2]], dim=-1).float()
    up_lines = get_up_lines(up, xy_grav)  # (B, N, 2, D)
    vvp = calculate_vvp(up_lines[..., 0, :], up_lines[..., 1, :])  # (B, N, 3)
    vvp = vvp / vvp[..., (2,)]
    return vvp


def get_up_samples(pred, xs, ys):
    B, N = xs.shape[:2]
    batch_indices = torch.arange(B).unsqueeze(1).unsqueeze(2).expand(B, N, 3).to(xs.device)
    zeros = torch.zeros_like(xs).to(xs.device)
    ones = torch.ones_like(xs).to(xs.device)
    sample_indices_x = torch.stack([batch_indices, zeros, ys, xs], dim=-1).long()  # (B, N, 3, 4)
    sample_indices_y = torch.stack([batch_indices, ones, ys, xs], dim=-1).long()  # (B, N, 3, 4)
    up_x = pred["up_field"][sample_indices_x[..., (0, 1), :].unbind(-1)]  # (B, N, 2)
    up_y = pred["up_field"][sample_indices_y[..., (0, 1), :].unbind(-1)]  # (B, N, 2)
    return torch.stack([up_x, up_y], dim=-1)  # (B, N, 2, D)


def get_latitude_samples(pred, xs, ys):
    # Setup latitude
    B, N = xs.shape[:2]
    batch_indices = torch.arange(B).unsqueeze(1).unsqueeze(2).expand(B, N, 3).to(xs.device)
    zeros = torch.zeros_like(xs).to(xs.device)
    sample_indices = torch.stack([batch_indices, zeros, ys, xs], dim=-1).long()  # (B, N, 3, 4)
    latitude = pred["latitude_field"][sample_indices[..., 2, :].unbind(-1)]
    return torch.sin(latitude)  # (B, N)


class MinimalSolver:
    def __init__(self):
        pass

    @staticmethod
    def solve_focal(
        L: torch.Tensor, xy: torch.Tensor, vvp: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve for focal length.

        Args:
            L (torch.Tensor): Latitude samples.
            xy (torch.Tensor): xy of latitude samples of shape (..., 2).
            vvp (torch.Tensor): Vertical vanishing points of shape (..., 3).
            c (torch.Tensor): Principal points of shape (..., 2).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Positive and negative solution of focal length.
        """
        c = c.unsqueeze(1)
        u, v = (xy - c).unbind(-1)

        vx, vy, vz = vvp.unbind(-1)
        cx, cy = c.unbind(-1)
        vx = vx - cx * vz
        vy = vy - cy * vz

        # Solve quadratic equation
        a0 = (L**2 - 1) * vz**2
        a1 = L**2 * (vz**2 * (u**2 + v**2) + vx**2 + vy**2) - 2 * vz * (vx * u + vy * v)
        a2 = L**2 * (v**2 + u**2) * (vx**2 + vy**2) - (u * vx + v * vy) ** 2

        a0 = torch.where(a0 == 0, torch.ones_like(a0) * 1e-6, a0)

        f2_pos = -a1 / (2 * a0) + torch.sqrt(a1**2 - 4 * a0 * a2) / (2 * a0)
        f2_neg = -a1 / (2 * a0) - torch.sqrt(a1**2 - 4 * a0 * a2) / (2 * a0)

        f_pos, f_neg = torch.sqrt(f2_pos), torch.sqrt(f2_neg)

        return f_pos, f_neg

    @staticmethod
    def solve_scale(
        L: torch.Tensor, xy: torch.Tensor, vvp: torch.Tensor, c: torch.Tensor, f: torch.Tensor
    ) -> torch.Tensor:
        """Solve for scale of homogeneous vector.

        Args:
            L (torch.Tensor): Latitude samples.
            xy (torch.Tensor): xy of latitude samples of shape (..., 2).
            vvp (torch.Tensor): Vertical vanishing points of shape (..., 3).
            c (torch.Tensor): Principal points of shape (..., 2).
            f (torch.Tensor): Focal lengths.

        Returns:
            torch.Tensor: Estimated scales.
        """
        c = c.unsqueeze(1)
        u, v = (xy - c).unbind(-1)

        vx, vy, vz = vvp.unbind(-1)
        cx, cy = c.unbind(-1)
        vx = vx - cx * vz
        vy = vy - cy * vz

        w2 = (f**2 * L**2 * (u**2 + v**2 + f**2)) / (vx * u + vy * v + vz * f**2) ** 2
        return torch.sqrt(w2)

    @staticmethod
    def solve_abc(
        vvp: torch.Tensor, c: torch.Tensor, f: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Solve for abc vector (solution to homogeneous equation).

        Args:
            vvp (torch.Tensor): Vertical vanishing points of shape (..., 3).
            c (torch.Tensor): Principal points of shape (..., 2).
            f (torch.Tensor): Focal lengths.
            w (torch.Tensor): Scales.

        Returns:
            torch.Tensor: Estimated abc vector.
        """
        vx, vy, vz = vvp.unbind(-1)
        cx, cy = c.unsqueeze(1).unbind(-1)
        vx = vx - cx * vz
        vy = vy - cy * vz

        a = vx / f
        b = vy / f
        c = vz

        abc = torch.stack((a, b, c), dim=-1)

        return F.normalize(abc, dim=-1) if w is None else abc * w.unsqueeze(-1)

    @staticmethod
    def solve_rp(abc: torch.Tensor) -> torch.Tensor:
        """Solve for roll, pitch.

        Args:
            abc (torch.Tensor): Estimated abc vector.

        Returns:
            torch.Tensor: Estimated roll, pitch, focal length.
        """
        a, _, c = abc.unbind(-1)
        roll = torch.asin(-a / torch.sqrt(1 - c**2))
        pitch = torch.asin(c)
        return roll, pitch


class RPFSolver(BaseModel):
    default_conf = {
        "n_iter": 1000,
        "up_inlier_th": 1,
        "latitude_inlier_th": 1,
        "error_fn": "angle",  # angle or mse
        "up_weight": 1,
        "latitude_weight": 1,
        "loss_weight": 1,
        "use_latitude": True,
    }

    def _init(self, conf):
        self.solver = MinimalSolver()

    def check_up_inliers(self, pred, est_camera, est_gravity, N=1):
        pred_up = pred["up_field"]
        # expand from from (B, 1, H, W) to (B * N, 1, H, W)
        B = pred_up.shape[0]
        pred_up = pred_up.unsqueeze(1).expand(-1, N, -1, -1, -1)
        pred_up = pred_up.reshape(B * N, *pred_up.shape[2:])

        est_up = get_up_field(est_camera, est_gravity).permute(0, 3, 1, 2)

        if self.conf.error_fn == "angle":
            mse = up_error(est_up, pred_up)
        elif self.conf.error_fn == "mse":
            mse = F.mse_loss(est_up, pred_up, reduction="none").mean(1)
        else:
            raise ValueError(f"Unknown error function: {self.conf.error_fn}")

        # shape (B, H, W)
        conf = pred.get("up_confidence", pred_up.new_ones(pred_up.shape[0], *pred_up.shape[-2:]))
        # shape (B, N, H, W)
        conf = conf.unsqueeze(1).expand(-1, N, -1, -1)
        # shape (B * N, H, W)
        conf = conf.reshape(B * N, *conf.shape[-2:])

        return (mse < self.conf.up_inlier_th) * conf

    def check_latitude_inliers(self, pred, est_camera, est_gravity, N=1):
        B = pred["up_field"].shape[0]
        pred_latitude = pred.get("latitude_field")

        if pred_latitude is None:
            shape = (B * N, *pred["up_field"].shape[-2:])
            return est_camera.new_zeros(shape)

        # expand from from (B, 1, H, W) to (B * N, 1, H, W)
        pred_latitude = pred_latitude.unsqueeze(1).expand(-1, N, -1, -1, -1)
        pred_latitude = pred_latitude.reshape(B * N, *pred_latitude.shape[2:])

        est_latitude = get_latitude_field(est_camera, est_gravity).permute(0, 3, 1, 2)

        if self.conf.error_fn == "angle":
            error = latitude_error(est_latitude, pred_latitude)
        elif self.conf.error_fn == "mse":
            error = F.mse_loss(est_latitude, pred_latitude, reduction="none").mean(1)
        else:
            raise ValueError(f"Unknown error function: {self.conf.error_fn}")

        conf = pred.get(
            "latitude_confidence",
            pred_latitude.new_ones(pred_latitude.shape[0], *pred_latitude.shape[-2:]),
        )
        conf = conf.unsqueeze(1).expand(-1, N, -1, -1)
        conf = conf.reshape(B * N, *conf.shape[-2:])
        return (error < self.conf.latitude_inlier_th) * conf

    def get_best_index(self, data, camera, gravity, inliers=None):
        B, _, H, W = data["up_field"].shape
        N = self.conf.n_iter

        up_inliers = self.check_up_inliers(data, camera, gravity, N)
        latitude_inliers = self.check_latitude_inliers(data, camera, gravity, N)

        up_inliers = up_inliers.reshape(B, N, H, W)
        latitude_inliers = latitude_inliers.reshape(B, N, H, W)

        if inliers is not None:
            up_inliers = up_inliers * inliers.unsqueeze(1)
            latitude_inliers = latitude_inliers * inliers.unsqueeze(1)

        up_inliers = up_inliers.sum((2, 3))
        latitude_inliers = latitude_inliers.sum((2, 3))

        total_inliers = (
            self.conf.up_weight * up_inliers + self.conf.latitude_weight * latitude_inliers
        )

        best_idx = total_inliers.argmax(-1)

        return best_idx, total_inliers[torch.arange(B), best_idx]

    def solve_rpf(self, pred, xs, ys, principal_points, focal=None):
        device = pred["up_field"].device

        # Get samples
        up = get_up_samples(pred, xs, ys)

        # Calculate vvps
        vvp = calculate_vvps(xs, ys, up).to(device)

        # Solve for focal length
        xy = torch.stack([xs[..., 2], ys[..., 2]], dim=-1).float()
        if focal is not None:
            f = focal.new_ones(xs[..., 2].shape) * focal.unsqueeze(-1)
            f_pos, f_neg = f, f
        else:
            L = get_latitude_samples(pred, xs, ys)
            f_pos, f_neg = self.solver.solve_focal(L, xy, vvp, principal_points)

        # Solve for abc
        abc_pos = self.solver.solve_abc(vvp, principal_points, f_pos)
        abc_neg = self.solver.solve_abc(vvp, principal_points, f_neg)

        # Solve for roll, pitch
        roll_pos, pitch_pos = self.solver.solve_rp(abc_pos)
        roll_neg, pitch_neg = self.solver.solve_rp(abc_neg)

        rpf_pos = torch.stack([roll_pos, pitch_pos, f_pos], dim=-1)
        rpf_neg = torch.stack([roll_neg, pitch_neg, f_neg], dim=-1)

        return rpf_pos, rpf_neg

    def get_camera_and_gravity(self, pred, rpf):
        B, _, H, W = pred["up_field"].shape
        N = rpf.shape[1]

        w = pred["up_field"].new_ones(B, N) * W
        h = pred["up_field"].new_ones(B, N) * H
        cx = w / 2.0
        cy = h / 2.0

        roll, pitch, focal = rpf.unbind(-1)

        params = torch.stack([w, h, focal, focal, cx, cy], dim=-1)
        params = params.reshape(B * N, params.shape[-1])
        cam = Pinhole(params)

        roll, pitch = roll.reshape(B * N), pitch.reshape(B * N)
        gravity = Gravity.from_rp(roll, pitch)

        return cam, gravity

    def _forward(self, data):
        device = data["up_field"].device
        B, _, H, W = data["up_field"].shape

        principal_points = torch.tensor([H / 2.0, W / 2.0]).expand(B, 2).to(device)

        if not self.conf.use_latitude and "latitude_field" in data:
            data.pop("latitude_field")

        if "inliers" in data:
            indices = torch.nonzero(data["inliers"] == 1, as_tuple=False)
            batch_indices = torch.unique(indices[:, 0])

            sampled_indices = []
            for batch_index in batch_indices:
                batch_mask = indices[:, 0] == batch_index

                batch_indices_sampled = np.random.choice(
                    batch_mask.sum(), self.conf.n_iter * 3, replace=True
                )
                batch_indices_sampled = batch_indices_sampled.reshape(self.conf.n_iter, 3)
                sampled_indices.append(indices[batch_mask][batch_indices_sampled][:, :, 1:])

            ys, xs = torch.stack(sampled_indices, dim=0).unbind(-1)

        else:
            xs = torch.randint(0, W, (B, self.conf.n_iter, 3)).to(device)
            ys = torch.randint(0, H, (B, self.conf.n_iter, 3)).to(device)

        rpf_pos, rpf_neg = self.solve_rpf(
            data, xs, ys, principal_points, focal=data.get("prior_focal")
        )

        cams_pos, gravity_pos = self.get_camera_and_gravity(data, rpf_pos)
        cams_neg, gravity_neg = self.get_camera_and_gravity(data, rpf_neg)

        inliers = data.get("inliers", None)
        best_pos, score_pos = self.get_best_index(data, cams_pos, gravity_pos, inliers)
        best_neg, score_neg = self.get_best_index(data, cams_neg, gravity_neg, inliers)

        rpf = rpf_pos[torch.arange(B), best_pos]
        rpf[score_neg > score_pos] = rpf_neg[torch.arange(B), best_neg][score_neg > score_pos]

        cam, gravity = self.get_camera_and_gravity(data, rpf.unsqueeze(1))

        return {
            "camera_opt": cam,
            "gravity_opt": gravity,
            "up_inliers": self.check_up_inliers(data, cam, gravity),
            "latitude_inliers": self.check_latitude_inliers(data, cam, gravity),
        }

    def metrics(self, pred, data):
        pred_cam, gt_cam = pred["camera_opt"], data["camera"]
        pred_gravity, gt_gravity = pred["gravity_opt"], data["gravity"]

        return {
            "roll_opt_error": roll_error(pred_gravity, gt_gravity),
            "pitch_opt_error": pitch_error(pred_gravity, gt_gravity),
            "vfov_opt_error": vfov_error(pred_cam, gt_cam),
        }

    def loss(self, pred, data):
        pred_cam, gt_cam = pred["camera_opt"], data["camera"]
        pred_gravity, gt_gravity = pred["gravity_opt"], data["gravity"]

        h = data["camera"].size[0, 0]

        gravity_loss = F.l1_loss(pred_gravity.vec3d, gt_gravity.vec3d, reduction="none")
        focal_loss = F.l1_loss(pred_cam.f, gt_cam.f, reduction="none").sum(-1) / h

        total_loss = gravity_loss.sum(-1)
        if self.conf.estimate_focal:
            total_loss = total_loss + focal_loss

        losses = {
            "opt_gravity": gravity_loss.sum(-1),
            "opt_focal": focal_loss,
            "opt_param_total": total_loss,
        }

        losses = {k: v * self.conf.loss_weight for k, v in losses.items()}
        return losses, self.metrics(pred, data)
