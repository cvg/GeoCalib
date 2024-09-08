import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from siclib.geometry.camera import Pinhole as Camera
from siclib.geometry.gravity import Gravity
from siclib.geometry.perspective_fields import get_perspective_field
from siclib.models.base_model import BaseModel
from siclib.models.utils.metrics import pitch_error, roll_error, vfov_error
from siclib.utils.conversions import deg2rad

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


class PerspectiveParamOpt(BaseModel):
    default_conf = {
        "max_steps": 1000,
        "lr": 0.01,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "options": {"mode": "min", "patience": 3},
        },
        "patience": 3,
        "abs_tol": 1e-7,
        "rel_tol": 1e-9,
        "lamb": 0.5,
        "verbose": False,
    }

    required_data_keys = ["up_field", "latitude_field"]

    def _init(self, conf):
        pass

    def cost_function(self, pred, target):
        """Compute cost function for perspective parameter optimization."""
        eps = 1e-7

        lat_loss = F.l1_loss(pred["latitude_field"], target["latitude_field"], reduction="none")
        lat_loss = lat_loss.squeeze(1)

        up_loss = F.cosine_similarity(pred["up_field"], target["up_field"], dim=1)
        up_loss = torch.acos(torch.clip(up_loss, -1 + eps, 1 - eps))

        cost = (self.conf.lamb * lat_loss) + ((1 - self.conf.lamb) * up_loss)
        return {
            "total": torch.mean(cost),
            "up": torch.mean(up_loss),
            "latitude": torch.mean(lat_loss),
        }

    def check_convergence(self, loss, losses_prev):
        """Check if optimization has converged."""

        if loss["total"].item() <= self.conf.abs_tol:
            return True, losses_prev

        if len(losses_prev) < self.conf.patience:
            losses_prev.append(loss["total"].item())

        elif np.abs(loss["total"].item() - losses_prev[0]) < self.conf.rel_tol:
            return True, losses_prev

        else:
            losses_prev.append(loss["total"].item())
            losses_prev = losses_prev[-self.conf.patience :]

        return False, losses_prev

    def _update_estimate(self, camera: Camera, gravity: Gravity):
        """Update camera estimate based on current parameters."""

        camera = Camera.from_dict(
            {"height": camera.size[..., 1], "width": camera.size[..., 0], "vfov": self.vfov_opt}
        )
        gravity = Gravity.from_rp(self.roll_opt, self.pitch_opt)
        return camera, gravity

    def optimize(self, data, camera_init, gravity_init):
        """Optimize camera parameters to minimize cost function."""
        device = data["up_field"].device
        self.roll_opt = nn.Parameter(gravity_init.roll, requires_grad=True).to(device)
        self.pitch_opt = nn.Parameter(gravity_init.pitch, requires_grad=True).to(device)
        self.vfov_opt = nn.Parameter(camera_init.vfov, requires_grad=True).to(device)

        optimizer = torch.optim.Adam(
            [self.roll_opt, self.pitch_opt, self.vfov_opt], lr=self.conf.lr
        )

        lr_scheduler = None
        if self.conf.lr_scheduler["name"] is not None:
            lr_scheduler = getattr(torch.optim.lr_scheduler, self.conf.lr_scheduler["name"])(
                optimizer, **self.conf.lr_scheduler["options"]
            )

        losses_prev = []

        loop = range(self.conf.max_steps)
        if self.conf.verbose:
            pbar = tqdm(loop, desc="Optimizing", total=len(loop), ncols=100)

        with torch.set_grad_enabled(True):
            self.train()
            for _ in loop:
                optimizer.zero_grad()

                camera_opt, gravity_opt = self._update_estimate(camera_init, gravity_init)

                up, lat = get_perspective_field(camera_opt, gravity_opt)
                pred = {"up_field": up, "latitude_field": lat}

                loss = self.cost_function(pred, data)
                loss["total"].backward()
                optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step(loss["total"])

                if self.conf.verbose:
                    pbar.set_postfix({k[:3]: v.item() for k, v in loss.items()})
                    pbar.update(1)

                converged, losses_prev = self.check_convergence(loss, losses_prev)
                if converged:
                    if self.conf.verbose:
                        pbar.close()
                    break

        camera_opt, gravity_opt = self._update_estimate(camera_init, gravity_init)
        return {"camera_opt": camera_opt, "gravity_opt": gravity_opt}

    def _get_init_params(self, data) -> Tuple[Camera, Gravity]:
        """Get initial camera parameters for optimization."""
        up_ref = data["up_field"]
        latitude_ref = data["latitude_field"]

        h, w = latitude_ref.shape[-2:]

        # init roll is angle of the up vector at the center of the image
        init_r = -torch.arctan2(
            up_ref[:, 0, int(h / 2), int(w / 2)],
            -up_ref[:, 1, int(h / 2), int(w / 2)],
        )

        # init pitch is the value at the center of the latitude map
        init_p = latitude_ref[:, 0, int(h / 2), int(w / 2)]

        # init vfov is the difference between the central top and bottom of the latitude map
        init_vfov = latitude_ref[:, 0, 0, int(w / 2)] - latitude_ref[:, 0, -1, int(w / 2)]
        init_vfov = torch.abs(init_vfov)
        init_vfov = init_vfov.clamp(min=deg2rad(20), max=deg2rad(120))

        h, w = (
            latitude_ref.new_ones(latitude_ref.shape[0]) * h,
            latitude_ref.new_ones(latitude_ref.shape[0]) * w,
        )
        params = {"width": w, "height": h, "vfov": init_vfov}
        camera = Camera.from_dict(params)
        gravity = Gravity.from_rp(init_r, init_p)
        return camera, gravity

    def _forward(self, data):
        """Forward pass of optimization model."""

        assert data["up_field"].shape[0] == 1, "Batch size must be 1 for optimization model."

        # detach all tensors to avoid backprop
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.detach()

        camera_init, gravity_init = self._get_init_params(data)
        return self.optimize(data, camera_init, gravity_init)

    def metrics(self, pred, data):
        pred_cam, gt_cam = pred["camera_opt"], data["camera"]
        pred_grav, gt_grav = pred["gravity_opt"], data["gravity"]

        return {
            "roll_opt_error": roll_error(pred_grav, gt_grav),
            "pitch_opt_error": pitch_error(pred_grav, gt_grav),
            "vfov_opt_error": vfov_error(pred_cam, gt_cam),
        }

    def loss(self, pred, data):
        """No loss function for this optimization model."""
        return {"opt_param_total": 0}, self.metrics(pred, data)
