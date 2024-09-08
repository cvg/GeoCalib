import logging
import time
from typing import Dict, Tuple

import torch
from torch import nn

import siclib.models.optimization.losses as losses
from siclib.geometry.base_camera import BaseCamera
from siclib.geometry.camera import camera_models
from siclib.geometry.gravity import Gravity
from siclib.geometry.jacobians import J_focal2fov
from siclib.geometry.perspective_fields import J_perspective_field, get_perspective_field
from siclib.models import get_model
from siclib.models.base_model import BaseModel
from siclib.models.optimization.utils import (
    early_stop,
    get_initial_estimation,
    optimizer_step,
    update_lambda,
)
from siclib.models.utils.metrics import (
    dist_error,
    gravity_error,
    pitch_error,
    roll_error,
    vfov_error,
)
from siclib.utils.conversions import rad2deg

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


class LMOptimizer(BaseModel):
    default_conf = {
        # Camera model parameters
        "camera_model": "pinhole",  # {"pinhole", "simple_radial", "simple_spherical"}
        "shared_intrinsics": False,  # share focal length across all images in batch
        # LM optimizer parameters
        "num_steps": 10,
        "lambda_": 0.1,
        "fix_lambda": False,
        "early_stop": False,
        "atol": 1e-8,
        "rtol": 1e-8,
        "use_spherical_manifold": True,  # use spherical manifold for gravity optimization
        "use_log_focal": True,  # use log focal length for optimization
        # Loss function parameters
        "loss_fn": "squared_loss",  # {"squared_loss", "huber_loss"}
        "up_loss_fn_scale": 1e-2,
        "lat_loss_fn_scale": 1e-2,
        "init_conf": {"name": "trivial"},  # pass config of other models to use as initializer
        # Misc
        "loss_weight": 1,
        "verbose": False,
    }

    def _init(self, conf):
        self.loss_fn = getattr(losses, conf.loss_fn)
        self.num_steps = conf.num_steps

        self.set_camera_model(conf.camera_model)

        self.setup_optimization_and_priors(shared_intrinsics=conf.shared_intrinsics)

        self.initializer = None
        if self.conf.init_conf.name not in ["trivial", "heuristic"]:
            self.initializer = get_model(conf.init_conf.name)(conf.init_conf)

    def set_camera_model(self, camera_model: str) -> None:
        """Set the camera model to use for the optimization.

        Args:
            camera_model (str): Camera model to use.
        """
        assert (
            camera_model in camera_models.keys()
        ), f"Unknown camera model: {camera_model} not in {camera_models.keys()}"
        self.camera_model = camera_models[camera_model]
        self.camera_has_distortion = hasattr(self.camera_model, "dist")

        logger.debug(
            f"Using camera model: {camera_model} (with distortion: {self.camera_has_distortion})"
        )

    def setup_optimization_and_priors(
        self, data: Dict[str, torch.Tensor] = None, shared_intrinsics: bool = False
    ) -> None:
        """Setup the optimization and priors for the LM optimizer.

        Args:
            data (Dict[str, torch.Tensor], optional): Dict potentially containing priors. Defaults
            to None.
            shared_intrinsics (bool, optional): Whether to share the intrinsics across the batch.
            Defaults to False.
        """
        if data is None:
            data = {}
        self.shared_intrinsics = shared_intrinsics

        if shared_intrinsics:  # si => must use pinhole
            assert (
                self.camera_model == camera_models["pinhole"]
            ), f"Shared intrinsics only supported with pinhole camera model: {self.camera_model}"

        self.estimate_gravity = True
        if "prior_gravity" in data:
            self.estimate_gravity = False
            logger.debug("Using provided gravity as prior.")

        self.estimate_focal = True
        if "prior_focal" in data:
            self.estimate_focal = False
            logger.debug("Using provided focal as prior.")

        self.estimate_k1 = True
        if "prior_k1" in data:
            self.estimate_k1 = False
            logger.debug("Using provided k1 as prior.")

        self.gravity_delta_dims = (0, 1) if self.estimate_gravity else (-1,)
        self.focal_delta_dims = (
            (max(self.gravity_delta_dims) + 1,) if self.estimate_focal else (-1,)
        )
        self.k1_delta_dims = (max(self.focal_delta_dims) + 1,) if self.estimate_k1 else (-1,)

        logger.debug(f"Camera Model:       {self.camera_model}")
        logger.debug(f"Optimizing gravity: {self.estimate_gravity} ({self.gravity_delta_dims})")
        logger.debug(f"Optimizing focal:   {self.estimate_focal} ({self.focal_delta_dims})")
        logger.debug(f"Optimizing k1:      {self.estimate_k1} ({self.k1_delta_dims})")

        logger.debug(f"Shared intrinsics:  {self.shared_intrinsics}")

    def calculate_residuals(
        self, camera: BaseCamera, gravity: Gravity, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calculate the residuals for the optimization.

        Args:
            camera (BaseCamera): Optimized camera.
            gravity (Gravity): Optimized gravity.
            data (Dict[str, torch.Tensor]): Input data containing the up and latitude fields.

        Returns:
            Dict[str, torch.Tensor]: Residuals for the optimization.
        """
        perspective_up, perspective_lat = get_perspective_field(camera, gravity)
        perspective_lat = torch.sin(perspective_lat)

        residuals = {}
        if "up_field" in data:
            up_residual = (data["up_field"] - perspective_up).permute(0, 2, 3, 1)
            residuals["up_residual"] = up_residual.reshape(up_residual.shape[0], -1, 2)

        if "latitude_field" in data:
            target_lat = torch.sin(data["latitude_field"])
            lat_residual = (target_lat - perspective_lat).permute(0, 2, 3, 1)
            residuals["latitude_residual"] = lat_residual.reshape(lat_residual.shape[0], -1, 1)

        return residuals

    def calculate_costs(
        self, residuals: torch.Tensor, data: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Calculate the costs and weights for the optimization.

        Args:
            residuals (torch.Tensor): Residuals for the optimization.
            data (Dict[str, torch.Tensor]): Input data containing the up and latitude confidence.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Costs and weights for the
            optimization.
        """
        costs, weights = {}, {}

        if "up_residual" in residuals:
            up_cost = (residuals["up_residual"] ** 2).sum(dim=-1)
            up_cost, up_weight, _ = losses.scaled_loss(
                up_cost, self.loss_fn, self.conf.up_loss_fn_scale
            )

            if "up_confidence" in data:
                up_conf = data["up_confidence"].reshape(up_weight.shape[0], -1)
                up_weight = up_weight * up_conf
                up_cost = up_cost * up_conf

            costs["up_cost"] = up_cost
            weights["up_weights"] = up_weight

        if "latitude_residual" in residuals:
            lat_cost = (residuals["latitude_residual"] ** 2).sum(dim=-1)
            lat_cost, lat_weight, _ = losses.scaled_loss(
                lat_cost, self.loss_fn, self.conf.lat_loss_fn_scale
            )

            if "latitude_confidence" in data:
                lat_conf = data["latitude_confidence"].reshape(lat_weight.shape[0], -1)
                lat_weight = lat_weight * lat_conf
                lat_cost = lat_cost * lat_conf

            costs["latitude_cost"] = lat_cost
            weights["latitude_weights"] = lat_weight

        return costs, weights

    def calculate_gradient_and_hessian(
        self,
        J: torch.Tensor,
        residuals: torch.Tensor,
        weights: torch.Tensor,
        shared_intrinsics: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the gradient and Hessian for given the Jacobian, residuals, and weights.

        Args:
            J (torch.Tensor): Jacobian.
            residuals (torch.Tensor): Residuals.
            weights (torch.Tensor): Weights.
            shared_intrinsics (bool): Whether to share the intrinsics across the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Gradient and Hessian.
        """
        dims = ()
        if self.estimate_gravity:
            dims = (0, 1)
        if self.estimate_focal:
            dims += (2,)
        if self.camera_has_distortion and self.estimate_k1:
            dims += (3,)
        assert dims, "No parameters to optimize"

        J = J[..., dims]

        Grad = torch.einsum("...Njk,...Nj->...Nk", J, residuals)
        Grad = weights[..., None] * Grad
        Grad = Grad.sum(-2)  # (B, N_params)

        if shared_intrinsics:
            # reshape to (1, B * (N_params-1) + 1)
            Grad_g = Grad[..., :2].reshape(1, -1)
            Grad_f = Grad[..., 2].reshape(1, -1).sum(-1, keepdim=True)
            Grad = torch.cat([Grad_g, Grad_f], dim=-1)

        Hess = torch.einsum("...Njk,...Njl->...Nkl", J, J)
        Hess = weights[..., None, None] * Hess
        Hess = Hess.sum(-3)

        if shared_intrinsics:
            H_g = torch.block_diag(*list(Hess[..., :2, :2]))
            J_fg = Hess[..., :2, 2].flatten()
            J_gf = Hess[..., 2, :2].flatten()
            J_f = Hess[..., 2, 2].sum()
            dims = H_g.shape[-1] + 1
            Hess = Hess.new_zeros((dims, dims), dtype=torch.float32)
            Hess[:-1, :-1] = H_g
            Hess[-1, :-1] = J_gf
            Hess[:-1, -1] = J_fg
            Hess[-1, -1] = J_f
            Hess = Hess.unsqueeze(0)

        return Grad, Hess

    def setup_system(
        self,
        camera: BaseCamera,
        gravity: Gravity,
        residuals: Dict[str, torch.Tensor],
        weights: Dict[str, torch.Tensor],
        as_rpf: bool = False,
        shared_intrinsics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the gradient and Hessian for the optimization.

        Args:
            camera (BaseCamera): Optimized camera.
            gravity (Gravity): Optimized gravity.
            residuals (Dict[str, torch.Tensor]): Residuals for the optimization.
            weights (Dict[str, torch.Tensor]): Weights for the optimization.
            as_rpf (bool, optional): Wether to calculate the gradient and Hessian with respect to
            roll, pitch, and focal length. Defaults to False.
            shared_intrinsics (bool, optional): Whether to share the intrinsics across the batch.
            Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Gradient and Hessian for the optimization.
        """
        J_up, J_lat = J_perspective_field(
            camera,
            gravity,
            spherical=self.conf.use_spherical_manifold and not as_rpf,
            log_focal=self.conf.use_log_focal and not as_rpf,
        )

        J_up = J_up.reshape(J_up.shape[0], -1, J_up.shape[-2], J_up.shape[-1])  # (B, N, 2, 3)
        J_lat = J_lat.reshape(J_lat.shape[0], -1, J_lat.shape[-2], J_lat.shape[-1])  # (B, N, 1, 3)

        n_params = (
            2 * self.estimate_gravity
            + self.estimate_focal
            + (self.camera_has_distortion and self.estimate_k1)
        )
        Grad = J_up.new_zeros(J_up.shape[0], n_params)
        Hess = J_up.new_zeros(J_up.shape[0], n_params, n_params)

        if shared_intrinsics:
            N_params = Grad.shape[0] * (n_params - 1) + 1
            Grad = Grad.new_zeros(1, N_params)
            Hess = Hess.new_zeros(1, N_params, N_params)

        if "up_residual" in residuals:
            Up_Grad, Up_Hess = self.calculate_gradient_and_hessian(
                J_up, residuals["up_residual"], weights["up_weights"], shared_intrinsics
            )

            if self.conf.verbose:
                logger.info(f"Up J:\n{Up_Grad.mean(0)}")

            Grad = Grad + Up_Grad
            Hess = Hess + Up_Hess

        if "latitude_residual" in residuals:
            Lat_Grad, Lat_Hess = self.calculate_gradient_and_hessian(
                J_lat,
                residuals["latitude_residual"],
                weights["latitude_weights"],
                shared_intrinsics,
            )

            if self.conf.verbose:
                logger.info(f"Lat J:\n{Lat_Grad.mean(0)}")

            Grad = Grad + Lat_Grad
            Hess = Hess + Lat_Hess

        return Grad, Hess

    def estimate_uncertainty(
        self,
        camera_opt: BaseCamera,
        gravity_opt: Gravity,
        errors: Dict[str, torch.Tensor],
        weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Estimate the uncertainty of the optimized camera and gravity at the final step.

        Args:
            camera_opt (BaseCamera): Final optimized camera.
            gravity_opt (Gravity): Final optimized gravity.
            errors (Dict[str, torch.Tensor]): Costs for the optimization.
            weights (Dict[str, torch.Tensor]): Weights for the optimization.

        Returns:
            Dict[str, torch.Tensor]: Uncertainty estimates for the optimized camera and gravity.
        """
        _, Hess = self.setup_system(
            camera_opt, gravity_opt, errors, weights, as_rpf=True, shared_intrinsics=False
        )
        Cov = torch.inverse(Hess)

        roll_uncertainty = Cov.new_zeros(Cov[..., 0, 0].shape)
        pitch_uncertainty = Cov.new_zeros(Cov[..., 0, 0].shape)
        gravity_uncertainty = Cov.new_zeros(Cov[..., 0, 0].shape)
        if self.estimate_gravity:
            roll_uncertainty = Cov[..., 0, 0]
            pitch_uncertainty = Cov[..., 1, 1]

            try:
                delta_uncertainty = Cov[..., :2, :2]
                eigenvalues = torch.linalg.eigvalsh(delta_uncertainty.cpu())
                gravity_uncertainty = torch.max(eigenvalues, dim=-1).values.to(Cov.device)
            except RuntimeError:
                logger.warning("Could not calculate gravity uncertainty")
                gravity_uncertainty = Cov.new_zeros(Cov.shape[0])

        focal_uncertainty = Cov.new_zeros(Cov[..., 0, 0].shape)
        fov_uncertainty = Cov.new_zeros(Cov[..., 0, 0].shape)
        if self.estimate_focal:
            focal_uncertainty = Cov[..., self.focal_delta_dims[0], self.focal_delta_dims[0]]
            fov_uncertainty = (
                J_focal2fov(camera_opt.f[..., 1], camera_opt.size[..., 1]) ** 2 * focal_uncertainty
            )

        return {
            "covariance": Cov,
            "roll_uncertainty": torch.sqrt(roll_uncertainty),
            "pitch_uncertainty": torch.sqrt(pitch_uncertainty),
            "gravity_uncertainty": torch.sqrt(gravity_uncertainty),
            "focal_uncertainty": torch.sqrt(focal_uncertainty) / 2,
            "vfov_uncertainty": torch.sqrt(fov_uncertainty / 2),
        }

    def update_estimate(
        self, camera: BaseCamera, gravity: Gravity, delta: torch.Tensor
    ) -> Tuple[BaseCamera, Gravity]:
        """Update the camera and gravity estimates with the given delta.

        Args:
            camera (BaseCamera): Optimized camera.
            gravity (Gravity): Optimized gravity.
            delta (torch.Tensor): Delta to update the camera and gravity estimates.

        Returns:
            Tuple[BaseCamera, Gravity]: Updated camera and gravity estimates.
        """
        delta_gravity = (
            delta[..., self.gravity_delta_dims]
            if self.estimate_gravity
            else delta.new_zeros(delta.shape[:-1] + (2,))
        )
        new_gravity = gravity.update(delta_gravity, spherical=self.conf.use_spherical_manifold)

        delta_f = (
            delta[..., self.focal_delta_dims]
            if self.estimate_focal
            else delta.new_zeros(delta.shape[:-1] + (1,))
        )
        new_camera = camera.update_focal(delta_f, as_log=self.conf.use_log_focal)

        delta_dist = (
            delta[..., self.k1_delta_dims]
            if self.camera_has_distortion and self.estimate_k1
            else delta.new_zeros(delta.shape[:-1] + (1,))
        )
        if self.camera_has_distortion:
            new_camera = new_camera.update_dist(delta_dist)

        return new_camera, new_gravity

    def optimize(
        self,
        data: Dict[str, torch.Tensor],
        camera_opt: BaseCamera,
        gravity_opt: Gravity,
    ) -> Tuple[BaseCamera, Gravity, Dict[str, torch.Tensor]]:
        """Optimize the camera and gravity estimates.

        Args:
            data (Dict[str, torch.Tensor]): Input data.
            camera_opt (BaseCamera): Optimized camera.
            gravity_opt (Gravity): Optimized gravity.

        Returns:
            Tuple[BaseCamera, Gravity, Dict[str, torch.Tensor]]: Optimized camera, gravity
            estimates and optimization information.
        """
        key = list(data.keys())[0]
        B = data[key].shape[0]

        lamb = data[key].new_ones(B) * self.conf.lambda_
        if self.shared_intrinsics:
            lamb = data[key].new_ones(1) * self.conf.lambda_

        infos = {"stop_at": self.num_steps}
        for i in range(self.num_steps):
            if self.conf.verbose:
                logger.info(f"Step {i+1}/{self.num_steps}")

            errors = self.calculate_residuals(camera_opt, gravity_opt, data)
            costs, weights = self.calculate_costs(errors, data)

            if i == 0:
                prev_cost = sum(c.mean(-1) for c in costs.values())
                for k, c in costs.items():
                    infos[f"initial_{k}"] = c.mean(-1)

                infos["initial_cost"] = prev_cost

            Grad, Hess = self.setup_system(
                camera_opt,
                gravity_opt,
                errors,
                weights,
                shared_intrinsics=self.shared_intrinsics,
            )
            delta = optimizer_step(Grad, Hess, lamb)  # (B, N_params)

            if self.shared_intrinsics:
                delta_g = delta[..., :-1].reshape(B, 2)
                delta_f = delta[..., -1].expand(B, 1)
                delta = torch.cat([delta_g, delta_f], dim=-1)

            # calculate new cost
            camera_opt, gravity_opt = self.update_estimate(camera_opt, gravity_opt, delta)
            new_cost, _ = self.calculate_costs(
                self.calculate_residuals(camera_opt, gravity_opt, data), data
            )
            new_cost = sum(c.mean(-1) for c in new_cost.values())

            if not self.conf.fix_lambda and not self.shared_intrinsics:
                lamb = update_lambda(lamb, prev_cost, new_cost)

            if self.conf.verbose:
                logger.info(f"Cost:\nPrev: {prev_cost}\nNew:  {new_cost}")
                logger.info(f"Camera:\n{camera_opt._data}")

            if early_stop(new_cost, prev_cost, atol=self.conf.atol, rtol=self.conf.rtol):
                infos["stop_at"] = min(i + 1, infos["stop_at"])

                if self.conf.early_stop:
                    if self.conf.verbose:
                        logger.info(f"Early stopping at step {i+1}")
                    break

            prev_cost = new_cost

            if i == self.num_steps - 1 and self.conf.early_stop:
                logger.warning("Reached maximum number of steps without convergence.")

        final_errors = self.calculate_residuals(camera_opt, gravity_opt, data)  # (B, N, 3)
        final_cost, weights = self.calculate_costs(final_errors, data)  # (B, N)

        if not self.training:
            infos |= self.estimate_uncertainty(camera_opt, gravity_opt, final_errors, weights)

        infos["stop_at"] = camera_opt.new_ones(camera_opt.shape[0]) * infos["stop_at"]
        for k, c in final_cost.items():
            infos[f"final_{k}"] = c.mean(-1)

        infos["final_cost"] = sum(c.mean(-1) for c in final_cost.values())

        return camera_opt, gravity_opt, infos

    def _forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the LM optimization."""
        if self.initializer is None:
            camera_init, gravity_init = get_initial_estimation(
                data, self.camera_model, trivial_init=self.conf.init_conf.name == "trivial"
            )
        else:
            out = self.initializer(data)
            camera_init = out["camera"]
            gravity_init = out["gravity"]

        self.setup_optimization_and_priors(data, shared_intrinsics=self.shared_intrinsics)

        start = time.time()
        camera_opt, gravity_opt, infos = self.optimize(data, camera_init, gravity_init)

        if self.conf.verbose:
            logger.info(f"Optimization took {(time.time() - start)*1000:.2f} ms")

            logger.info(f"Initial camera:\n{rad2deg(camera_init.vfov)}")
            logger.info(f"Optimized camera:\n{rad2deg(camera_opt.vfov)}")

            logger.info(f"Initial gravity:\n{rad2deg(gravity_init.rp)}")
            logger.info(f"Optimized gravity:\n{rad2deg(gravity_opt.rp)}")

        return {"camera": camera_opt, "gravity": gravity_opt, **infos}

    def metrics(
        self, pred: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calculate the metrics for the optimization."""
        pred_cam, gt_cam = pred["camera"], data["camera"]
        pred_gravity, gt_gravity = pred["gravity"], data["gravity"]

        infos = {"stop_at": pred["stop_at"]}
        for k, v in pred.items():
            if "initial" in k or "final" in k:
                infos[k] = v

        return {
            "roll_error": roll_error(pred_gravity, gt_gravity),
            "pitch_error": pitch_error(pred_gravity, gt_gravity),
            "gravity_error": gravity_error(pred_gravity, gt_gravity),
            "vfov_error": vfov_error(pred_cam, gt_cam),
            "k1_error": dist_error(pred_cam, gt_cam),
            **infos,
        }

    def loss(
        self, pred: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Calculate the loss for the optimization."""
        pred_cam, gt_cam = pred["camera"], data["camera"]
        pred_gravity, gt_gravity = pred["gravity"], data["gravity"]

        loss_fn = nn.L1Loss(reduction="none")

        # loss will be 0 if estimate is false and prior is provided during training
        gravity_loss = loss_fn(pred_gravity.vec3d, gt_gravity.vec3d)

        h = data["camera"].size[0, 0]
        focal_loss = loss_fn(pred_cam.f, gt_cam.f).mean(-1) / h

        dist_loss = focal_loss.new_zeros(focal_loss.shape)
        if self.camera_has_distortion:
            dist_loss = loss_fn(pred_cam.dist, gt_cam.dist).sum(-1)

        losses = {
            "gravity": gravity_loss.sum(-1),
            "focal": focal_loss,
            "dist": dist_loss,
            "param_total": gravity_loss.sum(-1) + focal_loss + dist_loss,
        }

        losses = {k: v * self.conf.loss_weight for k, v in losses.items()}
        return losses, self.metrics(pred, data)
