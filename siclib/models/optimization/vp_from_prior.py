"""Wrapper for VP estimation with prior gravity using the VP-Estimation-with-Prior-Gravity library.

repo: https://github.com/cvg/VP-Estimation-with-Prior-Gravity
"""

import sys

sys.path.append("third_party/VP-Estimation-with-Prior-Gravity")
sys.path.append("third_party/VP-Estimation-with-Prior-Gravity/src/deeplsd")

import logging
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from vp_estimation_with_prior_gravity.evaluation import get_labels_from_vp, project_vp_to_image
from vp_estimation_with_prior_gravity.features.line_detector import LineDetector
from vp_estimation_with_prior_gravity.solvers import run_hybrid_uncalibrated
from vp_estimation_with_prior_gravity.visualization import plot_images, plot_lines, plot_vp

from siclib.geometry.camera import Pinhole
from siclib.geometry.gravity import Gravity
from siclib.models import BaseModel
from siclib.models.utils.metrics import gravity_error, pitch_error, roll_error, vfov_error

# flake8: noqa
# mypy: ignore-errors

logger = logging.getLogger(__name__)


class VPEstimator(BaseModel):
    # Which solvers to us for our hybrid solver:
    # 0 - 2lines 200g
    # 1 - 2lines 110g
    # 2 - 2lines 011g
    # 3 - 4lines 211
    # 4 - 4lines 220
    default_conf = {
        "SOLVER_FLAGS": [True, True, True, True, True],
        "th_pixels": 3,  # RANSAC inlier threshold
        "ls_refinement": 2,  # 3 uses the gravity in the LS refinement, 2 does not.
        "nms": 3,  # change to 3 to add a Ceres optimization after the non minimal solver (slower)
        "magsac_scoring": True,
        "line_type": "deeplsd",  # 'lsd' or 'deeplsd'
        "min_lines": 5,  # only trust images with at least this many lines
        "verbose": False,
    }

    def _init(self, conf):
        if conf.SOLVER_FLAGS in [
            [True, False, False, False, False],
            [False, False, True, False, False],
        ]:
            self.vertical = np.array([random.random() / 1e12, 1, random.random() / 1e12])
            self.vertical /= np.linalg.norm(self.vertical)
        else:
            self.vertical = np.array([0.0, 1, 0.0])

        self.line_detector = LineDetector(line_detector=conf.line_type)

        self.verbose = conf.verbose

    def visualize_lines(self, vp, lines, img, K):
        vp_labels = get_labels_from_vp(
            lines[:, :, [1, 0]], project_vp_to_image(vp, K), threshold=self.conf.th_pixels
        )[0]

        plot_images([img, img])
        plot_lines([lines, np.empty((0, 2, 2))])
        plot_vp([np.empty((0, 2, 2)), lines], [[], vp_labels])

        plt.show()

    def get_vvp(self, vp, K):
        best_idx, best_cossim = 0, -1
        for i, point in enumerate(vp):
            cossim = np.dot(self.vertical, point) / np.linalg.norm(point)
            point = -point * np.dot(self.vertical, point)
            try:
                gravity = Gravity(np.linalg.inv(K) @ point)
            except:
                continue

            if (
                np.abs(cossim) > best_cossim
                and gravity.pitch.abs() <= np.pi / 4
                and gravity.roll.abs() <= np.pi / 4
            ):
                best_idx, best_cossim = i, np.abs(cossim)

        vvp = vp[best_idx]
        return -vvp * np.sign(np.dot(self.vertical, vvp))

    def _forward(self, data):
        device = data["image"].device
        images = data["image"].cpu()

        estimations = []
        for idx, img in enumerate(images.unbind(0)):
            if "prior_gravity" in data:
                self.vertical = -data["prior_gravity"][idx].vec3d.cpu().numpy()
            else:
                self.vertical = np.array([0.0, 1, 0.0])

            img = img.numpy().transpose(1, 2, 0) * 255
            img = img.astype(np.uint8)
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            lines = self.line_detector.detect_lines(gray_img)[:, :, [1, 0]]

            if len(lines) < self.conf.min_lines:
                logger.warning("Not enough lines detected! Skipping...")
                gravity = Gravity.from_rp(np.nan, np.nan)
                camera = Pinhole.from_dict(
                    {"f": np.nan, "height": img.shape[0], "width": img.shape[1]}
                )
                estimations.append({"camera": camera, "gravity": gravity})
                continue

            principle_point = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0])
            f, vp = run_hybrid_uncalibrated(
                lines - principle_point[None, None, :],
                self.vertical,
                th_pixels=self.conf.th_pixels,
                ls_refinement=self.conf.ls_refinement,
                nms=self.conf.nms,
                magsac_scoring=self.conf.magsac_scoring,
                sprt=True,
                solver_flags=self.conf.SOLVER_FLAGS,
            )
            vp[:, 1] *= -1

            K = np.array(
                [[f, 0.0, principle_point[0]], [0.0, f, principle_point[1]], [0.0, 0.0, 1.0]]
            )

            if self.verbose:
                self.visualize_lines(vp, lines, img, K)

            vp_labels = get_labels_from_vp(
                lines[:, :, [1, 0]], project_vp_to_image(vp, K), threshold=self.conf.th_pixels
            )[0]
            out = {"vp": vp, "lines": lines, "K": K, "vp_labels": vp_labels}

            vp = project_vp_to_image(vp, K)

            vvp = self.get_vvp(vp, K)

            vvp = -vvp * np.sign(np.dot(self.vertical, vvp))
            try:
                K_inv = np.linalg.inv(K)
                gravity = Gravity(K_inv @ vvp)
            except np.linalg.LinAlgError:
                gravity = Gravity.from_rp(np.nan, np.nan)

            camera = Pinhole.from_dict({"f": f, "height": img.shape[0], "width": img.shape[1]})
            estimations.append({"camera": camera, "gravity": gravity})

        if len(estimations) == 0:
            return {}

        gravity = torch.stack([Gravity(est["gravity"].vec3d) for est in estimations], dim=0)
        camera = torch.stack([Pinhole(est["camera"]._data) for est in estimations], dim=0)

        return {"camera": camera.float().to(device), "gravity": gravity.float().to(device)} | out

    def metrics(self, pred, data):
        pred_cam, gt_cam = pred["camera_opt"], data["camera"]
        pred_gravity, gt_gravity = pred["gravity_opt"], data["gravity"]

        return {
            "roll_opt_error": roll_error(pred_gravity, gt_gravity),
            "pitch_opt_error": pitch_error(pred_gravity, gt_gravity),
            "gravity_opt_error": gravity_error(pred_gravity, gt_gravity),
            "vfov_opt_error": vfov_error(pred_cam, gt_cam),
        }

    def loss(self, pred, data):
        return {}, self.metrics(pred, data)
