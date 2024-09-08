import logging
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import Identity

from siclib.geometry.camera import SimpleRadial
from siclib.geometry.gravity import Gravity
from siclib.models.base_model import BaseModel
from siclib.models.utils.metrics import dist_error, pitch_error, roll_error, vfov_error
from siclib.models.utils.modules import _DenseBlock, _Transition
from siclib.utils.conversions import deg2rad, pitch2rho, rho2pitch

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


def get_centers_and_edges(min: float, max: float, num_bins: int) -> Tuple[np.ndarray, torch.Tensor]:
    centers = torch.linspace(min, max + ((max - min) / (num_bins - 1)), num_bins + 1).float()
    edges = centers.detach() - ((centers.detach()[1] - centers[0]) / 2.0)
    return centers, edges


class DeepCalib(BaseModel):
    default_conf = {
        "name": "densenet",
        "model": "densenet161",
        "loss": "NLL",
        "num_bins": 256,
        "freeze_batch_normalization": False,
        "model": "densenet161",
        "pretrained": True,  # whether to use ImageNet weights
        "heads": ["roll", "rho", "vfov", "k1_hat"],
        "flip": [],  # keys of predictions to flip the sign of
        "rpf_scales": [1, 1, 1],
        "bounds": {
            "roll": [-45, 45],
            "rho": [-1, 1],
            "vfov": [20, 105],
            "k1_hat": [-0.7, 0.7],
        },
        "use_softamax": False,
    }

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    strict_conf = False

    required_data_keys = ["image", "image_size"]

    def _init(self, conf):
        self.is_classification = True if self.conf.loss in ["NLL"] else False

        self.num_bins = conf.num_bins

        self.roll_centers, self.roll_edges = get_centers_and_edges(
            deg2rad(conf.bounds.roll[0]), deg2rad(conf.bounds.roll[1]), self.num_bins
        )

        self.rho_centers, self.rho_edges = get_centers_and_edges(
            conf.bounds.rho[0], conf.bounds.rho[1], self.num_bins
        )

        self.fov_centers, self.fov_edges = get_centers_and_edges(
            deg2rad(conf.bounds.vfov[0]), deg2rad(conf.bounds.vfov[1]), self.num_bins
        )

        self.k1_hat_centers, self.k1_hat_edges = get_centers_and_edges(
            conf.bounds.k1_hat[0], conf.bounds.k1_hat[1], self.num_bins
        )

        Model = getattr(torchvision.models, conf.model)
        weights = "DEFAULT" if self.conf.pretrained else None
        self.model = Model(weights=weights)

        layers = []

        # 2208 for 161 layers. 1024 for 121
        num_features = self.model.classifier.in_features
        head_layers = 3
        layers.append(_Transition(num_features, num_features // 2))
        num_features = num_features // 2
        growth_rate = 32
        layers.append(
            _DenseBlock(
                num_layers=head_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=4,
                drop_rate=0,
            )
        )
        layers.append(nn.BatchNorm2d(num_features + head_layers * growth_rate))
        layers.append(nn.ReLU())
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(num_features + head_layers * growth_rate, 512))
        layers.append(nn.ReLU())
        self.model.classifier = Identity()
        self.model.features.norm5 = Identity()

        if self.is_classification:
            layers.append(nn.Linear(512, self.num_bins))
            layers.append(nn.LogSoftmax(dim=1))
        else:
            layers.append(nn.Linear(512, 1))
            layers.append(nn.Tanh())

        self.roll_head = nn.Sequential(*deepcopy(layers))
        self.rho_head = nn.Sequential(*deepcopy(layers))
        self.vfov_head = nn.Sequential(*deepcopy(layers))
        self.k1_hat_head = nn.Sequential(*deepcopy(layers))

    def bins_to_val(self, centers, pred):
        if centers.device != pred.device:
            centers = centers.to(pred.device)

        if not self.conf.use_softamax:
            return centers[pred.argmax(1)]

        beta = 1e-0
        pred_softmax = F.softmax(pred / beta, dim=1)
        weighted_centers = centers[:-1].unsqueeze(0) * pred_softmax
        val = weighted_centers.sum(dim=1)
        return val

    def _forward(self, data):
        image = data["image"]
        mean, std = image.new_tensor(self.mean), image.new_tensor(self.std)
        image = (image - mean[:, None, None]) / std[:, None, None]
        shared_features = self.model.features(image)
        pred = {}

        if "roll" in self.conf.heads:
            pred["roll"] = self.roll_head(shared_features)
        if "rho" in self.conf.heads:
            pred["rho"] = self.rho_head(shared_features)
        if "vfov" in self.conf.heads:
            pred["vfov"] = self.vfov_head(shared_features)
            if "vfov" in self.conf.flip:
                pred["vfov"] = pred["vfov"] * -1
        if "k1_hat" in self.conf.heads:
            pred["k1_hat"] = self.k1_hat_head(shared_features)

        size = data["image_size"]
        w, h = size[:, 0], size[:, 1]

        if self.is_classification:
            parameters = {
                "roll": self.bins_to_val(self.roll_centers, pred["roll"]),
                "rho": self.bins_to_val(self.rho_centers, pred["rho"]),
                "vfov": self.bins_to_val(self.fov_centers, pred["vfov"]),
                "k1_hat": self.bins_to_val(self.k1_hat_centers, pred["k1_hat"]),
                "width": w,
                "height": h,
            }

            for k in self.conf.flip:
                parameters[k] = parameters[k] * -1

            for i, k in enumerate(["roll", "rho", "vfov"]):
                parameters[k] = parameters[k] * self.conf.rpf_scales[i]

            camera = SimpleRadial.from_dict(parameters)

            roll, pitch = parameters["roll"], rho2pitch(parameters["rho"], camera.f[..., 1], h)
            gravity = Gravity.from_rp(roll, pitch)

        else:  # regression
            if "roll" in self.conf.heads:
                pred["roll"] = pred["roll"] * deg2rad(45)
            if "vfov" in self.conf.heads:
                pred["vfov"] = (pred["vfov"] + 1) * deg2rad((105 - 20) / 2 + 20)

            camera = SimpleRadial.from_dict(pred | {"width": w, "height": h})
            gravity = Gravity.from_rp(pred["roll"], pred["pitch"])

        return pred | {"camera": camera, "gravity": gravity}

    def loss(self, pred, data):
        loss = {"total": 0}
        if self.conf.loss == "Huber":
            loss_fn = nn.HuberLoss(reduction="none")
        elif self.conf.loss == "L1":
            loss_fn = nn.L1Loss(reduction="none")
        elif self.conf.loss == "L2":
            loss_fn = nn.MSELoss(reduction="none")
        elif self.conf.loss == "NLL":
            loss_fn = nn.NLLLoss(reduction="none")

        gt_cam = data["camera"]

        if "roll" in self.conf.heads:
            # nbins softmax values if classification, else scalar value
            gt_roll = data["gravity"].roll.float()
            pred_roll = pred["roll"].float()

            if gt_roll.device != self.roll_edges.device:
                self.roll_edges = self.roll_edges.to(gt_roll.device)
                self.roll_centers = self.roll_centers.to(gt_roll.device)

            if self.is_classification:
                gt_roll = (
                    torch.bucketize(gt_roll.contiguous(), self.roll_edges) - 1
                )  # converted to class

                assert (gt_roll >= 0).all(), gt_roll
                assert (gt_roll < self.num_bins).all(), gt_roll
            else:
                assert pred_roll.dim() == gt_roll.dim()

            loss_roll = loss_fn(pred_roll, gt_roll)
            loss["roll"] = loss_roll
            loss["total"] += loss_roll

        if "rho" in self.conf.heads:
            gt_rho = pitch2rho(data["gravity"].pitch, gt_cam.f[..., 1], gt_cam.size[..., 1]).float()
            pred_rho = pred["rho"].float()

            if gt_rho.device != self.rho_edges.device:
                self.rho_edges = self.rho_edges.to(gt_rho.device)
                self.rho_centers = self.rho_centers.to(gt_rho.device)

            if self.is_classification:
                gt_rho = torch.bucketize(gt_rho.contiguous(), self.rho_edges) - 1

                assert (gt_rho >= 0).all(), gt_rho
                assert (gt_rho < self.num_bins).all(), gt_rho
            else:
                assert pred_rho.dim() == gt_rho.dim()

            # print(f"Rho: {gt_rho.shape}, {pred_rho.shape}")
            loss_rho = loss_fn(pred_rho, gt_rho)
            loss["rho"] = loss_rho
            loss["total"] += loss_rho

        if "vfov" in self.conf.heads:
            gt_vfov = gt_cam.vfov.float()
            pred_vfov = pred["vfov"].float()

            if gt_vfov.device != self.fov_edges.device:
                self.fov_edges = self.fov_edges.to(gt_vfov.device)
                self.fov_centers = self.fov_centers.to(gt_vfov.device)

            if self.is_classification:
                gt_vfov = torch.bucketize(gt_vfov.contiguous(), self.fov_edges) - 1

                assert (gt_vfov >= 0).all(), gt_vfov
                assert (gt_vfov < self.num_bins).all(), gt_vfov
            else:
                min_vfov = deg2rad(self.conf.bounds.vfov[0])
                max_vfov = deg2rad(self.conf.bounds.vfov[1])
                gt_vfov = (2 * (gt_vfov - min_vfov) / (max_vfov - min_vfov)) - 1
                assert pred_vfov.dim() == gt_vfov.dim()

            loss_vfov = loss_fn(pred_vfov, gt_vfov)
            loss["vfov"] = loss_vfov
            loss["total"] += loss_vfov

        if "k1_hat" in self.conf.heads:
            gt_k1_hat = data["camera"].k1_hat.float()
            pred_k1_hat = pred["k1_hat"].float()

            if gt_k1_hat.device != self.k1_hat_edges.device:
                self.k1_hat_edges = self.k1_hat_edges.to(gt_k1_hat.device)
                self.k1_hat_centers = self.k1_hat_centers.to(gt_k1_hat.device)

            if self.is_classification:
                gt_k1_hat = torch.bucketize(gt_k1_hat.contiguous(), self.k1_hat_edges) - 1

                assert (gt_k1_hat >= 0).all(), gt_k1_hat
                assert (gt_k1_hat < self.num_bins).all(), gt_k1_hat
            else:
                assert pred_k1_hat.dim() == gt_k1_hat.dim()

            loss_k1_hat = loss_fn(pred_k1_hat, gt_k1_hat)
            loss["k1_hat"] = loss_k1_hat
            loss["total"] += loss_k1_hat

        return loss, self.metrics(pred, data)

    def metrics(self, pred, data):
        pred_cam, gt_cam = pred["camera"], data["camera"]
        pred_gravity, gt_gravity = pred["gravity"], data["gravity"]

        return {
            "roll_error": roll_error(pred_gravity, gt_gravity),
            "pitch_error": pitch_error(pred_gravity, gt_gravity),
            "vfov_error": vfov_error(pred_cam, gt_cam),
            "k1_error": dist_error(pred_cam, gt_cam),
        }
