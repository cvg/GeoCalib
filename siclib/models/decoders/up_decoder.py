"""up decoder head.

Adapted from https://github.com/jinlinyi/PerspectiveFields
"""

import logging

import torch
from torch import nn
from torch.nn import functional as F

from siclib.models import get_model
from siclib.models.base_model import BaseModel
from siclib.models.utils.metrics import up_error
from siclib.models.utils.perspective_encoding import decode_up_bin
from siclib.utils.conversions import deg2rad

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


class UpDecoder(BaseModel):
    default_conf = {
        "loss_type": "l1",
        "use_loss": True,
        "use_uncertainty_loss": True,
        "loss_weight": 1.0,
        "recall_thresholds": [1, 3, 5, 10],
        "decoder": {"name": "decoders.light_hamburger", "predict_uncertainty": True},
    }

    required_data_keys = ["features"]

    def _init(self, conf):
        self.loss_type = conf.loss_type
        self.loss_weight = conf.loss_weight

        self.use_uncertainty_loss = conf.use_uncertainty_loss
        self.predict_uncertainty = conf.decoder.predict_uncertainty

        self.num_classes = 2
        self.is_classification = self.conf.loss_type == "classification"
        if self.is_classification:
            self.num_classes = 73

        self.decoder = get_model(conf.decoder.name)(conf.decoder)
        self.linear_pred_up = nn.Conv2d(self.decoder.out_channels, self.num_classes, kernel_size=1)

    def calculate_losses(self, predictions, targets, confidence=None):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163

        residuals = predictions - targets
        if self.loss_type == "l2":
            loss = (residuals**2).sum(axis=1)
        elif self.loss_type == "l1":
            loss = residuals.abs().sum(axis=1)
        elif self.loss_type == "dot":
            loss = 1 - (residuals * targets).sum(axis=1)
        elif self.loss_type == "cauchy":
            c = 0.007  # -> corresponds to about 5 degrees
            residuals = (residuals**2).sum(axis=1)
            loss = c**2 / 2 * torch.log(1 + residuals / c**2)
        elif self.loss_type == "huber":
            c = deg2rad(1)
            loss = nn.HuberLoss(reduction="none", delta=c)(predictions, targets).sum(axis=1)
        else:
            raise NotImplementedError(f"Unknown loss type {self.conf.loss_type}")

        if confidence is not None and self.use_uncertainty_loss:
            conf_weight = confidence / confidence.sum(axis=(-2, -1), keepdims=True)
            conf_weight = conf_weight * (conf_weight.size(-1) * conf_weight.size(-2))
            loss = loss * conf_weight.detach()

        losses = {f"up-{self.loss_type}-loss": loss.mean(axis=(1, 2))}
        losses = {k: v * self.loss_weight for k, v in losses.items()}

        return losses

    def _forward(self, data):
        out = {}
        x, log_confidence = self.decoder(data["features"])
        up = self.linear_pred_up(x)

        if self.predict_uncertainty:
            out["up_confidence"] = torch.sigmoid(log_confidence)

        if self.is_classification:
            out["up_field"] = decode_up_bin(up.argmax(dim=1), self.num_classes)
            return out

        up = F.normalize(up, dim=1)

        out["up_field"] = up
        return out

    def loss(self, pred, data):
        if not self.conf.use_loss or self.is_classification:
            return {}, self.metrics(pred, data)

        predictions = pred["up_field"]
        targets = data["up_field"]

        losses = self.calculate_losses(predictions, targets, pred.get("up_confidence"))

        total = 0 + losses[f"up-{self.loss_type}-loss"]
        losses |= {"up_total": total}
        return losses, self.metrics(pred, data)

    def metrics(self, pred, data):
        predictions = pred["up_field"]
        targets = data["up_field"]

        mask = predictions.sum(axis=1) != 0

        error = up_error(predictions, targets) * mask
        out = {"up_angle_error": error.mean(axis=(1, 2))}

        if "up_confidence" in pred:
            weighted_error = (error * pred["up_confidence"]).sum(axis=(1, 2))
            out["up_angle_error_weighted"] = weighted_error / pred["up_confidence"].sum(axis=(1, 2))

        for th in self.conf.recall_thresholds:
            rec = (error < th).float().mean(axis=(1, 2))
            out[f"up_angle_recall@{th}"] = rec

        return out
