"""Latitude decoder head.

Adapted from https://github.com/jinlinyi/PerspectiveFields
"""

import logging

import torch
from torch import nn

from siclib.models import get_model
from siclib.models.base_model import BaseModel
from siclib.models.utils.metrics import latitude_error
from siclib.models.utils.perspective_encoding import decode_bin_latitude
from siclib.utils.conversions import deg2rad

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


class LatitudeDecoder(BaseModel):
    default_conf = {
        "loss_type": "l1",
        "use_loss": True,
        "use_uncertainty_loss": True,
        "loss_weight": 1.0,
        "recall_thresholds": [1, 3, 5, 10],
        "use_tanh": True,  # backward compatibility to original perspective weights
        "decoder": {"name": "decoders.light_hamburger", "predict_uncertainty": True},
    }

    required_data_keys = ["features"]

    def _init(self, conf):
        self.loss_type = conf.loss_type
        self.loss_weight = conf.loss_weight

        self.use_uncertainty_loss = conf.use_uncertainty_loss
        self.predict_uncertainty = conf.decoder.predict_uncertainty

        self.num_classes = 1
        self.is_classification = self.conf.loss_type == "classification"
        if self.is_classification:
            self.num_classes = 180

        self.decoder = get_model(conf.decoder.name)(conf.decoder)
        self.linear_pred_latitude = nn.Conv2d(
            self.decoder.out_channels, self.num_classes, kernel_size=1
        )

    def calculate_losses(self, predictions, targets, confidence=None):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163

        residuals = predictions - targets
        if self.loss_type == "l2":
            loss = (residuals**2).sum(axis=1)
        elif self.loss_type == "l1":
            loss = residuals.abs().sum(axis=1)
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

        losses = {f"latitude-{self.loss_type}-loss": loss.mean(axis=(1, 2))}
        losses = {k: v * self.loss_weight for k, v in losses.items()}

        return losses

    def _forward(self, data):
        out = {}
        x, log_confidence = self.decoder(data["features"])
        lat = self.linear_pred_latitude(x)

        if self.predict_uncertainty:
            out["latitude_confidence"] = torch.sigmoid(log_confidence)

        if self.is_classification:
            out["latitude_field_logits"] = lat
            out["latitude_field"] = decode_bin_latitude(
                lat.argmax(dim=1), self.num_classes
            ).unsqueeze(1)
            return out

        eps = 1e-5  # avoid nan in backward of asin
        lat = torch.tanh(lat) if self.conf.use_tanh else lat
        lat = torch.asin(torch.clamp(lat, -1 + eps, 1 - eps))

        out["latitude_field"] = lat
        return out

    def loss(self, pred, data):
        if not self.conf.use_loss or self.is_classification:
            return {}, self.metrics(pred, data)

        predictions = pred["latitude_field"]
        targets = data["latitude_field"]

        losses = self.calculate_losses(predictions, targets, pred.get("latitude_confidence"))

        total = 0 + losses[f"latitude-{self.loss_type}-loss"]
        losses |= {"latitude_total": total}
        return losses, self.metrics(pred, data)

    def metrics(self, pred, data):
        predictions = pred["latitude_field"]
        targets = data["latitude_field"]

        error = latitude_error(predictions, targets)
        out = {"latitude_angle_error": error.mean(axis=(1, 2))}

        if "latitude_confidence" in pred:
            weighted_error = (error * pred["latitude_confidence"]).sum(axis=(1, 2))
            out["latitude_angle_error_weighted"] = weighted_error / pred["latitude_confidence"].sum(
                axis=(1, 2)
            )

        for th in self.conf.recall_thresholds:
            rec = (error < th).float().mean(axis=(1, 2))
            out[f"latitude_angle_recall@{th}"] = rec

        return out
