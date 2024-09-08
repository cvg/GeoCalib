"""Perspective fields decoder heads.

Adapted from https://github.com/jinlinyi/PerspectiveFields
"""

import logging

from siclib.models import get_model
from siclib.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


class PerspectiveDecoder(BaseModel):
    default_conf = {
        "up_decoder": {"name": "decoders.up_decoder"},
        "latitude_decoder": {"name": "decoders.latitude_decoder"},
    }

    required_data_keys = ["features"]

    def _init(self, conf):
        logger.debug(f"Initializing PerspectiveDecoder with config: {conf}")
        self.use_up = conf.up_decoder is not None
        self.use_latitude = conf.latitude_decoder is not None

        if self.use_up:
            self.up_head = get_model(conf.up_decoder.name)(conf.up_decoder)

        if self.use_latitude:
            self.latitude_head = get_model(conf.latitude_decoder.name)(conf.latitude_decoder)

    def _forward(self, data):
        out_up = self.up_head(data) if self.use_up else {}
        out_lat = self.latitude_head(data) if self.use_latitude else {}
        return out_up | out_lat

    def loss(self, pred, data):
        ref = data["up_field"] if self.use_up else data["latitude_field"]

        total = ref.new_zeros(ref.shape[0])
        losses, metrics = {}, {}
        if self.use_up:
            up_losses, up_metrics = self.up_head.loss(pred, data)
            losses |= up_losses
            metrics |= up_metrics
            total = total + losses.get("up_total", 0)

        if self.use_latitude:
            latitude_losses, latitude_metrics = self.latitude_head.loss(pred, data)
            losses |= latitude_losses
            metrics |= latitude_metrics
            total = total + losses.get("latitude_total", 0)

        losses["perspective_total"] = total
        return losses, metrics
