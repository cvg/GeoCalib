import logging

from siclib.models import get_model
from siclib.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


class GeoCalib(BaseModel):
    default_conf = {
        "backbone": {"name": "encoders.mscan"},
        "ll_enc": {"name": "encoders.low_level_encoder"},
        "perspective_decoder": {"name": "decoders.perspective_decoder"},
        "optimizer": {"name": "optimization.lm_optimizer"},
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        logger.debug(f"Initializing GeoCalib with {conf}")
        self.backbone = get_model(conf.backbone["name"])(conf.backbone)
        self.ll_enc = get_model(conf.ll_enc["name"])(conf.ll_enc) if conf.ll_enc else None

        self.perspective_decoder = get_model(conf.perspective_decoder["name"])(
            conf.perspective_decoder
        )

        self.optimizer = (
            get_model(conf.optimizer["name"])(conf.optimizer) if conf.optimizer else None
        )

    def _forward(self, data):
        backbone_out = self.backbone(data)
        features = {"hl": backbone_out["features"], "padding": backbone_out.get("padding", None)}

        if self.ll_enc is not None:
            features["ll"] = self.ll_enc(data)["features"]  # low level features

        out = self.perspective_decoder({"features": features})

        out |= {
            k: data[k]
            for k in ["image", "scales", "prior_gravity", "prior_focal", "prior_k1"]
            if k in data
        }

        if self.optimizer is not None:
            out |= self.optimizer(out)

        return out

    def loss(self, pred, data):
        losses, metrics = self.perspective_decoder.loss(pred, data)
        total = losses["perspective_total"]

        if self.optimizer is not None:
            opt_losses, param_metrics = self.optimizer.loss(pred, data)
            losses |= opt_losses
            metrics |= param_metrics
            total = total + opt_losses["param_total"]

        losses["total"] = total
        return losses, metrics
