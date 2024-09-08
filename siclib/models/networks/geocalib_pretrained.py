"""Interface for GeoCalib inference package."""

from geocalib import GeoCalib
from siclib.models.base_model import BaseModel


# mypy: ignore-errors
class GeoCalibPretrained(BaseModel):
    """GeoCalib pretrained model."""

    default_conf = {
        "camera_model": "pinhole",
        "model_weights": "pinhole",
    }

    def _init(self, conf):
        """Initialize pretrained GeoCalib model."""
        self.model = GeoCalib(weights=conf.model_weights)

    def _forward(self, data):
        """Forward pass."""
        priors = {}
        if "prior_gravity" in data:
            priors["gravity"] = data["prior_gravity"]

        if "prior_focal" in data:
            priors["focal"] = data["prior_focal"]

        results = self.model.calibrate(
            data["image"], camera_model=self.conf.camera_model, priors=priors
        )

        return results

    def metrics(self, pred, data):
        """Compute metrics."""
        raise NotImplementedError("GeoCalibPretrained does not support metrics computation.")

    def loss(self, pred, data):
        """Compute loss."""
        raise NotImplementedError("GeoCalibPretrained does not support loss computation.")
