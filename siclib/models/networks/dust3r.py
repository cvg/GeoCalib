"""Wrapper for DUSt3R model to estimate focal length.

DUSt3R: Geometric 3D Vision Made Easy, https://arxiv.org/abs/2312.14132
"""

import sys

sys.path.append("third_party/dust3r")

import torch
from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images

from siclib.geometry.base_camera import BaseCamera
from siclib.geometry.gravity import Gravity
from siclib.models import BaseModel

# mypy: ignore-errors


class Dust3R(BaseModel):
    """DUSt3R model for focal length estimation."""

    default_conf = {
        "model_path": "weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "device": "cuda",
        "batch_size": 1,
        "schedule": "cosine",
        "lr": 0.01,
        "niter": 300,
        "show_scene": False,
    }

    required_data_keys = ["path"]

    def _init(self, conf):
        """Initialize the DUSt3R model."""
        self.model = load_model(conf["model_path"], conf["device"])

    def _forward(self, data):
        """Forward pass of the DUSt."""
        assert len(data["path"]) == 1, f"Only batch size of 1 is supported (bs={len(data['path'])}"

        path = data["path"][0]
        images = [path] * 2

        with torch.enable_grad():
            images = load_images(images, size=512)
            pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
            output = inference(
                pairs, self.model, self.conf["device"], batch_size=self.conf["batch_size"]
            )
            scene = global_aligner(
                output, device=self.conf["device"], mode=GlobalAlignerMode.PointCloudOptimizer
            )
            _ = scene.compute_global_alignment(
                init="mst",
                niter=self.conf["niter"],
                schedule=self.conf["schedule"],
                lr=self.conf["lr"],
            )

        # retrieve useful values from scene:
        focals = scene.get_focals().mean(dim=0)

        h, w = images[0]["true_shape"][:, 0], images[0]["true_shape"][:, 1]
        h, w = focals.new_tensor(h), focals.new_tensor(w)

        camera = BaseCamera.from_dict({"height": h, "width": w, "f": focals})
        gravity = Gravity.from_rp([0.0], [0.0])

        if self.conf["show_scene"]:
            scene.show()

        return {"camera": camera, "gravity": gravity}

    def loss(self, pred, data):
        """Loss function for DUSt3R model."""
        return {}, {}
