import logging

from siclib.models.optimization.lm_optimizer import LMOptimizer

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


class InferenceOptimizer(LMOptimizer):
    default_conf = {
        # Camera model parameters
        "camera_model": "pinhole",  # {"pinhole", "simple_radial", "simple_spherical"}
        "shared_intrinsics": False,  # share focal length across all images in batch
        "estimate_gravity": True,
        "estimate_focal": True,
        "estimate_k1": True,  # will be ignored if camera_model is pinhole
        # LM optimizer parameters
        "num_steps": 30,
        "lambda_": 0.1,
        "fix_lambda": False,
        "early_stop": True,
        "atol": 1e-8,
        "rtol": 1e-8,
        "use_spherical_manifold": True,  # use spherical manifold for gravity optimization
        "use_log_focal": True,  # use log focal length for optimization
        # Loss function parameters
        "loss_fn": "huber_loss",  # {"squared_loss", "huber_loss"}
        "up_loss_fn_scale": 1e-2,
        "lat_loss_fn_scale": 1e-2,
        "init_conf": {"name": "trivial"},  # pass config of other models to use as initializer
        # Misc
        "loss_weight": 1,
        "verbose": False,
    }
