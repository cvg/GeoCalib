"""Entrypoint for torch hub."""

dependencies = ["torch", "torchvision", "opencv-python", "kornia", "matplotlib"]

from geocalib import GeoCalib


def model(*args, **kwargs):
    """Pre-trained Geocalib model.

    Args:
        weights (str): trained variant, "pinhole" (default) or "distorted".
    """
    return GeoCalib(*args, **kwargs)
