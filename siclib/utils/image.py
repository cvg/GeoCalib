"""Image preprocessing utilities."""

import collections.abc as collections
from pathlib import Path
from typing import Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf

from siclib.utils.tensor import fit_features_to_multiple

# mypy: ignore-errors


class ImagePreprocessor:
    """Preprocess images for calibration."""

    default_conf = {
        "resize": 320,  # target edge length, None for no resizing
        "edge_divisible_by": None,
        "side": "short",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
        "square_crop": False,
        "add_padding_mask": False,
        "resize_backend": "kornia",  # torchvision, kornia
    }

    def __init__(self, conf) -> None:
        """Initialize the image preprocessor."""
        super().__init__()
        default_conf = OmegaConf.create(self.default_conf)
        OmegaConf.set_struct(default_conf, True)
        self.conf = OmegaConf.merge(default_conf, conf)

    def __call__(self, img: torch.Tensor, interpolation: Optional[str] = None) -> dict:
        """Resize and preprocess an image, return image and resize scale."""
        h, w = img.shape[-2:]
        size = h, w

        if self.conf.square_crop:
            min_size = min(h, w)
            offset = (h - min_size) // 2, (w - min_size) // 2
            img = img[:, offset[0] : offset[0] + min_size, offset[1] : offset[1] + min_size]
            size = img.shape[-2:]

        if self.conf.resize is not None:
            if interpolation is None:
                interpolation = self.conf.interpolation
            size = self.get_new_image_size(h, w)
            img = self.resize(img, size, interpolation)

        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        T = np.diag([scale[0].cpu(), scale[1].cpu(), 1])

        data = {
            "scales": scale,
            "image_size": np.array(size[::-1]),
            "transform": T,
            "original_image_size": np.array([w, h]),
        }

        if self.conf.edge_divisible_by is not None:
            # crop to make the edge divisible by a number
            w_, h_ = img.shape[-1], img.shape[-2]
            img, _ = fit_features_to_multiple(img, self.conf.edge_divisible_by, crop=True)
            crop_pad = torch.Tensor([img.shape[-1] - w_, img.shape[-2] - h_]).to(img)
            data["crop_pad"] = crop_pad
            data["image_size"] = np.array([img.shape[-1], img.shape[-2]])

        data["image"] = img
        return data

    def resize(self, img: torch.Tensor, size: Tuple[int, int], interpolation: str) -> torch.Tensor:
        """Resize an image using the specified backend."""
        if self.conf.resize_backend == "kornia":
            return kornia.geometry.transform.resize(
                img,
                size,
                side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners,
                interpolation=interpolation,
            )
        elif self.conf.resize_backend == "torchvision":
            return torchvision.transforms.Resize(size, antialias=self.conf.antialias)(img)
        else:
            raise ValueError(f"{self.conf.resize_backend} not implemented.")

    def load_image(self, image_path: Path) -> dict:
        """Load an image from a path and preprocess it."""
        return self(load_image(image_path))

    def get_new_image_size(self, h: int, w: int) -> Tuple[int, int]:
        """Get the new image size after resizing."""
        side = self.conf.side
        if isinstance(self.conf.resize, collections.Iterable):
            assert len(self.conf.resize) == 2
            return tuple(self.conf.resize)
        side_size = self.conf.resize
        aspect_ratio = w / h
        if side not in ("short", "long", "vert", "horz"):
            raise ValueError(
                f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'"
            )
        return (
            (side_size, int(side_size * aspect_ratio))
            if side == "vert" or (side != "horz" and (side == "short") ^ (aspect_ratio < 1.0))
            else (int(side_size / aspect_ratio), side_size)
        )


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def torch_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Normalize and reorder the dimensions of an image tensor."""
    if image.ndim == 3:
        image = image.permute((1, 2, 0))  # CxHxW to HxWxC
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return (image.cpu().detach().numpy() * 255).astype(np.uint8)


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def write_image(img: torch.Tensor, path: Path):
    """Write an image tensor to a file."""
    img = torch_image_to_numpy(img) if isinstance(img, torch.Tensor) else img
    cv2.imwrite(str(path), img[..., ::-1])


def load_image(path: Path, grayscale: bool = False, return_tensor: bool = True) -> torch.Tensor:
    """Load an image from a path and return as a tensor."""
    image = read_image(path, grayscale=grayscale)
    if return_tensor:
        return numpy_image_to_torch(image)

    assert image.ndim in [2, 3], f"Not an image: {image.shape}"
    image = image[None] if image.ndim == 2 else image
    return torch.tensor(image.copy(), dtype=torch.uint8)
