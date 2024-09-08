from typing import Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from omegaconf import OmegaConf


# flake8: noqa
# mypy: ignore-errors
class IdentityTransform(A.ImageOnlyTransform):
    def apply(self, img, **params):
        return img

    def get_transform_init_args_names(self):
        return ()


class RandomAdditiveShade(A.ImageOnlyTransform):
    def __init__(
        self,
        nb_ellipses=10,
        transparency_limit=[-0.5, 0.8],
        kernel_size_limit=[150, 350],
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.nb_ellipses = nb_ellipses
        self.transparency_limit = transparency_limit
        self.kernel_size_limit = kernel_size_limit

    def apply(self, img, **params):
        if img.dtype == np.float32:
            shaded = self._py_additive_shade(img * 255.0)
            shaded /= 255.0
        elif img.dtype == np.uint8:
            shaded = self._py_additive_shade(img.astype(np.float32))
            shaded = shaded.astype(np.uint8)
        else:
            raise NotImplementedError(f"Data augmentation not available for type: {img.dtype}")
        return shaded

    def _py_additive_shade(self, img):
        grayscale = len(img.shape) == 2
        if grayscale:
            img = img[None]
        min_dim = min(img.shape[:2]) / 4
        mask = np.zeros(img.shape[:2], img.dtype)
        for i in range(self.nb_ellipses):
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
            y = np.random.randint(max_rad, img.shape[0] - max_rad)
            angle = np.random.rand() * 90
            cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

        transparency = np.random.uniform(*self.transparency_limit)
        ks = np.random.randint(*self.kernel_size_limit)
        if (ks % 2) == 0:  # kernel_size has to be odd
            ks += 1
        mask = cv2.GaussianBlur(mask.astype(np.float32), (ks, ks), 0)
        shaded = img * (1 - transparency * mask[..., np.newaxis] / 255.0)
        out = np.clip(shaded, 0, 255)
        if grayscale:
            out = out.squeeze(0)
        return out

    def get_transform_init_args_names(self):
        return "transparency_limit", "kernel_size_limit", "nb_ellipses"


def kw(entry: Union[float, dict], n=None, **default):
    if not isinstance(entry, dict):
        entry = {"p": entry}
    entry = OmegaConf.create(entry)
    if n is not None:
        entry = default.get(n, entry)
    return OmegaConf.merge(default, entry)


def kwi(entry: Union[float, dict], n=None, **default):
    conf = kw(entry, n=n, **default)
    return {k: conf[k] for k in set(default.keys()).union(set(["p"]))}


def replay_str(transforms, s="Replay:\n", log_inactive=True):
    for t in transforms:
        if "transforms" in t.keys():
            s = replay_str(t["transforms"], s=s)
        elif t["applied"] or log_inactive:
            s += t["__class_fullname__"] + " " + str(t["applied"]) + "\n"
    return s


class BaseAugmentation(object):
    base_default_conf = {
        "name": "???",
        "shuffle": False,
        "p": 1.0,
        "verbose": False,
        "dtype": "uint8",  # (byte, float)
    }

    default_conf = {}

    def __init__(self, conf={}):
        """Perform some logic and call the _init method of the child model."""
        default_conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf),
            OmegaConf.create(self.default_conf),
        )
        OmegaConf.set_struct(default_conf, True)
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(self.conf, True)
        self._init(self.conf)

        self.conf = OmegaConf.merge(self.conf, conf)
        if self.conf.verbose:
            self.compose = A.ReplayCompose
        else:
            self.compose = A.Compose
        if self.conf.dtype == "uint8":
            self.dtype = np.uint8
            self.preprocess = A.FromFloat(always_apply=True, dtype="uint8")
            self.postprocess = A.ToFloat(always_apply=True)
        elif self.conf.dtype == "float32":
            self.dtype = np.float32
            self.preprocess = A.ToFloat(always_apply=True)
            self.postprocess = IdentityTransform()
        else:
            raise ValueError(f"Unsupported dtype {self.conf.dtype}")
        self.to_tensor = ToTensorV2()

    def _init(self, conf):
        """Child class overwrites this, setting up a list of transforms"""
        self.transforms = []

    def __call__(self, image, return_tensor=False):
        """image as HW or HWC"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        data = {"image": image}
        if image.dtype != self.dtype:
            data = self.preprocess(**data)
        transforms = self.transforms
        if self.conf.shuffle:
            order = [i for i, _ in enumerate(transforms)]
            np.random.shuffle(order)
            transforms = [transforms[i] for i in order]
        transformed = self.compose(transforms, p=self.conf.p)(**data)
        if self.conf.verbose:
            print(replay_str(transformed["replay"]["transforms"]))
        transformed = self.postprocess(**transformed)
        if return_tensor:
            return self.to_tensor(**transformed)["image"]
        else:
            return transformed["image"]


class IdentityAugmentation(BaseAugmentation):
    default_conf = {}

    def _init(self, conf):
        self.transforms = [IdentityTransform(p=1.0)]


class DarkAugmentation(BaseAugmentation):
    default_conf = {"p": 0.75}

    def _init(self, conf):
        bright_contr = 0.5
        blur = 0.1
        random_gamma = 0.1
        hue = 0.1
        self.transforms = [
            A.RandomRain(p=0.2),
            A.RandomBrightnessContrast(
                **kw(
                    bright_contr,
                    brightness_limit=(-0.4, 0.0),
                    contrast_limit=(-0.3, 0.0),
                )
            ),
            A.OneOf(
                [
                    A.Blur(**kwi(blur, p=0.1, blur_limit=(3, 9), n="blur")),
                    A.MotionBlur(**kwi(blur, p=0.2, blur_limit=(3, 25), n="motion_blur")),
                    A.ISONoise(),
                    A.ImageCompression(),
                ],
                **kwi(blur, p=0.1),
            ),
            A.RandomGamma(**kw(random_gamma, gamma_limit=(15, 65))),
            A.OneOf(
                [
                    A.Equalize(),
                    A.CLAHE(p=0.2),
                    A.ToGray(),
                    A.ToSepia(p=0.1),
                    A.HueSaturationValue(**kw(hue, val_shift_limit=(-100, -40))),
                ],
                p=0.5,
            ),
        ]


class DefaultAugmentation(BaseAugmentation):
    default_conf = {"p": 1.0}

    def _init(self, conf):
        self.transforms = [
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.ToGray(p=0.2),
            A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ],
                p=0.2,
            ),
        ]


class PerspectiveAugmentation(BaseAugmentation):
    default_conf = {"p": 1.0}

    def _init(self, conf):
        self.transforms = [
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.ToGray(p=0.2),
            A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ],
                p=0.2,
            ),
        ]


class DeepCalibAugmentations(BaseAugmentation):
    default_conf = {"p": 1.0}

    def _init(self, conf):
        self.transforms = [
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(var_limit=(5.0, 112.0), mean=0, per_channel=True, p=0.75),
            A.Downscale(
                scale_min=0.5,
                scale_max=0.95,
                interpolation=dict(downscale=cv2.INTER_AREA, upscale=cv2.INTER_LINEAR),
                p=0.5,
            ),
            A.Downscale(scale_min=0.5, scale_max=0.95, interpolation=cv2.INTER_LINEAR, p=0.5),
            A.ImageCompression(quality_lower=20, quality_upper=85, p=1, always_apply=True),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.4),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
            A.ToGray(always_apply=False, p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, p=0.25),
            A.MotionBlur(blur_limit=5, allow_shifted=True, p=0.25),
            A.MultiplicativeNoise(multiplier=[0.85, 1.15], elementwise=True, p=0.5),
        ]


class GeoCalibAugmentations(BaseAugmentation):
    default_conf = {"p": 1.0}

    def _init(self, conf):
        self.color_transforms = [
            A.RandomGamma(gamma_limit=(80, 180), p=0.8),
            A.RandomToneCurve(scale=0.1, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.4),
            A.OneOf([A.ToGray(p=0.1), A.ToSepia(p=0.1), IdentityTransform(p=0.8)], p=1),
        ]

        self.noise_transforms = [
            A.GaussNoise(var_limit=(5.0, 112.0), mean=0, per_channel=True, p=0.75),
            A.ImageCompression(quality_lower=20, quality_upper=100, p=1),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            A.OneOrOther(
                first=A.Compose(
                    [
                        A.AdvancedBlur(
                            p=1,
                            blur_limit=(3, 7),
                            sigmaX_limit=(0.2, 1.0),
                            sigmaY_limit=(0.2, 1.0),
                            rotate_limit=(-90, 90),
                            beta_limit=(0.5, 8.0),
                            noise_limit=(0.9, 1.1),
                        ),
                        A.Sharpen(p=0.5, alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
                    ]
                ),
                second=A.Compose(
                    [
                        A.Sharpen(p=0.5, alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
                        A.AdvancedBlur(
                            p=1,
                            blur_limit=(3, 7),
                            sigmaX_limit=(0.2, 1.0),
                            sigmaY_limit=(0.2, 1.0),
                            rotate_limit=(-90, 90),
                            beta_limit=(0.5, 8.0),
                            noise_limit=(0.9, 1.1),
                        ),
                    ]
                ),
            ),
        ]

        self.image_transforms = [
            A.OneOf(
                [
                    A.Downscale(
                        scale_min=0.5,
                        scale_max=0.99,
                        interpolation=dict(downscale=down, upscale=up),
                        p=1,
                    )
                    for down, up in [
                        (cv2.INTER_AREA, cv2.INTER_LINEAR),
                        (cv2.INTER_LINEAR, cv2.INTER_CUBIC),
                        (cv2.INTER_CUBIC, cv2.INTER_LINEAR),
                        (cv2.INTER_LINEAR, cv2.INTER_AREA),
                    ]
                ],
                p=1,
            )
        ]

        self.transforms = [
            *self.color_transforms,
            *self.noise_transforms,
            *self.image_transforms,
        ]


augmentations = {
    "default": DefaultAugmentation,
    "dark": DarkAugmentation,
    "perspective": PerspectiveAugmentation,
    "deepcalib": DeepCalibAugmentations,
    "geocalib": GeoCalibAugmentations,
    "identity": IdentityAugmentation,
}
