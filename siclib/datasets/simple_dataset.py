"""Dataset for images created with 'create_dataset_from_pano.py'."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from omegaconf import DictConfig

from siclib.datasets.augmentations import IdentityAugmentation, augmentations
from siclib.datasets.base_dataset import BaseDataset
from siclib.geometry.camera import SimpleRadial
from siclib.geometry.gravity import Gravity
from siclib.geometry.perspective_fields import get_perspective_field
from siclib.utils.conversions import fov2focal
from siclib.utils.image import ImagePreprocessor, load_image
from siclib.utils.tools import fork_rng

logger = logging.getLogger(__name__)

# mypy: ignore-errors


def load_csv(
    csv_file: Path, img_root: Path
) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
    """Load a CSV file containing image information.

    Args:
        csv_file (str): Path to the CSV file.
        img_root (str): Path to the root directory containing the images.

    Returns:
        list: List of dictionaries containing the image paths and camera parameters.
    """
    df = pd.read_csv(csv_file)

    infos, params, gravity = [], [], []
    for _, row in df.iterrows():
        h = row["height"]
        w = row["width"]
        px = row.get("px", w / 2)
        py = row.get("py", h / 2)
        vfov = row["vfov"]
        f = fov2focal(torch.tensor(vfov), h)
        k1 = row.get("k1", 0)
        k2 = row.get("k2", 0)
        params.append(torch.tensor([w, h, f, f, px, py, k1, k2]))

        roll = row["roll"]
        pitch = row["pitch"]
        gravity.append(torch.tensor([roll, pitch]))

        infos.append({"name": row["fname"], "file_name": str(img_root / row["fname"])})

    params = torch.stack(params).float()
    gravity = torch.stack(gravity).float()
    return infos, params, gravity


class SimpleDataset(BaseDataset):
    """Dataset for images created with 'create_dataset_from_pano.py'."""

    default_conf = {
        # paths
        "dataset_dir": "???",
        "train_img_dir": "${.dataset_dir}/train",
        "val_img_dir": "${.dataset_dir}/val",
        "test_img_dir": "${.dataset_dir}/test",
        "train_csv": "${.dataset_dir}/train.csv",
        "val_csv": "${.dataset_dir}/val.csv",
        "test_csv": "${.dataset_dir}/test.csv",
        # data options
        "use_up": True,
        "use_latitude": True,
        "use_prior_focal": False,
        "use_prior_gravity": False,
        "use_prior_k1": False,
        # image options
        "grayscale": False,
        "preprocessing": ImagePreprocessor.default_conf,
        "augmentations": {"name": "geocalib", "verbose": False},
        "p_rotate": 0.0,  # probability to rotate image by +/- 90°
        "reseed": False,
        "seed": 0,
        # data loader options
        "num_workers": 8,
        "prefetch_factor": 2,
        "train_batch_size": 32,
        "val_batch_size": 32,
        "test_batch_size": 32,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split: str) -> torch.utils.data.Dataset:
        """Return a dataset for a given split."""
        return _SimpleDataset(self.conf, split)


class _SimpleDataset(torch.utils.data.Dataset):
    """Dataset for dataset for images created with 'create_dataset_from_pano.py'."""

    def __init__(self, conf: DictConfig, split: str):
        """Initialize the dataset."""
        self.conf = conf
        self.split = split
        self.img_dir = Path(conf.get(f"{split}_img_dir"))

        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        # load image information
        assert f"{split}_csv" in conf, f"Missing {split}_csv in conf"
        infos_path = self.conf.get(f"{split}_csv")
        self.infos, self.parameters, self.gravity = load_csv(infos_path, self.img_dir)

        # define augmentations
        aug_name = conf.augmentations.name
        assert (
            aug_name in augmentations.keys()
        ), f'{aug_name} not in {" ".join(augmentations.keys())}'

        if self.split == "train":
            self.augmentation = augmentations[aug_name](conf.augmentations)
        else:
            self.augmentation = IdentityAugmentation()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        if not self.conf.reseed:
            return self.getitem(idx)
        with fork_rng(self.conf.seed + idx, False):
            return self.getitem(idx)

    def _read_image(
        self, infos: Dict[str, Any], parameters: torch.Tensor, gravity: torch.Tensor
    ) -> Dict[str, Any]:
        path = Path(str(infos["file_name"]))

        # load image as uint8 and HWC for augmentation
        image = load_image(path, self.conf.grayscale, return_tensor=False)
        image = self.augmentation(image, return_tensor=True)

        # create radial camera -> same as pinhole if k1 = 0
        camera = SimpleRadial(parameters[None]).float()

        roll, pitch = gravity[None].unbind(-1)
        gravity = Gravity.from_rp(roll, pitch)

        # preprocess
        data = self.preprocessor(image)
        camera = camera.scale(data["scales"])
        camera = camera.crop(data["crop_pad"]) if "crop_pad" in data else camera

        priors = {"prior_gravity": gravity} if self.conf.use_prior_gravity else {}
        priors |= {"prior_focal": camera.f[..., 1]} if self.conf.use_prior_focal else {}
        priors |= {"prior_k1": camera.k1} if self.conf.use_prior_k1 else {}
        return {
            "name": infos["name"],
            "path": str(path),
            "camera": camera[0],
            "gravity": gravity[0],
            **priors,
            **data,
        }

    def _get_perspective(self, data):
        """Get perspective field."""
        camera = data["camera"]
        gravity = data["gravity"]

        up_field, lat_field = get_perspective_field(
            camera, gravity, use_up=self.conf.use_up, use_latitude=self.conf.use_latitude
        )

        out = {}
        if self.conf.use_up:
            out["up_field"] = up_field[0]
        if self.conf.use_latitude:
            out["latitude_field"] = lat_field[0]

        return out

    def getitem(self, idx: int):
        """Return a sample from the dataset."""
        infos = self.infos[idx]
        parameters = self.parameters[idx]
        gravity = self.gravity[idx]
        data = self._read_image(infos, parameters, gravity)

        if self.conf.use_up or self.conf.use_latitude:
            data |= self._get_perspective(data)

        return data


if __name__ == "__main__":
    # Create a dump of the dataset
    import argparse

    import matplotlib.pyplot as plt

    from siclib.visualization.visualize_batch import make_perspective_figures

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--n_rows", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_intermixed_args()

    dconf = SimpleDataset.default_conf
    dconf["name"] = args.name
    dconf["num_workers"] = 0
    dconf["prefetch_factor"] = None

    dconf["dataset_dir"] = args.data_dir
    dconf[f"{args.split}_batch_size"] = args.n_rows

    torch.set_grad_enabled(False)

    dataset = SimpleDataset(dconf)
    loader = dataset.get_data_loader(args.split, args.shuffle)

    with fork_rng(seed=42):
        for data in loader:
            pred = data
            break
        fig = make_perspective_figures(pred, data, n_pairs=args.n_rows)

    plt.show()
