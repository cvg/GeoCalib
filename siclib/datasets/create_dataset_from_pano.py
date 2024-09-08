"""Script to create a dataset from panorama images."""

import hashlib
import logging
from concurrent import futures
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from siclib.geometry.camera import camera_models
from siclib.geometry.gravity import Gravity
from siclib.utils.conversions import deg2rad, focal2fov, fov2focal, rad2deg
from siclib.utils.image import load_image, write_image

logger = logging.getLogger(__name__)


# mypy: ignore-errors


def max_radius(a, b):
    """Compute the maximum radius of a Brown distortion model."""
    discrim = a * a - 4 * b
    # if torch.isfinite(discrim) and discrim >= 0.0:
    #     discrim = np.sqrt(discrim) - a
    #     if discrim > 0.0:
    #         return 2.0 / discrim

    valid = torch.isfinite(discrim) & (discrim >= 0.0)
    discrim = torch.sqrt(discrim) - a
    valid &= discrim > 0.0
    return 2.0 / torch.where(valid, discrim, 0)


def brown_max_radius(k1, k2):
    """Compute the maximum radius of a Brown distortion model."""
    # fold the constants from the derivative into a and b
    a = k1 * 3
    b = k2 * 5
    return torch.sqrt(max_radius(a, b))


class ParallelProcessor:
    """Generic parallel processor class."""

    def __init__(self, max_workers):
        """Init processor and pbars."""
        self.max_workers = max_workers
        self.executor = futures.ProcessPoolExecutor(max_workers=self.max_workers)
        self.pbars = {}

    def update_pbar(self, pbar_key):
        """Update progressbar."""
        pbar = self.pbars.get(pbar_key)
        pbar.update(1)

    def submit_tasks(self, task_func, task_args, pbar_key):
        """Submit tasks."""
        pbar = tqdm(total=len(task_args), desc=f"Processing {pbar_key}", ncols=80)
        self.pbars[pbar_key] = pbar

        def update_pbar(future):
            self.update_pbar(pbar_key)

        futures = []
        for args in task_args:
            future = self.executor.submit(task_func, *args)
            future.add_done_callback(update_pbar)
            futures.append(future)

        return futures

    def wait_for_completion(self, futures):
        """Wait for completion and return results."""
        results = []
        for f in futures:
            results += f.result()

        for key in self.pbars.keys():
            self.pbars[key].close()

        return results

    def shutdown(self):
        """Close the executer."""
        self.executor.shutdown()


class DatasetGenerator:
    """Dataset generator class to create perspective datasets from panoramas."""

    default_conf = {
        "name": "???",
        # paths
        "base_dir": "???",
        "pano_dir": "${.base_dir}/panoramas",
        "pano_train": "${.pano_dir}/train",
        "pano_val": "${.pano_dir}/val",
        "pano_test": "${.pano_dir}/test",
        "perspective_dir": "${.base_dir}/${.name}",
        "perspective_train": "${.perspective_dir}/train",
        "perspective_val": "${.perspective_dir}/val",
        "perspective_test": "${.perspective_dir}/test",
        "train_csv": "${.perspective_dir}/train.csv",
        "val_csv": "${.perspective_dir}/val.csv",
        "test_csv": "${.perspective_dir}/test.csv",
        # data options
        "camera_model": "pinhole",
        "parameter_dists": {
            "roll": {
                "type": "uniform",
                "options": {"loc": deg2rad(-45), "scale": deg2rad(90)},  # in [-45, 45]
            },
            "pitch": {
                "type": "uniform",
                "options": {"loc": deg2rad(-45), "scale": deg2rad(90)},  # in [-45, 45]
            },
            "vfov": {
                "type": "uniform",
                "options": {"loc": deg2rad(20), "scale": deg2rad(85)},  # in [20, 105]
            },
            "resize_factor": {
                "type": "uniform",
                "options": {"loc": 1.0, "scale": 1.0},  # factor in [1.0, 2.0]
            },
            "shape": {"type": "fix", "value": (640, 640)},
        },
        "images_per_pano": 16,
        "n_workers": 10,
        "device": "cpu",
        "overwrite": False,
    }

    def __init__(self, conf):
        """Init the class by merging and storing the config."""
        self.conf = OmegaConf.merge(
            OmegaConf.create(self.default_conf),
            OmegaConf.create(conf),
        )
        logger.info(f"Config:\n{OmegaConf.to_yaml(self.conf)}")

        self.infos = {}
        self.device = self.conf.device

        self.camera_model = camera_models[self.conf.camera_model]

    def sample_value(self, parameter_name, seed=None):
        """Sample a value from the specified distribution."""
        param_conf = self.conf["parameter_dists"][parameter_name]

        if param_conf.type == "fix":
            return torch.tensor(param_conf.value)

        # fix seed for reproducibility
        generator = None
        if seed:
            if not isinstance(seed, (int, float)):
                seed = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (2**32)
            generator = np.random.default_rng(seed)

        sampler = getattr(scipy.stats, param_conf.type)
        return torch.tensor(sampler.rvs(random_state=generator, **param_conf.options))

    def plot_distributions(self):
        """Plot parameter distributions."""
        fig, ax = plt.subplots(3, 3, figsize=(15, 10))
        for i, split in enumerate(["train", "val", "test"]):
            roll_vals = [rad2deg(row["roll"]) for row in self.infos[split]]
            ax[i, 0].hist(roll_vals, bins=100)
            ax[i, 0].set_xlabel("Roll (°)")
            ax[i, 0].set_ylabel(f"Count {split}")

            pitch_vals = [rad2deg(row["pitch"]) for row in self.infos[split]]
            ax[i, 1].hist(pitch_vals, bins=100)
            ax[i, 1].set_xlabel("Pitch (°)")
            ax[i, 1].set_ylabel(f"Count {split}")

            vfov_vals = [rad2deg(row["vfov"]) for row in self.infos[split]]
            ax[i, 2].hist(vfov_vals, bins=100)
            ax[i, 2].set_xlabel("vFoV (°)")
            ax[i, 2].set_ylabel(f"Count {split}")

        plt.tight_layout()
        plt.savefig(Path(self.conf.perspective_dir) / "distributions.pdf")

        fig, ax = plt.subplots(3, 3, figsize=(15, 10))
        for i, k1 in enumerate(["roll", "pitch", "vfov"]):
            for j, k2 in enumerate(["roll", "pitch", "vfov"]):
                ax[i, j].scatter(
                    [rad2deg(row[k1]) for row in self.infos["train"]],
                    [rad2deg(row[k2]) for row in self.infos["train"]],
                    s=1,
                    label="train",
                )

                ax[i, j].scatter(
                    [rad2deg(row[k1]) for row in self.infos["val"]],
                    [rad2deg(row[k2]) for row in self.infos["val"]],
                    s=1,
                    label="val",
                )

                ax[i, j].scatter(
                    [rad2deg(row[k1]) for row in self.infos["test"]],
                    [rad2deg(row[k2]) for row in self.infos["test"]],
                    s=1,
                    label="test",
                )

                ax[i, j].set_xlabel(k1)
                ax[i, j].set_ylabel(k2)
                ax[i, j].legend()

        plt.tight_layout()
        plt.savefig(Path(self.conf.perspective_dir) / "distributions_scatter.pdf")

    def generate_images_from_pano(self, pano_path: Path, out_dir: Path):
        """Generate perspective images from a single panorama."""
        infos = []

        pano = load_image(pano_path).to(self.device)

        yaws = np.linspace(0, 2 * np.pi, self.conf.images_per_pano, endpoint=False)
        params = {
            k: [self.sample_value(k, pano_path.stem + k + str(i)) for i in yaws]
            for k in self.conf.parameter_dists
            if k != "shape"
        }
        shapes = [self.sample_value("shape", pano_path.stem + "shape") for _ in yaws]
        params |= {
            "height": [shape[0] for shape in shapes],
            "width": [shape[1] for shape in shapes],
        }

        if "k1_hat" in params:
            height = torch.tensor(params["height"])
            width = torch.tensor(params["width"])
            k1_hat = torch.tensor(params["k1_hat"])
            vfov = torch.tensor(params["vfov"])
            focal = fov2focal(vfov, height)
            focal = focal
            rel_focal = focal / height
            k1 = k1_hat * rel_focal

            # distance to image corner
            # r_max_im = f_px * r_max * (1 + k1*r_max**2)
            # function of r_max_im: f_px = r_max_im / (r_max * (1 + k1*r_max**2))
            min_permissible_rmax = torch.sqrt((height / 2) ** 2 + (width / 2) ** 2)
            r_max = brown_max_radius(k1=k1, k2=0)
            lowest_possible_f_px = min_permissible_rmax / (r_max * (1 + k1 * r_max**2))
            valid = lowest_possible_f_px <= focal

            f = torch.where(valid, focal, lowest_possible_f_px)
            vfov = focal2fov(f, height)

            params["vfov"] = vfov
            params |= {"k1": k1}

        cam = self.camera_model.from_dict(params).float().to(self.device)
        gravity = Gravity.from_rp(params["roll"], params["pitch"]).float().to(self.device)

        if (out_dir / f"{pano_path.stem}_0.jpg").exists() and not self.conf.overwrite:
            for i in range(self.conf.images_per_pano):
                perspective_name = f"{pano_path.stem}_{i}.jpg"
                info = {"fname": perspective_name} | {k: v[i].item() for k, v in params.items()}
                infos.append(info)

            logger.info(f"Perspectives for {pano_path.stem} already exist.")

            return infos

        perspective_images = cam.get_img_from_pano(
            pano_img=pano, gravity=gravity, yaws=yaws, resize_factor=params["resize_factor"]
        )

        for i, perspective_image in enumerate(perspective_images):
            perspective_name = f"{pano_path.stem}_{i}.jpg"

            n_pixels = perspective_image.shape[-2] * perspective_image.shape[-1]
            valid = (torch.sum(perspective_image.sum(0) == 0) / n_pixels) < 0.01
            if not valid:
                logger.debug(f"Perspective {perspective_name} has too many black pixels.")
                continue

            write_image(perspective_image, out_dir / perspective_name)

            info = {"fname": perspective_name} | {k: v[i].item() for k, v in params.items()}
            infos.append(info)

        return infos

    def generate_split(self, split: str, parallel_processor: ParallelProcessor):
        """Generate a single split of a dataset."""
        self.infos[split] = []
        panorama_paths = [
            path
            for path in Path(self.conf[f"pano_{split}"]).glob("*")
            if not path.name.startswith(".")
        ]

        out_dir = Path(self.conf[f"perspective_{split}"])
        logger.info(f"Writing perspective images to {str(out_dir)}")
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        futures = parallel_processor.submit_tasks(
            self.generate_images_from_pano, [(f, out_dir) for f in panorama_paths], split
        )
        self.infos[split] = parallel_processor.wait_for_completion(futures)
        # parallel_processor.shutdown()

        metadata = pd.DataFrame(data=self.infos[split])
        metadata.to_csv(self.conf[f"{split}_csv"])

    def generate_dataset(self):
        """Generate all splits of a dataset."""
        out_dir = Path(self.conf.perspective_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        OmegaConf.save(self.conf, out_dir / "config.yaml")

        processor = ParallelProcessor(self.conf.n_workers)
        for split in ["train", "val", "test"]:
            self.generate_split(split=split, parallel_processor=processor)

        processor.shutdown()

        for split in ["train", "val", "test"]:
            logger.info(f"Generated {len(self.infos[split])} {split} images.")

        self.plot_distributions()


@hydra.main(version_base=None, config_path="configs", config_name="SUN360")
def main(cfg: DictConfig) -> None:
    """Run dataset generation."""
    generator = DatasetGenerator(conf=cfg)
    generator.generate_dataset()


if __name__ == "__main__":
    main()
