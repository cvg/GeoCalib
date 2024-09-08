import logging
import resource
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from siclib.datasets import get_dataset
from siclib.eval.eval_pipeline import EvalPipeline
from siclib.eval.io import get_eval_parser, load_model, parse_eval_args
from siclib.eval.utils import download_and_extract_benchmark, plot_scatter_grid
from siclib.geometry.base_camera import BaseCamera
from siclib.geometry.camera import Pinhole
from siclib.geometry.gravity import Gravity
from siclib.models.cache_loader import CacheLoader
from siclib.models.utils.metrics import (
    gravity_error,
    latitude_error,
    pitch_error,
    roll_error,
    up_error,
    vfov_error,
)
from siclib.settings import EVAL_PATH
from siclib.utils.conversions import rad2deg
from siclib.utils.export_predictions import export_predictions
from siclib.utils.tensor import add_batch_dim
from siclib.utils.tools import AUCMetric, set_seed
from siclib.visualization import visualize_batch, viz2d

# flake8: noqa
# mypy: ignore-errors

logger = logging.getLogger(__name__)

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.set_grad_enabled(False)


def calculate_pixel_projection_error(
    camera_pred: BaseCamera, camera_gt: BaseCamera, N: int = 500, distortion_only: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the pixel projection error between two cameras.

    1. Project a grid of points with the ground truth camera to the image plane.
    2. Project the same grid of points with the estimated camera to the image plane.
    3. Calculate the pixel distance between the ground truth and estimated points.

    Args:
        camera_pred (Camera): Predicted camera.
        camera_gt (Camera): Ground truth camera.
        N (int, optional): Number of points in the grid. Defaults to 500.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Pixel distance and valid pixels.
    """
    H, W = camera_gt.size.unbind(-1)
    H, W = H.int(), W.int()

    assert torch.allclose(
        camera_gt.size, camera_pred.size
    ), f"Cameras must have the same size: {camera_gt.size} != {camera_pred.size}"

    if distortion_only:
        params = camera_gt._data.clone()
        params[..., -2:] = camera_pred._data[..., -2:]
        CameraModel = type(camera_gt)
        camera_pred = CameraModel(params)

    x_gt, y_gt = torch.meshgrid(
        torch.linspace(0, H - 1, N), torch.linspace(0, W - 1, N), indexing="xy"
    )
    xy = torch.stack((x_gt, y_gt), dim=-1).reshape(-1, 2)

    camera_pin_gt = camera_gt.pinhole()
    uv_pin, _ = camera_pin_gt.image2world(xy)

    # gt
    xy_undist_gt, valid_dist_gt = camera_gt.world2image(uv_pin)
    # pred
    xy_undist, valid_dist = camera_pred.world2image(uv_pin)

    valid = valid_dist_gt & valid_dist

    dist = (xy_undist - xy_undist_gt) ** 2
    dist = (dist.sum(-1)).sqrt()

    return dist[valid_dist_gt], valid[valid_dist_gt]


def compute_camera_metrics(
    camera_pred: BaseCamera, camera_gt: BaseCamera, thresholds: List[float]
) -> Dict[str, float]:
    results = defaultdict(list)
    results["vfov"].append(rad2deg(camera_pred.vfov).item())
    results["vfov_error"].append(vfov_error(camera_pred, camera_gt).item())

    results["focal"].append(camera_pred.f[..., 1].item())
    focal_error = torch.abs(camera_pred.f[..., 1] - camera_gt.f[..., 1])
    results["focal_error"].append(focal_error.item())

    rel_focal_error = torch.abs(camera_pred.f[..., 1] - camera_gt.f[..., 1]) / camera_gt.f[..., 1]
    results["rel_focal_error"].append(rel_focal_error.item())

    if hasattr(camera_pred, "k1"):
        results["k1"].append(camera_pred.k1.item())
        k1_error = torch.abs(camera_pred.k1 - camera_gt.k1)
        results["k1_error"].append(k1_error.item())

        if thresholds is None:
            return results

        err, valid = calculate_pixel_projection_error(camera_pred, camera_gt, distortion_only=False)
        for th in thresholds:
            results[f"pixel_projection_error@{th}"].append(
                ((err[valid] < th).sum() / len(valid)).float().item()
            )

        err, valid = calculate_pixel_projection_error(camera_pred, camera_gt, distortion_only=True)
        for th in thresholds:
            results[f"pixel_distortion_error@{th}"].append(
                ((err[valid] < th).sum() / len(valid)).float().item()
            )
    return results


def compute_gravity_metrics(gravity_pred: Gravity, gravity_gt: Gravity) -> Dict[str, float]:
    results = defaultdict(list)
    results["roll"].append(rad2deg(gravity_pred.roll).item())
    results["pitch"].append(rad2deg(gravity_pred.pitch).item())

    results["roll_error"].append(roll_error(gravity_pred, gravity_gt).item())
    results["pitch_error"].append(pitch_error(gravity_pred, gravity_gt).item())
    results["gravity_error"].append(gravity_error(gravity_pred[None], gravity_gt[None]).item())
    return results


class SimplePipeline(EvalPipeline):
    default_conf = {
        "data": {},
        "model": {},
        "eval": {
            "thresholds": [1, 5, 10],
            "pixel_thresholds": [0.5, 1, 3, 5],
            "num_vis": 10,
            "verbose": True,
        },
        "url": None,  # url to benchmark.zip
    }

    export_keys = [
        "camera",
        "gravity",
    ]

    optional_export_keys = [
        "focal_uncertainty",
        "vfov_uncertainty",
        "roll_uncertainty",
        "pitch_uncertainty",
        "gravity_uncertainty",
        "up_field",
        "up_confidence",
        "latitude_field",
        "latitude_confidence",
    ]

    def _init(self, conf):
        self.verbose = conf.eval.verbose
        self.num_vis = self.conf.eval.num_vis

        self.CameraModel = Pinhole

        if conf.url is not None:
            ds_dir = Path(conf.data.dataset_dir)
            download_and_extract_benchmark(ds_dir.name, conf.url, ds_dir.parent)

    @classmethod
    def get_dataloader(cls, data_conf=None, batch_size=None):
        """Returns a data loader with samples for each eval datapoint"""
        data_conf = data_conf or cls.default_conf["data"]

        if batch_size is not None:
            data_conf["test_batch_size"] = batch_size

        do_shuffle = data_conf["test_batch_size"] > 1
        dataset = get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test", shuffle=do_shuffle)

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        # set_seed(0)
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
                verbose=self.verbose,
            )
        return pred_file

    def get_figures(self, results):
        figures = {}

        if self.num_vis == 0:
            return figures

        gl = ["up", "latitude"]
        rpf = ["roll", "pitch", "vfov"]

        # check if rpf in results
        if all(k in results for k in rpf):
            x_keys = [f"{k}_gt" for k in rpf]

            # gt vs error
            y_keys = [f"{k}_error" for k in rpf]
            fig, _ = plot_scatter_grid(results, x_keys, y_keys, show_means=False)
            figures |= {"rpf_gt_error": fig}

            # gt vs pred
            y_keys = [f"{k}" for k in rpf]
            fig, _ = plot_scatter_grid(results, x_keys, y_keys, diag=True, show_means=False)
            figures |= {"rpf_gt_pred": fig}

        if all(f"{k}_error" in results for k in gl):
            x_keys = [f"{k}_gt" for k in rpf]
            y_keys = [f"{k}_error" for k in gl]
            fig, _ = plot_scatter_grid(results, x_keys, y_keys, show_means=False)
            figures |= {"gl_gt_error": fig}

        return figures

    def run_eval(self, loader, pred_file):
        conf = self.conf.eval
        results = defaultdict(list)

        save_to = Path(pred_file).parent / "figures"
        if not save_to.exists() and self.num_vis > 0:
            save_to.mkdir()

        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()

        if not self.verbose:
            logger.info(f"Evaluating {pred_file}")

        for i, data in enumerate(
            tqdm(loader, desc="Evaluating", total=len(loader), ncols=80, disable=not self.verbose)
        ):
            # NOTE: data is batched but pred is not
            pred = cache_loader(data)

            results["names"].append(data["name"][0])

            gt_cam = data["camera"][0]
            gt_gravity = data["gravity"][0]
            # add gt parameters
            results["roll_gt"].append(rad2deg(gt_gravity.roll).item())
            results["pitch_gt"].append(rad2deg(gt_gravity.pitch).item())
            results["vfov_gt"].append(rad2deg(gt_cam.vfov).item())
            results["focal_gt"].append(gt_cam.f[1].item())

            results["k1_gt"].append(gt_cam.k1.item())

            if "camera" in pred:
                # pred["camera"] is a tensor of the parameters
                pred_cam = self.CameraModel(pred["camera"])

                pred_camera = pred_cam[None].undo_scale_crop(data)[0]
                gt_camera = gt_cam[None].undo_scale_crop(data)[0]

                camera_metrics = compute_camera_metrics(
                    pred_camera, gt_camera, conf.pixel_thresholds
                )

                for k, v in camera_metrics.items():
                    results[k].extend(v)

                if "focal_uncertainty" in pred:
                    focal_uncertainty = pred["focal_uncertainty"]
                    results["focal_uncertainty"].append(focal_uncertainty.item())

                if "vfov_uncertainty" in pred:
                    vfov_uncertainty = rad2deg(pred["vfov_uncertainty"])
                    results["vfov_uncertainty"].append(vfov_uncertainty.item())

            if "gravity" in pred:
                # pred["gravity"] is a tensor of the parameters
                pred_gravity = Gravity(pred["gravity"])

                gravity_metrics = compute_gravity_metrics(pred_gravity, gt_gravity)
                for k, v in gravity_metrics.items():
                    results[k].extend(v)

                if "roll_uncertainty" in pred:
                    roll_uncertainty = rad2deg(pred["roll_uncertainty"])
                    results["roll_uncertainty"].append(roll_uncertainty.item())

                if "pitch_uncertainty" in pred:
                    pitch_uncertainty = rad2deg(pred["pitch_uncertainty"])
                    results["pitch_uncertainty"].append(pitch_uncertainty.item())

                if "gravity_uncertainty" in pred:
                    gravity_uncertainty = rad2deg(pred["gravity_uncertainty"])
                    results["gravity_uncertainty"].append(gravity_uncertainty.item())

            if "up_field" in pred:
                up_err = up_error(pred["up_field"].unsqueeze(0), data["up_field"])
                results["up_error"].append(up_err.mean(axis=(1, 2)).item())
                results["up_med_error"].append(up_err.median().item())

                if "up_confidence" in pred:
                    up_confidence = pred["up_confidence"].unsqueeze(0)
                    weighted_error = (up_err * up_confidence).sum(axis=(1, 2))
                    weighted_error = weighted_error / up_confidence.sum(axis=(1, 2))
                    results["up_weighted_error"].append(weighted_error.item())

                if i < self.num_vis:
                    pred_batched = add_batch_dim(pred)
                    up_fig = visualize_batch.make_up_figure(pred=pred_batched, data=data)
                    up_fig = up_fig["up"]
                    plt.tight_layout()
                    viz2d.save_plot(save_to / f"up-{i}-{up_err.median().item():.3f}.jpg")
                    plt.close()

            if "latitude_field" in pred:
                lat_err = latitude_error(
                    pred["latitude_field"].unsqueeze(0), data["latitude_field"]
                )
                results["latitude_error"].append(lat_err.mean(axis=(1, 2)).item())
                results["latitude_med_error"].append(lat_err.median().item())

                if "latitude_confidence" in pred:
                    lat_confidence = pred["latitude_confidence"].unsqueeze(0)
                    weighted_error = (lat_err * lat_confidence).sum(axis=(1, 2))
                    weighted_error = weighted_error / lat_confidence.sum(axis=(1, 2))
                    results["latitude_weighted_error"].append(weighted_error.item())

                if i < self.num_vis:
                    pred_batched = add_batch_dim(pred)
                    lat_fig = visualize_batch.make_latitude_figure(pred=pred_batched, data=data)
                    lat_fig = lat_fig["latitude"]
                    plt.tight_layout()
                    viz2d.save_plot(save_to / f"latitude-{i}-{lat_err.median().item():.3f}.jpg")
                    plt.close()

        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue

            if k.endswith("_error") or "recall" in k or "pixel" in k:
                summaries[f"mean_{k}"] = round(np.nanmean(arr), 3)
                summaries[f"median_{k}"] = round(np.nanmedian(arr), 3)

                if any(keyword in k for keyword in ["roll", "pitch", "vfov", "gravity"]):
                    if not conf.thresholds:
                        continue

                    auc = AUCMetric(
                        elements=arr, thresholds=list(conf.thresholds), min_error=1
                    ).compute()
                    for i, t in enumerate(conf.thresholds):
                        summaries[f"auc_{k}@{t}"] = round(auc[i], 3)

        return summaries, self.get_figures(results), results


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(SimplePipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(dataset_name, args, "configs/", default_conf)

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = SimplePipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir, overwrite=args.overwrite, overwrite_eval=args.overwrite_eval
    )

    pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
