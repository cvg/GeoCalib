import resource
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from siclib.eval.io import get_eval_parser, parse_eval_args
from siclib.eval.simple_pipeline import SimplePipeline
from siclib.settings import EVAL_PATH

# flake8: noqa
# mypy: ignore-errors

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.set_grad_enabled(False)


class Lamar2k(SimplePipeline):
    default_conf = {
        "data": {
            "name": "simple_dataset",
            "dataset_dir": "data/lamar2k",
            "test_img_dir": "${.dataset_dir}/images",
            "test_csv": "${.dataset_dir}/images.csv",
            "augmentations": {"name": "identity"},
            "preprocessing": {"resize": 320, "edge_divisible_by": 32},
            "test_batch_size": 1,
        },
        "model": {},
        "eval": {
            "thresholds": [1, 5, 10],
            "pixel_thresholds": [0.5, 1, 3, 5],
            "num_vis": 10,
            "verbose": True,
        },
        "url": "https://cvg-data.inf.ethz.ch/GeoCalib_ECCV2024/lamar2k.zip",
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


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(Lamar2k.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(dataset_name, args, "configs/", default_conf)

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = Lamar2k(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
