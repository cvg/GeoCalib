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


class OpenPano(SimplePipeline):
    default_conf = {
        "data": {
            "name": "simple_dataset",
            "dataset_dir": "data/poly+maps+laval/poly+maps+laval",
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
        "url": None,
    }


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(OpenPano.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(dataset_name, args, "configs/", default_conf)

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = OpenPano(conf)
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
