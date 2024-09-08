import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


def download_and_extract_benchmark(name: str, url: Path, output: Path) -> None:
    benchmark_dir = output / name
    if not output.exists():
        output.mkdir(parents=True)

    if benchmark_dir.exists():
        logger.info(f"Benchmark {name} already exists at {benchmark_dir}, skipping download.")
        return

    if name == "stanford2d3d":
        # prompt user to sign data sharing and usage terms
        txt = "\n" + "#" * 108 + "\n\n"
        txt += "To download the Stanford2D3D dataset, you must agree to the terms of use:\n\n"
        txt += (
            "https://docs.google.com/forms/d/e/"
            + "1FAIpQLScFR0U8WEUtb7tgjOhhnl31OrkEs73-Y8bQwPeXgebqVKNMpQ/viewform?c=0&w=1\n\n"
        )
        txt += "#" * 108 + "\n\n"
        txt += "Did you fill out the data sharing and usage terms? [y/n] "
        choice = input(txt)
        if choice.lower() != "y":
            raise ValueError(
                "You must agree to the terms of use to download the Stanford2D3D dataset."
            )

    zip_file = output / f"{name}.zip"

    if not zip_file.exists():
        logger.info(f"Downloading benchmark {name} to {zip_file} from {url}.")
        torch.hub.download_url_to_file(url, zip_file)

    logger.info(f"Extracting benchmark {name} in {output}.")
    shutil.unpack_archive(zip_file, output, format="zip")
    zip_file.unlink()


def check_keys_recursive(d, pattern):
    if isinstance(pattern, dict):
        {check_keys_recursive(d[k], v) for k, v in pattern.items()}
    else:
        for k in pattern:
            assert k in d.keys()


def plot_scatter_grid(
    results, x_keys, y_keys, name=None, diag=False, ax=None, line_idx=0, show_means=True
):  # sourcery skip: low-code-quality
    if ax is None:
        N, M = len(y_keys), len(x_keys)
        fig, ax = plt.subplots(N, M, figsize=(M * 6, N * 5))

        if N == 1:
            ax = np.array(ax)
            ax = ax.reshape(1, -1)

        if M == 1:
            ax = np.array(ax)
            ax = ax.reshape(-1, 1)
    else:
        fig = None

    for j, kx in enumerate(x_keys):
        for i, ky in enumerate(y_keys):
            ax[i, j].scatter(
                results[kx],
                results[ky],
                s=1,
                alpha=0.5,
                label=name or None,
            )

            ax[i, j].set_xlabel(f"{' '.join(kx.split('_')).title()}")
            ax[i, j].set_ylabel(f"{' '.join(ky.split('_')).title()}")

            low = min(ax[i, j].get_xlim()[0], ax[i, j].get_ylim()[0])
            high = max(ax[i, j].get_xlim()[1], ax[i, j].get_ylim()[1])
            if diag == "all" or (i == j and diag):
                ax[i, j].plot([low, high], [low, high], ls="--", c="red", label="y=x")

            if name or diag == "all" or (i == j and diag):
                ax[i, j].legend()

    if not show_means:
        return fig, ax

    means = {"y": {}, "x": {}}
    for kx in x_keys:
        for ky in y_keys:
            means["x"][kx] = np.mean(results[kx])
            means["y"][ky] = np.mean(results[ky])

    for j, kx in enumerate(x_keys):
        for i, ky in enumerate(y_keys):
            xlim = np.min(results[kx]), np.max(results[kx])
            ylim = np.min(results[ky]), np.max(results[ky])
            means_x = [means["x"][kx]]
            means_y = [means["y"][ky]]
            color = plt.cm.tab10(line_idx)
            ax[i, j].vlines(means_x, *ylim, colors=[color])
            ax[i, j].hlines(means_y, *xlim, colors=[color])

    return fig, ax
