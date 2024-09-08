"""Visualization of predicted and ground truth for a single batch."""

from typing import Any, Dict

import numpy as np
import torch

from siclib.geometry.perspective_fields import get_latitude_field
from siclib.models.utils.metrics import latitude_error, up_error
from siclib.utils.conversions import rad2deg
from siclib.utils.tensor import batch_to_device
from siclib.visualization.viz2d import (
    plot_confidences,
    plot_heatmaps,
    plot_image_grid,
    plot_latitudes,
    plot_vector_fields,
)


def make_up_figure(
    pred: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor], n_pairs: int = 2
) -> Dict[str, Any]:
    """Get predicted and ground truth up fields and errors.

    Args:
        pred (Dict[str, torch.Tensor]): Predicted up field.
        data (Dict[str, torch.Tensor]): Ground truth up field.
        n_pairs (int): Number of pairs to visualize.

    Returns:
        Dict[str, Any]: Dictionary with figure.
    """
    pred = batch_to_device(pred, "cpu", detach=True)
    data = batch_to_device(data, "cpu", detach=True)

    n_pairs = min(n_pairs, len(data["image"]))

    if "up_field" not in pred.keys():
        return {}

    errors = up_error(pred["up_field"], data["up_field"])

    up_fields = []
    for i in range(n_pairs):
        row = [data["up_field"][i], pred["up_field"][i], errors[i]]
        titles = ["Up GT", "Up Pred", "Up Error"]

        if "up_confidence" in pred.keys():
            row += [pred["up_confidence"][i]]
            titles += ["Up Confidence"]

        row = [r.float().numpy() if isinstance(r, torch.Tensor) else r for r in row]
        up_fields.append(row)

    # create figure
    N, M = len(up_fields), len(up_fields[0]) + 1
    imgs = [[data["image"][i].permute(1, 2, 0).cpu().clip(0, 1)] * M for i in range(n_pairs)]
    fig, ax = plot_image_grid(imgs, titles=[["Image"] + titles] * N, return_fig=True, set_lim=True)
    ax = np.array(ax)

    for i in range(n_pairs):
        plot_vector_fields(up_fields[i][:2], axes=ax[i, [1, 2]])
        plot_heatmaps([up_fields[i][2]], cmap="turbo", colorbar=True, axes=ax[i, [3]])

        if "up_confidence" in pred.keys():
            plot_confidences([up_fields[i][3]], axes=ax[i, [4]])

    return {"up": fig}


def make_latitude_figure(
    pred: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor], n_pairs: int = 2
) -> Dict[str, Any]:
    """Get predicted and ground truth latitude fields and errors.

    Args:
        pred (Dict[str, torch.Tensor]): Predicted latitude field.
        data (Dict[str, torch.Tensor]): Ground truth latitude field.
        n_pairs (int, optional): Number of pairs to visualize. Defaults to 2.

    Returns:
        Dict[str, Any]: Dictionary with figure.
    """
    pred = batch_to_device(pred, "cpu", detach=True)
    data = batch_to_device(data, "cpu", detach=True)

    n_pairs = min(n_pairs, len(data["image"]))
    latitude_fields = []

    if "latitude_field" not in pred.keys():
        return {}

    errors = latitude_error(pred["latitude_field"], data["latitude_field"])
    for i in range(n_pairs):
        row = [
            rad2deg(data["latitude_field"][i][0]),
            rad2deg(pred["latitude_field"][i][0]),
            errors[i],
        ]
        titles = ["Latitude GT", "Latitude Pred", "Latitude Error"]

        if "latitude_confidence" in pred.keys():
            row += [pred["latitude_confidence"][i]]
            titles += ["Latitude Confidence"]

        row = [r.float().numpy() if isinstance(r, torch.Tensor) else r for r in row]
        latitude_fields.append(row)

    # create figure
    N, M = len(latitude_fields), len(latitude_fields[0]) + 1
    imgs = [[data["image"][i].permute(1, 2, 0).cpu().clip(0, 1)] * M for i in range(n_pairs)]
    fig, ax = plot_image_grid(imgs, titles=[["Image"] + titles] * N, return_fig=True, set_lim=True)
    ax = np.array(ax)

    for i in range(n_pairs):
        plot_latitudes(latitude_fields[i][:2], is_radians=False, axes=ax[i, [1, 2]])
        plot_heatmaps([latitude_fields[i][2]], cmap="turbo", colorbar=True, axes=ax[i, [3]])

        if "latitude_confidence" in pred.keys():
            plot_confidences([latitude_fields[i][3]], axes=ax[i, [4]])

    return {"latitude": fig}


def make_camera_figure(
    pred: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor], n_pairs: int = 2
) -> Dict[str, Any]:
    """Get predicted and ground truth camera parameters.

    Args:
        pred (Dict[str, torch.Tensor]): Predicted camera parameters.
        data (Dict[str, torch.Tensor]): Ground truth camera parameters.
        n_pairs (int, optional): Number of pairs to visualize. Defaults to 2.

    Returns:
        Dict[str, Any]: Dictionary with figure.
    """
    pred = batch_to_device(pred, "cpu", detach=True)
    data = batch_to_device(data, "cpu", detach=True)

    n_pairs = min(n_pairs, len(data["image"]))

    if "camera" not in pred.keys():
        return {}

    latitudes = []
    for i in range(n_pairs):
        titles = ["Cameras GT"]
        row = [get_latitude_field(data["camera"][i], data["gravity"][i])]

        if "camera" in pred.keys() and "gravity" in pred.keys():
            row += [get_latitude_field(pred["camera"][i], pred["gravity"][i])]
            titles += ["Cameras Pred"]

        row = [rad2deg(r).squeeze(-1).float().numpy()[0] for r in row]
        latitudes.append(row)

    # create figure
    N, M = len(latitudes), len(latitudes[0]) + 1
    imgs = [[data["image"][i].permute(1, 2, 0).cpu().clip(0, 1)] * M for i in range(n_pairs)]
    fig, ax = plot_image_grid(imgs, titles=[["Image"] + titles] * N, return_fig=True, set_lim=True)
    ax = np.array(ax)

    for i in range(n_pairs):
        plot_latitudes(latitudes[i], is_radians=False, axes=ax[i, 1:])

    return {"camera": fig}


def make_perspective_figures(
    pred: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor], n_pairs: int = 2
) -> Dict[str, Any]:
    """Get predicted and ground truth perspective fields.

    Args:
        pred (Dict[str, torch.Tensor]): Predicted perspective fields.
        data (Dict[str, torch.Tensor]): Ground truth perspective fields.
        n_pairs (int, optional): Number of pairs to visualize. Defaults to 2.

    Returns:
        Dict[str, Any]: Dictionary with figure.
    """
    n_pairs = min(n_pairs, len(data["image"]))
    figures = make_up_figure(pred, data, n_pairs)
    figures |= make_latitude_figure(pred, data, n_pairs)
    figures |= make_camera_figure(pred, data, n_pairs)

    {f.tight_layout() for f in figures.values()}

    return figures
