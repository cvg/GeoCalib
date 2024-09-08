import inspect
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.widgets import Button, RadioButtons

from siclib.geometry.camera import SimpleRadial as Camera
from siclib.geometry.gravity import Gravity
from siclib.geometry.perspective_fields import (
    get_latitude_field,
    get_perspective_field,
    get_up_field,
)
from siclib.models.utils.metrics import latitude_error, up_error
from siclib.utils.conversions import rad2deg
from siclib.visualization.viz2d import (
    add_text,
    plot_confidences,
    plot_heatmaps,
    plot_horizon_lines,
    plot_latitudes,
    plot_vector_fields,
)

# flake8: noqa
# mypy: ignore-errors

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plt.rcParams["toolbar"] = "toolmanager"


class RadioHideTool(ToolToggleBase):
    """Show lines with a given gid."""

    default_keymap = "R"
    description = "Show by gid"
    default_toggled = False
    radio_group = "default"

    def __init__(self, *args, options=[], active=None, callback_fn=None, keymap="R", **kwargs):
        super().__init__(*args, **kwargs)
        self.f = 1.0
        self.options = options
        self.callback_fn = callback_fn
        self.active = self.options.index(active) if active else 0
        self.default_keymap = keymap

        self.enabled = self.default_toggled

    def build_radios(self):
        w = 0.2
        self.radios_ax = self.figure.add_axes([1.0 - w, 0.4, w, 0.5], zorder=1)
        # self.radios_ax = self.figure.add_axes([0.5-w/2, 1.0-0.2, w, 0.2], zorder=1)
        self.radios = RadioButtons(self.radios_ax, self.options, active=self.active)
        self.radios.on_clicked(self.on_radio_clicked)

    def enable(self, *args):
        size = self.figure.get_size_inches()
        size[0] *= self.f
        self.build_radios()
        self.figure.canvas.draw_idle()
        self.enabled = True

    def disable(self, *args):
        size = self.figure.get_size_inches()
        size[0] /= self.f
        self.radios_ax.remove()
        self.radios = None
        self.figure.canvas.draw_idle()
        self.enabled = False

    def on_radio_clicked(self, value):
        self.active = self.options.index(value)
        enabled = self.enabled
        if enabled:
            self.disable()
        if self.callback_fn is not None:
            self.callback_fn(value)
        if enabled:
            self.enable()


class ToggleTool(ToolToggleBase):
    """Show lines with a given gid."""

    default_keymap = "t"
    description = "Show by gid"

    def __init__(self, *args, callback_fn=None, keymap="t", **kwargs):
        super().__init__(*args, **kwargs)
        self.f = 1.0
        self.callback_fn = callback_fn
        self.default_keymap = keymap
        self.enabled = self.default_toggled

    def enable(self, *args):
        self.callback_fn(True)

    def disable(self, *args):
        self.callback_fn(False)


def add_whitespace_left(fig, factor):
    w, h = fig.get_size_inches()
    left = fig.subplotpars.left
    fig.set_size_inches([w * (1 + factor), h])
    fig.subplots_adjust(left=(factor + left) / (1 + factor))


def add_whitespace_bottom(fig, factor):
    w, h = fig.get_size_inches()
    b = fig.subplotpars.bottom
    fig.set_size_inches([w, h * (1 + factor)])
    fig.subplots_adjust(bottom=(factor + b) / (1 + factor))
    fig.canvas.draw_idle()


class ImagePlot:
    plot_name = "image"
    required_keys = ["image"]

    def __init__(self, fig, axes, data, preds):
        pass


class HorizonLinePlot:
    plot_name = "horizon_line"
    required_keys = ["camera", "gravity"]

    def __init__(self, fig, axes, data, preds):
        for idx, name in enumerate(preds):
            pred = preds[name]
            gt_cam = data["camera"][0].detach().cpu()
            gt_gravity = data["gravity"][0].detach().cpu()
            plot_horizon_lines([gt_cam], [gt_gravity], line_colors="r", ax=[axes[0][idx]])

            if "camera" in pred and "gravity" in pred:
                pred_cam = Camera(pred["camera"][0].detach().cpu())
                gravity = Gravity(pred["gravity"][0].detach().cpu())
                plot_horizon_lines([pred_cam], [gravity], line_colors="yellow", ax=[axes[0][idx]])


class LatitudePlot:
    plot_name = "latitude"
    required_keys = ["latitude_field"]

    def __init__(self, fig, axes, data, preds):
        self.artists = []
        self.gt_mode = False  # Flag to track whether to display ground truth or predicted
        self.text_objects = []  # To store text objects

        self.fig = fig
        self.axes = axes
        self.data = data
        self.preds = preds

        # Create a toggle button on the lower left corner of the first axis
        self.ax_button = self.fig.add_axes([0.01, 0.02, 0.2, 0.06])
        self.button = Button(self.ax_button, "Toggle GT")
        self.button.on_clicked(self.toggle_display)

        self.update_plot()

    def toggle_display(self, event):
        # Toggle between ground truth and predicted latitudes
        self.gt_mode = not self.gt_mode
        self.update_plot()

    def update_plot(self):
        for x in self.artists:
            x.remove()
        for text in self.text_objects:
            text.remove()

        self.artists = []
        self.text_objects = []

        for idx, name in enumerate(self.preds):
            pred = self.preds[name]

            if self.gt_mode:
                latitude = self.data["latitude_field"][0][0]
                text = "\nGT"
            else:
                if "latitude_field" not in pred:
                    continue
                latitude = pred["latitude_field"][0][0]
                text = "\nPrediction"

            self.artists += plot_latitudes([latitude], axes=[self.axes[0][idx]])

            self.text_objects.append(add_text(idx, text))

        # Update the plot
        self.fig.canvas.draw()

    def clear(self):
        # Remove the button
        self.button.disconnect_events()
        self.ax_button.remove()

        for x in self.artists:
            x.remove()
        for text in self.text_objects:
            text.remove()

        self.artists = []
        self.text_objects = []


class LatitudeErrorPlot:
    plot_name = "latitude_error"
    required_keys = ["latitude_field"]

    def __init__(self, fig, axes, data, preds):
        self.artists = []
        for idx, name in enumerate(preds):
            pred = preds[name]
            gt = data["latitude_field"].detach().cpu()

            if "latitude_field" in pred:
                lat = pred["latitude_field"].detach().cpu()
                error = latitude_error(lat, gt)[0].numpy()

                if "latitude_confidence" in pred:
                    confidence = pred["latitude_confidence"].detach().cpu().numpy()
                    confidence = np.log10(confidence).clip(-5)
                    confidence = (confidence + 5) / (confidence.max() + 5)
                    arts = plot_heatmaps(
                        [error], cmap="turbo", axes=[axes[0][idx]], colorbar=True, a=confidence
                    )
                else:
                    arts = plot_heatmaps([error], cmap="turbo", axes=[axes[0][idx]], colorbar=True)
                self.artists += arts

    def clear(self):
        for x in self.artists:
            x.remove()
            x.colorbar.remove()

        self.artists = []


class LatitudeConfidencePlot:
    plot_name = "latitude_confidence"
    required_keys = []
    # required_keys = ["latitude_confidence"]

    def __init__(self, fig, axes, data, preds):
        self.artists = []
        for idx, name in enumerate(preds):
            pred = preds[name]

            if "latitude_confidence" in pred:
                arts = plot_confidences([pred["latitude_confidence"][0]], axes=[axes[0][idx]])
                self.artists += arts

    def clear(self):
        for x in self.artists:
            x.remove()
            x.colorbar.remove()

        self.artists = []


class UpPlot:
    plot_name = "up"
    required_keys = ["up_field"]

    def __init__(self, fig, axes, data, preds):
        self.artists = []
        self.gt_mode = False  # Flag to track whether to display ground truth or predicted
        self.text_objects = []  # To store text objects

        self.fig = fig
        self.axes = axes
        self.data = data
        self.preds = preds

        # Create a toggle button on the lower left corner of the first axis
        self.ax_button = self.fig.add_axes([0.01, 0.02, 0.2, 0.06])
        self.button = Button(self.ax_button, "Toggle GT")
        self.button.on_clicked(self.toggle_display)

        self.update_plot()

    def toggle_display(self, event):
        # Toggle between ground truth and predicted latitudes
        self.gt_mode = not self.gt_mode
        self.update_plot()

    def update_plot(self):
        for x in self.artists:
            x.remove()
        for text in self.text_objects:
            text.remove()

        self.artists = []
        self.text_objects = []

        for idx, name in enumerate(self.preds):
            pred = self.preds[name]

            if self.gt_mode:
                up = self.data["up_field"][0]
                text = "\nGT"
            else:
                if "up_field" not in pred:
                    continue
                up = pred["up_field"][0]
                text = "\nPrediction"

            # Plot up
            self.artists += plot_vector_fields([up], axes=[self.axes[0][idx]])

            self.text_objects.append(add_text(idx, text))

        # Update the plot
        self.fig.canvas.draw()

    def clear(self):
        # Remove the button
        self.button.disconnect_events()
        self.ax_button.remove()

        for x in self.artists:
            x.remove()
        for text in self.text_objects:
            text.remove()

        self.artists = []
        self.text_objects = []


class UpErrorPlot:
    plot_name = "up_error"
    required_keys = ["up_field"]

    def __init__(self, fig, axes, data, preds):
        self.artists = []
        for idx, name in enumerate(preds):
            pred = preds[name]
            gt = data["up_field"].detach().cpu()

            if "up_field" in pred:
                up = pred["up_field"].detach().cpu()
                error = up_error(up, gt)[0].numpy()

                if "up_confidence" in pred:
                    confidence = pred["up_confidence"].detach().cpu().numpy()
                    confidence = np.log10(confidence).clip(-5)
                    confidence = (confidence + 5) / (confidence.max() + 5)
                    arts = plot_heatmaps(
                        [error], cmap="turbo", axes=[axes[0][idx]], colorbar=True, a=confidence
                    )
                else:
                    arts = plot_heatmaps([error], cmap="turbo", axes=[axes[0][idx]], colorbar=True)
                self.artists += arts

    def clear(self):
        for x in self.artists:
            x.remove()
            x.colorbar.remove()

        self.artists = []


class UpConfidencePlot:
    plot_name = "up_confidence"
    required_keys = []
    # required_keys = ["up_confidence"]

    def __init__(self, fig, axes, data, preds):
        self.artists = []
        for idx, name in enumerate(preds):
            pred = preds[name]

            if "up_confidence" in pred:
                arts = plot_confidences([pred["up_confidence"][0]], axes=[axes[0][idx]])
                self.artists += arts

    def clear(self):
        for x in self.artists:
            x.remove()
            x.colorbar.remove()

        self.artists = []


class PerspectiveField:
    plot_name = "perspective_field"
    required_keys = ["camera", "gravity"]

    def __init__(self, fig, axes, data, preds):
        self.artists = []
        self.gt_mode = False  # Flag to track whether to display ground truth or predicted
        self.text_objects = []  # To store text objects

        self.fig = fig
        self.axes = axes
        self.data = data
        self.preds = preds

        # Create a toggle button on the lower left corner of the first axis
        self.ax_button = self.fig.add_axes([0.01, 0.02, 0.2, 0.06])
        self.button = Button(self.ax_button, "Toggle GT")
        self.button.on_clicked(self.toggle_display)

        self.update_plot()

    def toggle_display(self, event):
        # Toggle between ground truth and predicted latitudes
        self.gt_mode = not self.gt_mode
        self.update_plot()

    def update_plot(self):
        for x in self.artists:
            x.remove()
        for text in self.text_objects:
            text.remove()

        self.artists = []
        self.text_objects = []

        for idx, name in enumerate(self.preds):
            pred = self.preds[name]

            if self.gt_mode:
                camera = self.data["camera"]
                gravity = self.data["gravity"]
                text = "\nGT"
            else:
                camera = pred["camera"]
                gravity = pred["gravity"]
                text = "\nPrediction"
                camera = Camera(camera)
                gravity = Gravity(gravity)

            up, latitude = get_perspective_field(camera, gravity)

            self.artists += plot_latitudes([latitude[0, 0]], axes=[self.axes[0][idx]])
            self.artists += plot_vector_fields([up[0]], axes=[self.axes[0][idx]])

            self.text_objects.append(add_text(idx, text))

        # Update the plot
        self.fig.canvas.draw()

    def clear(self):
        # Remove the button
        self.button.disconnect_events()
        self.ax_button.remove()

        for x in self.artists:
            x.remove()
        for text in self.text_objects:
            text.remove()

        self.artists = []
        self.text_objects = []


__plot_dict__ = {
    obj.plot_name: obj
    for _, obj in inspect.getmembers(sys.modules[__name__], predicate=inspect.isclass)
    if hasattr(obj, "plot_name")
}
