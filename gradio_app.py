"""Gradio app for GeoCalib inference."""

from copy import deepcopy
from time import time

import gradio as gr
import numpy as np
import spaces
import torch

from geocalib import logger, viz2d
from geocalib.camera import camera_models
from geocalib.extractor import GeoCalib
from geocalib.perspective_fields import get_perspective_field
from geocalib.utils import rad2deg

# flake8: noqa
# mypy: ignore-errors

description = """
<p align="center">
  <h1 align="center"><ins>GeoCalib</ins> 📸<br>Single-image Calibration with Geometric Optimization</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/alexander-veicht/">Alexander Veicht</a>
    ·
    <a href="https://psarlin.com/">Paul-Edouard&nbsp;Sarlin</a>
    ·
    <a href="https://www.linkedin.com/in/philipplindenberger/">Philipp Lindenberger</a>
    ·
    <a href="https://www.microsoft.com/en-us/research/people/mapoll/">Marc&nbsp;Pollefeys</a>
  </p>
  <h2 align="center">
    <p>ECCV 2024</p>
    <a href="https://arxiv.org/pdf/2409.06704" align="center">Paper</a> |
    <a href="https://github.com/cvg/GeoCalib" align="center">Code</a> |
    <a href="https://colab.research.google.com/drive/1oMzgPGppAPAIQxe-s7SRd_q8r7dVfnqo#scrollTo=etdzQZQzoo-K" align="center">Colab</a>
  </h2>
</p>

## Getting Started
GeoCalib accurately estimates the camera intrinsics and gravity direction from a single image by 
combining geometric optimization with deep learning.

To get started, upload an image or select one of the examples below.
You can choose between different camera models and visualize the calibration results.

"""

example_images = [
    ["assets/pinhole-church.jpg"],
    ["assets/pinhole-garden.jpg"],
    ["assets/fisheye-skyline.jpg"],
    ["assets/fisheye-dog-pool.jpg"],
]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoCalib().to(device)


def format_output(results):
    camera, gravity = results["camera"], results["gravity"]
    vfov = rad2deg(camera.vfov)
    roll, pitch = rad2deg(gravity.rp).unbind(-1)

    txt = "Estimated parameters:\n"
    txt += f"Roll:  {roll.item():.2f}° (± {rad2deg(results['roll_uncertainty']).item():.2f})°\n"
    txt += f"Pitch: {pitch.item():.2f}° (± {rad2deg(results['pitch_uncertainty']).item():.2f})°\n"
    txt += f"vFoV:  {vfov.item():.2f}° (± {rad2deg(results['vfov_uncertainty']).item():.2f})°\n"
    txt += (
        f"Focal: {camera.f[0, 1].item():.2f} px (± {results['focal_uncertainty'].item():.2f} px)\n"
    )
    if hasattr(camera, "k1"):
        txt += f"K1:    {camera.k1[0].item():.2f}\n"
    return txt


@spaces.GPU(duration=10)
def inference(img, camera_model):
    out = model.calibrate(img.to(device), camera_model=camera_model)
    save_keys = ["camera", "gravity"] + [
        f"{k}_uncertainty" for k in ["roll", "pitch", "vfov", "focal"]
    ]
    res = {k: v.cpu() for k, v in out.items() if k in save_keys}
    # not converting to numpy results in gpu abort
    res["up_confidence"] = out["up_confidence"].cpu().numpy()
    res["latitude_confidence"] = out["latitude_confidence"].cpu().numpy()
    return res


def process_results(
    image_path,
    camera_model,
    plot_up,
    plot_up_confidence,
    plot_latitude,
    plot_latitude_confidence,
    plot_undistort,
):
    """Process the image and return the calibration results."""

    if image_path is None:
        raise gr.Error("Please upload an image first.")

    img = model.load_image(image_path)
    start = time()
    inference_result = inference(img, camera_model)
    logger.info(f"Calibration took {time() - start:.2f} sec. ({camera_model})")
    inference_result["image"] = img.cpu()

    if inference_result is None:
        return ("", np.ones((128, 256, 3)), None)

    plot_img = update_plot(
        inference_result,
        plot_up,
        plot_up_confidence,
        plot_latitude,
        plot_latitude_confidence,
        plot_undistort,
    )

    return format_output(inference_result), plot_img, inference_result


def update_plot(
    inference_result,
    plot_up,
    plot_up_confidence,
    plot_latitude,
    plot_latitude_confidence,
    plot_undistort,
):
    """Update the plot based on the selected options."""
    if inference_result is None:
        gr.Error("Please calibrate an image first.")
        return np.ones((128, 256, 3))

    camera, gravity = inference_result["camera"], inference_result["gravity"]
    img = inference_result["image"].permute(1, 2, 0).numpy()

    if plot_undistort:
        if not hasattr(camera, "k1"):
            return img

        return camera.undistort_image(inference_result["image"][None])[0].permute(1, 2, 0).numpy()

    up, lat = get_perspective_field(camera, gravity)

    fig = viz2d.plot_images([img], pad=0)
    ax = fig.get_axes()

    if plot_up:
        viz2d.plot_vector_fields([up[0]], axes=[ax[0]])

    if plot_latitude:
        viz2d.plot_latitudes([lat[0, 0]], axes=[ax[0]])

    if plot_up_confidence:
        viz2d.plot_confidences([torch.tensor(inference_result["up_confidence"][0])], axes=[ax[0]])

    if plot_latitude_confidence:
        viz2d.plot_confidences(
            [torch.tensor(inference_result["latitude_confidence"][0])], axes=[ax[0]]
        )

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())

    return img


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            gr.Markdown("""## Input Image""")
            image_path = gr.Image(label="Upload image to calibrate", type="filepath")
            choice_input = gr.Dropdown(
                choices=list(camera_models.keys()), label="Choose a camera model.", value="pinhole"
            )
            submit_btn = gr.Button("Calibrate 📸")
            gr.Examples(examples=example_images, inputs=[image_path, choice_input])

        with gr.Column():
            gr.Markdown("""## Results""")
            image_output = gr.Image(label="Calibration Results")
            gr.Markdown("### Plot Options")
            plot_undistort = gr.Checkbox(
                label="undistort",
                value=False,
                info="Undistorted image "
                + "(this is only available for models with distortion "
                + "parameters and will overwrite other options).",
            )

            with gr.Row():
                plot_up = gr.Checkbox(label="up-vectors", value=True)
                plot_up_confidence = gr.Checkbox(label="up confidence", value=False)
                plot_latitude = gr.Checkbox(label="latitude", value=True)
                plot_latitude_confidence = gr.Checkbox(label="latitude confidence", value=False)

            gr.Markdown("### Calibration Results")
            text_output = gr.Textbox(label="Estimated parameters", type="text", lines=5)

    # Define the action when the button is clicked
    inference_state = gr.State()
    plot_inputs = [
        inference_state,
        plot_up,
        plot_up_confidence,
        plot_latitude,
        plot_latitude_confidence,
        plot_undistort,
    ]
    submit_btn.click(
        fn=process_results,
        inputs=[image_path, choice_input] + plot_inputs[1:],
        outputs=[text_output, image_output, inference_state],
    )

    # Define the action when the plot checkboxes are clicked
    plot_up.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)
    plot_up_confidence.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)
    plot_latitude.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)
    plot_latitude_confidence.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)
    plot_undistort.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)


# Launch the app
demo.launch()
