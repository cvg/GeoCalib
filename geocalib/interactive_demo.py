import argparse
import logging
import queue
import threading
import time
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from geocalib.extractor import GeoCalib
from geocalib.perspective_fields import get_perspective_field
from geocalib.utils import get_device, rad2deg

# flake8: noqa
# mypy: ignore-errors


description = """
-------------------------
GeoCalib Interactive Demo
-------------------------

This script is an interactive demo for GeoCalib. It will open a window showing the camera feed and 
the calibration results. 

Arguments:
- '--camera_id': Camera ID to use. If none, will ask for ip of droidcam (https://droidcam.app)

You can toggle different features using the following keys:

- 'h': Toggle horizon line
- 'u': Toggle up vector field
- 'l': Toggle latitude heatmap
- 'c': Toggle confidence heatmap
- 'd': Toggle undistorted image
- 'g': Toggle grid of points
- 'b': Toggle box object

You can also change the camera model using the following keys:

- '1': Pinhole
- '2': Simple Radial
- '3': Simple Divisional

Press 'q' to quit the demo.
"""


# Custom VideoCapture class to get the most recent frame instead FIFO
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return 1, self.q.get()

    def isOpened(self):
        return self.cap.isOpened()


def add_text(frame, text, align_left=True, align_top=True):
    """Add text to a plot."""
    h, w = frame.shape[:2]
    sc = min(h / 640.0, 2.0)
    Ht = int(40 * sc)  # text height

    for i, l in enumerate(text.split("\n")):
        max_line = len(max([l for l in text.split("\n")], key=len))
        x = int(8 * sc if align_left else w - (max_line) * sc * 18)
        y = Ht * (i + 1) if align_top else h - Ht * (len(text.split("\n")) - i - 1) - int(8 * sc)

        c_back, c_front = (0, 0, 0), (255, 255, 255)
        font, style = cv2.FONT_HERSHEY_DUPLEX, cv2.LINE_AA
        cv2.putText(frame, l, (x, y), font, 1.0 * sc, c_back, int(6 * sc), style)
        cv2.putText(frame, l, (x, y), font, 1.0 * sc, c_front, int(1 * sc), style)
    return frame


def is_corner(p, h, w):
    """Check if a point is a corner."""
    return p in [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]


def plot_latitude(frame, latitude):
    """Plot latitude heatmap."""
    if not isinstance(latitude, np.ndarray):
        latitude = latitude.cpu().numpy()

    cmap = plt.get_cmap("seismic")
    h, w = frame.shape[0], frame.shape[1]
    sc = min(h / 640.0, 2.0)

    vmin, vmax = -90, 90
    latitude = (latitude - vmin) / (vmax - vmin)

    colors = (cmap(latitude)[..., :3] * 255).astype(np.uint8)[..., ::-1]
    frame = cv2.addWeighted(frame, 1 - 0.4, colors, 0.4, 0)

    for contour_line in np.linspace(vmin, vmax, 15):
        contour_line = (contour_line - vmin) / (vmax - vmin)

        mask = (latitude > contour_line).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            color = (np.array(cmap(contour_line))[:3] * 255).astype(np.uint8)[::-1]

            # remove corners
            contour = [p for p in contour if not is_corner(tuple(p[0]), h, w)]
            for index, item in enumerate(contour[:-1]):
                cv2.line(frame, item[0], contour[index + 1][0], color.tolist(), int(5 * sc))

    return frame


def draw_horizon_line(frame, heatmap):
    """Draw a horizon line."""
    if not isinstance(heatmap, np.ndarray):
        heatmap = heatmap.cpu().numpy()

    h, w = frame.shape[0], frame.shape[1]
    sc = min(h / 640.0, 2.0)

    color = (0, 255, 255)
    vmin, vmax = -90, 90
    heatmap = (heatmap - vmin) / (vmax - vmin)

    contours, _ = cv2.findContours(
        (heatmap > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        contour = [p for p in contours[0] if not is_corner(tuple(p[0]), h, w)]
        for index, item in enumerate(contour[:-1]):
            cv2.line(frame, item[0], contour[index + 1][0], color, int(5 * sc))
    return frame


def plot_confidence(frame, confidence):
    """Plot confidence heatmap."""
    if not isinstance(confidence, np.ndarray):
        confidence = confidence.cpu().numpy()

    confidence = np.log10(confidence.clip(1e-6)).clip(-4)
    confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())

    cmap = plt.get_cmap("turbo")
    colors = (cmap(confidence)[..., :3] * 255).astype(np.uint8)[..., ::-1]
    return cv2.addWeighted(frame, 1 - 0.4, colors, 0.4, 0)


def plot_vector_field(frame, vector_field):
    """Plot a vector field."""
    if not isinstance(vector_field, np.ndarray):
        vector_field = vector_field.cpu().numpy()

    H, W = frame.shape[:2]
    sc = min(H / 640.0, 2.0)

    subsample = min(W, H) // 10
    offset_x = ((W % subsample) + subsample) // 2
    samples_x = np.arange(offset_x, W, subsample)
    samples_y = np.arange(int(subsample * 0.9), H, subsample)

    vec_len = 40 * sc
    x_grid, y_grid = np.meshgrid(samples_x, samples_y)
    x, y = vector_field[:, samples_y][:, :, samples_x]
    for xi, yi, xi_dir, yi_dir in zip(x_grid.ravel(), y_grid.ravel(), x.ravel(), y.ravel()):
        start = (xi, yi)
        end = (int(xi + xi_dir * vec_len), int(yi + yi_dir * vec_len))
        cv2.arrowedLine(
            frame, start, end, (0, 255, 0), int(5 * sc), line_type=cv2.LINE_AA, tipLength=0.3
        )

    return frame


def plot_box(frame, gravity, camera):
    """Plot a box object."""
    pts = np.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    )
    pts = pts - np.array([0.5, 1, 0.5])
    rotation_vec = cv2.Rodrigues(gravity.R.numpy()[0])[0]
    t = np.array([0, 0, 1], dtype=float)
    K = camera.K[0].cpu().numpy().astype(float)
    dist = np.zeros(4, dtype=float)
    axis_points, _ = cv2.projectPoints(
        0.1 * pts.reshape(-1, 3).astype(float), rotation_vec, t, K, dist
    )

    h = frame.shape[0]
    sc = min(h / 640.0, 2.0)

    color = (85, 108, 228)
    for p in axis_points:
        center = tuple((int(p[0][0]), int(p[0][1])))
        frame = cv2.circle(frame, center, 10, color, -1, cv2.LINE_AA)

    for i in range(0, 4):
        p1 = axis_points[i].astype(int)
        p2 = axis_points[i + 4].astype(int)
        frame = cv2.line(frame, tuple(p1[0]), tuple(p2[0]), color, int(5 * sc), cv2.LINE_AA)

        p1 = axis_points[i].astype(int)
        p2 = axis_points[(i + 1) % 4].astype(int)
        frame = cv2.line(frame, tuple(p1[0]), tuple(p2[0]), color, int(5 * sc), cv2.LINE_AA)

        p1 = axis_points[i + 4].astype(int)
        p2 = axis_points[(i + 1) % 4 + 4].astype(int)
        frame = cv2.line(frame, tuple(p1[0]), tuple(p2[0]), color, int(5 * sc), cv2.LINE_AA)

    return frame


def plot_grid(frame, gravity, camera, grid_size=0.2, num_points=5):
    """Plot a grid of points."""
    h = frame.shape[0]
    sc = min(h / 640.0, 2.0)

    samples = np.linspace(-grid_size, grid_size, num_points)
    xz = np.meshgrid(samples, samples)
    pts = np.stack((xz[0].ravel(), np.zeros_like(xz[0].ravel()), xz[1].ravel()), axis=-1)

    # project points
    rotation_vec = cv2.Rodrigues(gravity.R.numpy()[0])[0]
    t = np.array([0, 0, 1], dtype=float)
    K = camera.K[0].cpu().numpy().astype(float)
    dist = np.zeros(4, dtype=float)
    axis_points, _ = cv2.projectPoints(pts.reshape(-1, 3).astype(float), rotation_vec, t, K, dist)

    color = (192, 77, 58)
    # draw points
    for p in axis_points:
        center = tuple((int(p[0][0]), int(p[0][1])))
        frame = cv2.circle(frame, center, 10, color, -1, cv2.LINE_AA)

    # draw lines
    for i in range(num_points):
        for j in range(num_points - 1):
            p1 = axis_points[i * num_points + j].astype(int)
            p2 = axis_points[i * num_points + j + 1].astype(int)
            frame = cv2.line(frame, tuple(p1[0]), tuple(p2[0]), color, int(5 * sc), cv2.LINE_AA)

            p1 = axis_points[j * num_points + i].astype(int)
            p2 = axis_points[(j + 1) * num_points + i].astype(int)
            frame = cv2.line(frame, tuple(p1[0]), tuple(p2[0]), color, int(5 * sc), cv2.LINE_AA)

    return frame


def undistort_image(img, camera, padding=0.3):
    """Undistort an image."""
    W, H = camera.size.unbind(-1)
    H, W = H.int().item(), W.int().item()

    pad_h, pad_w = int(H * padding), int(W * padding)
    x, y = torch.meshgrid(torch.arange(0, W + pad_w), torch.arange(0, H + pad_h), indexing="xy")
    coords = torch.stack((x, y), dim=-1).reshape(-1, 2) - torch.tensor([pad_w / 2, pad_h / 2])

    p3d, _ = camera.pinhole().image2world(coords.to(camera.device).to(camera.dtype))
    p2d, _ = camera.world2image(p3d)

    p2d = p2d.float().numpy().reshape(H + pad_h, W + pad_w, 2)
    img = cv2.remap(img, p2d[..., 0], p2d[..., 1], cv2.INTER_LINEAR, borderValue=(254, 254, 254))
    return cv2.resize(img, (W, H))


class InteractiveDemo:
    def __init__(self, capture: VideoCapture, device: str) -> None:
        self.cap = capture

        self.device = torch.device(device)
        self.model = GeoCalib().to(device)

        self.up_toggle = False
        self.lat_toggle = False
        self.conf_toggle = False

        self.hl_toggle = False
        self.grid_toggle = False
        self.box_toggle = False

        self.undist_toggle = False

        self.camera_model = "pinhole"

    def render_frame(self, frame, calibration):
        """Render the frame with the calibration results."""
        camera, gravity = calibration["camera"].cpu(), calibration["gravity"].cpu()

        if self.undist_toggle:
            return undistort_image(frame, camera)

        up, lat = get_perspective_field(camera, gravity)

        if gravity.pitch[0] > 0:
            frame = plot_box(frame, gravity, camera) if self.box_toggle else frame
            frame = plot_grid(frame, gravity, camera) if self.grid_toggle else frame
        else:
            frame = plot_grid(frame, gravity, camera) if self.grid_toggle else frame
            frame = plot_box(frame, gravity, camera) if self.box_toggle else frame

        frame = draw_horizon_line(frame, lat[0, 0]) if self.hl_toggle else frame

        if self.conf_toggle and self.up_toggle:
            frame = plot_confidence(frame, calibration["up_confidence"][0])
        frame = plot_vector_field(frame, up[0]) if self.up_toggle else frame

        if self.conf_toggle and self.lat_toggle:
            frame = plot_confidence(frame, calibration["latitude_confidence"][0])
        frame = plot_latitude(frame, rad2deg(lat)[0, 0]) if self.lat_toggle else frame

        return frame

    def format_results(self, calibration):
        """Format the calibration results."""
        camera, gravity = calibration["camera"].cpu(), calibration["gravity"].cpu()

        vfov, focal = camera.vfov[0].item(), camera.f[0, 0].item()
        fov_unc = rad2deg(calibration["vfov_uncertainty"].item())
        f_unc = calibration["focal_uncertainty"].item()

        roll, pitch = gravity.rp[0].unbind(-1)
        roll, pitch, vfov = rad2deg(roll), rad2deg(pitch), rad2deg(vfov)
        roll_unc = rad2deg(calibration["roll_uncertainty"].item())
        pitch_unc = rad2deg(calibration["pitch_uncertainty"].item())

        text = f"{self.camera_model.replace('_', ' ').title()}\n"
        text += f"Roll:  {roll:.2f} (+- {roll_unc:.2f})\n"
        text += f"Pitch: {pitch:.2f} (+- {pitch_unc:.2f})\n"
        text += f"vFoV:  {vfov:.2f} (+- {fov_unc:.2f})\n"
        text += f"Focal: {focal:.2f} (+- {f_unc:.2f})"

        if hasattr(camera, "k1"):
            text += f"\nK1:    {camera.k1[0].item():.2f}"

        return text

    def update_toggles(self):
        """Update the toggles."""
        key = cv2.waitKey(100) & 0xFF
        if key == ord("h"):
            self.hl_toggle = not self.hl_toggle
        elif key == ord("u"):
            self.up_toggle = not self.up_toggle
        elif key == ord("l"):
            self.lat_toggle = not self.lat_toggle
        elif key == ord("c"):
            self.conf_toggle = not self.conf_toggle
        elif key == ord("d"):
            self.undist_toggle = not self.undist_toggle
        elif key == ord("g"):
            self.grid_toggle = not self.grid_toggle
        elif key == ord("b"):
            self.box_toggle = not self.box_toggle

        elif key == ord("1"):
            self.camera_model = "pinhole"
        elif key == ord("2"):
            self.camera_model = "simple_radial"
        elif key == ord("3"):
            self.camera_model = "simple_divisional"

        elif key == ord("q"):
            return True

        return False

    def run(self):
        """Run the interactive demo."""
        while True:
            start = time()
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Failed to retrieve frame.")
                break

            # create tensor from frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = torch.tensor(img).permute(2, 0, 1) / 255.0

            calibration = self.model.calibrate(img.to(self.device), camera_model=self.camera_model)

            # render results to the frame
            frame = self.render_frame(frame, calibration)
            frame = add_text(frame, self.format_results(calibration))

            end = time()
            frame = add_text(
                frame, f"FPS: {1 / (end - start):04.1f}", align_left=False, align_top=False
            )

            cv2.imshow("GeoCalib Demo", frame)

            if self.update_toggles():
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera_id",
        type=int,
        default=None,
        help="Camera ID to use. If none, will ask for ip of droidcam.",
    )
    args = parser.parse_args()

    print(description)

    device = get_device()
    print(f"Running on: {device}")

    # setup video capture
    if args.camera_id is not None:
        cap = VideoCapture(args.camera_id)
    else:
        ip = input("Enter the IP address of the camera: ")
        cap = VideoCapture(f"http://{ip}:4747/video/force/1920x1080")

    if not cap.isOpened():
        raise ValueError("Error: Could not open camera.")

    demo = InteractiveDemo(cap, device)
    demo.run()


if __name__ == "__main__":
    main()
