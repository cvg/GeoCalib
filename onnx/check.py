"""Check GeoCalib models exported to ONNX against the PyTorch implementation.

Runs ONNX Runtime and PyTorch (with the given checkpoint) on the asset images
across landscape, portrait, and square aspect ratios, and compares:
- the dense perspective fields (up_field, latitude_field) and their confidences
  (up_confidence, latitude_confidence);
- the camera intrinsics (vfov, radial distortion k1) and gravity (roll, pitch)
  optimized by LMOptimizer from the predicted perspective fields.

Requires onnx and onnxruntime (onnxruntime-gpu on GPU).

Usage (from the root of the repository):
    python onnx/check.py --model onnx/models/geocalib_distorted.onnx
    python onnx/check.py --model onnx/models/geocalib_pinhole.onnx --weights pinhole
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch

# Also run from a clone of the repository without installing it.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from export import FIELD_OUTPUTS  # noqa: E402
from geocalib import GeoCalib  # noqa: E402
from geocalib.utils import load_image, rad2deg  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
CHECK_IMAGES = [
    ("pinhole-church.jpg", "pinhole"),
    ("pinhole-garden.jpg", "pinhole"),
    ("fisheye-dog-pool.jpg", "simple_radial"),
    ("fisheye-skyline.jpg", "simple_radial"),
]

# Maximum absolute error tolerances for float32 on CPU and CUDA (where ONNX
# Runtime uses TF32 for float32 matrix multiplications by default).
TOLERANCES = {
    "cpu": dict(field=1e-4, confidence=1e-4, angle_deg=1e-2, dist=1e-3),
    "cuda": dict(field=5e-3, confidence=5e-3, angle_deg=0.1, dist=1e-2),
}


def onnx_providers(cpu: bool) -> List[str]:
    """Select ONNX Runtime execution providers."""
    if not cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def time_session(
    session: ort.InferenceSession, feeds: Dict[str, np.ndarray], warmup: int = 2, runs: int = 10
) -> float:
    """Average run time of an ONNX Runtime session in milliseconds."""
    for _ in range(warmup):
        session.run(None, feeds)
    start = time.time()
    for _ in range(runs):
        session.run(None, feeds)
    return (time.time() - start) * 1000 / runs


def predict_pytorch(
    extractor: GeoCalib, img_data: Dict[str, torch.Tensor], camera_model: str
) -> Dict[str, torch.Tensor]:
    """Run PyTorch perspective field prediction and LM optimization."""
    device = next(extractor.model.parameters()).device
    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in img_data.items()}
    extractor.model.optimizer.set_camera_model(camera_model)
    with torch.no_grad():
        return extractor.model(data)


def optimize_from_onnx_fields(
    extractor: GeoCalib,
    img_data: Dict[str, torch.Tensor],
    ort_fields: Dict[str, np.ndarray],
    camera_model: str,
) -> Dict[str, torch.Tensor]:
    """Run the PyTorch LMOptimizer on perspective fields predicted by ONNX Runtime."""
    device = next(extractor.model.parameters()).device
    opt_input = {
        "image": img_data["image"].to(device),
        "scales": img_data["scales"].to(device),
        "up_field": torch.from_numpy(ort_fields["up_field"]).to(device),
        "up_confidence": torch.from_numpy(ort_fields["up_confidence"]).squeeze(1).to(device),
        "latitude_field": torch.from_numpy(ort_fields["latitude_field"]).to(device),
        "latitude_confidence": (
            torch.from_numpy(ort_fields["latitude_confidence"]).squeeze(1).to(device)
        ),
    }
    extractor.model.optimizer.set_camera_model(camera_model)
    with torch.no_grad():
        return extractor.model.optimizer(opt_input)


def compare_fields(
    pt_out: Dict[str, torch.Tensor], ort_fields: Dict[str, np.ndarray], tol: Dict[str, float]
) -> Tuple[bool, str]:
    """Compare dense perspective fields and confidences."""
    up_err = float(np.abs(pt_out["up_field"].cpu().numpy() - ort_fields["up_field"]).max())
    lat_err = float(
        np.abs(pt_out["latitude_field"].cpu().numpy() - ort_fields["latitude_field"]).max()
    )
    up_conf_err = float(
        np.abs(
            pt_out["up_confidence"].unsqueeze(1).cpu().numpy() - ort_fields["up_confidence"]
        ).max()
    )
    lat_conf_err = float(
        np.abs(
            pt_out["latitude_confidence"].unsqueeze(1).cpu().numpy()
            - ort_fields["latitude_confidence"]
        ).max()
    )
    conf_err = max(up_conf_err, lat_conf_err)
    ok = up_err <= tol["field"] and lat_err <= tol["field"] and conf_err <= tol["confidence"]
    msg = f"up={up_err:.1e} lat={lat_err:.1e} conf={conf_err:.1e}"
    return bool(ok), msg


def compare_calibration(
    pt_out: Dict[str, torch.Tensor], ort_opt: Dict[str, torch.Tensor], tol: Dict[str, float]
) -> Tuple[bool, str]:
    """Compare calibrated camera and gravity parameters."""
    vfov_err = float(rad2deg((pt_out["camera"].vfov - ort_opt["camera"].vfov).abs()).max().item())
    rp_err = float(rad2deg((pt_out["gravity"].rp - ort_opt["gravity"].rp).abs()).max().item())
    ok = vfov_err <= tol["angle_deg"] and rp_err <= tol["angle_deg"]
    msg = f"vfov={vfov_err:.1e}deg rp={rp_err:.1e}deg"
    if hasattr(pt_out["camera"], "dist") and pt_out["camera"].dist is not None:
        dist_err = float((pt_out["camera"].dist - ort_opt["camera"].dist).abs().max().item())
        ok &= dist_err <= tol["dist"]
        msg += f" dist={dist_err:.1e}"
    return bool(ok), msg


def check_input(
    session: ort.InferenceSession,
    extractor: GeoCalib,
    label: str,
    img_data: Dict[str, torch.Tensor],
    camera_model: str,
    tol: Dict[str, float],
    timing: bool,
) -> bool:
    """Compare ONNX Runtime and PyTorch on one preprocessed image."""
    feeds = {"image": img_data["image"].numpy()}
    outputs = session.run(FIELD_OUTPUTS, feeds)
    ort_fields = dict(zip(FIELD_OUTPUTS, outputs, strict=True))

    pt_out = predict_pytorch(extractor, img_data, camera_model)
    ort_opt = optimize_from_onnx_fields(extractor, img_data, ort_fields, camera_model)

    fields_ok, fields_msg = compare_fields(pt_out, ort_fields, tol)
    calib_ok, calib_msg = compare_calibration(pt_out, ort_opt, tol)
    ok = fields_ok and calib_ok

    h, w = img_data["image"].shape[-2:]
    line = f"  {label:<18s} ({h}x{w}, {camera_model}): {fields_msg} {calib_msg}"
    if timing:
        line += f" ort_time={time_session(session, feeds):.1f}ms"
    print(f"{line} {'OK' if ok else 'FAIL'}")
    return bool(ok)


def square_crop(image: torch.Tensor) -> torch.Tensor:
    """Center-crop a (..., H, W) image tensor to a square."""
    h, w = image.shape[-2:]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return image[..., top : top + s, left : left + s]


def check_image(
    session: ort.InferenceSession,
    extractor: GeoCalib,
    name: str,
    camera_model: str,
    tol: Dict[str, float],
    timing: bool,
) -> bool:
    """Check an asset image in landscape, portrait (rotated 90°), and square shapes."""
    print(f"[{name}]")
    image = load_image(ASSETS / name)[None]
    variants = [
        ("landscape", extractor.image_processor(image)),
        ("portrait (rot90)", extractor.image_processor(torch.rot90(image, 1, [-2, -1]))),
        ("square", extractor.image_processor(square_crop(image))),
    ]
    all_ok = True
    for label, img_data in variants:
        all_ok &= check_input(session, extractor, label, img_data, camera_model, tol, timing)
    return all_ok


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="Exported ONNX model path.")
    parser.add_argument(
        "--weights",
        default="distorted",
        help="PyTorch checkpoint ('pinhole', 'distorted', or path).",
    )
    parser.add_argument("--timing", action="store_true", help="Time the ONNX model.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution provider.")
    return parser.parse_args()


def main() -> None:
    """Run the ONNX vs PyTorch verification check."""
    args = parse_args()
    onnx_model = onnx.load(str(args.model))
    onnx.checker.check_model(onnx_model)

    providers = onnx_providers(args.cpu)
    device = "cuda" if providers[0].startswith("CUDA") else "cpu"
    print(f"ONNX Runtime {ort.__version__}, providers: {providers}")

    session = ort.InferenceSession(str(args.model), providers=providers)
    extractor = GeoCalib(weights=args.weights).to(device)
    extractor.model.freeze_nmf_bases()
    extractor.model.eval()

    tol = TOLERANCES[device]
    all_ok = True
    for name, camera_model in CHECK_IMAGES:
        all_ok &= check_image(session, extractor, name, camera_model, tol, args.timing)
    print("CHECK", "PASSED" if all_ok else "FAILED")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
