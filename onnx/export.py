"""Export GeoCalib's perspective field network to ONNX for COLMAP.

The exported graph takes an RGB image tensor:
    image                [1, 3, H, W] float32 in [0, 1], H and W divisible by 32
and predicts the dense perspective fields and their pixel-wise confidences:
    up_field             [1, 2, H, W] float32, unit 2D up-vectors
    up_confidence        [1, 1, H, W] float32, in [0, 1]
    latitude_field       [1, 1, H, W] float32, radians in (-pi/2, pi/2)
    latitude_confidence  [1, 1, H, W] float32, in [0, 1]

The graph is traced directly from GeoCalib.predict_fields (MSCAN backbone,
low-level encoder, and PerspectiveDecoder) with deterministic NMF2D basis
initialization (GeoCalib.freeze_nmf_bases).

Requires onnx.

Usage (from the root of the repository):
    python onnx/export.py
    python onnx/export.py --weights pinhole
The models are written to onnx/models/ by default. onnx/check.py checks them
against PyTorch.
"""

import argparse
import copy
import hashlib
import sys
import time
from pathlib import Path
from typing import Tuple

import onnx
import torch
from torch import nn

# Also run from a clone of the repository without installing it.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from geocalib import GeoCalib  # noqa: E402
from geocalib.geocalib import GeoCalib as Model  # noqa: E402

FIELD_OUTPUTS = [
    "up_field",
    "up_confidence",
    "latitude_field",
    "latitude_confidence",
]


class PerspectiveFieldsONNX(nn.Module):
    """GeoCalib's perspective field network with (1, C, H, W) tensor outputs."""

    def __init__(self, model: Model) -> None:
        """Initialize the ONNX wrapper around a GeoCalib model."""
        super().__init__()
        self.model = model

    def forward(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict perspective fields and confidences for an input image."""
        out = self.model.predict_fields(image)
        return (
            out["up_field"],
            out["up_confidence"].unsqueeze(1),
            out["latitude_field"],
            out["latitude_confidence"].unsqueeze(1),
        )


def sha256sum(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_single_file(path: Path) -> None:
    """Internalize any external data into a single self-contained ONNX file."""
    model = onnx.load(str(path), load_external_data=True)
    data_file = Path(f"{path}.data")
    if data_file.exists():
        onnx.save_model(model, str(path), save_as_external_data=False)
        data_file.unlink()
    onnx.checker.check_model(model)


def export_perspective_fields(model: Model, path: Path, opset: int = 18) -> None:
    """Export a copy of GeoCalib's perspective field network to ONNX."""
    model = copy.deepcopy(model).cpu().eval()
    model.freeze_nmf_bases()
    wrapper = PerspectiveFieldsONNX(model).eval()
    dummy_image = torch.rand(1, 3, 320, 480, dtype=torch.float32)
    dynamic_axes = {name: {2: "height", 3: "width"} for name in ["image", *FIELD_OUTPUTS]}

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_image,),
        str(path),
        input_names=["image"],
        output_names=FIELD_OUTPUTS,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,
    )
    save_single_file(path)


def print_exported(path: Path, start: float) -> None:
    """Print the path, elapsed time, size, and SHA-256 of an exported model."""
    print(
        f"Exported {path} in {time.time() - start:.1f}s "
        f"({path.stat().st_size / 1e6:.1f}MB, sha256={sha256sum(path)})"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--weights",
        default="distorted",
        help="Pretrained weights ('pinhole' or 'distorted') or path to a checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
        help="Directory where the exported model is written when --output is not set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output .onnx file path.",
    )
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    """Export the GeoCalib perspective field model to ONNX."""
    args = parse_args()
    output_path = args.output or (args.output_dir / f"geocalib_{Path(args.weights).stem}.onnx")
    extractor = GeoCalib(weights=args.weights)
    start = time.time()
    export_perspective_fields(extractor.model, output_path, opset=args.opset)
    print_exported(output_path, start)


if __name__ == "__main__":
    main()
