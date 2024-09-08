import argparse
import subprocess
from pathlib import Path

# flake8: noqa
# mypy: ignore-errors

parser = argparse.ArgumentParser(description="Aligns a COLMAP model and plots the horizon lines")
parser.add_argument(
    "--base_dir", type=str, help="Path to the base directory of the MegaDepth dataset"
)
parser.add_argument("--out_dir", type=str, help="Path to the output directory")
args = parser.parse_args()

base_dir = Path(args.base_dir)
out_dir = Path(args.out_dir)

scenes = [d.name for d in base_dir.iterdir() if d.is_dir()]
print(scenes[:3], len(scenes))

# exit()

for scene in scenes:
    image_dir = base_dir / scene / "images"
    sfm_dir = base_dir / scene / "sparse" / "manhattan" / "0"

    # Align model
    align_dir = out_dir / scene / "sparse" / "align"
    align_dir.mkdir(exist_ok=True, parents=True)

    print(f"image_dir ({image_dir.exists()}): {image_dir}")
    print(f"sfm_dir   ({sfm_dir.exists()}): {sfm_dir}")
    print(f"align_dir ({align_dir.exists()}): {align_dir}")

    cmd = (
        "colmap model_orientation_aligner "
        + f"--image_path {image_dir} "
        + f"--input_path {sfm_dir} "
        + f"--output_path {str(align_dir)}"
    )
    subprocess.run(cmd, shell=True)
