"""Helper script to download and extract OpenPano dataset."""

import argparse
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from siclib import logger

PANO_URL = "https://polybox.ethz.ch/index.php/s/XK4oM1l6ZqSIXw9/download"
# PANO_URL = "https://cvg-data.inf.ethz.ch/GeoCalib_ECCV2024/openpano.zip"


def download_and_extract_dataset(name: str, url: Path, output: Path) -> None:
    """Download and extract a dataset from a URL."""
    dataset_dir = output / name
    if not output.exists():
        output.mkdir(parents=True)

    if dataset_dir.exists():
        logger.info(f"Dataset {name} already exists at {dataset_dir}, skipping download.")
        return

    zip_file = output / f"{name}.zip"

    if not zip_file.exists():
        logger.info(f"Downloading dataset {name} to {zip_file} from {url}.")
        torch.hub.download_url_to_file(url, zip_file)

    logger.info(f"Extracting dataset {name} in {output}.")
    shutil.unpack_archive(zip_file, output, format="zip")
    zip_file.unlink()


def main():
    """Prepare the OpenPano dataset."""
    parser = argparse.ArgumentParser(description="Download and extract OpenPano dataset.")
    parser.add_argument("--name", type=str, default="openpano", help="Name of the dataset.")
    parser.add_argument(
        "--laval_dir", type=str, default="data/laval-tonemap", help="Path the Laval dataset."
    )

    args = parser.parse_args()

    out_dir = Path("data")
    download_and_extract_dataset(args.name, PANO_URL, out_dir)

    pano_dir = out_dir / args.name / "panoramas"
    for split in ["train", "test", "val"]:
        with open(pano_dir / f"{split}_panos.txt", "r") as f:
            pano_list = f.readlines()
            pano_list = [fname.strip() for fname in pano_list]

        for fname in tqdm(pano_list, ncols=80, desc=f"Copying {split} panoramas"):
            laval_path = Path(args.laval_dir) / fname
            target_path = pano_dir / split / fname

            # pano either exists in laval or is in split
            if target_path.exists():
                continue

            if laval_path.exists():
                shutil.copy(laval_path, target_path)
            else:  # not in laval and not in split
                logger.warning(f"Panorama {fname} not found in {args.laval_dir} or {split} split.")

    n_train = len(list(pano_dir.glob("train/*.jpg")))
    n_test = len(list(pano_dir.glob("test/*.jpg")))
    n_val = len(list(pano_dir.glob("val/*.jpg")))
    logger.info(f"{args.name} contains {n_train}/{n_test}/{n_val} train/test/val panoramas.")


if __name__ == "__main__":
    main()
