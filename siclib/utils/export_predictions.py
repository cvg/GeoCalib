"""
Export the predictions of a model for a given dataloader (e.g. ImageFolder).
Use a standalone script with `python3 -m geocalib.scipts.export_predictions dir`
or call from another script.
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from siclib.utils.tensor import batch_to_device
from siclib.utils.tools import get_device

# flake8: noqa
# mypy: ignore-errors

logger = logging.getLogger(__name__)


@torch.no_grad()
def export_predictions(
    loader,
    model,
    output_file,
    as_half=False,
    keys="*",
    callback_fn=None,
    optional_keys=None,
    verbose=True,
):  # sourcery skip: low-code-quality
    if optional_keys is None:
        optional_keys = []

    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    hfile = h5py.File(str(output_file), "w")
    device = get_device()
    model = model.to(device).eval()

    if not verbose:
        logger.info(f"Exporting predictions to {output_file}")

    for data_ in tqdm(loader, desc="Exporting", total=len(loader), ncols=80, disable=not verbose):
        data = batch_to_device(data_, device, non_blocking=True)
        pred = model(data)
        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}
        if keys != "*":
            if len(set(keys) - set(pred.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
            pred = {k: v for k, v in pred.items() if k in keys + optional_keys}

        # assert len(pred) > 0, "No predictions found"

        for idx in range(len(data["name"])):
            pred_ = {k: v[idx].cpu().numpy() for k, v in pred.items()}

            if as_half:
                for k in pred_:
                    dt = pred_[k].dtype
                    if (dt == np.float32) and (dt != np.float16):
                        pred_[k] = pred_[k].astype(np.float16)
            try:
                name = data["name"][idx]
                try:
                    grp = hfile.create_group(name)
                except ValueError as e:
                    raise ValueError(f"Group already exists {name}") from e

                # grp = hfile.create_group(name)
                for k, v in pred_.items():
                    grp.create_dataset(k, data=v)
            except RuntimeError:
                print(f"Failed to export {name}")
                continue

        del pred

    hfile.close()
    return output_file
