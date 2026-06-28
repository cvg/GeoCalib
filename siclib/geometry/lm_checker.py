"""Check the LM optimizer across calibration setups on panorama-rendered images.

Each crop from a panorama is a camera in a rigid rig with known gravity/intrinsics, so we
render ``--n-samples`` random rigs and run GeoCalib for every setup in ``SETUPS`` (vanilla,
shared-intrinsics, focal/gravity priors, rig modes), reporting gravity/vFoV error, runtime,
and win-rate over the vanilla baseline.
Add a setup by appending one ``Setup(...)`` to ``SETUPS``.

Run: python -m siclib.geometry.lm_checker --pano-path PANO.jpg [--camera-model simple_radial]
"""

import argparse
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from geocalib import GeoCalib
from geocalib.camera import camera_models
from geocalib.gravity import Gravity
from geocalib.utils import deg2rad, get_device, load_image, rad2rotmat

# mypy: ignore-errors

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("lm_checker")
logging.getLogger("geocalib").setLevel(logging.ERROR)

H = W = 320  # rendered crop size (GeoCalib runs at 320px anyway)


@dataclass
class Setup:
    """A calibration setup. ``priors`` is a subset of {"focal", "gravity"} fed as GT."""

    name: str
    use_rig: bool = False
    shared_intrinsics: bool = False
    priors: Tuple[str, ...] = ()
    camera_model: Optional[str] = None  # None -> use the global --camera-model


SETUPS = [
    Setup("vanilla"),  # baseline
    Setup("shared_intrinsics", shared_intrinsics=True),
    Setup("prior_focal", priors=("focal",)),
    Setup("prior_gravity", priors=("gravity",)),
    Setup("rig_unshared", use_rig=True, shared_intrinsics=False),  # shared gravity only
    Setup("rig_shared", use_rig=True, shared_intrinsics=True),  # shared gravity + intrinsics
]
BASELINE = "vanilla"


def sample_rig(
    n_cameras: int, camera_model: str, device: torch.device, gen: torch.Generator
) -> dict:
    """Sample a random multi-camera rig of crops from one panorama with shared intrinsics."""
    rand = lambda lo, hi, n=n_cameras: lo + (hi - lo) * torch.rand(n, generator=gen)
    rolls, pitches, yaws = deg2rad(rand(-45, 45)), deg2rad(rand(-45, 45)), deg2rad(rand(-180, 180))
    vfov = deg2rad(rand(40, 90, 1)).expand(n_cameras).clone()  # shared across the rig

    Cam = camera_models[camera_model]
    params = {
        "height": torch.full((n_cameras,), float(H)),
        "width": torch.full((n_cameras,), float(W)),
        "vfov": vfov,
    }
    if hasattr(Cam, "num_dist_params"):  # distorted model -> mild shared barrel distortion
        params["k1"] = rand(-0.2, -0.05, 1).expand(n_cameras).clone()
        if Cam.num_dist_params() == 2:
            params["k2"] = rand(0.0, 0.02, 1).expand(n_cameras).clone()
    cam = Cam.from_dict(params).float().to(device)
    gravity = Gravity.from_rp(rolls, pitches).float().to(device)

    R = rad2rotmat(rolls, pitches, yaws)  # cam_i_from_world
    rf = deg2rad(rand(-180, 180, 3))  # arbitrary rig frame, not aligned to any camera (R0 != I)
    R_rig = rad2rotmat(rf[0:1], rf[1:2], rf[2:3])
    camera_R_rig = (R @ R_rig.transpose(-1, -2)).float().to(device)

    return {
        "cam": cam,
        "gravity": gravity,
        "yaws": yaws.to(device),
        "vfov": vfov.to(device),
        "gt_focal": cam.f[..., 1],
        "camera_R_rig": camera_R_rig,
    }


def render(pano: torch.Tensor, rig: dict) -> torch.Tensor:
    """Render the rig's crops from the panorama -> (B, 3, H, W) in [0, 1]."""
    return rig["cam"].get_img_from_pano(
        pano_img=pano,
        gravity=rig["gravity"],
        yaws=rig["yaws"],
        resize_factor=[1.0] * rig["cam"].shape[0],
    )


def calibrate_kwargs(setup: Setup, rig: dict, camera_model: str) -> dict:
    """Translate a Setup + rig into kwargs for ``model.calibrate``."""
    kw = dict(
        camera_model=setup.camera_model or camera_model, shared_intrinsics=setup.shared_intrinsics
    )
    if setup.use_rig:
        kw["camera_R_rig"] = rig["camera_R_rig"]
    priors = {}
    if "focal" in setup.priors:
        priors["focal"] = rig["gt_focal"]
    if "gravity" in setup.priors:
        priors["gravity"] = rig["gravity"].vec3d
    if priors:
        kw["priors"] = priors
    return kw


def angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between batched unit vectors."""
    return torch.rad2deg(torch.arccos((a * b).sum(-1).clamp(-1, 1)))


def timed_calibrate(model: GeoCalib, crops: torch.Tensor, kw: dict, device: torch.device):
    """Run calibrate, returning (output, milliseconds)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.calibrate(crops, **kw)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1e3


def _mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _winrate(setup_xs, base_xs):
    wins = [s < b for s, b in zip(setup_xs, base_xs) if not (math.isnan(s) or math.isnan(b))]
    return f"{100 * sum(wins) / len(wins):.0f}%" if wins else "-"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pano-path", required=True, help="Path to an equirectangular panorama.")
    p.add_argument("--n-samples", type=int, default=20, help="Number of random rigs to evaluate.")
    p.add_argument("--n-cameras", type=int, default=4, help="Cameras per rig.")
    p.add_argument(
        "--camera-model",
        default="pinhole",
        choices=list(camera_models),
        help="GT + estimation model.",
    )
    p.add_argument(
        "--weights", default=None, help='GeoCalib weights; default picks "pinhole"/"distorted".'
    )
    p.add_argument("--device", default=get_device())
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    gen = torch.Generator().manual_seed(args.seed)
    weights = args.weights or ("pinhole" if args.camera_model == "pinhole" else "distorted")

    model = GeoCalib(weights=weights).to(device).eval()
    pano = load_image(args.pano_path).to(device)
    logger.info(
        "pano %s | model=%s weights=%s | %d samples x %d cams on %s",
        tuple(pano.shape),
        args.camera_model,
        weights,
        args.n_samples,
        args.n_cameras,
        device,
    )

    results = {s.name: {"grav": [], "vfov": [], "ms": []} for s in SETUPS}

    # warmup so the first timed call is not skewed by lazy CUDA/cudnn init
    warm = sample_rig(args.n_cameras, args.camera_model, device, gen)
    with torch.no_grad():
        model.calibrate(render(pano, warm), **calibrate_kwargs(SETUPS[0], warm, args.camera_model))

    t_start = time.perf_counter()
    for i in range(args.n_samples):
        rig = sample_rig(args.n_cameras, args.camera_model, device, gen)
        crops = render(pano, rig)
        for s in SETUPS:
            try:
                with torch.no_grad():
                    out, ms = timed_calibrate(
                        model, crops, calibrate_kwargs(s, rig, args.camera_model), device
                    )
                ge = angle_deg(out["gravity"].vec3d, rig["gravity"].vec3d).mean().item()
                ve = torch.rad2deg((out["camera"].vfov - rig["vfov"]).abs()).mean().item()
            except Exception as e:  # noqa: BLE001 -- a broken path should not abort the sweep
                logger.warning("sample %d setup '%s' failed: %s", i, s.name, e)
                ge = ve = ms = float("nan")
            results[s.name]["grav"].append(ge)
            results[s.name]["vfov"].append(ve)
            results[s.name]["ms"].append(ms)
        logger.info("sample %d/%d done", i + 1, args.n_samples)

    base = results[BASELINE]
    print(
        f"\n{'setup':<18}{'grav(deg)':>11}{'vfov(deg)':>11}"
        + f"{'time(ms)':>10}{'grav win':>10}{'vfov win':>10}"
    )
    print("-" * 70)
    for s in SETUPS:
        r = results[s.name]
        gm, vm, tm = _mean(r["grav"]), _mean(r["vfov"]), _mean(r["ms"])
        gw, vw = (
            ("-", "-")
            if s.name == BASELINE
            else (_winrate(r["grav"], base["grav"]), _winrate(r["vfov"], base["vfov"]))
        )
        ok = "" if not math.isnan(gm) else "  <- FAILED (all samples)"
        print(f"{s.name:<18}{gm:>11.3f}{vm:>11.3f}{tm:>10.1f}{gw:>10}{vw:>10}{ok}")

    total_time = time.perf_counter() - t_start
    print(f"\nwin = fraction of samples beating '{BASELINE}'. total wall time {total_time:.1f}s.")


if __name__ == "__main__":
    main()
