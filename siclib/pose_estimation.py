import pickle
from pathlib import Path

import numpy as np
import poselib
import pycolmap

from siclib.models.extractor import VP

from .models.extractor import GeoCalib
from .utils.image import load_image

# flake8: noqa
# mypy: ignore-errors


class AbsolutePoseEstimator:
    default_opts = {
        "ransac": "poselib_gravity",  # pycolmap, poselib, poselib_gravity
        "refinement": "pycolmap_gravity",  # pycolmap, pycolmap_gravity, none
        "gravity_weight": 50000,
        "max_reproj_error": 48.0,
        "loss_function_scale": 1.0,
        "use_vp": False,
        "max_uncertainty": 10.0 / 180.0 * 3.1415,  # radians
        "cache_path": "../../outputs/inloc/calib.pickle",
    }

    def __init__(self, pose_opts=None):
        pose_opts = {} if pose_opts is None else pose_opts
        self.opts = {**self.default_opts, **pose_opts}
        self.device = "cuda"

        if self.opts["use_vp"]:
            self.calib = VP().to(self.device)
            self.cache_path = str(self.opts["cache_path"]).replace(".pickle", "_vp.pickle")
        else:
            self.calib = GeoCalib().to(self.device)
            self.cache_path = str(self.opts["cache_path"])

        # self.read_cache()
        self.cache = {}

    def read_cache(self):
        print(f"Reading cache from {self.cache_path} ({Path(self.cache_path).exists()})")
        if not Path(self.cache_path).exists():
            self.cache = {}
            return
        with open(self.cache_path, "rb") as handle:
            self.cache = pickle.load(handle)

    def write_cache(self):
        with open(self.cache_path, "wb") as handle:
            pickle.dump(self.cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __call__(self, query_path, p2d, p3d, camera_dict):
        focal_length = pycolmap.Camera(camera_dict).mean_focal_length()

        if query_path in self.cache:
            calib = self.cache[query_path]
        else:
            calib = self.calib.calibrate(
                load_image(query_path).to(self.device), priors={"f": focal_length}
            )
            calib = {k: v[0].detach().cpu().numpy() for k, v in calib.items()}
            self.cache[query_path] = calib
            # self.write_cache()

        if self.opts["ransac"] == "pycolmap":
            ret = pycolmap.absolute_pose_estimation(
                p2d, p3d, camera_dict, self.opts["max_reproj_error"]  # , do_refine=False
            )
        elif self.opts["ransac"] == "poselib":
            M, ret = poselib.estimate_absolute_pose(
                p2d,
                p3d,
                camera_dict,
                ransac_opt={"max_reproj_error": self.opts["max_reproj_error"]},
            )
            ret["success"] = M is not None
            ret["qvec"] = M.q
            ret["tvec"] = M.t
        elif self.opts["ransac"] == "poselib_gravity":
            g_q = calib["gravity"].vec3d
            g_qu = calib.get("gravity_uncertainty", self.opts["max_uncertainty"])
            M, ret = poselib.estimate_absolute_pose_gravity(
                p2d,
                p3d,
                camera_dict,
                g_q,
                g_qu * 2 * 180 / 3.1415,  # convert to scalar
                ransac_opt={"max_reproj_error": self.opts["max_reproj_error"]},
            )
            ret["success"] = M is not None
            ret["qvec"] = M.q
            ret["tvec"] = M.t
        else:
            raise NotImplementedError(self.opts["ransac"])
        r_opts = {
            "refine_focal_length": False,
            "refine_extra_params": False,
            "print_summary": False,
            "loss_function_scale": self.opts["loss_function_scale"],
        }
        if self.opts["refinement"] == "pycolmap_gravity":
            g_q = calib["gravity"].vec3d
            g_qu = calib.get("gravity_uncertainty", self.opts["max_uncertainty"])
            if g_qu <= self.opts["max_uncertainty"]:
                g_gt = np.array([0, 0, 1])  # world frame
                ret_ref = pycolmap.pose_refinement_gravity(
                    ret["tvec"],
                    ret["qvec"],
                    p2d,
                    p3d,
                    ret["inliers"],
                    camera_dict,
                    g_q,
                    g_gt,
                    self.opts["gravity_weight"],
                    r_opts,
                )
            else:
                ret_ref = pycolmap.pose_refinement(
                    ret["tvec"],
                    ret["qvec"],
                    p2d,
                    p3d,
                    ret["inliers"],
                    camera_dict,
                    r_opts,
                )
        elif self.opts["refinement"] == "pycolmap":
            ret_ref = pycolmap.pose_refinement(
                ret["tvec"],
                ret["qvec"],
                p2d,
                p3d,
                ret["inliers"],
                camera_dict,
                r_opts,
            )
        elif self.opts["refinement"] == "none":
            ret_ref = {}
        else:
            raise NotImplementedError(self.opts["refinement"])
        ret = {**ret, **ret_ref}
        ret["camera_dict"] = camera_dict
        return ret, calib
