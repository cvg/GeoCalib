"""Test gradient implementations."""

import logging
import unittest

import torch
from torch.func import jacfwd, vmap

from siclib.geometry.camera import camera_models
from siclib.geometry.gravity import Gravity
from siclib.geometry.jacobians import J_up_projection
from siclib.geometry.manifolds import SphericalManifold
from siclib.geometry.perspective_fields import J_perspective_field, get_perspective_field
from siclib.models.optimization.lm_optimizer import LMOptimizer
from siclib.utils.conversions import deg2rad, fov2focal

# flake8: noqa E731
# mypy: ignore-errors

H, W = 320, 320

K1 = -0.1

# CAMERA_MODEL = "pinhole"
CAMERA_MODEL = "simple_radial"
# CAMERA_MODEL = "simple_divisional"

Camera = camera_models[CAMERA_MODEL]

# detect anomaly
torch.autograd.set_detect_anomaly(True)


logger = logging.getLogger("geocalib.models.base_model")
logger.setLevel("ERROR")


def get_toy_rpf(roll=None, pitch=None, vfov=None) -> torch.Tensor:
    """Return a random roll, pitch, focal length if not specified."""
    if roll is None:
        roll = deg2rad((torch.rand(1) - 0.5) * 90)  # -45 ~ 45
    elif not isinstance(roll, torch.Tensor):
        roll = torch.tensor(deg2rad(roll)).unsqueeze(0)

    if pitch is None:
        pitch = deg2rad((torch.rand(1) - 0.5) * 90)  # -45 ~ 45
    elif not isinstance(pitch, torch.Tensor):
        pitch = torch.tensor(deg2rad(pitch)).unsqueeze(0)

    if vfov is None:
        vfov = deg2rad(5 + torch.rand(1) * 75)  # 5 ~ 80
    elif not isinstance(vfov, torch.Tensor):
        vfov = torch.tensor(deg2rad(vfov)).unsqueeze(0)

    return torch.stack([roll, pitch, fov2focal(vfov, H)], dim=-1).float()


class TestJacobianFunctions(unittest.TestCase):
    """Test the jacobian functions."""

    eps = 5e-3

    def validate(self, J: torch.Tensor, J_auto: torch.Tensor):
        """Check if the jacobians are close and finite."""
        self.assertTrue(torch.all(torch.isfinite(J)), "found nan in numerical")
        self.assertTrue(torch.all(torch.isfinite(J_auto)), "found nan in auto")

        text_j = f" > {self.eps}\nJ:\n{J[0, 0].numpy()}\nJ_auto:\n{J_auto[0, 0].numpy()}"
        max_diff = torch.max(torch.abs(J - J_auto))
        text = f"Overall - max diff is {max_diff:.4f}" + text_j
        self.assertTrue(torch.allclose(J, J_auto, atol=self.eps), text)

    def test_spherical_plus(self):
        """Test the spherical plus operator."""
        rpf = get_toy_rpf()
        gravity = Gravity.from_rp(rpf[..., 0], rpf[..., 1])
        J = gravity.J_update(spherical=True)

        # auto jacobian
        delta = gravity.vec3d.new_zeros(gravity.vec3d.shape)[..., :-1]

        def spherical_plus(delta: torch.Tensor) -> torch.Tensor:
            """Plus operator."""
            return SphericalManifold.plus(gravity.vec3d, delta)

        J_auto = vmap(jacfwd(spherical_plus))(delta).squeeze(0)

        self.validate(J, J_auto)

    def test_up_projection_uv(self):
        """Test the up projection jacobians."""
        rpf = get_toy_rpf()

        r, p, f = rpf.unbind(dim=-1)
        camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": [K1]})
        gravity = Gravity.from_rp(r, p)
        uv = camera.normalize(camera.pixel_coordinates())

        J = J_up_projection(uv, gravity.vec3d, "uv")

        # auto jacobian
        def projection_uv(uv: torch.Tensor) -> torch.Tensor:
            """Projection."""
            abc = gravity.vec3d
            projected_up2d = abc[..., None, :2] - abc[..., 2, None, None] * uv
            return projected_up2d[0, 0]

        J_auto = vmap(jacfwd(projection_uv))(uv[0])[None]

        self.validate(J, J_auto)

    def test_up_projection_abc(self):
        """Test the up projection jacobians."""
        rpf = get_toy_rpf()

        r, p, f = rpf.unbind(dim=-1)
        camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": [K1]})
        gravity = Gravity.from_rp(r, p)
        uv = camera.normalize(camera.pixel_coordinates())
        J = J_up_projection(uv, gravity.vec3d, "abc")

        # auto jacobian
        def projection_abc(abc: torch.Tensor) -> torch.Tensor:
            """Projection."""
            return abc[..., None, :2] - abc[..., 2, None, None] * uv

        J_auto = vmap(jacfwd(projection_abc))(gravity.vec3d)[0]

        self.validate(J, J_auto)

    def test_undistort_pts(self):
        """Test the undistortion jacobians."""
        if CAMERA_MODEL == "pinhole":
            return

        rpf = get_toy_rpf()
        _, _, f = rpf.unbind(dim=-1)
        camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": [K1]})
        uv = camera.normalize(camera.pixel_coordinates())
        J = camera.J_undistort(uv, "pts")

        # auto jacobian
        def func_pts(pts):
            return camera.undistort(pts)[0][0]

        J_auto = vmap(jacfwd(func_pts))(uv[0])[None].squeeze(-3)

        self.validate(J, J_auto)

    def test_undistort_k1(self):
        """Test the undistortion jacobians."""
        if CAMERA_MODEL == "pinhole":
            return

        rpf = get_toy_rpf()
        _, _, f = rpf.unbind(dim=-1)
        camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": [K1]})
        uv = camera.normalize(camera.pixel_coordinates())
        J = camera.J_undistort(uv, "dist")

        # auto jacobian
        def func_k1(k1):
            camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": k1})
            return camera.undistort(uv)[0][0]

        J_auto = vmap(jacfwd(func_k1))(camera.dist[..., :1]).squeeze(-1)

        self.validate(J, J_auto)

    def test_up_projection_offset(self):
        """Test the up projection offset jacobians."""
        if CAMERA_MODEL == "pinhole":
            return

        rpf = get_toy_rpf()
        # J = up_projection_offset(rpf)
        _, _, f = rpf.unbind(dim=-1)
        camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": [K1]})
        uv = camera.normalize(camera.pixel_coordinates())
        J = camera.up_projection_offset(uv)

        # auto jacobian
        def projection_uv(uv: torch.Tensor) -> torch.Tensor:
            """Projection."""
            s, _ = camera.distort(uv, return_scale=True)
            return s[0, 0, 0]

        J_auto = vmap(jacfwd(projection_uv))(uv[0])[None].squeeze(-2)

        self.validate(J, J_auto)

    def test_J_up_projection_offset_uv(self):
        """Test the up projection offset jacobians."""
        if CAMERA_MODEL == "pinhole":
            return

        rpf = get_toy_rpf()
        _, _, f = rpf.unbind(dim=-1)
        camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": [K1]})
        uv = camera.normalize(camera.pixel_coordinates())
        J = camera.J_up_projection_offset(uv, "uv")

        # auto jacobian
        def projection_uv(uv: torch.Tensor) -> torch.Tensor:
            """Projection."""
            return camera.up_projection_offset(uv)[0, 0]

        J_auto = vmap(jacfwd(projection_uv))(uv[0])[None]

        # print(J.shape, J_auto.shape)

        self.validate(J, J_auto)


class TestEuclidean(unittest.TestCase):
    """Test the Euclidean manifold jacobians."""

    eps = 5e-3

    def validate(self, J: torch.Tensor, J_auto: torch.Tensor):
        """Check if the jacobians are close and finite."""
        self.assertTrue(torch.all(torch.isfinite(J)), "found nan in numerical")
        self.assertTrue(torch.all(torch.isfinite(J_auto)), "found nan in auto")

        # print(f"analytical:\n{J[0, 0, 0].numpy()}\nauto:\n{J_auto[0, 0, 0].numpy()}")

        text_j = f" > {self.eps}\nJ:\n{J[0, 0, 0].numpy()}\nJ_auto:\n{J_auto[0, 0, 0].numpy()}"

        J_up2grav = J[..., :2, :2]
        J_up2grav_auto = J_auto[..., :2, :2]
        max_diff = torch.max(torch.abs(J_up2grav - J_up2grav_auto))
        text = f"UP - GRAV max diff is {max_diff:.4f}" + text_j
        self.assertTrue(torch.allclose(J_up2grav, J_up2grav_auto, atol=self.eps), text)

        J_up2focal = J[..., :2, 2]
        J_up2focal_auto = J_auto[..., :2, 2]
        max_diff = torch.max(torch.abs(J_up2focal - J_up2focal_auto))
        text = f"UP - FOCAL max diff is {max_diff:.4f}" + text_j
        self.assertTrue(torch.allclose(J_up2focal, J_up2focal_auto, atol=self.eps), text)

        if CAMERA_MODEL != "pinhole":
            J_up2k1 = J[..., :2, 3]
            J_up2k1_auto = J_auto[..., :2, 3]
            max_diff = torch.max(torch.abs(J_up2k1 - J_up2k1_auto))
            text = f"UP - K1 max diff is {max_diff:.4f}" + text_j
            self.assertTrue(torch.allclose(J_up2k1, J_up2k1_auto, atol=self.eps), text)

        J_lat2grav = J[..., 2:, :2]
        J_lat2grav_auto = J_auto[..., 2:, :2]
        max_diff = torch.max(torch.abs(J_lat2grav - J_lat2grav_auto))
        text = f"LAT - GRAV max diff is {max_diff:.4f}" + text_j
        self.assertTrue(torch.allclose(J_lat2grav, J_lat2grav_auto, atol=self.eps), text)

        J_lat2focal = J[..., 2:, 2]
        J_lat2focal_auto = J_auto[..., 2:, 2]
        max_diff = torch.max(torch.abs(J_lat2focal - J_lat2focal_auto))
        text = f"LAT - FOCAL max diff is {max_diff:.4f}" + text_j
        self.assertTrue(torch.allclose(J_lat2focal, J_lat2focal_auto, atol=self.eps), text)

        if CAMERA_MODEL != "pinhole":
            J_lat2k1 = J[..., 2:, 3]
            J_lat2k1_auto = J_auto[..., 2:, 3]
            max_diff = torch.max(torch.abs(J_lat2k1 - J_lat2k1_auto))
            text = f"LAT - K1 max diff is {max_diff:.4f}" + text_j
            self.assertTrue(torch.allclose(J_lat2k1, J_lat2k1_auto, atol=self.eps), text)

        max_diff = torch.max(torch.abs(J - J_auto[..., : J.shape[-1]]))
        text = f"Overall - max diff is {max_diff:.4f}" + text_j
        self.assertTrue(torch.allclose(J, J_auto[..., : J.shape[-1]], atol=self.eps), text)

    def local_pf_calc(self, rpfk: torch.Tensor):
        """Calculate the perspective field."""
        r, p, f, k1 = rpfk.unbind(dim=-1)
        camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": k1})
        gravity = Gravity.from_rp(r, p)
        up, lat = get_perspective_field(camera, gravity)
        persp = torch.cat([up, torch.sin(lat)], dim=-3)
        return persp.permute(0, 2, 3, 1).reshape(1, -1, 3)

    def test_random(self):
        """Random rpf."""
        rpf = get_toy_rpf()
        rpfk = torch.cat([rpf, torch.tensor([[K1]])], dim=-1)
        r, p, f, k1 = rpfk.unbind(dim=-1)
        camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": k1})
        gravity = Gravity.from_rp(r, p)

        J = torch.cat(J_perspective_field(camera, gravity, spherical=False), -2)
        J_auto = jacfwd(self.local_pf_calc)(rpfk).squeeze(-2, -3).reshape(1, H, W, 3, 4)

        self.validate(J, J_auto)

    def test_zero_roll(self):
        """Roll = 0."""
        rpf = get_toy_rpf(roll=0)
        rpfk = torch.cat([rpf, torch.tensor([[K1]])], dim=-1)
        r, p, f, k1 = rpfk.unbind(dim=-1)
        camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": k1})
        gravity = Gravity.from_rp(r, p)

        J = torch.cat(J_perspective_field(camera, gravity, spherical=False), -2)
        J_auto = jacfwd(self.local_pf_calc)(rpfk).squeeze(-2, -3).reshape(1, H, W, 3, 4)

        self.validate(J, J_auto)

    def test_zero_pitch(self):
        """Pitch = 0."""
        rpf = get_toy_rpf(pitch=0)
        rpfk = torch.cat([rpf, torch.tensor([[K1]])], dim=-1)
        r, p, f, k1 = rpfk.unbind(dim=-1)
        camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": k1})
        gravity = Gravity.from_rp(r, p)

        J = torch.cat(J_perspective_field(camera, gravity, spherical=False), -2)
        J_auto = jacfwd(self.local_pf_calc)(rpfk).squeeze(-2, -3).reshape(1, H, W, 3, 4)

        self.validate(J, J_auto)

    def test_max_roll(self):
        """Roll = -45, 45."""
        for roll in [-45, 45]:
            rpf = get_toy_rpf(roll=roll)
            rpfk = torch.cat([rpf, torch.tensor([[K1]])], dim=-1)
            r, p, f, k1 = rpfk.unbind(dim=-1)
            camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": k1})
            gravity = Gravity.from_rp(r, p)

            J = torch.cat(J_perspective_field(camera, gravity, spherical=False), -2)
            J_auto = jacfwd(self.local_pf_calc)(rpfk).squeeze(-2, -3).reshape(1, H, W, 3, 4)

            self.validate(J, J_auto)

    def test_max_pitch(self):
        """Pitch = -45, 45."""
        for pitch in [-45, 45]:
            rpf = get_toy_rpf(pitch=pitch)
            rpfk = torch.cat([rpf, torch.tensor([[K1]])], dim=-1)
            r, p, f, k1 = rpfk.unbind(dim=-1)
            camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": k1})
            gravity = Gravity.from_rp(r, p)

            J = torch.cat(J_perspective_field(camera, gravity, spherical=False), -2)
            J_auto = jacfwd(self.local_pf_calc)(rpfk).squeeze(-2, -3).reshape(1, H, W, 3, 4)

            self.validate(J, J_auto)


class TestSpherical(unittest.TestCase):
    """Test the spherical manifold jacobians."""

    eps = 5e-3

    def validate(self, J: torch.Tensor, J_auto: torch.Tensor):
        """Check if the jacobians are close and finite."""
        self.assertTrue(torch.all(torch.isfinite(J)), "found nan in numerical")
        self.assertTrue(torch.all(torch.isfinite(J_auto)), "found nan in auto")

        text_j = f" > {self.eps}\nJ:\n{J[0, 0, 0].numpy()}\nJ_auto:\n{J_auto[0, 0, 0].numpy()}"

        J_up2grav = J[..., :2, :2]
        J_up2grav_auto = J_auto[..., :2, :2]
        max_diff = torch.max(torch.abs(J_up2grav - J_up2grav_auto))
        text = f"UP - GRAV max diff is {max_diff:.4f}" + text_j

        self.assertTrue(torch.allclose(J_up2grav, J_up2grav_auto, atol=self.eps), text)

        J_up2focal = J[..., :2, 2]
        J_up2focal_auto = J_auto[..., :2, 2]
        max_diff = torch.max(torch.abs(J_up2focal - J_up2focal_auto))
        text = f"UP - FOCAL max diff is {max_diff:.4f}" + text_j

        self.assertTrue(torch.allclose(J_up2focal, J_up2focal_auto, atol=self.eps), text)

        if CAMERA_MODEL != "pinhole":
            J_up2k1 = J[..., :2, 3]
            J_up2k1_auto = J_auto[..., :2, 3]
            max_diff = torch.max(torch.abs(J_up2k1 - J_up2k1_auto))
            text = f"UP - K1 max diff is {max_diff:.4f}" + text_j
            self.assertTrue(torch.allclose(J_up2k1, J_up2k1_auto, atol=self.eps), text)

        J_lat2grav = J[..., 2:, :2]
        J_lat2grav_auto = J_auto[..., 2:, :2]
        max_diff = torch.max(torch.abs(J_lat2grav - J_lat2grav_auto))
        text = f"LAT - GRAV max diff is {max_diff:.4f}" + text_j
        self.assertTrue(torch.allclose(J_lat2grav, J_lat2grav_auto, atol=self.eps), text)

        J_lat2focal = J[..., 2:, 2]
        J_lat2focal_auto = J_auto[..., 2:, 2]
        max_diff = torch.max(torch.abs(J_lat2focal - J_lat2focal_auto))
        text = f"LAT - FOCAL max diff is {max_diff:.4f}" + text_j
        self.assertTrue(torch.allclose(J_lat2focal, J_lat2focal_auto, atol=self.eps), text)

        if CAMERA_MODEL != "pinhole":
            J_lat2k1 = J[..., 2:, 3]
            J_lat2k1_auto = J_auto[..., 2:, 3]
            max_diff = torch.max(torch.abs(J_lat2k1 - J_lat2k1_auto))
            text = f"LAT - K1 max diff is {max_diff:.4f}" + text_j
            self.assertTrue(torch.allclose(J_lat2k1, J_lat2k1_auto, atol=self.eps), text)

        max_diff = torch.max(torch.abs(J - J_auto[..., : J.shape[-1]]))
        text = f"Overall - max diff is {max_diff:.4f}" + text_j
        self.assertTrue(torch.allclose(J, J_auto[..., : J.shape[-1]], atol=self.eps), text)

    def local_pf_calc(self, uvfk: torch.Tensor, gravity: Gravity):
        """Calculate the perspective field."""
        delta, f, k1 = uvfk[..., :2], uvfk[..., 2], uvfk[..., 3]
        cam = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": k1})
        up, lat = get_perspective_field(cam, gravity.update(delta, spherical=True))
        persp = torch.cat([up, torch.sin(lat)], dim=-3)
        return persp.permute(0, 2, 3, 1).reshape(1, -1, 3)

    def test_random(self):
        """Test random rpf."""
        rpf = get_toy_rpf()
        rpfk = torch.cat([rpf, torch.tensor([[K1]])], dim=-1)
        r, p, f, k1 = rpfk.unbind(dim=-1)
        camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": k1})
        gravity = Gravity.from_rp(r, p)

        J = torch.cat(J_perspective_field(camera, gravity, spherical=True), -2)

        uvfk = torch.zeros_like(rpfk)
        uvfk[..., 2] = f
        uvfk[..., 3] = k1
        func = lambda uvfk: self.local_pf_calc(uvfk, gravity)
        J_auto = jacfwd(func)(uvfk).squeeze(-2).reshape(1, H, W, 3, 4)

        self.validate(J, J_auto)

    def test_zero_roll(self):
        """Test roll = 0."""
        rpf = get_toy_rpf(roll=0)
        rpfk = torch.cat([rpf, torch.tensor([[K1]])], dim=-1)
        r, p, f, k1 = rpfk.unbind(dim=-1)
        camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": k1})
        gravity = Gravity.from_rp(r, p)

        J = torch.cat(J_perspective_field(camera, gravity, spherical=True), -2)

        uvfk = torch.zeros_like(rpfk)
        uvfk[..., 2] = f
        uvfk[..., 3] = k1
        func = lambda uvfk: self.local_pf_calc(uvfk, gravity)
        J_auto = jacfwd(func)(uvfk).squeeze(-2).reshape(1, H, W, 3, 4)

        self.validate(J, J_auto)

    def test_zero_pitch(self):
        """Test pitch = 0."""
        rpf = get_toy_rpf(pitch=0)
        rpfk = torch.cat([rpf, torch.tensor([[K1]])], dim=-1)
        r, p, f, k1 = rpfk.unbind(dim=-1)
        camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": k1})
        gravity = Gravity.from_rp(r, p)

        J = torch.cat(J_perspective_field(camera, gravity, spherical=True), -2)

        uvfk = torch.zeros_like(rpfk)
        uvfk[..., 2] = f
        uvfk[..., 3] = k1
        func = lambda uvfk: self.local_pf_calc(uvfk, gravity)
        J_auto = jacfwd(func)(uvfk).squeeze(-2).reshape(1, H, W, 3, 4)

        self.validate(J, J_auto)

    def test_max_roll(self):
        """Test roll = -45, 45."""
        for roll in [-45, 45]:
            rpf = get_toy_rpf(roll=roll)
            rpfk = torch.cat([rpf, torch.tensor([[K1]])], dim=-1)
            r, p, f, k1 = rpfk.unbind(dim=-1)
            camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": k1})
            gravity = Gravity.from_rp(r, p)

            J = torch.cat(J_perspective_field(camera, gravity, spherical=True), -2)

            uvfk = torch.zeros_like(rpfk)
            uvfk[..., 2] = f
            uvfk[..., 3] = k1
            func = lambda uvfk: self.local_pf_calc(uvfk, gravity)
            J_auto = jacfwd(func)(uvfk).squeeze(-2).reshape(1, H, W, 3, 4)

            self.validate(J, J_auto)

    def test_max_pitch(self):
        """Test pitch = -45, 45."""
        for pitch in [-45, 45]:
            rpf = get_toy_rpf(pitch=pitch)
            rpfk = torch.cat([rpf, torch.tensor([[K1]])], dim=-1)
            r, p, f, k1 = rpfk.unbind(dim=-1)
            camera = Camera.from_dict({"height": [H], "width": [W], "f": f, "k1": k1})
            gravity = Gravity.from_rp(r, p)

            J = torch.cat(J_perspective_field(camera, gravity, spherical=True), -2)

            uvfk = torch.zeros_like(rpfk)
            uvfk[..., 2] = f
            uvfk[..., 3] = k1
            func = lambda uvfk: self.local_pf_calc(uvfk, gravity)
            J_auto = jacfwd(func)(uvfk).squeeze(-2).reshape(1, H, W, 3, 4)

            self.validate(J, J_auto)


class TestLM(unittest.TestCase):
    """Test the LM optimizer."""

    eps = 1e-3

    def test_random_spherical(self):
        """Test random rpf."""
        rpf = get_toy_rpf()
        gravity = Gravity.from_rp(rpf[..., 0], rpf[..., 1])
        camera = Camera.from_dict({"height": [H], "width": [W], "f": rpf[..., 2], "k1": [K1]})

        up, lat = get_perspective_field(camera, gravity)

        lm = LMOptimizer({"use_spherical_manifold": True, "camera_model": CAMERA_MODEL})

        out = lm({"up_field": up, "latitude_field": lat})

        cam_opt = out["camera"]
        gravity_opt = out["gravity"]

        if hasattr(cam_opt, "k1"):
            text = f"cam_opt: {cam_opt.k1.numpy()} | rpf: {[K1]}"
            self.assertTrue(
                torch.allclose(cam_opt.k1, torch.tensor([K1]).float(), atol=self.eps), text
            )

        text = f"cam_opt: {cam_opt.f[..., 1].numpy()} | rpf: {rpf[..., 2].numpy()}"
        self.assertTrue(torch.allclose(cam_opt.f[..., 1], rpf[..., 2], atol=self.eps), text)

        text = f"gravity_opt.roll: {gravity_opt.roll.numpy()} | rpf: {rpf[..., 0].numpy()}"
        self.assertTrue(torch.allclose(gravity_opt.roll, rpf[..., 0], atol=self.eps), text)

        text = f"gravity_opt.pitch: {gravity_opt.pitch.numpy()} | rpf: {rpf[..., 1].numpy()}"
        self.assertTrue(torch.allclose(gravity_opt.pitch, rpf[..., 1], atol=self.eps), text)

    def test_random(self):
        """Test random rpf."""
        rpf = get_toy_rpf()
        gravity = Gravity.from_rp(rpf[..., 0], rpf[..., 1])
        camera = Camera.from_dict({"height": [H], "width": [W], "f": rpf[..., 2], "k1": [K1]})

        up, lat = get_perspective_field(camera, gravity)

        lm = LMOptimizer({"use_spherical_manifold": False, "camera_model": CAMERA_MODEL})
        out = lm({"up_field": up, "latitude_field": lat})

        cam_opt = out["camera"]
        gravity_opt = out["gravity"]

        if hasattr(cam_opt, "k1"):
            text = f"cam_opt: {cam_opt.k1.numpy()} | rpf: {[K1]}"
            self.assertTrue(
                torch.allclose(cam_opt.k1, torch.tensor([K1]).float(), atol=self.eps), text
            )

        text = f"cam_opt: {cam_opt.f[..., 1].numpy()} | rpf: {rpf[..., 2].numpy()}"
        self.assertTrue(torch.allclose(cam_opt.f[..., 1], rpf[..., 2], atol=self.eps), text)

        text = f"gravity_opt.roll: {gravity_opt.roll.numpy()} | rpf: {rpf[..., 0].numpy()}"
        self.assertTrue(torch.allclose(gravity_opt.roll, rpf[..., 0], atol=self.eps), text)

        text = f"gravity_opt.pitch: {gravity_opt.pitch.numpy()} | rpf: {rpf[..., 1].numpy()}"
        self.assertTrue(torch.allclose(gravity_opt.pitch, rpf[..., 1], atol=self.eps), text)


if __name__ == "__main__":
    unittest.main()
