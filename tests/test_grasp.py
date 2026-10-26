"""motion/grasp.py: 물체 자세 → 대기(P)·집기(G) 좌표, 회전 표기 변환, 랜덤 배치."""

import unittest

import numpy as np

from cad.model import approach_mm, bounds_mm, load_mesh_m
from motion.grasp import (
    LIE_Z_MM,
    RANDOM_PLACE_MODES,
    SLANT_Z_MM,
    STAND_Z_MM,
    PlacePose,
    T_to_xyzrpy,
    _mesh_above_floor,
    _place_xy_in_region,
    _rpy_slant_about_s,
    canonicalize_extrinsic_rpy,
    compute_auto_pre,
    pose_to_T,
    rpy_body_xyz_to_extrinsic,
    rpy_extrinsic_to_body_xyz,
)
from motion.robot_kinematics import DEFAULT_URDF, RobotKinematics, rpy_deg_to_rotmat


def _R(rpy):
    return rpy_deg_to_rotmat(float(rpy[0]), float(rpy[1]), float(rpy[2]))


class RotationNotationTest(unittest.TestCase):
    def test_pose_T_round_trip(self):
        xyz = np.array([180.0, -40.0, 25.0])
        rpy = np.array([10.0, -20.0, 30.0])
        xyz2, rpy2 = T_to_xyzrpy(pose_to_T(xyz, rpy))
        np.testing.assert_allclose(xyz2, xyz, atol=1e-9)
        np.testing.assert_allclose(_R(rpy2), _R(rpy), atol=1e-9)

    def test_lie_examples_from_docstring(self):
        # grasp.py 머리 주석의 예: 물체 축 (±90, 30, 0) → 베이스 (±90, 0, ±30).
        np.testing.assert_allclose(rpy_body_xyz_to_extrinsic(90, 30, 0), (90, 0, 30), atol=1e-9)
        np.testing.assert_allclose(rpy_body_xyz_to_extrinsic(-90, 30, 0), (-90, 0, -30), atol=1e-9)

    def test_body_extrinsic_round_trip_is_same_rotation(self):
        for rpy in [(10, 20, 30), (90, 0, 45), (-120, 60, 170), (0, 89.9, 0)]:
            back = rpy_body_xyz_to_extrinsic(*rpy_extrinsic_to_body_xyz(rpy))
            np.testing.assert_allclose(_R(back), _R(rpy), atol=1e-6)

    def test_canonicalize_keeps_rotation(self):
        for rpy in [(0, 90, 30), (180, 45, 180), (-90, 0, -30)]:
            np.testing.assert_allclose(_R(canonicalize_extrinsic_rpy(rpy)), _R(rpy), atol=1e-6)

    def test_slant_base_rpy(self):
        # 비스듬히: 베이스 Rx=180, Ry=45 고정, Rz=wrap(180+φ).
        for phi, rz in [(0, 180), (90, -90)]:
            np.testing.assert_allclose(_R(_rpy_slant_about_s(phi)), _R((180, 45, rz)), atol=1e-9)


class AutoPreTest(unittest.TestCase):
    """대기(P)는 물체 위, 집기(G)는 P 에서 a mm 바로 아래, 두 자세는 같은 손목 방향."""

    @classmethod
    def setUpClass(cls):
        cls.mins, cls.maxs = bounds_mm()
        cls.d, cls.a = approach_mm()
        cls.kin = RobotKinematics(DEFAULT_URDF)
        cls.q = cls.kin.q_home()

    def _check(self, place, axis):
        r = compute_auto_pre(place, self.mins, self.maxs, self.d, self.a, kin=self.kin, q=self.q)
        p, _ = T_to_xyzrpy(r.T_base)
        g, _ = T_to_xyzrpy(r.T_grasp_base)
        self.assertEqual(r.axis_name, axis)
        np.testing.assert_allclose(r.T_base[:3, :3], r.T_grasp_base[:3, :3], atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(p - g)), self.a, places=6)
        self.assertGreater(p[2], g[2])
        # 바로 위에서 내려간다 (x, y 같음).
        np.testing.assert_allclose(p[:2], g[:2], atol=1e-6)
        return p, g

    def test_standing(self):
        p, g = self._check(PlacePose((200.0, 0.0, STAND_Z_MM), (0.0, 0.0, 30.0)), "Z")
        # P 는 윗면(CAD max Z + 놓인 높이) 에서 d 위.
        self.assertAlmostEqual(p[2], STAND_Z_MM + self.maxs[2] + self.d, places=6)
        self.assertAlmostEqual(g[2], p[2] - self.a, places=6)

    def test_lying(self):
        self._check(PlacePose((200.0, 50.0, LIE_Z_MM), rpy_body_xyz_to_extrinsic(90, 30, 0)), "Y")

    def test_slanted(self):
        self._check(PlacePose((180.0, -40.0, SLANT_Z_MM), _rpy_slant_about_s(20)), "N")

    def test_without_robot_uses_home(self):
        r = compute_auto_pre(PlacePose((200.0, 0.0, STAND_Z_MM), (0, 0, 0)), self.mins, self.maxs, self.d, self.a)
        self.assertEqual(r.axis_name, "Z")


class RandomPlaceTest(unittest.TestCase):
    def test_every_mode_lands_in_region_above_floor(self):
        from motion.grasp import sample_random_place

        verts, _ = load_mesh_m()
        rng = np.random.default_rng(0)
        for mode in RANDOM_PLACE_MODES:
            for _ in range(5):
                place = sample_random_place(verts, mode=mode, rng=rng)
                self.assertIsNotNone(place, mode)
                self.assertTrue(_mesh_above_floor(verts, place), mode)
                if mode != "free":
                    self.assertTrue(_place_xy_in_region(place.xyz_mm[0], place.xyz_mm[1]), mode)


if __name__ == "__main__":
    unittest.main()
