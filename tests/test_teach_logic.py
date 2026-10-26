"""pendant/teach_logic.py: 사물 위치 패널의 계산 (화면 없이)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pendant"))

from cad.model import approach_mm, bounds_mm, load_mesh_m  # noqa: E402
from motion.grasp import PlacePose, rpy_body_xyz_to_extrinsic  # noqa: E402
from motion.robot_kinematics import DEFAULT_URDF, RobotKinematics, rpy_deg_to_rotmat  # noqa: E402
from teach_logic import PlaceRpy, auto_label, auto_pre, pick_place_steps, random_place  # noqa: E402


def _R(rpy):
    return rpy_deg_to_rotmat(*(float(v) for v in rpy))


class PlaceRpyTest(unittest.TestCase):
    def test_body_and_base_show_the_same_rotation(self):
        st = PlaceRpy.from_place(rpy_body_xyz_to_extrinsic(90, 30, 0))
        self.assertEqual(st.frame, "body")
        body = st.shown()
        np.testing.assert_allclose([body["roll"], body["pitch"], body["yaw"]], [90, 30, 0], atol=1e-6)
        self.assertTrue(st.set_frame("base"))
        self.assertFalse(st.set_frame("base"))
        base = st.refresh()
        np.testing.assert_allclose([base["roll"], base["pitch"], base["yaw"]], [90, 0, 30], atol=1e-6)

    def test_typing_an_axis_sets_that_axis(self):
        st = PlaceRpy.from_place((0.0, 0.0, 0.0))
        shown = st.set_axis("yaw", 45.0)
        self.assertAlmostEqual(shown["yaw"], 45.0, places=6)
        np.testing.assert_allclose(_R(st.base_rpy), _R((0, 0, 45)), atol=1e-9)

    def test_dragging_rotates_by_the_difference_and_keeps_the_dragged_slider(self):
        st = PlaceRpy.from_place((0.0, 0.0, 30.0))
        st.nudge_axis("yaw", 50.0)  # +20°
        np.testing.assert_allclose(_R(st.base_rpy), _R((0, 0, 50)), atol=1e-9)
        self.assertEqual(st.slider["yaw"], 50.0)
        # 179 → -179 는 2° 만 돈다 (한 바퀴 반대로 돌지 않는다).
        st = PlaceRpy.from_place((0.0, 0.0, 179.0))
        st.nudge_axis("yaw", -179.0)
        np.testing.assert_allclose(_R(st.base_rpy), _R((0, 0, 181)), atol=1e-9)


class PlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mins, cls.maxs = bounds_mm()
        cls.d, cls.a = approach_mm()
        cls.kin = RobotKinematics(DEFAULT_URDF)

    def test_auto_pre_and_steps(self):
        auto = auto_pre(PlacePose((200.0, 0.0, 2.0), (0, 0, 30)), self.mins, self.maxs, self.d, self.a, kin=self.kin, ctrl=None)
        self.assertEqual(auto_label(auto, "pre"), "대기 위치 (Z pitch+)")
        self.assertEqual(auto_label(auto, "grasp"), f"집기 위치 (Z a={self.a:.0f} pitch+)")
        steps = pick_place_steps(auto, (150.0, -100.0))
        self.assertEqual(len(steps), 10)
        lower = next(s for s in steps if s.name == "하강")
        np.testing.assert_allclose(lower.xyz_mm[:2], [150.0, -100.0])

    def test_random_place_checks_collision(self):
        verts, _ = load_mesh_m()
        res = random_place(verts, "stand", np.random.default_rng(1), kin=self.kin, ctrl=None)
        self.assertIsNotNone(res.place)
        self.assertEqual(res.error, "")
        self.assertIsInstance(res.collided, bool)


if __name__ == "__main__":
    unittest.main()
