"""motion/controller.py: Go-to 도중 속도 % 를 바꾸면 남은 구간이 새 속도로 간다."""

import unittest

import numpy as np

from motion.controller import Controller
from motion.robot_kinematics import DEFAULT_URDF, INIT_POSE_JOINTS_DEG, RobotKinematics

_KIN = None


def _ctrl() -> Controller:
    global _KIN
    if _KIN is None:
        _KIN = RobotKinematics(DEFAULT_URDF)
    c = Controller(_KIN)  # 제어 루프 스레드는 켜지 않고 _goto_step 을 직접 부른다
    c.snap_to_joints(INIT_POSE_JOINTS_DEG)
    return c


def _target() -> dict:
    t = dict(INIT_POSE_JOINTS_DEG)
    t["S1"] += 40.0
    return t


def _steps_to_finish(c: Controller, *, change_at: int | None = None, pct: float = 100.0) -> int:
    for k in range(10_000):
        if not c.goto_active():
            return k
        if k == change_at:
            c.set_speed_pct(pct)
        c._goto_step()
    raise AssertionError("이동이 끝나지 않음")


class GotoSpeedTest(unittest.TestCase):
    def test_joint_goto_speeds_up_midway(self):
        c = _ctrl()
        c.set_speed_pct(10)
        c.start_joint_goto(_target())
        slow = _steps_to_finish(c)

        c = _ctrl()
        c.set_speed_pct(10)
        c.start_joint_goto(_target())
        mixed = _steps_to_finish(c, change_at=slow // 4, pct=100)
        # 1/4 은 느리게, 나머지 3/4 은 100% 속도. 끝까지 10% 로 간 것보다 훨씬 빨라야 한다.
        self.assertLess(mixed, slow * 0.5)
        np.testing.assert_allclose(c.snapshot().joints_deg["S1"], _target()["S1"], atol=1e-6)

    def test_slows_down_midway(self):
        c = _ctrl()
        c.set_speed_pct(100)
        c.start_joint_goto(_target())
        fast = _steps_to_finish(c)

        c = _ctrl()
        c.set_speed_pct(100)
        c.start_joint_goto(_target())
        self.assertGreater(_steps_to_finish(c, change_at=1, pct=10), fast * 3)

    def test_given_duration_ignores_slider(self):
        c = _ctrl()
        c.set_speed_pct(10)
        c.start_joint_goto(_target(), duration_s=1.0)
        self.assertEqual(_steps_to_finish(c, change_at=3, pct=100), 30)

    def test_ee_goto_follows_slider(self):
        c = _ctrl()
        p = c.snapshot().pose
        c.set_speed_pct(10)
        c.set_tcp_jog_mm_s(5)
        c.start_ee_goto(np.asarray(p.xyz_mm) + [0.0, 0.0, -30.0], np.asarray(p.rpy_deg))
        n0 = c._goto["n"]
        c.set_tcp_jog_mm_s(45)
        steps = _steps_to_finish(c)
        self.assertLess(steps, n0 * 0.3)


if __name__ == "__main__":
    unittest.main()
