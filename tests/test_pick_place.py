"""motion/pick_place.py: 픽앤플레이스 단계 만들기, 실패 시 내려놓기."""

import unittest
from types import SimpleNamespace

import numpy as np

from motion.pick_place import (
    GRIPPER_CLOSE_100,
    GRIPPER_OPEN_100,
    PICK_LIFT_MM,
    PLACE_APPROACH_MM,
    SAFE_LIFT_MM,
    SAFE_LOWER_MPS,
    PickPlaceRunner,
    _grasp_z,
    build_pick_place_steps,
    build_safe_putdown_steps,
    rpy_tcp_rx_parallel_base_x,
)
from motion.robot_kinematics import rpy_deg_to_rotmat

P = np.array([200.0, 10.0, 60.0])
G = np.array([200.0, 10.0, 30.0])
G_RPY = np.array([90.0, 0.0, 30.0])
DROP = np.array([150.0, -100.0])


def _steps(**kw):
    return build_pick_place_steps(
        p_xyz_mm=P, p_rpy_deg=G_RPY, g_xyz_mm=G, g_rpy_deg=G_RPY, drop_xy_mm=DROP, **kw
    )


class BuildStepsTest(unittest.TestCase):
    def test_ten_steps_in_order(self):
        names = [s.name for s in _steps()]
        self.assertEqual(
            names,
            ["P", "그리퍼 열기", "G", "그리퍼 닫기", "집기 리프트", "드롭 XY", "하강", "그리퍼 열기", "놓기 리프트", "초기자세"],
        )

    def test_heights_and_targets(self):
        s = {st.name: st for st in _steps()}
        z_g = G[2]
        np.testing.assert_allclose(s["G"].xyz_mm, G)
        np.testing.assert_allclose(s["집기 리프트"].xyz_mm, [G[0], G[1], z_g + PICK_LIFT_MM])
        # 놓을 곳 위로는 관절 이동, 높이는 집은 높이 + 접근 높이.
        self.assertEqual(s["드롭 XY"].kind, "ee_joints")
        np.testing.assert_allclose(s["드롭 XY"].xyz_mm, [DROP[0], DROP[1], z_g + PLACE_APPROACH_MM])
        # 놓는 높이 = 집은 높이.
        np.testing.assert_allclose(s["하강"].xyz_mm, [DROP[0], DROP[1], z_g])
        np.testing.assert_allclose(s["놓기 리프트"].xyz_mm, [DROP[0], DROP[1], z_g + PLACE_APPROACH_MM])
        self.assertEqual(s["초기자세"].kind, "joints")

    def test_gripper_and_holding_flags(self):
        grips = [st for st in _steps() if st.kind == "gripper"]
        self.assertEqual([g.grip_100 for g in grips], [GRIPPER_OPEN_100, GRIPPER_CLOSE_100, GRIPPER_OPEN_100])
        # 닫을 때 "쥠", 놓을 때 "놓음". 처음 여는 단계는 바꾸지 않는다.
        self.assertEqual([g.hold for g in grips], [None, True, False])

    def test_extra_step_when_lift_is_higher_than_approach(self):
        names = [s.name for s in _steps(pick_lift_mm=80.0, place_approach_mm=50.0)]
        self.assertIn("드롭 위", names)
        self.assertEqual(len(names), 11)

    def test_drop_wrist_has_tcp_x_along_base_x(self):
        for st in _steps():
            if st.name in ("드롭 XY", "하강", "놓기 리프트"):
                R = rpy_deg_to_rotmat(*(float(v) for v in st.rpy_deg))
                self.assertAlmostEqual(abs(float(R[0, 0])), 1.0, places=9)

    def test_parallel_rpy_is_closest_choice(self):
        # 이미 TCP X ∥ 베이스 X 이면 그대로 둔다.
        rpy = np.array([90.0, 0.0, 0.0])
        R0 = rpy_deg_to_rotmat(*rpy)
        R1 = rpy_deg_to_rotmat(*(float(v) for v in rpy_tcp_rx_parallel_base_x(rpy)))
        np.testing.assert_allclose(R1, R0, atol=1e-9)

    def test_grasp_z_is_height_before_closing(self):
        self.assertAlmostEqual(_grasp_z(_steps()), G[2])


class SafePutdownTest(unittest.TestCase):
    def test_lowers_slowly_then_opens_then_lifts(self):
        steps = build_safe_putdown_steps(np.array([180.0, 0.0, 90.0]), G_RPY, place_z_mm=30.0)
        self.assertEqual([s.kind for s in steps], ["ee", "gripper", "ee"])
        self.assertEqual(steps[0].lin_mps, SAFE_LOWER_MPS)
        self.assertAlmostEqual(steps[0].xyz_mm[2], 30.0)
        self.assertIs(steps[1].hold, False)
        self.assertAlmostEqual(steps[2].xyz_mm[2], 30.0 + SAFE_LIFT_MM)

    def test_no_lowering_when_already_at_place_height(self):
        steps = build_safe_putdown_steps(np.array([180.0, 0.0, 30.5]), G_RPY, place_z_mm=30.0)
        self.assertEqual([s.kind for s in steps], ["gripper", "ee"])


class FakeController:
    """PickPlaceRunner 가 부르는 Controller 메서드만 흉내 낸다. 이동은 바로 끝난다."""

    def __init__(self):
        self._holding = False
        self.fault = ""
        self.launched = []
        self.stopped = False

    def holding(self):
        return self._holding

    def set_holding(self, on):
        self._holding = bool(on)

    def interrupt_id(self):
        return 0

    def goto_active(self):
        return False

    def snapshot(self):
        pose = SimpleNamespace(xyz_mm=np.array([180.0, 0.0, 90.0]), rpy_deg=G_RPY.copy())
        return SimpleNamespace(fault=self.fault, pose=pose, joints_deg={"S7": 0.0})

    def _go(self, kind, *a, **k):
        # 진짜 Controller 처럼 새 이동을 시작하면 실패 표시를 지운다.
        self.launched.append(kind)
        self.fault = ""
        return 0.5

    def start_ee_goto(self, *a, **k):
        return self._go("ee", *a, **k)

    def start_ee_goto_joints(self, *a, **k):
        return self._go("ee_joints", *a, **k)

    def start_joint_goto(self, *a, **k):
        return self._go("joints", *a, **k)

    def stop_all(self):
        self.stopped = True


def _run(runner, ctrl, fault_at=None, fault="", max_ticks=200):
    """단계 사이 대기를 없애고, fault_at 번째 단계에서 실패를 낸다."""
    import motion.pick_place as pp

    old = pp.STEP_PAUSE_S
    pp.STEP_PAUSE_S = 0.0
    try:
        for _ in range(max_ticks):
            if fault_at is not None and runner._i == fault_at and not runner.recovering:
                ctrl.fault = fault
            phase = runner.tick()
            ctrl.fault = ""
            if phase in ("done", "fault") and not runner.busy():
                return phase
    finally:
        pp.STEP_PAUSE_S = old
    return runner.phase


class RunnerTest(unittest.TestCase):
    def test_runs_to_done_and_releases(self):
        ctrl = FakeController()
        r = PickPlaceRunner(ctrl)
        self.assertTrue(r.start(_steps()))
        self.assertEqual(_run(r, ctrl), "done")
        self.assertFalse(ctrl.holding())
        self.assertEqual(len(ctrl.launched), 10)

    def test_refuses_to_start_while_holding(self):
        ctrl = FakeController()
        ctrl.set_holding(True)
        r = PickPlaceRunner(ctrl)
        self.assertFalse(r.start(_steps()))
        self.assertEqual(r.phase, "fault")
        self.assertEqual(ctrl.launched, [])

    def test_failure_while_holding_puts_object_down(self):
        ctrl = FakeController()
        r = PickPlaceRunner(ctrl)
        r.start(_steps())
        # 5번째 단계(집기 리프트, 이미 쥔 상태)에서 IK 실패 → 그 자리 아래로 내려놓고 fault 로 끝.
        phase = _run(r, ctrl, fault_at=4, fault="IK 실패")
        self.assertEqual(phase, "fault")
        self.assertIn("내려놓고 멈춤", r.status)
        self.assertFalse(ctrl.holding())

    def test_comm_failure_while_holding_keeps_object(self):
        ctrl = FakeController()
        r = PickPlaceRunner(ctrl)
        r.start(_steps())
        phase = _run(r, ctrl, fault_at=4, fault="통신 오류 — 읽기 실패")
        self.assertEqual(phase, "fault")
        self.assertIn("쥔 채 멈춤", r.status)
        self.assertTrue(ctrl.holding())


if __name__ == "__main__":
    unittest.main()
