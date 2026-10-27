"""vision/handeye/zero_fix.py: 손눈 사진으로 관절 0° 오차 찾기 (정답을 아는 가짜 사진으로)."""

import unittest

import numpy as np

from motion.base_frame import tcp_pose_user_from_urdf
from motion.robot_kinematics import DEFAULT_URDF, RobotKinematics, rpy_deg_to_rotmat
from vision.handeye.compute import Sample, invert_T, mean_rigid, solve_eye_to_hand
from vision.handeye.zero_fix import FIT_JOINTS, estimate


def _fk(kin, joints):
    p = tcp_pose_user_from_urdf(kin.forward_tcp(kin.q_from_deg(joints)))
    T = np.eye(4)
    T[:3, :3] = p.rotation
    T[:3, 3] = p.xyz_mm
    return T


def _pose(rpy, xyz):
    T = np.eye(4)
    T[:3, :3] = rpy_deg_to_rotmat(*rpy)
    T[:3, 3] = xyz
    return T


class ZeroFixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = RobotKinematics(DEFAULT_URDF)
        rng = np.random.default_rng(0)
        cls.joint_sets = []
        for _ in range(16):
            cls.joint_sets.append({
                "S1": rng.uniform(-45, 45), "S2": rng.uniform(-70, -5), "S3": rng.uniform(20, 80),
                "S4": rng.uniform(-20, 20), "S5": rng.uniform(-60, 60), "S6": rng.uniform(-40, 40), "S7": -30.0,
            })
        # 책상 위 카메라 (아래를 봄), 손끝에 붙은 보드.
        cls.T_bc = _pose((180.0, 2.0, 90.0), (160.0, -10.0, 440.0))
        cls.T_tb = _pose((5.0, -3.0, 20.0), (10.0, -30.0, 25.0))

    def _samples(self, true_dq):
        out = []
        for i, j in enumerate(self.joint_sets):
            real = dict(j)
            for name, d in true_dq.items():
                real[name] += d
            T_cb = invert_T(self.T_bc) @ _fk(self.kin, real) @ self.T_tb  # 카메라가 본 보드 (진짜 자세)
            out.append(Sample(f"s{i}.png", None, T_cb, _fk(self.kin, j), 0.1, {"joints_deg": j}))  # 팔은 틀린 각도를 보고
        return out

    def _estimate(self, samples):
        T_bc = solve_eye_to_hand([s.T_base_tcp for s in samples], [s.T_cam_board for s in samples])
        T_tb = mean_rigid([invert_T(s.T_base_tcp) @ T_bc @ s.T_cam_board for s in samples])
        return estimate(samples, T_bc, T_tb, self.kin)

    def test_finds_the_planted_offsets(self):
        true_dq = {"S2": -4.0, "S3": 3.0, "S4": 1.0, "S5": -2.5}
        res = self._estimate(self._samples(true_dq))
        self.assertTrue(res["ok"], res["reason"])
        for name in FIT_JOINTS:
            self.assertAlmostEqual(res["dq_deg"][name], true_dq[name], delta=0.05, msg=name)
        self.assertLess(res["after_mm"][0], 0.05)
        self.assertGreater(res["before_mm"][0], 2.0)

    def test_nothing_to_fix_when_calibration_is_right(self):
        res = self._estimate(self._samples({}))
        self.assertFalse(res["ok"])
        self.assertTrue(res.get("nothing"))


if __name__ == "__main__":
    unittest.main()
