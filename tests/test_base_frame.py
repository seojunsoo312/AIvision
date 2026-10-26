"""motion/base_frame.py: 사용자 베이스 좌표(+X 앞, +Y 왼쪽, +Z 위) ↔ URDF 좌표."""

import unittest

import numpy as np

from motion.base_frame import (
    LEGACY_BASE_FRAME,
    USER_BASE_FRAME,
    T_user_from_stored,
    ee_target_urdf_from_user,
    rot_urdf_from_user,
    rot_user_from_urdf,
    xyz_urdf_from_user,
    xyz_user_from_urdf,
)


class BaseFrameTest(unittest.TestCase):
    def test_axes_map_by_minus_90_about_z(self):
        # p_urdf = Rz(-90°) @ p_user: 사용자 앞(+X) 은 URDF -Y, 사용자 왼쪽(+Y) 은 URDF +X.
        np.testing.assert_allclose(xyz_urdf_from_user([1, 0, 0]), [0, -1, 0])
        np.testing.assert_allclose(xyz_urdf_from_user([0, 1, 0]), [1, 0, 0])
        np.testing.assert_allclose(xyz_urdf_from_user([0, 0, 1]), [0, 0, 1])

    def test_xyz_round_trip(self):
        p = np.array([0.123, -0.045, 0.067])
        np.testing.assert_allclose(xyz_user_from_urdf(xyz_urdf_from_user(p)), p)

    def test_rotation_round_trip(self):
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(rot_user_from_urdf(rot_urdf_from_user(R)), R)

    def test_stored_in_user_frame_is_unchanged(self):
        T = np.eye(4)
        T[:3, 3] = [0.1, 0.2, 0.3]
        np.testing.assert_allclose(T_user_from_stored(T, USER_BASE_FRAME), T)

    def test_stored_in_legacy_frame_is_rotated(self):
        # 예전 좌표(+X 오른쪽, +Y 앞)의 "앞 0.1 m" 는 지금 좌표의 +X 0.1 m.
        T = np.eye(4)
        T[:3, 3] = [0.0, 0.1, 0.0]
        out = T_user_from_stored(T, LEGACY_BASE_FRAME)
        np.testing.assert_allclose(out[:3, 3], [0.1, 0.0, 0.0], atol=1e-12)
        # base_frame 이 없으면 예전 좌표로 본다.
        np.testing.assert_allclose(T_user_from_stored(T, None), out)

    def test_unknown_stored_frame_raises(self):
        with self.assertRaises(ValueError):
            T_user_from_stored(np.eye(4), "no_such_frame")

    def test_ee_target_converts_mm_to_m(self):
        xyz, R = ee_target_urdf_from_user(np.array([200.0, 0.0, 50.0]), np.zeros(3))
        np.testing.assert_allclose(xyz, [0.0, -0.2, 0.05], atol=1e-12)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
