"""yolo/pose/register.py: 책상 위 x, y, yaw 만 찾는 정합 (register_rest_icp)."""

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from motion.robot_kinematics import rpy_deg_to_rotmat
from yolo.pose import register as R

CAMERA = np.array([150.0, -130.0, 460.0])  # 베이스 mm, 책상 위 위쪽
DESK = np.array([0.0, 0.0, 1.0, 0.0])


def _scene(cad, T, rng, noise_mm=0.7):
    """카메라에서 보이는 CAD 면만, 1.5mm 간격 + 노이즈. 책상 근처 4.5mm 는 뺀다."""
    pts = R._voxel(R.visible_xyz(R.apply_T(cad, T), CAMERA), 1.5)
    pts = pts + rng.normal(0.0, noise_mm, pts.shape)
    return pts[pts[:, 2] > 4.5]


def _adds(cad, Ta, Tb) -> float:
    return float(np.mean(R._nn_dists(R.apply_T(cad, Tb), R.apply_T(cad, Ta))))


class RestRegisterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.cad = R.load_cad_xyz(voxel_mm=1.0)
            cls.dense = R.load_cad_xyz(voxel_mm=0.5)

    def _check(self, class_id: int, yaw: float, xy, *, give_class: bool):
        rng = np.random.default_rng(class_id * 1000 + int(yaw) + 500)
        T = np.eye(4)
        T[:3, :3] = R._rot_z(yaw) @ rpy_deg_to_rotmat(*R.REST_POSES[class_id][0])
        T[:3, 3] = [xy[0], xy[1], 0.0]
        T = R._seat_z(T, self.dense, DESK)
        scene = _scene(self.dense, T, rng)
        out = R.register_rest_icp(
            self.cad,
            scene,
            desk_plane=DESK,
            class_id=class_id if give_class else None,
            camera_origin=CAMERA,
        )
        self.assertTrue(out["aligned_ok"])
        self.assertEqual(out["class_id"], class_id)
        self.assertLess(_adds(self.cad, T, out["T"]), 1.5)

    def test_each_rest_without_class(self):
        for cid, yaw, xy in ((0, 37.0, (140, -120)), (1, -110.0, (180, -90)), (2, 160.0, (130, -160))):
            with self.subTest(class_id=cid):
                self._check(cid, yaw, xy, give_class=False)

    def test_with_class(self):
        self._check(2, 75.0, (160, -140), give_class=True)

    def test_register_pose_rest_method(self):
        rng = np.random.default_rng(3)
        T = np.eye(4)
        T[:3, :3] = R._rot_z(20.0) @ rpy_deg_to_rotmat(*R.REST_POSES[0][0])
        T[:3, 3] = [150.0, -100.0, 0.0]
        T = R._seat_z(T, self.dense, DESK)
        with redirect_stdout(io.StringIO()):
            pose = R.register_pose(
                _scene(self.dense, T, rng),
                voxel_mm=1.0,
                desk_plane=DESK,
                camera_origin=CAMERA,
                method="rest",
            )
        self.assertTrue(pose["aligned_ok"])
        self.assertEqual(pose["class_id"], 0)
        self.assertLess(_adds(self.cad, T, pose["T"]), 1.5)


if __name__ == "__main__":
    unittest.main()
