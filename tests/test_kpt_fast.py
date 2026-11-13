"""yolo/train/kpt_label.py: 자세 여러 개를 한 번에 투영 (project_many) 이 하나씩 (project_base) 과 같은지."""

import unittest

import numpy as np

from yolo.train import kpt_label as KL


class ProjectManyTest(unittest.TestCase):
    def setUp(self):
        T_base_cam = np.eye(4)
        T_base_cam[:3, :3] = np.diag([1.0, -1.0, -1.0])
        T_base_cam[:3, 3] = [150.0, -20.0, 450.0]
        K = np.array([[454.3, 0.0, 315.1], [0.0, 454.3, 235.5], [0.0, 0.0, 1.0]])
        self.cam = KL.Camera(K, np.array([0.0476, -0.0650, 0.001, -0.0008, 0.01]), T_base_cam, "test")
        self.pts = np.random.default_rng(0).uniform(-20, 40, (10, 3))

    def test_matches_project_base(self):
        rng = np.random.default_rng(1)
        A = np.eye(4)
        poses = KL._desk_poses(A, rng.uniform(0, 30, 20), rng.uniform(-50, 250, 20), rng.uniform(-200, 200, 20),
                               rng.uniform(-np.pi, np.pi, 20))
        many = KL.project_many(poses, self.pts, self.cam)
        one = np.stack([KL.project_base(T, self.pts, self.cam) for T in poses])
        self.assertLess(float(np.abs(many - one).max()), 1e-6)

    def test_desk_poses_matches_desk_pose(self):
        A = np.eye(4)
        A[:3, :3] = KL._Rz(0.7)[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        got = KL._desk_poses(A, np.array([12.0]), np.array([100.0]), np.array([-30.0]), np.array([1.1]))[0]
        self.assertTrue(np.allclose(got, KL.desk_pose(A, 12.0, 100.0, -30.0, 1.1)))


if __name__ == "__main__":
    unittest.main()
