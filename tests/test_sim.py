"""vision/sim + yolo/pose/sim_eval: 노이즈, 대칭, 정답 비교. MuJoCo 없이 돈다."""

import unittest

import numpy as np

from vision.sim.noise import NO_NOISE, NoiseConfig, add_noise
from vision.sim.symmetry import cad_symmetries
from yolo.pose.sim_eval import pose_error, seg_box

K = np.array([[454.0, 0, 319.5], [0, 454.0, 239.5], [0, 0, 1]])


def _box_scene() -> np.ndarray:
    """책상 452 mm + 가운데 40 mm 높이 상자."""
    depth = np.full((480, 640), 452.0, np.float32)
    depth[200:280, 280:360] = 412.0
    return depth


class NoiseTest(unittest.TestCase):
    def test_off_is_copy(self):
        d = _box_scene()
        out = add_noise(d, K, np.random.default_rng(0), NO_NOISE)
        np.testing.assert_array_equal(out, d)
        self.assertIsNot(out, d)

    def test_same_seed_same_noise(self):
        d = _box_scene()
        a = add_noise(d, K, np.random.default_rng(3))
        b = add_noise(d, K, np.random.default_rng(3))
        np.testing.assert_array_equal(a, b)

    def test_flying_only_near_edges(self):
        d = _box_scene()
        cfg = NoiseConfig(sigma_mm=0.0, edge_hole_p=0.0, grazing_hole_p=0.0)
        out = add_noise(d, K, np.random.default_rng(0), cfg)
        between = (out > 413.0) & (out < 451.0)
        self.assertGreater(int(between.sum()), 50)
        ys, xs = np.nonzero(between)
        # 상자 테두리에서 2 px 안쪽에만 생긴다.
        near_x = (np.abs(xs - 280) <= 2) | (np.abs(xs - 359) <= 2)
        near_y = (np.abs(ys - 200) <= 2) | (np.abs(ys - 279) <= 2)
        self.assertTrue(np.all(near_x | near_y))


class PoseErrorTest(unittest.TestCase):
    def test_symmetry_counts_as_same(self):
        syms = cad_symmetries()
        self.assertGreaterEqual(len(syms), 2)
        T_gt = np.eye(4)
        T_gt[:3, 3] = [200.0, 10.0, 0.0]
        center = np.array([12.7, 0.0, 12.7])
        for S in syms:
            pos, rot = pose_error(T_gt @ S, T_gt, center)
            self.assertLess(pos, 1e-6)
            self.assertLess(rot, 1e-3)

    def test_90_deg_flip_is_error(self):
        T_gt = np.eye(4)
        T_est = np.eye(4)
        T_est[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        _pos, rot = pose_error(T_est, T_gt, np.array([12.7, 0.0, 12.7]))
        self.assertGreater(rot, 60.0)

    def test_seg_box(self):
        seg = np.zeros((10, 10), np.uint8)
        seg[2:5, 3:7] = 1
        self.assertEqual(seg_box(seg, 1), [3.0, 2.0, 7.0, 5.0])
        self.assertIsNone(seg_box(seg, 2))


class KptLabelTest(unittest.TestCase):
    """키포인트 라벨 도구: 사람이 대칭 반대쪽 번호로 찍어도 저장하면 시뮬 라벨 규칙과 같아진다."""

    def test_flipped_numbering_is_fixed_on_save(self):
        import tempfile
        from pathlib import Path

        from vision.sim.keypoints import KEYPOINTS_CAD_MM, canonical_T, project
        from yolo.train import kpt_label as KL

        K = np.array([[454.2, 0.0, 319.5], [0.0, 454.2, 239.5], [0.0, 0.0, 1.0]])
        T_base_cam = np.array([[-0.0745, -0.9965, -0.0388, 149.4], [-0.9972, 0.0741, 0.0106, -21.3],
                               [-0.0077, 0.0394, -0.9992, 452.3], [0.0, 0.0, 0.0, 1.0]])
        T_base_cam[:3, :3] = np.linalg.svd(T_base_cam[:3, :3])[0] @ np.linalg.svd(T_base_cam[:3, :3])[2]
        cam = KL.Camera(K, np.zeros(5), T_base_cam, "test")
        syms = cad_symmetries()
        S = [m for m in syms if not np.allclose(m, np.eye(4))][0]
        perm = KL.symmetry_perm(S)
        for yaw in (0.0, 50.0, 130.0, 250.0):
            c, s_ = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
            T = np.eye(4)
            T[:3, :3] = [[c, -s_, 0], [s_, c, 0], [0, 0, 1]]
            T[:3, 3] = [170.0, 20.0, 2.0]
            want = project(K, T_base_cam, KEYPOINTS_CAD_MM @ canonical_T(T, T_base_cam, syms)[:3, :3].T
                           + canonical_T(T, T_base_cam, syms)[:3, 3])[0]
            for flip in (False, True):
                inst = KL.Instance()
                inst.uv = want[perm].copy() if flip else want.copy()
                inst.vis = np.full(len(want), KL.VISIBLE)
                with tempfile.TemporaryDirectory() as d:
                    path = Path(d) / "a.txt"
                    self.assertEqual(KL.save_labels(path, [inst], cam, 640, 480), [])
                    got = KL.load_labels(path, 640, 480)[0]
                np.testing.assert_allclose(got.uv, want, atol=0.05)


if __name__ == "__main__":
    unittest.main()
