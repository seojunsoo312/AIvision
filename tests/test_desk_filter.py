"""vision/depth_cloud.py: 책상 제거 높이 자동 결정과 가장자리 튀는 뎁스 제거."""

import unittest

import numpy as np

from vision.depth_cloud import (
    PLANE_HEIGHT_MAX_MM,
    PLANE_HEIGHT_MIN_MM,
    auto_plane_height_mm,
    depth_edge_mask,
    fit_plane_ransac,
    plane_foreground_mask,
    plane_noise_mm,
    points_from_mask,
)

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
DESK_MM = 450.0
TOP_MM = 410.0  # 물체 윗면 (책상 위 40 mm)


def _scene(noise_sd: float, seed: int = 0) -> np.ndarray:
    """책상(노이즈 noise_sd) + 가운데 물체 + 물체 둘레 한 줄의 중간 깊이 픽셀."""
    rng = np.random.default_rng(seed)
    depth = DESK_MM + rng.normal(0.0, noise_sd, (480, 640))
    depth[200:280, 280:360] = TOP_MM + rng.normal(0.0, 0.5, (80, 80))
    ring = np.zeros((480, 640), bool)
    ring[199:281, 279:361] = True
    ring[200:280, 280:360] = False
    depth[ring] = (DESK_MM + TOP_MM) / 2.0
    return depth.astype(np.float32), ring


def _plane(depth):
    xyz, _ = points_from_mask(depth, K, np.ones(depth.shape, bool), stride=5)
    plane, _ = fit_plane_ransac(xyz)
    return plane, xyz


def _auto_height(depth) -> float:
    plane, xyz = _plane(depth)
    return auto_plane_height_mm(plane_noise_mm(xyz, plane))


class DeskHeightTest(unittest.TestCase):
    def test_noisier_desk_gets_higher_threshold(self):
        quiet = _auto_height(_scene(0.5)[0])
        noisy = _auto_height(_scene(2.0)[0])
        self.assertEqual(quiet, PLANE_HEIGHT_MIN_MM)
        self.assertGreater(noisy, quiet + 2.0)
        self.assertLessEqual(noisy, PLANE_HEIGHT_MAX_MM)

    def test_auto_height_removes_desk_keeps_object(self):
        depth, _ = _scene(2.0)
        plane, _ = _plane(depth)
        h = _auto_height(depth)
        fg = plane_foreground_mask(depth, K, plane, height_mm=h)
        desk = np.ones(depth.shape, bool)
        desk[190:290, 270:370] = False
        self.assertLess(fg[desk].mean(), 0.002)  # 책상 픽셀은 거의 다 지움
        self.assertGreater(fg[205:275, 285:355].mean(), 0.99)  # 물체는 남김

    def test_no_measurement_falls_back_to_min(self):
        self.assertEqual(auto_plane_height_mm(None), PLANE_HEIGHT_MIN_MM)


class EdgeMaskTest(unittest.TestCase):
    def test_drops_middle_ring_only(self):
        depth, ring = _scene(0.5)
        edge = depth_edge_mask(depth)
        self.assertGreater(edge[ring].mean(), 0.95)
        self.assertFalse(edge[205:275, 285:355].any())  # 물체 윗면 안쪽
        self.assertFalse(edge[:150].any())  # 멀리 떨어진 책상


if __name__ == "__main__":
    unittest.main()
