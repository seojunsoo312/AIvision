"""vision/depth_cloud.py: 책상 제거 높이 자동 결정과 가장자리 튀는 뎁스 제거."""

import unittest

import numpy as np

from vision.depth_cloud import (
    PLANE_HEIGHT_MAX_MM,
    PLANE_HEIGHT_MIN_MM,
    DEPTH_VIS_FAR_MM,
    DEPTH_VIS_NEAR_MAX_MM,
    auto_plane_height_mm,
    colorize_depth,
    depth_vis_range,
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


class DepthVisRangeTest(unittest.TestCase):
    """화면 깊이 색: 책상보다 높은 브라켓과 로봇팔이 서로 다른 색이어야 한다."""

    def _scene(self):
        depth = np.full((480, 640), DESK_MM, np.float32)
        depth[200:280, 100:180] = DESK_MM - 42.0  # 브라켓 윗면
        depth[150:480, 300:400] = DESK_MM - 180.0  # 로봇팔
        return depth

    def test_bracket_and_arm_get_different_colors(self):
        depth = self._scene()
        low, high = depth_vis_range(depth)
        self.assertLessEqual(low, DESK_MM - 180.0)  # 팔까지 담음
        self.assertEqual(high, DESK_MM + DEPTH_VIS_FAR_MM)
        vis = colorize_depth(depth).astype(int)
        desk, bracket, arm = vis[100, 50], vis[240, 140], vis[400, 350]
        self.assertGreater(np.abs(bracket - arm).sum(), 60)
        self.assertGreater(np.abs(bracket - desk).sum(), 60)

    def test_range_is_capped_and_ignores_specks(self):
        depth = self._scene()
        depth[0:3, 0:3] = 100.0  # 몇 점 튀는 노이즈
        low, _ = depth_vis_range(depth)
        self.assertGreaterEqual(low, DESK_MM - DEPTH_VIS_NEAR_MAX_MM)

    def test_empty_depth(self):
        self.assertIsNone(depth_vis_range(np.zeros((480, 640), np.float32)))


if __name__ == "__main__":
    unittest.main()


class DepthBlobTest(unittest.TestCase):
    """YOLO 대신 뎁스 덩어리로 물체 찾기: 가장자리에 닿거나 높은 덩어리는 뺀다."""

    def test_keeps_object_drops_border_and_tall(self):
        from yolo.pose.instances import depth_blob_boxes

        depth = np.full((480, 640), DESK_MM, np.float32)
        depth[:, 600:] = 0.0  # 컬러 맞춤 뒤 오른쪽은 뎁스가 없다
        depth[200:240, 300:360] = DESK_MM - 35.0  # 브라켓
        depth[100:140, 585:600] = DESK_MM - 10.0  # 뎁스 끝에 닿은 책상 프레임
        depth[300:360, 100:160] = DESK_MM - 150.0  # 높은 것 (팔)
        plane = np.array([0.0, 0.0, 1.0, -DESK_MM])  # 카메라 z = 책상 깊이
        boxes = depth_blob_boxes(depth, K, plane, 4.0, None)
        self.assertEqual(len(boxes), 1)
        x1, y1, x2, y2 = boxes[0]
        self.assertTrue(295 <= x1 <= 305 and 195 <= y1 <= 205 and 355 <= x2 <= 365 and 235 <= y2 <= 245)
