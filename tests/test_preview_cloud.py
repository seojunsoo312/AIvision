"""v 키 Open3D 미리보기 점군 (_preview_cloud) 시험. 카메라 없이 가짜 브라켓 점군으로 확인한다."""

import unittest
from types import SimpleNamespace

import numpy as np

from yolo.pose.roi_cloud import _OTHER_CAD_RGBS, _TOP_CAD_RGB, _preview_cloud


def _inst(n, offset):
    xyz = np.random.default_rng(n).normal(size=(n, 3)).astype(np.float32) + np.asarray(offset, dtype=np.float32)
    rgb = np.full((n, 3), 100, dtype=np.uint8)
    return SimpleNamespace(xyz=xyz, rgb=rgb)


def _T(x, y, z):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


class PreviewCloudTest(unittest.TestCase):
    def setUp(self):
        self.cad = np.zeros((5, 3))  # CAD 점 5개 (모두 원점)
        self.insts = [_inst(30, (0, 0, 0)), _inst(40, (100, 0, 0)), _inst(50, (0, 100, 0))]

    def test_all_clouds_and_cads(self):
        others = [{"T": _T(100, 0, 0)}, {"T": _T(0, 100, 0)}]
        xyz, rgb = _preview_cloud(self.insts, {"T": _T(0, 0, 0)}, others, self.cad)
        # 점군 120 + CAD 5 × 3
        self.assertEqual(len(xyz), 120 + 15)
        self.assertEqual(len(rgb), len(xyz))
        cad_rgb = rgb[120:]
        np.testing.assert_array_equal(cad_rgb[:5], np.tile(_TOP_CAD_RGB, (5, 1)))
        np.testing.assert_array_equal(cad_rgb[5:10], np.tile(_OTHER_CAD_RGBS[0], (5, 1)))
        np.testing.assert_array_equal(cad_rgb[10:], np.tile(_OTHER_CAD_RGBS[1], (5, 1)))
        # CAD 는 각 자세 자리로 옮겨진다
        np.testing.assert_allclose(xyz[125:130], np.tile([100, 0, 0], (5, 1)))

    def test_shades_are_red(self):
        # 나머지 색은 모두 빨강 계열 (R 이 가장 크다), 맨 위와 다르다
        for c in _OTHER_CAD_RGBS:
            self.assertGreater(c[0], c[1])
            self.assertGreater(c[0], c[2])
            self.assertNotEqual(tuple(c), tuple(_TOP_CAD_RGB))

    def test_no_cad(self):
        # 키포인트 방식 · 정합 전: 점군만
        xyz, rgb = _preview_cloud(self.insts, None, [], None)
        self.assertEqual(len(xyz), 120)

    def test_failed_others_show_cloud_only(self):
        # 나머지 정합이 하나도 없으면 맨 위 CAD 만
        xyz, _ = _preview_cloud(self.insts, {"T": _T(0, 0, 0)}, [], self.cad)
        self.assertEqual(len(xyz), 125)

    def test_empty(self):
        xyz, rgb = _preview_cloud([], None, [], self.cad)
        self.assertEqual(xyz.shape, (0, 3))
        self.assertEqual(rgb.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
