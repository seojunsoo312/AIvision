"""yolo/pose/instances.py: 책상 위 전제로 튀는 뎁스를 버린다 (BRACKET_MAX_HEIGHT_MM)."""

import unittest

import numpy as np

from yolo.pose.instances import BRACKET_MAX_HEIGHT_MM, collect_instances

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
# 카메라가 책상 450 mm 위에서 똑바로 내려다본다 (카메라 z = 베이스 -z).
T_BASE_CAM = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 450.0], [0, 0, 0, 1.0]])
PLANE_CAM = np.array([0.0, 0.0, 1.0, -450.0])


def _scene():
    depth = np.full((480, 640), 450.0, np.float32)
    depth[220:260, 300:340] = 410.0  # 40 mm 높이 물체 (세운 브라켓 정도)
    depth[230:232, 310:312] = 250.0  # 흰 면에서 튀는 뎁스: 200 mm 높이
    depth[245, 325] = 300.0
    depth[100:130, 100:150] = 437.0  # 두 번째 물체 (13 mm 판): 상자가 여럿이어도 되는지
    bgr = np.full((480, 640, 3), 200, np.uint8)
    return depth, bgr


class HeightCapTest(unittest.TestCase):
    def test_drops_flying_depth_keeps_object(self):
        depth, bgr = _scene()
        out = collect_instances(depth=depth, bgr=bgr, K=K, result=None, plane=PLANE_CAM, pad=2, plane_mm=5.0,
                                stride=1, T_base_cam=T_BASE_CAM, xyxys=[[295, 215, 345, 265], [95, 95, 155, 135]])
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(float(np.median(out[1].xyz[:, 2])), 13.0, delta=1.0)
        z = out[0].xyz[:, 2]
        self.assertLessEqual(float(z.max()), BRACKET_MAX_HEIGHT_MM)
        self.assertAlmostEqual(float(np.median(z)), 40.0, delta=1.0)
        self.assertGreater(len(z), 1400)  # 물체 1600 px 중 튀는 5 px 와 가장자리만 빠진다
        self.assertAlmostEqual(out[0].height_score, 40.0, delta=1.0)  # 맨 위 고르기가 200 mm 에 속지 않는다


if __name__ == "__main__":
    unittest.main()
