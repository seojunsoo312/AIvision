"""카메라 창 자세 방식 공통 부분 (yolo/pose/pose_methods.py) 시험. 모델 없이 도는 것만."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from yolo.pose.pose_methods import ReplaySource, box_iou, pick_target, save_scene


class PoseMethodsTest(unittest.TestCase):
    def test_box_iou(self):
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)

    def test_pick_target_follows_cad_box(self):
        # 두 방식을 바꿔도 같은 브라켓: CAD 쪽 박스와 가장 겹치는 것을 고른다
        poses = [{"box": (0, 0, 50, 50)}, {"box": (100, 100, 120, 120)}]
        self.assertEqual(pick_target(poses, (98, 98, 121, 121)), 1)
        # 겹치는 것이 없으면 가장 큰 것
        self.assertEqual(pick_target(poses, (300, 300, 310, 310)), 0)
        self.assertEqual(pick_target(poses, None), 0)
        self.assertIsNone(pick_target([], None))

    def test_scene_save_and_replay(self):
        # s 로 저장한 장면을 --replay 로 그대로 다시 읽는다 (뎁스는 카메라처럼 mm float32)
        with tempfile.TemporaryDirectory() as tmp:
            bgr = np.full((48, 64, 3), 200, np.uint8)
            depth = np.full((48, 64), 450.5, np.float32)
            K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
            T = np.eye(4)
            T[2, 3] = 450
            for name in ("scene_a", "scene_b"):
                save_scene(Path(tmp) / name, bgr, depth, K, T)
            src = ReplaySource(Path(tmp))
            self.assertEqual(len(src.scenes), 2)
            K2, T2 = src.camera()
            np.testing.assert_allclose(K2, K)
            np.testing.assert_allclose(T2, T)
            b2, d2 = src.grab()
            self.assertEqual(d2.dtype, np.float32)
            np.testing.assert_allclose(d2, depth)
            np.testing.assert_array_equal(b2, bgr)
            src.step(5)
            self.assertEqual(src.name, "scene_b")


if __name__ == "__main__":
    unittest.main()
