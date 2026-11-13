"""「카메라로 집기」: 구역 · 가까운 순 고르기, 등록 요청의 조건 · 결과, 따로 쓰는 설정 파일, 바구니 단계."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
for _p in (PROJECT, PROJECT / "launcher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import demo_settings  # noqa: E402
from motion import register_link  # noqa: E402
from motion.pick_place import build_pick_drop_steps  # noqa: E402
from yolo.pose.instances import InstanceCloud, ordered_targets, select_topmost  # noqa: E402


def _inst(x: float, y: float, height: float) -> InstanceCloud:
    xyz = np.array([[x, y, 0.0]] * 30, dtype=np.float64) + np.random.default_rng(0).normal(0, 1, (30, 3))
    return InstanceCloud(xyxy=[0, 0, 1, 1], roi=np.zeros((1, 1)), xyz=xyz, rgb=np.zeros((30, 3)), height_score=height)


class SelectTest(unittest.TestCase):
    def test_skips_object_in_place_zone(self):
        placed, other = _inst(150, -180, 40.0), _inst(100, 120, 20.0)
        self.assertEqual(select_topmost([placed, other]), 0)
        self.assertEqual(select_topmost([placed, other], exclude=[(150, -180, 60)]), 1)

    def test_nothing_left(self):
        self.assertIsNone(select_topmost([_inst(150, -180, 40.0)], exclude=[(150, -180, 60)]))

    def test_region_and_nearest(self):
        far, near, outside = _inst(250, 150, 50.0), _inst(80, 100, 10.0), _inst(150, -150, 90.0)
        region = ((60, 300), (60, 260))
        order = ordered_targets([far, near, outside], region=region, nearest=True)
        self.assertEqual([i for i, _, _ in order], [1, 0])
        order = ordered_targets([far, near, outside], region=region)
        self.assertEqual([i for i, _, _ in order], [0, 1])  # 조건 없으면 높은 순


class RequestOptionsTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            req_path, st_path = Path(d) / "req.json", Path(d) / "st.json"
            with mock.patch.multiple(register_link, REGISTER_REQUEST_JSON=req_path, REGISTER_STATUS_JSON=st_path):
                req = register_link.RegisterRequest.send(region_xy_mm=((60, 300), (60, 260)), nearest=True)
                opts = register_link.read_request_options()
                self.assertEqual(opts.region, ((60.0, 300.0), (60.0, 260.0)))
                self.assertTrue(opts.nearest and opts.active)
                register_link.write_status(req.id, "ok", xyz=(1, 2, 3), rpy=(0, 0, 0),
                                           extra={"gray": 180.0, "targets_xy_mm": [[80, 100], [250, 150]]})
                self.assertEqual(req.poll(), "ok")
                self.assertEqual(req.gray, 180.0)
                self.assertEqual(req.targets_xy_mm, [(80.0, 100.0), (250.0, 150.0)])
                register_link.RegisterRequest.send()
                self.assertFalse(register_link.read_request_options().active)

    def test_bad_items_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "req.json"
            path.write_text(json.dumps({"id": "x", "exclude_xy_mm": [[1, 2], "a", [1, 2, 0], [1, 2, 3]],
                                        "region_xy_mm": [[5, 1], [0, 1]], "nearest": "yes"}))
            with mock.patch.object(register_link, "REGISTER_REQUEST_JSON", path):
                opts = register_link.read_request_options()
                self.assertEqual(opts.exclude, ((1.0, 2.0, 3.0),))
                self.assertIsNone(opts.region)
                self.assertFalse(opts.nearest)


class VisionSettingsTest(unittest.TestCase):
    def test_separate_file_with_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            demo, vision = Path(d) / "demo.json", Path(d) / "vision.json"
            demo_settings.ensure_settings_file(demo, vision)
            self.assertNotIn("white_bin_xyz_mm", json.loads(demo.read_text()))
            got = json.loads(vision.read_text())
            self.assertEqual(got["white_bin_xyz_mm"], list(demo_settings.VISION_BINS["white_bin_xyz_mm"]))
            self.assertEqual(got["pick_region_mm"], [list(p) for p in demo_settings.VISION_REGION])
            self.assertNotIn("cycles", got)

    def test_bad_value_uses_default(self):
        with tempfile.TemporaryDirectory() as d:
            vision = Path(d) / "vision.json"
            vision.write_text(json.dumps({"black_bin_xyz_mm": [500, 0, 80], "white_gray_min": 100,
                                          "pick_region_mm": [[300, 60], [60, 260]]}))
            with self.assertLogs("aivision.demo", "WARNING"):
                got = demo_settings.load_vision_settings(vision)
            self.assertEqual(got["black_bin_xyz_mm"], list(demo_settings.VISION_BINS["black_bin_xyz_mm"]))
            self.assertEqual(got["pick_region_mm"], [list(p) for p in demo_settings.VISION_REGION])
            self.assertEqual(got["white_gray_min"], 100)
            self.assertIn("approach_d_mm", json.loads(vision.read_text()))  # 빠진 키를 채워 씀


class DropStepsTest(unittest.TestCase):
    def test_lift_then_drop_in_air(self):
        steps = build_pick_drop_steps(p_xyz_mm=[150, 120, 40], p_rpy_deg=[180, 0, 0], g_xyz_mm=[150, 120, 10],
                                      g_rpy_deg=[180, 0, 0], drop_xyz_mm=[220, -150, 80])
        names = [s.name for s in steps]
        self.assertEqual(names[-3:], ["바구니 위", "떨어뜨리기", "초기자세"])
        lift = steps[names.index("집기 리프트")]
        self.assertAlmostEqual(float(lift.xyz_mm[2]), 80.0)  # 바구니 높이까지 바로 위로
        self.assertEqual(list(steps[names.index("바구니 위")].xyz_mm), [220, -150, 80])
        self.assertIs(steps[names.index("떨어뜨리기")].hold, False)


if __name__ == "__main__":
    unittest.main()
