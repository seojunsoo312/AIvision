"""모든 브라켓 요청 (load_all_pick_targets) 의 주고받기 시험. 카메라 창 없이 상태 파일을 직접 쓴다."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from motion import register_link as RL


class RegisterAllTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.patches = [mock.patch.object(RL, "REGISTER_REQUEST_JSON", d / "req.json"),
                        mock.patch.object(RL, "REGISTER_STATUS_JSON", d / "status.json")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_request_flag_and_objects(self):
        req = RL.RegisterRequest.send(all_objects=True)
        rid, want_all = RL.read_request_full()
        self.assertEqual(rid, req.id)
        self.assertTrue(want_all)
        objs = [{"xyz_mm": [100, 20, 17.5], "rpy_deg": [90, 0, -30], "pose": "눕히기", "color": "흰", "verdict": "불량"}]
        RL.write_status(req.id, "ok", objects=objs)
        self.assertEqual(req.poll(), "ok")
        self.assertEqual(req.objects[0]["verdict"], "불량")
        self.assertEqual(req.objects[0]["xyz_mm"], (100.0, 20.0, 17.5))

    def test_single_request_unchanged(self):
        # 예전 요청 (펜던트) 은 all 표시가 없고, 답도 xyz · rpy 하나
        req = RL.RegisterRequest.send()
        self.assertEqual(RL.read_request_full(), (req.id, False))
        RL.write_status(req.id, "ok", xyz=[1, 2, 3], rpy=[0, 0, 0])
        self.assertEqual(req.poll(), "ok")
        self.assertEqual(req.xyz_mm, (1.0, 2.0, 3.0))

    def test_all_request_without_objects_fails(self):
        req = RL.RegisterRequest.send(all_objects=True)
        RL.write_status(req.id, "ok", xyz=[1, 2, 3], rpy=[0, 0, 0])
        self.assertEqual(req.poll(), "fail")


if __name__ == "__main__":
    unittest.main()
