"""가중치 이름 (box_base · box_best · kpt_base · kpt_best) 과 예전 이름 대신 쓰기 시험. 빈 임시 폴더로 확인한다."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yolo.config as C
from yolo.train import kpt_label as KL


class WeightNamesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.w = Path(self.tmp.name) / "weights"
        self.runs = Path(self.tmp.name) / "pose_runs"
        self.w.mkdir()
        self.patches = [
            mock.patch.object(C, "BEST_PT", self.w / "box_best.pt"),
            mock.patch.object(C, "OLD_BEST_PT", self.w / "best.pt"),
            mock.patch.object(C, "BOX_BASE_PT", self.w / "box_base.pt"),
            mock.patch.object(C, "OLD_BOX_BASE_PT", self.w / "bracket_base.pt"),
            mock.patch.object(KL, "KPT_BASE_PT", self.w / "kpt_base.pt"),
            mock.patch.object(KL, "STUDENT_WEIGHTS", self.w / "kpt_best.pt"),
            mock.patch.object(KL, "POSE_RUNS", self.runs),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _touch(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    def test_names(self):
        # 실제 설정의 이름 네 개 (모두 yolo/weights/)
        self.assertEqual(
            [p.name for p in (C.WEIGHTS_DIR / "box_base.pt", C.WEIGHTS_DIR / "box_best.pt",
                              C.WEIGHTS_DIR / "kpt_base.pt", C.WEIGHTS_DIR / "kpt_best.pt")],
            ["box_base.pt", "box_best.pt", "kpt_base.pt", "kpt_best.pt"])
        self.assertEqual(C.KPT_BEST_PT.name, "kpt_best.pt")
        self.assertEqual(C.KPT_BASE_PT.parent, C.WEIGHTS_DIR)

    def test_box_best_new_and_old(self):
        self.assertEqual(C.box_best_pt().name, "box_best.pt")  # 둘 다 없으면 새 이름 (부르는 쪽이 안내)
        self._touch(self.w / "best.pt")
        self.assertEqual(C.box_best_pt().name, "best.pt")  # 예전 이름만
        self._touch(self.w / "box_best.pt")
        self.assertEqual(C.box_best_pt().name, "box_best.pt")  # 새 이름이 먼저

    def test_box_base_new_and_old(self):
        self.assertEqual(C.box_base_pt().name, "box_base.pt")
        self._touch(self.w / "bracket_base.pt")
        self.assertEqual(C.box_base_pt().name, "bracket_base.pt")
        self._touch(self.w / "box_base.pt")
        self.assertEqual(C.box_base_pt().name, "box_base.pt")

    def test_kpt_base_new_and_old_place(self):
        self.assertEqual(KL.base_weights().name, "kpt_base.pt")  # 없으면 새 경로로 안내
        self._touch(self.runs / "v4n_ft2" / "weights" / "best.pt")
        self.assertEqual(KL.base_weights().parent.parent.name, "v4n_ft2")  # 예전 자리
        self._touch(self.w / "kpt_base.pt")
        self.assertEqual(KL.base_weights().name, "kpt_base.pt")  # 새 자리가 먼저

    def test_kpt_student_first(self):
        self._touch(self.w / "kpt_base.pt")
        self.assertEqual(KL.pick_weights().name, "kpt_base.pt")
        self._touch(self.w / "kpt_best.pt")
        self.assertEqual(KL.pick_weights().name, "kpt_best.pt")


if __name__ == "__main__":
    unittest.main()
