"""yolo/pose/roi_cloud.py _CadTracks: 브라켓마다 자리를 기억해 바뀐 것만 다시 맞춘다."""

import unittest

import numpy as np

from yolo.pose.roi_cloud import _CadTracks


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _cloud(x, y, z=20.0, n=50, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([x, y, z]) + rng.normal(0, 1.0, (n, 3))


class CadTracksTest(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.calls = []
        self.fail_at: set[int] = set()  # 이 x (반올림) 자리는 정합 실패

        def fake_register(xyz, *, T_init=None, **_kw):
            x = int(round(float(np.median(xyz[:, 0]))))
            self.calls.append(("track" if T_init is not None else "full", x))
            T = np.eye(4)
            T[:3, 3] = np.median(xyz, axis=0)
            return {"T": T, "aligned_ok": x not in self.fail_at}

        self.tracks = _CadTracks(register_fn=fake_register, clock=self.clock, start_thread=False,
                                 move_mm=8.0, reset_mm=30.0, retry_s=8.0, drop_s=3.0, verbose=False)

    def _run(self, *clouds):
        self.calls.clear()
        self.tracks.process([(c, None) for c in clouds], None)
        return list(self.calls)

    def test_only_changed_brackets_are_fitted(self):
        self.assertEqual(self._run(_cloud(100, 0), _cloud(200, 0)), [("full", 100), ("full", 200)])
        self.clock.t += 0.5
        self.assertEqual(self._run(_cloud(100, 0, seed=1), _cloud(200, 0, seed=1)), [])  # 그대로: 계산 안 함
        self.clock.t += 0.5
        self.assertEqual(self._run(_cloud(112, 0), _cloud(200, 0)), [("track", 112)])  # 12 mm 옮김: 따라가기
        self.assertEqual(len([t for t in self.tracks.snapshot() if t["T"] is not None]), 2)

    def test_depth_height_jitter_is_not_a_move(self):
        self._run(_cloud(100, 0, z=40))
        self.clock.t += 0.5
        self.assertEqual(self._run(_cloud(100, 0, z=58)), [])  # 검은 브라켓처럼 높이만 18 mm 흔들림

    def test_failed_spot_retries_after_retry_s(self):
        self.fail_at = {150}
        self.assertEqual(self._run(_cloud(150, 0)), [("full", 150)])
        self.clock.t += 2.0
        self.assertEqual(self._run(_cloud(150, 0)), [])
        self.clock.t += 7.0
        self.assertEqual(self._run(_cloud(150, 0)), [("full", 150)])

    def test_missed_frames_keep_pose_until_drop(self):
        self._run(_cloud(100, 0), _cloud(200, 0))
        self.clock.t += 2.0
        self._run(_cloud(100, 0))  # 200 을 YOLO 가 놓침
        self.assertEqual(len(self.tracks.snapshot()), 2)
        self.clock.t += 2.0
        self._run(_cloud(100, 0))
        self.assertEqual([round(t["anchor"][0]) for t in self.tracks.snapshot()], [100])

    def test_set_pose_from_request(self):
        self.fail_at = {150}
        self._run(_cloud(150, 0))
        T = np.eye(4)
        self.tracks.set_pose(np.array([151.0, 0.0, 20.0]), {"T": T, "aligned_ok": True})
        (t,) = self.tracks.snapshot()
        self.assertIsNotNone(t["T"])


if __name__ == "__main__":
    unittest.main()
