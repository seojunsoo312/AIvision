"""카메라 창 (roi_cloud) 의 자세 방식 두 가지와, 방식과 상관없이 쓰는 뒷부분.

자세 방식
  cad : YOLO 박스 → 뎁스 점군 → CAD 정합 (roi_cloud 의 _CadWorker, 예전 그대로)
  kpt : YOLO-pose 키포인트 → 책상 붙이기 (컬러만. KptWorker)
두 방식 모두 같은 CAD (cad/model.py) 좌표계의 "로봇 베이스 mm 4×4 자세" 를 낸다.
그래서 뒷부분 (테두리 맞춤 · 흰/검 · 양/불, 펜던트 숫자) 은 하나로 쓴다.

리플레이: 카메라 대신 저장된 컬러 + 뎁스 장면 폴더 (rgb.png · depth_mm.npy · camera.json) 로 창을 돌린다.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# 판정 색: 초록 양품, 빨강 불량, 보라 검사 안 함 (검은 브라켓), 주황 브라켓 아님
VERDICT_BGR = {"양품": (0, 200, 0), "불량": (0, 0, 255), "검사 안 함": (255, 0, 200), "브라켓 아님": (0, 140, 255)}
POSE_METHODS = ("cad", "kpt")
METHOD_KO = {"cad": "CAD 정합", "kpt": "키포인트"}


def pendant_numbers(T_base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """베이스 mm 4×4 → 펜던트 칸 숫자 (xyz mm, 베이스 RPY, 물체축 RPY)."""
    from motion.grasp import canonicalize_extrinsic_rpy, rpy_extrinsic_to_body_xyz
    from yolo.pose.register import T_to_xyzrpy

    xyz, rpy_raw = T_to_xyzrpy(np.asarray(T_base, dtype=np.float64).reshape(4, 4))
    rpy_base = canonicalize_extrinsic_rpy(rpy_raw)
    rpy_body = rpy_extrinsic_to_body_xyz(rpy_base)
    return np.asarray(xyz, float).reshape(3), np.asarray(rpy_base, float).reshape(3), np.asarray(rpy_body, float).reshape(3)


def kpt_camera(K: np.ndarray, T_base_cam: np.ndarray, calib_dir: Path | None = None, dist=None):
    """키포인트 · 판정 쪽이 쓰는 카메라 (K · 왜곡 · 손눈). roi_cloud 와 같은 K, 같은 손눈을 쓴다.

    왜곡은 이 보드 캘리브 (180° 돌린 영상 기준) 에서 가져온다. 리플레이처럼 왜곡이 없으면 dist 를 0 으로 준다.
    """
    from yolo.train import kpt_label as KL

    if dist is None:
        try:
            dist = KL.calib_camera(calib_dir).dist
        except FileNotFoundError:
            dist = np.zeros(5)
    return KL.Camera(np.asarray(K, float), np.asarray(dist, float).reshape(-1),
                     np.asarray(T_base_cam, float), "roi_cloud")


def judge_all(bgr: np.ndarray, poses: list[np.ndarray], cam) -> list[dict]:
    """자세마다 테두리 맞춤 → 흰/검 → 양/불. 방식과 상관없다."""
    from yolo.pose.sim_defect_check import judge

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = []
    for T in poses:
        r = judge(bgr, T, cam, gray=gray)
        r["pendant"] = pendant_numbers(r["T_base"])
        out.append(r)
    return out


def draw_judged(bgr: np.ndarray, results: list[dict], title: str) -> np.ndarray:
    """판정 그림: 마커 픽셀 빨강, 윤곽은 판정 색."""
    from vision.text import put_text

    vis = bgr.copy()
    for k, r in enumerate(results):
        col = VERDICT_BGR[r["verdict"]]
        if r["white"]:
            vis[r["marks"]] = (0, 0, 255)
        for c in r["inst"].outline or []:
            cv2.polylines(vis, [c.astype(np.int32)], True, col, 2, cv2.LINE_AA)
        box = r["inst"].box
        if box is not None:
            put_text(vis, f"{k} {r['pose']} {r['verdict']}", (int(box[0]), max(16, int(box[1]) - 6)), 0.6, col, 2)
    put_text(vis, title, (8, 22), 0.6, (0, 255, 255), 2)
    return vis


def print_judged(results: list[dict], title: str) -> None:
    print(f"\n== {title}  브라켓 {len(results)}개")
    if not results:
        return
    print(f"{'#':>2s} {'자세':6s} {'색':3s} {'판정':8s} {'어두운px':>7s} {'테두리':>6s}   x mm    y mm    z mm "
          "|  베이스 Rx  Ry  Rz |  물체축 Rx  Ry  Rz")
    for k, r in enumerate(results):
        xyz, rb, ro = r["pendant"]
        print(f"{k:>2d} {r['pose']:6s} {'흰' if r['white'] else '검':3s} {r['verdict']:8s} {r['dark_px']:>7d} "
              f"{r['edge_px']:6.2f}  {xyz[0]:6.1f}  {xyz[1]:6.1f}  {xyz[2]:6.1f} | {rb[0]:6.1f} {rb[1]:6.1f} {rb[2]:6.1f} | "
              f"{ro[0]:6.1f} {ro[1]:6.1f} {ro[2]:6.1f}")


def refine_pose(bgr: np.ndarray, T_base: np.ndarray, cam) -> np.ndarray:
    """테두리 자동 맞춤으로 자세를 1~2 px 까지 다듬는다. 안 되면 그대로."""
    from yolo.train import kpt_label as KL

    res = KL.fit_edges(bgr, np.asarray(T_base, float), cam)
    return np.asarray(T_base, float) if res is None else res[0]


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def pick_target(kpt_poses: list[dict], ref_box=None) -> int | None:
    """키포인트 브라켓 중 집을 대상. CAD 쪽이 고른 박스 (맨 위 물체) 와 가장 겹치는 것, 없으면 가장 큰 것.

    두 방식을 바꿔 가며 볼 때 같은 브라켓을 가리키게 한다.
    """
    if not kpt_poses:
        return None
    if ref_box is not None:
        ious = [box_iou(p["box"], ref_box) for p in kpt_poses]
        if max(ious) > 0.1:
            return int(np.argmax(ious))
    areas = [(p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]) for p in kpt_poses]
    return int(np.argmax(areas))


class KptWorker:
    """키포인트 자세. 젯슨에서 한 장 0.3~0.4 초라 카메라 루프 밖에서 돈다.

    결과: 브라켓마다 {"T": 베이스 mm 4×4, "class_id", "box": (x1, y1, x2, y2), "outline", "uv", "vis"}.
    """

    def __init__(self, cam, weights: Path | None = None):
        from ultralytics import YOLO

        from yolo.train import kpt_label as KL

        self._KL = KL
        self.weights = Path(weights) if weights else KL.pick_weights()
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"키포인트 가중치가 없습니다: {self.weights}\n"
                "키포인트 기본 모델 yolo/weights/kpt_base.pt 를 USB 로 복사하세요 (노트북의 v4n_ft2 best.pt)."
            )
        self.cam = cam
        self.model = YOLO(str(self.weights))
        # 첫 장이 느리지 않게 모델 · CAD 광선 · 책상 자세 후보를 미리 불러 둔다
        self.model.predict(np.zeros((480, 640, 3), np.uint8), verbose=False)
        KL._ray_scene()
        KL.desk_candidates()
        self._lock = threading.Lock()
        self._job: np.ndarray | None = None
        self.result: list[dict] | None = None
        self.busy = False
        self.seconds = 0.0  # 마지막 한 장에 걸린 시간
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, bgr: np.ndarray) -> bool:
        with self._lock:
            if self.busy:
                return False
            self._job = bgr.copy()
            self.busy = True
        return True

    def estimate(self, bgr: np.ndarray) -> list[dict]:
        """한 장 → 브라켓 자세 목록 (스레드 없이 바로)."""
        KL = self._KL
        out = []
        for inst in KL.model_guess(self.model, bgr, self.cam):
            if inst.T_cam_obj is None or inst.box is None:
                continue
            out.append({"T": self.cam.T_base_cam @ inst.T_cam_obj, "class_id": int(inst.cls),
                        "box": tuple(float(v) for v in inst.box), "outline": inst.outline,
                        "uv": inst.uv.copy(), "vis": inst.vis.copy()})
        return out

    def _loop(self) -> None:
        while not self._stop:
            with self._lock:
                job = self._job
                self._job = None
            if job is None:
                time.sleep(0.01)
                continue
            t0 = time.perf_counter()
            try:
                res = self.estimate(job)
            except Exception as exc:  # 카메라 창은 계속 돈다
                print(f"키포인트 자세 실패: {exc}")
                res = []
            with self._lock:
                self.result = res
                self.seconds = time.perf_counter() - t0
                self.busy = False

    def take_result(self) -> list[dict] | None:
        """새 결과를 한 번만 꺼낸다."""
        with self._lock:
            res = self.result
            self.result = None
        return res

    def close(self) -> None:
        # 스레드가 GPU 를 쓰는 채로 프로그램이 끝나면 비정상 종료 (core dump) 한다. 끝날 때까지 기다린다.
        self._stop = True
        self._thread.join(timeout=5.0)


class ReplaySource:
    """카메라 대신 장면 폴더 (rgb.png · depth_mm.npy · camera.json) 를 돌린다. open_orbbec 과 같은 grab/flush/close.

    folder 가 장면 하나면 그것만, 장면 폴더들을 담은 폴더면 n / p 로 넘긴다.
    시뮬 장면은 이미 180° 돌린 영상 기준이라 회전하지 않는다. noise=True 면 시뮬 뎁스에 실물 같은 노이즈를 넣는다.
    """

    def __init__(self, folder: Path, noise: bool = False, seed: int = 0):
        folder = Path(folder)
        if (folder / "rgb.png").is_file():
            self.scenes = [folder]
        else:
            self.scenes = sorted(p for p in folder.iterdir() if (p / "rgb.png").is_file())
        if not self.scenes:
            raise SystemExit(f"장면이 없습니다 (rgb.png · depth_mm.npy): {folder}")
        self.idx = 0
        self.noise = noise
        self.seed = seed
        self._cache: tuple[int, np.ndarray, np.ndarray] | None = None

    def camera(self) -> tuple[np.ndarray, np.ndarray]:
        """(K, T_base_cam). 장면마다 같다고 보고 첫 장면 것을 쓴다."""
        d = json.loads((self.scenes[0] / "camera.json").read_text(encoding="utf-8"))
        return np.asarray(d["K"], float), np.asarray(d["T_base_cam"], float)

    @property
    def name(self) -> str:
        return self.scenes[self.idx].name

    def step(self, delta: int) -> None:
        self.idx = max(0, min(self.idx + delta, len(self.scenes) - 1))

    def flush(self) -> None:
        pass

    def grab(self, timeout_ms: int = 0):
        if self._cache is None or self._cache[0] != self.idx:
            folder = self.scenes[self.idx]
            bgr = cv2.imread(str(folder / "rgb.png"))
            depth = np.load(folder / "depth_mm.npy").astype(np.float32)
            if self.noise:
                from vision.sim.noise import add_noise

                K, _ = self.camera()
                depth = add_noise(depth, K, np.random.default_rng(self.seed + self.idx))
            # 카메라 (OrbbecV1.grab) 와 같은 mm float32
            self._cache = (self.idx, bgr, np.asarray(depth, np.float32))
        time.sleep(0.03)  # 카메라처럼 30 fps 정도로
        return self._cache[1].copy(), self._cache[2].copy()

    def close(self) -> None:
        pass


def save_scene(folder: Path, bgr: np.ndarray, depth: np.ndarray, K: np.ndarray, T_base_cam) -> None:
    """지금 화면을 리플레이 장면 폴더로 저장한다 (rgb.png · depth_mm.npy · camera.json)."""
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "rgb.png"), bgr)
    np.save(folder / "depth_mm.npy", np.asarray(depth, np.float32))
    cam = {"K": np.asarray(K, float).tolist(), "width": int(bgr.shape[1]), "height": int(bgr.shape[0])}
    if T_base_cam is not None:
        cam["T_base_cam"] = np.asarray(T_base_cam, float).tolist()
    (folder / "camera.json").write_text(json.dumps(cam, indent=2), encoding="utf-8")
