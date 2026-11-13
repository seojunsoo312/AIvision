#!/usr/bin/env python3
"""YOLO ROI → 인스턴스 점군 → 최상단 → (선택) CAD 등록.

V4L2 컬러만으로는 안 된다. Orbbec SDK(D2C) + 학습한 박스 모델 box_best.pt.
펜던트·Viewer·detect.py 와 동시에 켜지 말 것.

  python yolo/pose/roi_cloud.py                     # 자세 = CAD 정합 (기본)
  python yolo/pose/roi_cloud.py --pose kpt          # 자세 = 키포인트 (컬러만)
  python yolo/pose/roi_cloud.py --replay vision/sim/out/scenes   # 카메라 대신 저장한 장면 (n / p 넘기기)
  v=3D (브라켓 전부, CAD 는 맨 위 빨강 · 나머지 진하거나 연한 빨강)  c=지금 등록  s=PLY · 장면 저장  r=평면  q=종료
  k=자세 방식 바꾸기 (CAD 정합 ↔ 키포인트)  i=검사 (흰/검 · 양/불, 화면 멈춤. 아무 키나 → 다시)
  별도 창: NoiseRemoval / HoleFilter / HoleFilling / Color / DepthExp / Temporal

자세 방식만 다르고 뒷부분 (검사 · 펜던트 「물체 위치 불러오기」) 은 같다. 펜던트에는 지금 고른 방식의 자세가 간다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vision.gui_env import quiet_gtk

quiet_gtk()

from vision.calib import intrinsics_for_rotate180, load_K, load_T_base_cam
from vision.camera import FRAME_HEIGHT, FRAME_WIDTH, ROTATE_180
from vision.text import put_text
from vision.transforms import apply_T, invert_T, plane_tilt_from_z_deg, transform_plane
from vision.orbbec_filters import (
    NOISE_MAX_SIZE_DEFAULT,
    NOISE_MIN_DIFF_DEFAULT,
    OrbbecFilterPanel,
)
from yolo.config import (
    box_best_pt,
    KPT_BEST_PT,
    CLASS_NAME,
    DETECT_CONF,
    RUNS_DIR,
    pose_name,
)
from vision.depth_cloud import (
    PLANE_HEIGHT_MIN_MM,
    auto_plane_height_mm,
    colorize_depth,
    fit_plane_ransac,
    open_orbbec,
    plane_noise_mm,
    points_from_mask,
    rotate180,
    write_ply,
)
from yolo.pose.debug.local_plane import SLICE_MM, local_pose
from yolo.pose.instances import collect_instances, depth_blob_boxes, ordered_targets, select_topmost
from yolo.pose.pose_methods import (
    METHOD_KO,
    POSE_METHODS,
    KptWorker,
    ReplaySource,
    draw_judged,
    judge_all,
    kpt_camera,
    pick_target,
    print_judged,
    refine_pose,
    save_scene,
)
from motion.register_link import read_request, read_request_full, read_request_options, write_status
from motion import runlog  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

WIN = "YOLO box ROI cloud"
ROI_DIR = RUNS_DIR / "roi"
_POSE_BGR = {
    0: (80, 220, 80),
    1: (0, 180, 255),
    2: (80, 80, 255),
}


def _work_desk_plane(plane, T_bc) -> np.ndarray | None:
    """등록 프레임의 책상. --base 면 로봇 XY, 없으면 카메라 RANSAC."""
    if plane is not None:
        p = np.asarray(plane, dtype=np.float64).reshape(4)
        if T_bc is not None:
            p = transform_plane(p, T_bc)
        return p
    if T_bc is not None:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    return None


def _pose_mm_to_T(xyz_mm, rpy_deg) -> np.ndarray:
    from motion.robot_kinematics import rpy_deg_to_rotmat

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_deg_to_rotmat(
        float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])
    )
    T[:3, 3] = np.asarray(xyz_mm, dtype=np.float64).reshape(3)
    return T


def _load_grasp_cad() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """model.yaml `grasp` → T_cad_grasp(mm), xyz, rpy. 티칭 물체 기준 좌표계."""
    from cad.model import grasp_xyzrpy_gripper

    xyz_t, rpy_t, _gripper = grasp_xyzrpy_gripper()
    xyz = np.asarray(xyz_t, dtype=np.float64)
    rpy = np.asarray(rpy_t, dtype=np.float64)
    return _pose_mm_to_T(xyz, rpy), xyz, rpy


def _place_ui_xyzrpy(T_work_cad: np.ndarray):
    """펜던트 place UI용: xyz + 베이스 RPY + 물체축 RPY.

    베이스 숫자는 물체축→extrinsic 으로 고정(canonicalize)해서
    펜던트에 물체 RPY를 넣었을 때와 같은 베이스 칸 값이 나오게 한다.
    """
    from motion.grasp import canonicalize_extrinsic_rpy, rpy_extrinsic_to_body_xyz
    from yolo.pose.register import T_to_xyzrpy

    xyz, rpy_raw = T_to_xyzrpy(np.asarray(T_work_cad, dtype=np.float64).reshape(4, 4))
    rpy_base = canonicalize_extrinsic_rpy(rpy_raw)
    rpy_body = rpy_extrinsic_to_body_xyz(rpy_base)
    return (
        np.asarray(xyz, dtype=float).reshape(3),
        np.asarray(rpy_base, dtype=float).reshape(3),
        np.asarray(rpy_body, dtype=float).reshape(3),
    )


def _project_cam(xyz: np.ndarray, K: np.ndarray) -> tuple[int, int] | None:
    z = float(xyz[2])
    if z < 20.0:
        return None
    u = float(K[0, 0]) * float(xyz[0]) / z + float(K[0, 2])
    v = float(K[1, 1]) * float(xyz[1]) / z + float(K[1, 2])
    return int(round(u)), int(round(v))


def _draw_cad_axes(
    img: np.ndarray,
    T_cam: np.ndarray,
    K: np.ndarray,
    *,
    length_mm: float = 30.0,
) -> None:
    """카메라 mm 의 T_cam_cad 를 사진에 RGB 축으로 그린다."""
    T = np.asarray(T_cam, dtype=np.float64).reshape(4, 4)
    origin = T[:3, 3]
    po = _project_cam(origin, K)
    if po is None:
        return
    h, w = img.shape[:2]
    if not (0 <= po[0] < w and 0 <= po[1] < h):
        return
    colors_bgr = ((0, 0, 255), (0, 255, 0), (255, 80, 80))
    for i, color in enumerate(colors_bgr):
        tip = origin + T[:3, i] * length_mm
        pt = _project_cam(tip, K)
        if pt is None:
            continue
        cv2.arrowedLine(img, po, pt, color, 2, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(img, po, 4, (255, 255, 255), -1, cv2.LINE_AA)


def _has_dets(result) -> bool:
    boxes = getattr(result, "boxes", None)
    obb = getattr(result, "obb", None)
    return bool((boxes is not None and len(boxes)) or (obb is not None and len(obb)))


def _median_xyz(xyz: np.ndarray) -> np.ndarray:
    return np.median(np.asarray(xyz, dtype=np.float64).reshape(-1, 3), axis=0)


def _device() -> str | int:
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


class _CadOthers:
    """맨 위 하나 (펜던트 집기용, _CadWorker) 말고 나머지 브라켓도 CAD 정합한다. 화면 · 검사용.

    한 번에 브라켓 목록을 받아 차례로 맞춘다. 지난번 자리에서 move_mm 안이면 다시 계산하지 않고,
    reset_mm 안이면 지난 자세에서 ICP 만 (추적), 그보다 멀거나 새 브라켓이면 처음부터 등록한다.
    실패한 자리도 기억해서, 그 브라켓이 move_mm 넘게 움직이기 전까지는 다시 해 보지 않는다.
    """

    def __init__(self, *, voxel_mm: float, flip: bool, tries: int, camera_origin, method: str,
                 move_mm: float, reset_mm: float):
        from yolo.pose.register import register_pose

        self._register_pose = register_pose
        self.kw = dict(voxel_mm=voxel_mm, flip=flip, tries=tries, camera_origin=camera_origin, method=method)
        self.move_mm = float(move_mm)
        self.reset_mm = float(reset_mm)
        self._lock = threading.Lock()
        self._job: list | None = None
        self._desk = None
        self._cache: list[dict] = []  # 지난번 결과 {"anchor", "T" (실패면 None), "class_id"}
        self.result: list[dict] | None = None
        self.busy = False
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, clouds: list[tuple[np.ndarray, int | None]], desk_plane) -> bool:
        """clouds: 브라켓마다 (점군 베이스 mm, YOLO 자세 클래스 또는 None)."""
        with self._lock:
            if self.busy:
                return False
            self._job = [(np.asarray(x, dtype=np.float64).copy(), c) for x, c in clouds]
            self._desk = None if desk_plane is None else np.asarray(desk_plane, dtype=np.float64).copy()
            self.busy = True
        return True

    def _one(self, xyz: np.ndarray, class_id, desk) -> dict:
        anchor = _median_xyz(xyz)
        prev = None
        if self._cache:
            d = [float(np.linalg.norm(anchor - c["anchor"])) for c in self._cache]
            j = int(np.argmin(d))
            if d[j] < self.move_mm:
                return dict(self._cache[j])  # 그대로 있음 (성공이든 실패든)
            if d[j] < self.reset_mm and self._cache[j]["T"] is not None:
                prev = self._cache[j]
        try:
            pose = self._register_pose(
                xyz,
                T_base_cam=None,
                T_init=None if prev is None else prev["T"],
                T_prev=None if prev is None else prev["T"],
                desk_plane=desk,
                class_id=class_id if prev is None else None,
                **self.kw,
            )
        except Exception as exc:
            print(f"CAD 정합 실패 (다른 브라켓): {exc}")
            pose = None
        if pose is None or not pose.get("aligned_ok", False):
            return {"anchor": anchor, "T": None, "class_id": None}
        return {"anchor": anchor, "T": np.asarray(pose["T"], dtype=np.float64), "class_id": pose.get("class_id")}

    def _loop(self) -> None:
        while not self._stop:
            with self._lock:
                job, desk = self._job, self._desk
                self._job = None
            if job is None:
                time.sleep(0.03)
                continue
            done = []
            for xyz, cid in job:
                if self._stop:
                    break
                done.append(self._one(xyz, cid, desk))
            self._cache = done
            with self._lock:
                self.result = [r for r in done if r["T"] is not None]
                self.busy = False

    def take_result(self) -> list[dict] | None:
        with self._lock:
            res = self.result
            self.result = None
        return res

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=10.0)


class _CadWorker:
    """등록은 수 초 걸리므로 카메라 루프 밖에서 돌린다."""

    def __init__(
        self,
        *,
        voxel_mm: float,
        flip: bool,
        tries: int,
        camera_origin: np.ndarray | None = None,
        T_cad_grasp: np.ndarray | None = None,
        grasp_xyz_mm: np.ndarray | None = None,
        grasp_rpy_deg: np.ndarray | None = None,
        frame_name: str = "camera_mm",
        method: str = "rest",
    ):
        from yolo.pose.register import load_cad_xyz, register_pose, warm_rest_icp

        self._register_pose = register_pose
        self.voxel_mm = voxel_mm
        self.flip = flip
        self.tries = tries
        self.camera_origin = (
            None
            if camera_origin is None
            else np.asarray(camera_origin, dtype=np.float64).reshape(3)
        )
        self.T_cad_grasp = (
            np.eye(4, dtype=np.float64)
            if T_cad_grasp is None
            else np.asarray(T_cad_grasp, dtype=np.float64).reshape(4, 4)
        )
        self.grasp_xyz_mm = (
            np.zeros(3, dtype=np.float64)
            if grasp_xyz_mm is None
            else np.asarray(grasp_xyz_mm, dtype=np.float64).reshape(3)
        )
        self.grasp_rpy_deg = (
            np.zeros(3, dtype=np.float64)
            if grasp_rpy_deg is None
            else np.asarray(grasp_rpy_deg, dtype=np.float64).reshape(3)
        )
        self.frame_name = frame_name
        self._lock = threading.Lock()
        self._job: np.ndarray | None = None
        self._T_init: np.ndarray | None = None
        self._T_prev: np.ndarray | None = None
        self._desk_plane: np.ndarray | None = None
        self._class_id: int | None = None
        self.result: dict | None = None
        self.error: str | None = None
        self.busy = False
        self._stop = False
        self.method = method
        load_cad_xyz(voxel_mm=voxel_mm)
        warm_rest_icp(voxel_mm)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(
        self,
        xyz: np.ndarray,
        *,
        T_init: np.ndarray | None = None,
        T_prev: np.ndarray | None = None,
        desk_plane: np.ndarray | None = None,
        class_id: int | None = None,
    ) -> bool:
        if len(xyz) < 20:
            return False
        with self._lock:
            if self.busy:
                return False
            self._job = np.asarray(xyz, dtype=np.float64).copy()
            self._T_init = None if T_init is None else np.asarray(T_init, dtype=np.float64).copy()
            self._T_prev = None if T_prev is None else np.asarray(T_prev, dtype=np.float64).copy()
            self._desk_plane = (
                None
                if desk_plane is None
                else np.asarray(desk_plane, dtype=np.float64).reshape(4).copy()
            )
            self._class_id = None if class_id is None else int(class_id)
            self.busy = True
            self.error = None
            self.result = None
        return True

    def _loop(self) -> None:
        while not self._stop:
            with self._lock:
                xyz = self._job
                T_init = self._T_init
                T_prev = self._T_prev
                desk_plane = self._desk_plane
                class_id = self._class_id
                self._job = None
            if xyz is None:
                time.sleep(0.03)
                continue
            try:
                pose = self._register_pose(
                    xyz,
                    T_base_cam=None,
                    voxel_mm=self.voxel_mm,
                    flip=self.flip,
                    tries=self.tries,
                    T_init=T_init,
                    T_prev=T_prev,
                    camera_origin=self.camera_origin,
                    desk_plane=desk_plane,
                    class_id=None if T_init is not None else class_id,
                    method=self.method,
                )
                err = None
            except Exception as exc:
                pose = None
                err = str(exc)
            with self._lock:
                if pose is not None:
                    self.result = pose
                self.error = err
                self.busy = False
            if pose is not None:
                xyz_mm = pose["xyz_mm"]
                rpy = pose["rpy_deg"]
                kind = "추적" if str(pose.get("source", "")).endswith("track") else "등록"
                ok = "ok" if pose.get("aligned_ok") else "miss"
                print(
                    f"CAD {kind} {ok} {self.frame_name} "
                    f"xyz=[{xyz_mm[0]:.1f}, {xyz_mm[1]:.1f}, {xyz_mm[2]:.1f}]  "
                    f"rpy_ext=[{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}]  "
                    f"on={pose.get('scene_frac', 0):.2f} "
                    f"vis={pose.get('vis_frac', 0):.2f} "
                    f"iou={pose.get('xy_iou', 0):.2f}"
                )
                g = self.grasp_xyz_mm
                gr = self.grasp_rpy_deg
                print(
                    f"  grasp(티칭/물체축) xyz=[{g[0]:.1f}, {g[1]:.1f}, {g[2]:.1f}]  "
                    f"rpy=[{gr[0]:.1f}, {gr[1]:.1f}, {gr[2]:.1f}]"
                )
                if pose.get("aligned_ok", False):
                    p_xyz, p_base, p_body = _place_ui_xyzrpy(pose["T"])
                    print(
                        f"  물체 x 축: {p_xyz[0]:.1f} (mm),  y 축: {p_xyz[1]:.1f} (mm),  "
                        f"z 축: {p_xyz[2]:.1f} (mm)"
                    )
                    print(
                        f"  물체 Rx: {p_body[0]:.1f} ,  Ry: {p_body[1]:.1f} ,  Rz: {p_body[2]:.1f}"
                    )
                    print(
                        f"  베이스 기준 Rx: {p_base[0]:.1f} ,  Ry: {p_base[1]:.1f} ,  "
                        f"Rz: {p_base[2]:.1f}"
                    )
            elif err:
                print(f"CAD 등록 실패: {err}")

    def close(self) -> None:
        self._stop = True

    def take_result(self) -> dict | None:
        """완료 결과를 한 번만 소비한다."""
        with self._lock:
            result = self.result
            self.result = None
        return result

    def take_error(self) -> str | None:
        with self._lock:
            err = self.error
            self.error = None
        return err


# 점 간격과 화면. 실행할 때 바꾸지 않는다.
_STRIDE = 1
_PAD = 2
_VOXEL_MM = 1.0
_OVERLAY_VOXEL_MM = 0.5
# YOLO 자세가 이 프레임 수만큼 연속으로 달라야 다시 전체 등록한다(약 0.7초).
_CLASS_CHANGE_FRAMES = 10
_OUTLIER_STD = 1.5
_OUTLIER_NB = 20
# v 키 Open3D 미리보기의 CAD 점 색 (RGB). 맨 위 브라켓은 기본 빨강, 나머지는 같은 빨강 계열의
# 진하고 연한 색을 차례로 쓴다 (몇 번째 브라켓인지 구분만 되게).
_TOP_CAD_RGB = (220, 40, 40)
_OTHER_CAD_RGBS = (
    (140, 15, 15),
    (255, 130, 130),
    (180, 25, 25),
    (255, 90, 90),
    (105, 10, 10),
    (240, 170, 170),
)


def _preview_cloud(instances, cad_pose, others_poses, cad_xyz) -> tuple[np.ndarray, np.ndarray]:
    """v 키 미리보기 점군: 보이는 브라켓 점군 전부 + 정합된 CAD 점 (맨 위 하나 + 나머지).

    instances 의 점군과 자세 T 는 같은 좌표계 (캘리브가 있으면 베이스 mm) 여야 한다.
    cad_xyz 가 None 이면 CAD 는 겹치지 않는다. 정합에 실패한 브라켓은 점군만 나온다.
    """
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    for inst in instances:
        if len(inst.xyz) == 0:
            continue
        rgb = inst.rgb
        if rgb is None or len(rgb) != len(inst.xyz):
            rgb = np.full((len(inst.xyz), 3), 180, dtype=np.uint8)
        xyz_parts.append(np.asarray(inst.xyz, dtype=np.float32))
        rgb_parts.append(np.asarray(rgb, dtype=np.uint8))
    if cad_xyz is not None:
        cads = []
        if cad_pose is not None:
            cads.append((cad_pose["T"], _TOP_CAD_RGB))
        for j, op in enumerate(others_poses or []):
            cads.append((op["T"], _OTHER_CAD_RGBS[j % len(_OTHER_CAD_RGBS)]))
        for T, color in cads:
            aligned = apply_T(np.asarray(cad_xyz, dtype=np.float64), np.asarray(T, dtype=np.float64))
            xyz_parts.append(aligned.astype(np.float32))
            rgb_parts.append(np.tile(np.array([color], dtype=np.uint8), (len(aligned), 1)))
    if not xyz_parts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(xyz_parts), np.vstack(rgb_parts)


_log = runlog.get("camera")


def _cad_outline(T_base: np.ndarray, kcam, w: int, h: int) -> list | None:
    """CAD 자세 → 화면 윤곽 (키포인트 방식과 같은 그림)."""
    from yolo.train import kpt_label as KL

    T_cam = invert_T(kcam.T_base_cam) @ np.asarray(T_base, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(T_cam[:3, :3])
    v, f = KL.mesh_mm()
    pv = cv2.projectPoints(v, rvec, T_cam[:3, 3], kcam.K, kcam.dist)[0].reshape(-1, 2)
    x0, y0 = np.clip(pv.min(axis=0), 0, [w, h])
    x1, y1 = np.clip(pv.max(axis=0), 0, [w, h])
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    # 삼각형을 다 칠한 마스크의 테두리 (4배 해상도로 부드럽게)
    sc = 4.0
    ox, oy = int(x0) - 2, int(y0) - 2
    mask = np.zeros((int((y1 - oy + 4) * sc), int((x1 - ox + 4) * sc)), np.uint8)
    cv2.fillPoly(mask, list(((pv[f] - [ox, oy]) * sc).astype(np.int32)), 255)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2) / sc + [ox, oy] for c in cs]


class _NoFilters:
    """리플레이 때 Orbbec 필터 창 대신."""

    def sync(self, *, force: bool = False) -> None:
        pass


def main() -> None:
    runlog.setup("camera")
    parser = argparse.ArgumentParser(description="YOLO box → RGB-D point ROI")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=DETECT_CONF)
    parser.add_argument(
        "--plane-mm",
        type=float,
        default=None,
        help="책상 평면 이내 점 제거(mm). 안 주면 책상 노이즈를 재서 자동으로 정함",
    )
    parser.add_argument("--no-plane", action="store_true", help="RANSAC 책상 제거 안 함")
    parser.add_argument(
        "--base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="점군을 베이스 좌표(mm)로. 기본 켜짐. 끄려면 --no-base",
    )
    parser.add_argument(
        "--cad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="최상단 ROI vs CAD 등록. 화면에 xyzrpy. 기본 켜짐. 끄려면 --no-cad",
    )
    parser.add_argument(
        "--cad-every",
        type=float,
        default=2.5,
        metavar="SEC",
        help="물체가 움직인 뒤 ICP를 다시 하는 주기(초). 전체 검색은 한 번, c 로만 다시",
    )
    parser.add_argument(
        "--cad-move-mm",
        type=float,
        default=8.0,
        metavar="MM",
        help="점군 중심이 이만큼 움직여야 ICP를 다시 함. 정지 시 축 고정",
    )
    parser.add_argument(
        "--cad-reset-mm",
        type=float,
        default=30.0,
        metavar="MM",
        help="마지막 전체 등록 자리에서 이만큼 움직이면 추적 대신 처음부터 다시 등록",
    )
    parser.add_argument(
        "--no-blobs",
        action="store_true",
        help="YOLO 가 아무것도 못 찾을 때 뎁스 덩어리로 물체 찾기를 끈다",
    )
    parser.add_argument(
        "--register",
        choices=("rest", "class"),
        default="rest",
        help="rest=책상 위 세 자세로 x·y·회전만 찾기(빠름, 자세 클래스 불필요, 안 맞으면 class 로). "
        "class=예전 방식(YOLO 자세 클래스 + ICP)",
    )
    parser.add_argument(
        "--box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="YOLO 박스를 화면에 그림. 기본 켜짐. 끄려면 --no-box",
    )
    parser.add_argument(
        "--no-noise-filter",
        action="store_true",
        help="Orbbec NoiseRemovalFilter 끄기. 기본은 켜짐",
    )
    parser.add_argument(
        "--pose",
        choices=POSE_METHODS,
        default="cad",
        help="자세 방식. cad=뎁스 점군 + CAD 정합 (기본), kpt=키포인트 (컬러만). 창에서 k 로 바꾼다",
    )
    parser.add_argument(
        "--kpt-weights",
        type=Path,
        default=None,
        help="키포인트 모델. 기본: 학생이 학습한 yolo/weights/kpt_best.pt (없으면 키포인트 방식을 못 쓴다)",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="카메라 대신 장면 폴더 (rgb.png · depth_mm.npy · camera.json). n / p 로 넘긴다",
    )
    parser.add_argument(
        "--replay-noise",
        action="store_true",
        help="리플레이 뎁스에 실물 같은 노이즈를 넣는다 (시뮬 장면용)",
    )
    args = parser.parse_args()

    replay = None if args.replay is None else ReplaySource(args.replay, noise=args.replay_noise)
    rotate = ROTATE_180 and replay is None
    weights = args.weights if args.weights is not None else box_best_pt()
    if not Path(weights).exists():
        # 수업 흐름에 맞춰 학생이 학습한 모델로만 켠다 (기본 모델로 대신 켜지 않는다)
        raise SystemExit(
            f"박스 모델이 없습니다: {weights}\n"
            "5차시 박스 학습 (python yolo/train/train.py) 뒤에 켜집니다."
        )

    if replay is not None:
        K, T_replay = replay.camera()
        k_path = args.replay / "camera.json"
    else:
        K, k_path = load_K()
    if rotate:
        K, _ = intrinsics_for_rotate180(K, None, FRAME_WIDTH, FRAME_HEIGHT)
        print(
            f"K {k_path}  fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}  "
            f"cx={K[0, 2]:.1f} cy={K[1, 2]:.1f} (180°)"
        )
    else:
        print(f"K {k_path}  fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}")
    if K[0, 0] > 2000:
        print("주의: fx가 해상도 640에 비해 큽니다. 렌즈 캘리브를 다시 보는 편이 좋습니다.")

    T_bc = None
    frame_name = "camera_mm"
    if args.base:
        if replay is not None:
            T_bc, t_path = T_replay, k_path
        else:
            T_bc, t_path = load_T_base_cam()
        frame_name = "base_mm"
        print(f"T_base_cam {t_path}")
        _log.info("손눈 캘리브 %s", t_path)

    noise_min_diff = NOISE_MIN_DIFF_DEFAULT
    noise_max_size = NOISE_MAX_SIZE_DEFAULT

    # 카메라보다 먼저 로드. 스트림 중 긴 로드는 OpenNI 큐 폭주·USB 단절을 부른다.
    print("YOLO 로드 중...")
    from ultralytics import YOLO

    device = _device()
    model = YOLO(str(weights))
    _log.info("YOLO 가중치 %s  device=%s", weights, device)

    keys = "v=3D  s=save  r=plane  q=quit"
    if args.cad:
        keys = "c=등록  " + keys
    print(f"class={CLASS_NAME}  device={device}  {keys}")
    last_xyz = np.zeros((0, 3), dtype=np.float32)
    last_rgb = np.zeros((0, 3), dtype=np.uint8)
    plane = None
    plane_mm = PLANE_HEIGHT_MIN_MM if args.plane_mm is None else float(args.plane_mm)
    refit_plane = not args.no_plane
    last_pose = None
    cad_pose = None
    viewer_process = None
    cad_worker = None
    cad_others = None  # 맨 위 말고 나머지 브라켓 CAD 정합
    others_poses: list[dict] = []
    last_others_t = 0.0
    last_cad_t = 0.0
    last_cad_anchor = None
    cad_full_done = False
    # 마지막 전체 등록을 보낸 자리와 YOLO 자세. 많이 옮기거나 자세가 바뀌면 다시 전체 등록한다.
    last_full_anchor = None
    last_full_class = None
    # YOLO 자세는 프레임마다 흔들린다. 같은 새 자세가 이만큼 이어져야 바뀐 것으로 본다.
    class_streak = 0
    pending_req_id: str | None = None
    pending_req_extra: dict = {}
    seen_req_id = read_request()
    if args.cad:
        print("CAD 로드 중...")
        T_cad_grasp, grasp_xyz_mm, grasp_rpy_deg = _load_grasp_cad()
        print(
            f"obj(티칭) xyz=[{grasp_xyz_mm[0]:.1f}, {grasp_xyz_mm[1]:.1f}, {grasp_xyz_mm[2]:.1f}]  "
            f"rpy=[{grasp_rpy_deg[0]:.1f}, {grasp_rpy_deg[1]:.1f}, {grasp_rpy_deg[2]:.1f}]"
        )
        cad_worker = _CadWorker(
            voxel_mm=_VOXEL_MM,
            flip=bool(args.base),
            tries=4,
            camera_origin=None if T_bc is None else T_bc[:3, 3],
            T_cad_grasp=T_cad_grasp,
            grasp_xyz_mm=grasp_xyz_mm,
            grasp_rpy_deg=grasp_rpy_deg,
            frame_name=frame_name,
            method=args.register,
        )
        cad_others = _CadOthers(
            voxel_mm=_VOXEL_MM,
            flip=bool(args.base),
            tries=4,
            camera_origin=None if T_bc is None else T_bc[:3, 3],
            method=args.register,
            move_mm=args.cad_move_mm,
            reset_mm=args.cad_reset_mm,
        )
        if args.base:
            print(
                (
                    "CAD: 책상 위 세 자세(세우기/눕히기/비스듬히) 중에서 x·y·회전을 찾는다. "
                    "안 맞으면 YOLO 자세 + ICP 로 한 번 더. "
                    if args.register == "rest"
                    else "CAD: YOLO 자세(세우기/눕히기/비스듬히)로 한 번 맞춘다. "
                )
                + "맞으면 축을 고정하고, "
                f"{args.cad_move_mm:.0f}mm 이상 움직이면 ICP. "
                "클래스 후보가 모두 실패하면 전체 검색 한 번. "
                f"{args.cad_reset_mm:.0f}mm 이상 옮기거나 자세가 바뀌거나 추적이 실패하면 처음부터. "
                "miss면 옮길 때까지 멈춤. c 는 처음부터."
            )
        else:
            print(
                "CAD: 베이스가 아니라 클래스 각도를 쓰지 않는다. "
                "전체 검색은 한 번. 맞으면 축을 고정하고, "
                f"{args.cad_move_mm:.0f}mm 이상 움직이면 ICP. "
                "miss면 멈춤. c 는 처음부터."
            )
        print(
            f"밀도  stride={_STRIDE}  등록={_VOXEL_MM:g}mm  "
            f"표시={_OVERLAY_VOXEL_MM:g}mm"
        )
    else:
        T_cad_grasp = np.eye(4, dtype=np.float64)
        grasp_xyz_mm = np.zeros(3, dtype=np.float64)
        grasp_rpy_deg = np.zeros(3, dtype=np.float64)

    # 키포인트 방식도 카메라보다 먼저 로드한다 (스트림 중 긴 로드는 USB 단절을 부른다). 손눈이 있어야 쓴다.
    method = args.pose
    kpt_worker = None
    kcam = None
    kpt_poses: list[dict] = []
    kpt_target: int | None = None
    # 키포인트도 학생이 학습한 모델로만 쓴다 (기본 모델로 대신 켜지 않는다). 다른 모델은 --kpt-weights 로.
    kpt_weights = args.kpt_weights if args.kpt_weights is not None else KPT_BEST_PT
    kpt_why = ""  # 키포인트 방식을 못 쓰는 이유 (k 를 눌렀을 때 안내)
    if T_bc is None:
        kpt_why = "손눈 캘리브가 없습니다 (4차시)."
    elif not Path(kpt_weights).is_file():
        kpt_why = (
            f"키포인트 모델이 없습니다: {kpt_weights}\n"
            "6차시 키포인트 학습 (python yolo/train/kpt_train.py) 뒤에 쓸 수 있습니다."
        )
    else:
        kcam = kpt_camera(K, T_bc, dist=np.zeros(5) if replay is not None else None)
        try:
            print("키포인트 모델 로드 중...")
            kpt_worker = KptWorker(kcam, kpt_weights)
            print(f"키포인트 {kpt_worker.weights}")
        except (FileNotFoundError, OSError) as exc:
            kpt_why = f"키포인트 모델을 불러오지 못했습니다: {exc}"
    if kpt_why:
        print(f"키포인트 방식을 쓸 수 없습니다. {kpt_why}")
    if method == "kpt" and kpt_worker is None:
        raise SystemExit(f"--pose kpt 를 켤 수 없습니다. {kpt_why}")
    print(f"자세 방식: {METHOD_KO[method]}  (k 로 바꾸기, i 검사)")

    if replay is not None:
        cam = replay
        print(f"리플레이 {args.replay}  장면 {len(replay.scenes)}개  (n / p 넘기기)")
    else:
        cam = open_orbbec(
            noise_filter=not args.no_noise_filter,
            noise_min_diff=noise_min_diff,
            noise_max_size=noise_max_size,
            hole_filter=True,
        )
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    if replay is not None:
        filters = _NoFilters()
    else:
        filters = OrbbecFilterPanel(
            cam,
            noise_on=not args.no_noise_filter,
            min_diff=noise_min_diff,
            max_size=noise_max_size,
        )

    def current_poses() -> list[np.ndarray]:
        """지금 방식으로 구한 브라켓 자세들 (베이스 mm). CAD 정합은 맨 위 하나 + 나머지."""
        if method == "kpt":
            return [kp["T"] for kp in kpt_poses]
        poses = [op["T"] for op in others_poses]
        if cad_pose is not None and cad_pose.get("aligned_ok", False):
            poses = [cad_pose["T"]] + poses
        return poses

    def objects_for_request(bgr: np.ndarray, poses: list[np.ndarray]) -> list[dict]:
        """자세들 → 검사 (테두리 맞춤 · 흰/검 · 양/불) → 펜던트 숫자 목록. 브라켓 아님은 빼고, 로봇에서 가까운 순."""
        out = []
        for r in judge_all(bgr, poses, kcam):
            if r["verdict"] == "브라켓 아님":
                continue
            xyz_ui, rpy_ui, _body = _place_ui_xyzrpy(r["T_base"])
            out.append({
                "xyz_mm": [round(float(v), 2) for v in xyz_ui],
                "rpy_deg": [round(float(v), 2) for v in rpy_ui],
                "pose": r["pose"],
                "color": "흰" if r["white"] else "검",
                "verdict": r["verdict"],
            })
        out.sort(key=lambda o: float(np.hypot(o["xyz_mm"][0], o["xyz_mm"][1])))
        _log.info("브라켓 목록 보냄  %s", [(o["verdict"], o["xyz_mm"][:2]) for o in out])
        return out

    try:
        while True:
            cam.flush()
            bgr, depth = cam.grab(timeout_ms=800)
            if bgr is None or depth is None:
                vis = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(vis, "no RGB-D", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow(WIN, vis)
                filters.sync()
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            if rotate:
                bgr, depth = rotate180(bgr, depth)
            if depth.shape[:2] != bgr.shape[:2]:
                depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

            raw = bgr.copy()  # 그리기 전 영상 (키포인트 · 검사 · 저장용)
            h, w = bgr.shape[:2]
            if not args.no_plane and refit_plane:
                full = np.ones((h, w), dtype=bool)
                table_xyz, _ = points_from_mask(depth, K, full, stride=5)
                try:
                    plane, inliers = fit_plane_ransac(table_xyz)
                    extra = ""
                    if T_bc is not None:
                        desk = transform_plane(plane, T_bc)
                        extra = (
                            f"  vs 로봇 XY {plane_tilt_from_z_deg(desk):.2f}°"
                        )
                    print(
                        "책상 평면 "
                        f"[{plane[0]:.4f}, {plane[1]:.4f}, {plane[2]:.4f}, "
                        f"{plane[3]:.1f}]  inliers={inliers}{extra}"
                    )
                    if args.plane_mm is None:
                        noise = plane_noise_mm(table_xyz, plane)
                        plane_mm = auto_plane_height_mm(noise)
                        noise_txt = "?" if noise is None else f"{noise:.1f}"
                        print(f"책상 노이즈 p99={noise_txt}mm → 책상 제거 {plane_mm:.1f}mm (자동)")
                        _log.info("책상 노이즈 p99=%smm  책상 제거 %.1fmm (자동)", noise_txt, plane_mm)
                except ValueError as exc:
                    plane = None
                    print(f"책상 평면 실패: {exc}. r로 다시 시도하세요.")
                refit_plane = False

            results = model.predict(
                bgr, conf=args.conf, device=device, verbose=False, imgsz=640
            )
            result = results[0]
            depth_vis = colorize_depth(depth)
            blob_boxes = None
            if not args.no_blobs and plane is not None and not _has_dets(result):
                # 새 책상의 흰 브라켓은 예전 YOLO 가 못 찾는다. 흰색 YOLO 학습 전까지 뎁스 덩어리로.
                blob_boxes = depth_blob_boxes(depth, K, plane, plane_mm, T_bc)
            instances = collect_instances(
                depth=depth,
                bgr=bgr,
                K=K,
                result=result,
                plane=plane,
                pad=_PAD,
                plane_mm=plane_mm,
                stride=_STRIDE,
                T_base_cam=T_bc,
                xyxys=blob_boxes,
            )
            chosen = select_topmost(instances)
            n_det = len(instances)
            cad_class_id = None
            if chosen is not None and T_bc is not None:
                cad_class_id = instances[chosen].class_id
            last_pose = None
            if chosen is None:
                last_xyz = np.zeros((0, 3), dtype=np.float32)
                last_rgb = np.zeros((0, 3), dtype=np.uint8)
            else:
                pick = instances[chosen]
                last_xyz, last_rgb = pick.xyz, pick.rgb
                last_pose = local_pose(
                    pick.xyz,
                    desk_plane=None if T_bc is not None else plane,
                    band_mm=SLICE_MM,
                    in_base=T_bc is not None,
                )

            if method == "kpt" and kpt_worker is not None:
                kpt_worker.submit(raw)
                res = kpt_worker.take_result()
                if res is not None:
                    kpt_poses = res
                kpt_target = pick_target(
                    kpt_poses, None if chosen is None else instances[chosen].xyxy
                )

            if cad_others is not None and method == "cad":
                res = cad_others.take_result()
                if res is not None:
                    others_poses = res
                    if kcam is not None:
                        # 윤곽은 결과가 올 때 한 번만 그려 둔다
                        for op in others_poses:
                            op["outline"] = _cad_outline(op["T"], kcam, w, h)
                now_o = time.monotonic()
                if (now_o - last_others_t) >= max(0.2, float(args.cad_every)) and not cad_others.busy:
                    clouds = [
                        (inst.xyz, inst.class_id)
                        for i, inst in enumerate(instances)
                        if i != chosen and len(inst.xyz) >= 20
                    ]
                    if cad_others.submit(clouds, _work_desk_plane(plane, T_bc)):
                        last_others_t = now_o
                # 맨 위 브라켓이 바뀌어 겹쳐 보이는 것은 뺀다
                if len(last_xyz) >= 20:
                    top_anchor = _median_xyz(last_xyz)
                    others_poses = [
                        op for op in others_poses
                        if float(np.linalg.norm(op["anchor"] - top_anchor)) >= 15.0
                    ]

            if cad_worker is not None:
                nxt = cad_worker.take_result()
                if nxt is not None:
                    tracked = str(nxt.get("source", "")).endswith("track")
                    # 실패한 전체 검색은 숫자·축에 내지 않고, 다시 돌리지 않는다.
                    if nxt.get("aligned_ok", False):
                        cad_pose = nxt
                        cad_full_done = True
                        if len(last_xyz) >= 20:
                            last_cad_anchor = _median_xyz(last_xyz)
                    elif tracked and cad_pose is not None:
                        # 옛 축을 계속 보이면 물체를 옮긴 뒤에도 옛 자리가 남는다.
                        cad_pose = None
                        last_cad_anchor = None
                        cad_full_done = False
                        print("CAD 추적 miss. 처음부터 다시 등록합니다.")
                    else:
                        cad_pose = None
                        last_cad_anchor = None
                        cad_full_done = True
                        print("CAD miss. 전체 검색은 멈춤. 처음부터는 c.")
                    if pending_req_id is not None and not tracked:
                        _log.info(
                            "등록 끝  성공=%s  겹침=%.3f  오차=%.2fmm",
                            bool(nxt.get("aligned_ok", False)),
                            float(nxt.get("fitness", 0.0) or 0.0),
                            float(nxt.get("inlier_rmse", 0.0) or 0.0),
                        )
                        if nxt.get("aligned_ok", False) and T_bc is not None:
                            xyz_ui, rpy_ui, _body_ui = _place_ui_xyzrpy(nxt["T"])
                            write_status(
                                pending_req_id, "ok", xyz=xyz_ui, rpy=rpy_ui, extra=pending_req_extra
                            )
                        else:
                            write_status(pending_req_id, "fail")
                        pending_req_id = None
                elif pending_req_id is not None and not cad_worker.busy:
                    cad_worker.take_error()
                    write_status(pending_req_id, "fail")
                    pending_req_id = None
                now = time.monotonic()
                due = (now - last_cad_t) >= max(0.2, float(args.cad_every))
                pose_ok = cad_pose is not None and cad_pose.get("aligned_ok", False)
                if (
                    cad_class_id is not None
                    and last_full_class is not None
                    and cad_class_id != last_full_class
                ):
                    class_streak += 1
                else:
                    class_streak = 0
                # 키포인트 방식일 때는 CAD 자동 등록을 쉬어 젯슨 계산을 아낀다
                if len(last_xyz) >= 20 and due and method == "cad":
                    desk = _work_desk_plane(plane, T_bc)
                    anchor = _median_xyz(last_xyz)
                    far = last_full_anchor is not None and float(
                        np.linalg.norm(anchor - last_full_anchor)
                    ) >= max(1.0, float(args.cad_reset_mm))
                    class_changed = class_streak >= _CLASS_CHANGE_FRAMES
                    full = False
                    if pose_ok and (far or class_changed):
                        full = True
                    elif pose_ok:
                        moved = (
                            last_cad_anchor is None
                            or float(np.linalg.norm(anchor - last_cad_anchor))
                            >= max(1.0, float(args.cad_move_mm))
                        )
                        if moved and cad_worker.submit(
                            last_xyz,
                            T_init=cad_pose["T"],
                            T_prev=cad_pose["T"],
                            desk_plane=desk,
                        ):
                            last_cad_t = now
                    else:
                        # 실패한 자리에서는 반복하지 않고, 옮기거나 자세가 바뀌면 다시 해 본다.
                        full = not cad_full_done or far or class_changed
                    if full and cad_worker.submit(
                        last_xyz,
                        T_init=None,
                        T_prev=None,
                        desk_plane=desk,
                        class_id=cad_class_id,
                    ):
                        last_cad_t = now
                        cad_full_done = True
                        last_full_anchor = anchor
                        last_full_class = cad_class_id
                        class_streak = 0
                        if far:
                            print("CAD 물체를 옮김. 처음부터 다시 등록")
                        elif class_changed:
                            print("CAD 자세가 바뀜. 처음부터 다시 등록")

            for i, inst in enumerate(instances):
                x1, y1, x2, y2 = [int(v) for v in inst.xyxy[:4]]
                color = (0, 255, 255) if i == chosen else (0, 180, 0)
                tag_x, tag_y = x1, max(16, y1 - 6)
                shown_class = inst.class_id
                if shown_class is None and i == chosen and cad_pose is not None:
                    # 뎁스 덩어리 상자엔 YOLO 자세가 없다. 정합이 찾은 자세를 쓴다.
                    shown_class = cad_pose.get("class_id")
                pose = (
                    pose_name(shown_class, korean=True)
                    if shown_class is not None
                    else "물체"
                )
                pose_color = _POSE_BGR.get(shown_class, color)
                if args.box:
                    # 고른(가장 위) 물체는 굵게
                    cv2.rectangle(
                        bgr, (x1, y1), (x2, y2), pose_color, 2 if i == chosen else 1, cv2.LINE_AA
                    )
                put_text(bgr, pose, (tag_x, tag_y), 0.6, pose_color, 1)

            ztxt = ""
            if len(last_xyz):
                ztxt = f"  zmed={float(np.median(last_xyz[:, 2])):.0f}"
            ctxt = ""
            if last_pose is not None:
                c = last_pose["center_mm"]
                ctxt = f"  c={c[0]:.0f},{c[1]:.0f}"
            if plane is None:
                plane_text = "plane=off"
            else:
                auto = "auto" if args.plane_mm is None else ""
                plane_text = f"plane>{plane_mm:.1f}mm{auto}"
            hint = "v=3D s=save r=plane q=quit"
            if args.cad:
                hint = "c=CAD " + hint
            if kpt_worker is not None:
                hint = "k=방식 i=검사 " + hint
            if replay is not None:
                hint += f" n/p {replay.idx + 1}/{len(replay.scenes)}"
            pose_txt = ""
            if chosen is not None and instances[chosen].class_id is not None:
                pose_txt = "  " + pose_name(instances[chosen].class_id, korean=True)
            put_text(
                bgr,
                f"n={n_det} sel={0 if chosen is None else chosen}{pose_txt}"
                f"{ztxt}{ctxt} {plane_text} {frame_name} {hint}",
                (8, 24),
                0.45,
                (0, 255, 0),
                1,
            )
            if args.cad or method == "kpt":
                place_font = 0.58
                place_thick = 1
                if method == "kpt":
                    if kpt_target is not None:
                        p_xyz, p_base, p_body = _place_ui_xyzrpy(kpt_poses[kpt_target]["T"])
                        xyz_line = (
                            f"물체 x 축: {p_xyz[0]:.1f} (mm),  y 축: {p_xyz[1]:.1f} (mm),  "
                            f"z 축: {p_xyz[2]:.1f} (mm)"
                        )
                        body_line = (
                            f"물체 Rx: {p_body[0]:.1f} ,  Ry: {p_body[1]:.1f} ,  "
                            f"Rz: {p_body[2]:.1f}"
                        )
                        base_line = (
                            f"베이스 기준 Rx: {p_base[0]:.1f} ,  Ry: {p_base[1]:.1f} ,  "
                            f"Rz: {p_base[2]:.1f}"
                        )
                    else:
                        xyz_line = "물체 x/y/z 축: 브라켓 없음"
                        body_line = "물체 Rx/Ry/Rz: -"
                        base_line = "베이스 기준 Rx/Ry/Rz: -"
                    cad_line = (
                        f"[키포인트] 브라켓 {len(kpt_poses)}개  "
                        f"{kpt_worker.seconds * 1000:.0f} ms/장  (k=CAD 정합)"
                    )
                elif cad_pose is not None:
                    on = cad_pose.get("scene_frac")
                    vis = cad_pose.get("vis_frac")
                    iou = cad_pose.get("xy_iou")
                    ok = cad_pose.get("aligned_ok", False)
                    p_xyz, p_base, p_body = _place_ui_xyzrpy(cad_pose["T"])
                    xyz_line = (
                        f"물체 x 축: {p_xyz[0]:.1f} (mm),  y 축: {p_xyz[1]:.1f} (mm),  "
                        f"z 축: {p_xyz[2]:.1f} (mm)"
                    )
                    body_line = (
                        f"물체 Rx: {p_body[0]:.1f} ,  Ry: {p_body[1]:.1f} ,  "
                        f"Rz: {p_body[2]:.1f}"
                    )
                    base_line = (
                        f"베이스 기준 Rx: {p_base[0]:.1f} ,  Ry: {p_base[1]:.1f} ,  "
                        f"Rz: {p_base[2]:.1f}"
                    )
                    cad_line = (
                        f"CAD/{frame_name} on={on:.2f}" if on is not None else f"CAD/{frame_name}"
                    )
                    if vis is not None:
                        cad_line += f" vis={vis:.2f}"
                    if iou is not None:
                        cad_line += f" iou={iou:.2f}"
                    if not ok:
                        cad_line += "  MISS"
                elif cad_worker is not None and cad_worker.busy:
                    xyz_line = "물체 x/y/z 축: 등록 중 ..."
                    body_line = "물체 Rx/Ry/Rz: ..."
                    base_line = "베이스 기준 Rx/Ry/Rz: ..."
                    cad_line = "CAD ..."
                else:
                    xyz_line = "물체 x/y/z 축: 대기 (점군 나오면 시작)"
                    body_line = "물체 Rx/Ry/Rz: 대기"
                    base_line = "베이스 기준 Rx/Ry/Rz: 대기"
                    cad_line = "CAD 대기"
                if method == "cad" and others_poses:
                    cad_line += f"  외 {len(others_poses)}개"
                if method == "cad" and kpt_worker is not None:
                    cad_line = "[CAD 정합] " + cad_line + "  (k=키포인트)"
                overlay_lines = [
                    (50, xyz_line, (0, 255, 255)),
                    (78, body_line, (0, 255, 255)),
                    (106, base_line, (0, 255, 255)),
                    (134, cad_line, (180, 180, 80)),
                ]
                for y, text, color in overlay_lines:
                    if not text:
                        continue
                    put_text(bgr, text, (8, y), place_font, color, place_thick)
            if method == "kpt":
                for j, kp in enumerate(kpt_poses):
                    thick = 2 if j == kpt_target else 1
                    for c in kp["outline"] or []:
                        cv2.polylines(bgr, [c.astype(np.int32)], True, (255, 200, 0), thick, cv2.LINE_AA)
                if kpt_target is not None:
                    T_draw = invert_T(T_bc) @ np.asarray(kpt_poses[kpt_target]["T"], dtype=np.float64)
                    _draw_cad_axes(bgr, T_draw, K)
                    _draw_cad_axes(depth_vis, T_draw, K)
            elif cad_pose is not None or others_poses:
                for op in others_poses:
                    for c in op.get("outline") or []:
                        cv2.polylines(bgr, [c.astype(np.int32)], True, (255, 200, 0), 1, cv2.LINE_AA)
                    T_o = np.asarray(op["T"], dtype=np.float64)
                    if T_bc is not None:
                        T_o = invert_T(T_bc) @ T_o
                    _draw_cad_axes(bgr, T_o, K)
                    _draw_cad_axes(depth_vis, T_o, K)
            if method == "cad" and cad_pose is not None:
                T_draw = np.asarray(cad_pose["T"], dtype=np.float64)
                if T_bc is not None:
                    T_draw = invert_T(T_bc) @ T_draw
                _draw_cad_axes(bgr, T_draw, K)
                _draw_cad_axes(depth_vis, T_draw, K)
            filters.sync()
            panel = np.hstack([bgr, depth_vis])
            cv2.imshow(WIN, panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and not args.no_plane:
                refit_plane = True
            if replay is not None and key in (ord("n"), ord("p")):
                replay.step(1 if key == ord("n") else -1)
                refit_plane = not args.no_plane
                kpt_poses, kpt_target = [], None
                others_poses = []
                print(f"장면 {replay.name}")
            if key == ord("k"):
                if kpt_worker is None:
                    print(f"키포인트 방식을 쓸 수 없습니다. {kpt_why}")
                else:
                    method = "kpt" if method == "cad" else "cad"
                    kpt_poses, kpt_target = [], None
                    print(f"자세 방식: {METHOD_KO[method]}")
            if key == ord("i"):
                if kcam is None:
                    print("검사는 손눈 캘리브가 있어야 합니다.")
                else:
                    poses = current_poses()
                    if not poses:
                        print("검사할 자세가 없습니다.")
                    else:
                        judged = judge_all(raw, poses, kcam)
                        title = f"검사 ({METHOD_KO[method]} 자세). 아무 키나 → 다시"
                        print_judged(judged, title)
                        cv2.imshow(WIN, np.hstack([draw_judged(raw, judged, title), depth_vis]))
                        cv2.waitKey(0)
            incoming, want_all = read_request_full()
            req_id = None
            if incoming and incoming != seen_req_id:
                seen_req_id = incoming
                req_id = incoming
            if req_id is not None and want_all:
                # 모든 브라켓 (load_all_pick_targets): 지금 방식의 자세들을 검사까지 해서 목록으로 보낸다
                if kcam is None:
                    write_status(req_id, "fail")
                else:
                    write_status(req_id, "run")
                    poses = current_poses()
                    objects = [] if not poses else objects_for_request(raw, poses)
                    if objects:
                        write_status(req_id, "ok", objects=objects)
                    else:
                        write_status(req_id, "no_box")
                req_id = None
            # 런처 「카메라로 집기」 요청 (구역 · 가까운 순 조건) 은 키포인트 방식이어도 아래 CAD 정합으로 처리한다.
            if req_id is not None and method == "kpt" and not read_request_options().active:
                # 키포인트 자세: 이미 나와 있으니 테두리 맞춤으로 다듬어 바로 보낸다
                if kpt_target is None:
                    write_status(req_id, "no_box")
                else:
                    T_send = refine_pose(raw, kpt_poses[kpt_target]["T"], kcam)
                    xyz_ui, rpy_ui, _body_ui = _place_ui_xyzrpy(T_send)
                    write_status(req_id, "ok", xyz=xyz_ui, rpy=rpy_ui)
                    _log.info("키포인트 자세 보냄  xyz=%s", np.round(xyz_ui, 1).tolist())
                req_id = None
            req_T_prev = None if cad_pose is None else cad_pose["T"]
            req_extra: dict = {}
            if req_id is not None and T_bc is not None:
                opts = read_request_options()
                if opts.active:
                    # 요청의 조건 (제외 원 · 구역 · 가까운 순) 으로 다시 고른다.
                    # 고른 물체가 바뀌면 지금 추적 중인 자세는 쓰지 않는다.
                    order = ordered_targets(
                        instances, exclude=opts.exclude, region=opts.region, nearest=opts.nearest
                    )
                    picked = order[0][0] if order else None
                    if picked != chosen:
                        req_T_prev = None
                    chosen = picked
                    req_extra["targets_xy_mm"] = [[round(x, 1), round(y, 1)] for _, x, y in order]
                    if chosen is None:
                        last_xyz = np.zeros((0, 3), dtype=np.float32)
                        cad_class_id = None
                    else:
                        last_xyz, last_rgb = instances[chosen].xyz, instances[chosen].rgb
                        cad_class_id = instances[chosen].class_id
                        # 흰 · 검 판정용 밝기 (점 색의 채널 평균, 중앙값).
                        req_extra["gray"] = round(float(np.median(np.asarray(last_rgb, dtype=np.float64).mean(axis=1))), 1)
            if req_id is not None and cad_worker is None:
                write_status(req_id, "fail")
            elif (key == ord("c") and cad_worker is not None) or req_id is not None:
                if cad_worker.busy:
                    if req_id is not None:
                        write_status(req_id, "busy")
                    else:
                        print("이미 등록 중입니다.")
                elif req_id is not None and chosen is None:
                    write_status(req_id, "no_box", extra=req_extra)
                elif len(last_xyz) < 20:
                    if req_id is not None:
                        write_status(req_id, "fail")
                    else:
                        print("등록할 점이 없습니다.")
                elif not cad_worker.submit(
                    last_xyz,
                    T_init=None,
                    T_prev=req_T_prev,
                    desk_plane=_work_desk_plane(plane, T_bc),
                    class_id=cad_class_id,
                ):
                    if req_id is not None:
                        write_status(req_id, "busy")
                    else:
                        print("이미 등록 중입니다.")
                else:
                    last_cad_t = time.monotonic()
                    last_cad_anchor = None
                    cad_full_done = True
                    last_full_anchor = _median_xyz(last_xyz)
                    last_full_class = cad_class_id
                    if req_id is not None:
                        pending_req_id = req_id
                        pending_req_extra = req_extra
                        write_status(req_id, "run")
                    print("CAD 처음부터 다시 등록")
            if key == ord("v"):
                if len(last_xyz) == 0:
                    zok = instances[0].n_zok if instances else 0
                    if zok == 0:
                        print(
                            "표시할 점이 없습니다. 박스 안 뎁스가 0입니다 "
                            "(센서 홀). 흰 종이보다 회색 매트가 낫습니다."
                        )
                    else:
                        print(
                            f"표시할 점이 없습니다. 뎁스 {zok}px 있었으나 "
                            "책상 제거 뒤에 남은 점이 없습니다."
                        )
                    continue
                if viewer_process is not None and viewer_process.poll() is None:
                    print("Open3D 창이 이미 열려 있습니다.")
                    continue
                ROI_DIR.mkdir(parents=True, exist_ok=True)
                preview = ROI_DIR / f".preview_{frame_name}.ply"
                # 보이는 브라켓 전부. CAD 는 맨 위 (기본 빨강) + 나머지 (빨강의 진하고 연한 색)
                v_others = others_poses if method == "cad" else []
                cad_vis = None
                if cad_pose is not None or v_others:
                    from yolo.pose.register import load_cad_xyz

                    cad_vis = load_cad_xyz(voxel_mm=_OVERLAY_VOXEL_MM)
                o_xyz, o_rgb = _preview_cloud(instances, cad_pose, v_others, cad_vis)
                write_ply(preview, o_xyz, o_rgb)
                if v_others:
                    print(f"Open3D: 맨 위 CAD 1개 + 나머지 {len(v_others)}개 (빨강 진하기로 구분)")
                view_cmd = [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "view" / "view_cloud.py"),
                    str(preview),
                    "--no-obb",
                ]
                if cad_pose is not None:
                    # T[0,0] 이 음수면 "--cad-T" 다음 토큰이 새 옵션으로 파싱된다.
                    view_cmd.append(
                        "--cad-T="
                        + ",".join(f"{x:.9g}" for x in np.asarray(cad_pose["T"]).reshape(-1))
                    )
                if plane is not None:
                    desk = (
                        transform_plane(plane, T_bc)
                        if T_bc is not None
                        else np.asarray(plane, dtype=np.float64)
                    )
                    view_cmd.append(
                        "--desk-plane="
                        + ",".join(f"{x:.9g}" for x in np.asarray(desk).reshape(-1))
                    )
                view_cmd.extend(
                    [
                        "--point-size",
                        "2",
                        "--outlier-std",
                        str(_OUTLIER_STD),
                        "--outlier-nb",
                        str(_OUTLIER_NB),
                    ]
                )
                viewer_process = subprocess.Popen(view_cmd)
                print(f"Open3D: {preview}")
            if key == ord("s"):
                if len(last_xyz) == 0:
                    print("저장할 점이 없습니다.")
                    continue
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 리플레이로 다시 볼 수 있게 컬러 + 뎁스 장면도 남긴다
                save_scene(ROI_DIR / "scenes" / f"scene_{stamp}", raw, depth, K, T_bc)
                print(f"장면 저장 {ROI_DIR / 'scenes' / f'scene_{stamp}'}")
                out = ROI_DIR / f"roi_{stamp}_{frame_name}.ply"
                write_ply(out, last_xyz, last_rgb)
                print(f"저장 {out}  n={len(last_xyz)}")
                if last_pose is not None:
                    slice_xyz = last_pose.get("sliced_xyz")
                    if slice_xyz is not None and len(slice_xyz):
                        slice_out = out.with_name(f"{out.stem}_slice.ply")
                        write_ply(slice_out, slice_xyz)
                        print(f"저장 {slice_out}  n={len(slice_xyz)}")
                    c = last_pose["center_mm"]
                    print(
                        f"local_pose yaw={last_pose['yaw_deg']:.1f} "
                        f"center={np.round(c, 1).tolist()} mm  "
                        f"n={last_pose['n_points']}"
                    )
    finally:
        cam.close()
        if cad_worker is not None:
            cad_worker.close()
        if cad_others is not None:
            cad_others.close()
        if kpt_worker is not None:
            kpt_worker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
