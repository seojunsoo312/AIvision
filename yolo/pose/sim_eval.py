"""시뮬 장면에서 지금 정합 코드를 돌려 정답과 비교한다.

뎁스 처리는 실제 카메라 창(yolo/pose/roi_cloud.py, --base --cad)과 같은 순서다.
  책상 RANSAC → 책상 노이즈로 제거 높이 자동 → YOLO 대신 정답 분할의 외접 사각형 →
  collect_instances (책상 빼기 · 가장자리 빼기 · 가장 큰 덩어리 · 베이스 변환) → register_pose

    python -m yolo.pose.sim_eval --n 20                    # 노이즈 끔, 정답 자세 클래스
    python -m yolo.pose.sim_eval --n 100 --noise --cls none --workers 4
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vision.sim.data import OUT_DIR, Scene, load_scene
from vision.sim.noise import NO_NOISE, NoiseConfig, add_noise
from vision.sim.rgb_noise import (
    CLEAN,
    PRESETS,
    REAL_DIST,
    RgbConfig,
    degrade_rgb,
    distort,
)
from vision.sim.symmetry import cad_symmetries

# 성공 기준: 위치(CAD 중심) 오차와 회전 오차 둘 다 이하.
SUCCESS_POS_MM = 3.0
SUCCESS_ROT_DEG = 5.0

# 실제 카메라 창 값 (roi_cloud.py).
PAD_PX = 2
STRIDE = 1
VOXEL_MM = 1.0
TRIES = 4


# ---- YOLO 흉내 ----


class _Det:
    """ultralytics 박스 하나처럼 보이는 객체 (xyxy, cls)."""

    def __init__(self, xyxy, cls_id):
        self.xyxy = [np.asarray(xyxy, dtype=np.float32)]
        self.cls = None if cls_id is None else np.asarray([float(cls_id)])


class _Result:
    def __init__(self, dets):
        self.boxes = dets
        self.obb = None


def seg_box(seg: np.ndarray, obj_id: int) -> list[float] | None:
    """분할 마스크의 외접 사각형 (YOLO 박스 대신)."""
    ys, xs = np.nonzero(seg == obj_id)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


# ---- 오차 ----


def pose_error(T_est: np.ndarray, T_gt: np.ndarray, cad_center: np.ndarray) -> tuple[float, float]:
    """(위치 오차 mm, 회전 오차 도). 대칭으로 같은 자세 중 가장 가까운 것과 비교한다.

    위치는 CAD 원점이 아니라 CAD 중심으로 잰다 (원점은 모서리에 있어 회전 오차가 위치로 번진다).
    """
    best = (np.inf, np.inf)
    c = np.r_[cad_center, 1.0]
    for S in cad_symmetries():
        G = T_gt @ S
        R = T_est[:3, :3].T @ G[:3, :3]
        rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
        pos = float(np.linalg.norm((T_est @ c)[:3] - (G @ c)[:3]))
        if (rot, pos) < (best[1], best[0]):
            best = (pos, rot)
    return best


# ---- 한 장면 ----


@dataclass
class Settings:
    noise: NoiseConfig
    cls_mode: str  # "gt" = 정답 자세 클래스, "none" = 클래스 없이 (FPFH 전체 검색)
    voxel_mm: float = VOXEL_MM
    # "icp" = 지금 방식 (박스 → 점군 → CAD 후보 ICP)
    # "pnp" = YOLO-pose 키포인트 → PnP 만, "pnp_depth" = PnP + 뎁스 깊이 맞춤 + 설 수 있는 자세로 붙임,
    # "pnp_icp" = pnp_depth 를 처음 자세로 ICP 로 다듬기
    method: str = "icp"
    weights: str | None = None  # YOLO-pose 가중치 (pnp 계열)
    kpt_conf: float = 0.5  # PnP 에 쓸 키포인트 신뢰도 하한
    rgb: RgbConfig = CLEAN  # 컬러 영상 열화. distort 면 뎁스 · 분할도 렌즈 왜곡
    pnp_dist: bool = False  # PnP · 깊이 맞춤에 실제 렌즈 왜곡 계수를 넘긴다 (실제 카메라 창은 안 넘김)
    kpt_noise_px: float = 0.0  # 검출한 키포인트에 더하는 가우시안 오차 (픽셀)
    kpt_outlier_p: float = 0.0  # 키포인트가 박스 안 엉뚱한 곳으로 튈 확률
    he_mm: float = 0.0  # 손눈 캘리브 오차: 처리 코드가 쓰는 T_base_cam 을 이만큼 옮기고
    he_deg: float = 0.0  # 이만큼 돌린다 (렌더는 참값)


# ---- YOLO-pose + PnP ----

_POSE_MODELS: dict = {}
# YOLO-pose 박스와 정답 박스가 이만큼 겹쳐야 같은 물체로 본다.
MATCH_IOU = 0.5
PNP_REPROJ_PX = 4.0


def _pose_model(weights: str):
    if weights not in _POSE_MODELS:
        from ultralytics import YOLO

        _POSE_MODELS[weights] = YOLO(weights)
    return _POSE_MODELS[weights]


def detect_keypoints(bgr: np.ndarray, weights: str, conf: float = 0.25) -> list[dict]:
    """YOLO-pose 한 장 → [{xyxy, conf, kxy (N,2), kconf (N,)}]."""
    # 학습이 GPU 를 쓰는 중이면 SIM_POSE_DEVICE=cpu 로 추론을 CPU 에서 돌린다.
    device = os.environ.get("SIM_POSE_DEVICE") or None
    res = _pose_model(weights).predict(bgr, conf=conf, verbose=False, imgsz=640, device=device)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    kxy = res.keypoints.xy.cpu().numpy()
    kconf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])
    return [
        {"xyxy": boxes[i].tolist(), "conf": float(confs[i]), "kxy": kxy[i], "kconf": kconf[i]}
        for i in range(len(boxes))
    ]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_detections(scene: Scene, dets: list[dict]) -> dict[int, dict]:
    """정답 물체 id → 가장 많이 겹치는 검출 (IoU ≥ MATCH_IOU). 한 검출은 한 물체에만."""
    pairs = []
    for o in scene.objects:
        box = seg_box(scene.seg, o["id"])
        if box is None:
            continue
        for j, d in enumerate(dets):
            pairs.append((_iou(box, d["xyxy"]), o["id"], j))
    out: dict[int, dict] = {}
    used: set[int] = set()
    for iou, oid, j in sorted(pairs, reverse=True):
        if iou < MATCH_IOU or oid in out or j in used:
            continue
        out[oid] = dets[j]
        used.add(j)
    return out


def pnp_pose(
    det: dict, K: np.ndarray, T_base_cam: np.ndarray, min_conf: float, *, dist: np.ndarray | None = None
) -> dict | None:
    """키포인트 → PnP (RANSAC + LM 다듬기) → T_base_obj. 쓸 점이 4개 미만이거나 실패하면 None."""
    import cv2

    from vision.sim.keypoints import KEYPOINTS_CAD_MM

    sel = np.asarray(det["kconf"]) >= min_conf
    if int(sel.sum()) < 4:
        return None
    obj = KEYPOINTS_CAD_MM[sel].astype(np.float64)
    img = np.asarray(det["kxy"], np.float64)[sel]
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP, reprojectionError=PNP_REPROJ_PX, iterationsCount=100
    )
    if not ok or inl is None or len(inl) < 4:
        return None
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj[inl], rvec, tvec, K, dist)
    err = float(np.median(np.linalg.norm(proj.reshape(-1, 2) - img[inl], axis=1)))
    T_cam_obj = np.eye(4)
    T_cam_obj[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_cam_obj[:3, 3] = tvec.ravel()
    return {"T": T_base_cam @ T_cam_obj, "n_kpt": int(sel.sum()), "n_inlier": len(inl), "reproj_px": err}


def scene_clouds(scene: Scene, depth_mm: np.ndarray, cls_mode: str, boxes: dict[int, list[float]] | None = None):
    """실제 카메라 창과 같은 단계로 물체마다 베이스 mm 점군을 만든다. (물체, 점군, 책상 평면(베이스), 제거 높이)

    boxes 를 주면 (물체 id → xyxy) 그 박스를 쓰고, 없는 물체는 뺀다. 안 주면 정답 분할의 외접 사각형.
    """
    from vision.depth_cloud import (
        auto_plane_height_mm,
        fit_plane_ransac,
        plane_noise_mm,
        points_from_mask,
    )
    from vision.transforms import transform_plane
    from yolo.pose.instances import collect_instances

    h, w = depth_mm.shape
    full = np.ones((h, w), dtype=bool)
    table_xyz, _ = points_from_mask(depth_mm, scene.K, full, stride=5)
    plane, _ = fit_plane_ransac(table_xyz)
    plane_mm = auto_plane_height_mm(plane_noise_mm(table_xyz, plane))
    desk_base = transform_plane(plane, scene.T_base_cam)

    dets, objs = [], []
    for o in scene.objects:
        box = seg_box(scene.seg, o["id"]) if boxes is None else boxes.get(o["id"])
        if box is None:
            continue
        dets.append(_Det(box, o["pose_class"] if cls_mode == "gt" else None))
        objs.append(o)
    bgr = scene.rgb[..., ::-1].copy()
    inst = collect_instances(
        depth=depth_mm,
        bgr=bgr,
        K=scene.K,
        result=_Result(dets),
        plane=plane,
        pad=PAD_PX,
        plane_mm=plane_mm,
        stride=STRIDE,
        T_base_cam=scene.T_base_cam,
    )
    return [(o, i) for o, i in zip(objs, inst)], desk_base, plane_mm


def degrade_scene(scene: Scene, settings: Settings, rng: np.random.Generator, seed: int) -> tuple[Scene, np.ndarray]:
    """시험 조건 적용: 렌즈 왜곡 → 뎁스 노이즈 → 컬러 열화 → 손눈 오차. (처리 코드가 볼 장면, 노이즈 뎁스)

    정답(objects 의 T_base_obj)은 그대로다. 손눈 오차는 처리 코드가 받는 T_base_cam 만 바꾼다.
    """
    from dataclasses import replace

    depth0, seg = scene.depth_mm, scene.seg
    if settings.rgb.distort:
        depth0 = distort(depth0, scene.K, nearest=True)
        seg = distort(seg, scene.K, nearest=True)
    depth = add_noise(depth0, scene.K, rng, settings.noise)
    # 컬러 열화 난수는 따로 둔다. 조건을 바꿔도 뎁스 노이즈는 같게.
    rgb = degrade_rgb(scene.rgb, settings.rgb, np.random.default_rng(seed + 7_000_000), scene.K)
    T_used = scene.T_base_cam
    if settings.he_mm > 0 or settings.he_deg > 0:
        from vision.transforms import axis_angle_R

        hr = np.random.default_rng(seed + 9_000_000)
        T_used = np.asarray(scene.T_base_cam, float).copy()
        axis = hr.normal(size=3)
        T_used[:3, :3] = axis_angle_R(axis / np.linalg.norm(axis), np.deg2rad(settings.he_deg)) @ T_used[:3, :3]
        d = hr.normal(size=3)
        T_used[:3, 3] += d / np.linalg.norm(d) * settings.he_mm
    return replace(scene, rgb=rgb, depth_mm=depth0, seg=seg, T_base_cam=T_used), depth


def eval_scene(folder: Path, settings: Settings, seed: int) -> list[dict]:
    from yolo.pose.register import load_cad_xyz, register_pose

    scene = load_scene(folder)
    rng = np.random.default_rng(seed)
    scene, depth = degrade_scene(scene, settings, rng, seed)
    with contextlib.redirect_stdout(io.StringIO()):
        cad = load_cad_xyz(voxel_mm=settings.voxel_mm)
    center = cad.mean(axis=0)
    if settings.method == "icp":
        pairs, desk_base, plane_mm = scene_clouds(scene, depth, settings.cls_mode)
    else:
        return _eval_scene_pnp(folder.name, scene, depth, settings, center, cad, seed)
    rows = []
    for o, inst in pairs:
        T_gt = np.asarray(o["T_base_obj"], float)
        row = {
            "scene": folder.name,
            "obj": o["id"],
            "pose_class": o["pose_class"],
            "n_points": len(inst.xyz),
            "plane_mm": round(plane_mm, 2),
        }
        if len(inst.xyz) < 20:
            row.update(ok=False, aligned_ok=False, pos_mm=None, rot_deg=None, time_s=0.0, note="점 부족")
            rows.append(row)
            continue
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pose = register_pose(
                    np.asarray(inst.xyz, dtype=np.float64),
                    T_base_cam=None,
                    voxel_mm=settings.voxel_mm,
                    flip=True,
                    tries=TRIES,
                    camera_origin=scene.T_base_cam[:3, 3],
                    desk_plane=desk_base,
                    class_id=inst.class_id,
                )
        except RuntimeError as exc:
            # 실제 카메라 창도 이 경우 결과 없이 넘어간다. 실패로 센다.
            row.update(ok=False, aligned_ok=False, pos_mm=None, rot_deg=None,
                       time_s=round(time.perf_counter() - t0, 3), note=f"예외: {exc}"[:120])
            rows.append(row)
            continue
        dt = time.perf_counter() - t0
        pos, rot = pose_error(np.asarray(pose["T"], float), T_gt, center)
        ok = pos <= SUCCESS_POS_MM and rot <= SUCCESS_ROT_DEG
        row.update(
            ok=bool(ok),
            aligned_ok=bool(pose["aligned_ok"]),
            pos_mm=round(pos, 3),
            rot_deg=round(rot, 3),
            time_s=round(dt, 3),
            source=pose.get("source"),
            scene_frac=pose.get("scene_frac"),
            scene_med=pose.get("scene_med"),
            scene_p90=pose.get("scene_p90"),
            vis_frac=pose.get("vis_frac"),
        )
        rows.append(row)
    return rows


def depth_scale(T: np.ndarray, cad: np.ndarray, depth_mm: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
                *, gaps_mm: tuple[float, ...] = (60.0, 25.0), dist: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """PnP 자세를 카메라 시선 방향으로만 밀어 뎁스 영상과 깊이를 맞춘다. (새 T, 옮긴 깊이 mm)

    PnP 는 깊이가 약하다 (브라켓이 40 px 라 키포인트 1 px ≈ 10 mm). 보이는 CAD 점을 영상에 투영해
    그 픽셀의 실제 뎁스와의 차이 중앙값만큼 물체를 카메라에서 멀리/가까이 옮긴다. 시선 방향이라 영상 위치는 그대로다.
    책상 제거 전 뎁스 영상을 쓰므로, 노이즈로 점군에서 빠진 세운 판도 깊이 비교에 들어간다.
    PnP 깊이가 25 mm 넘게 틀리면 좁은 허용폭으로는 맞는 점이 하나도 없어 못 옮긴다. 그래서 넓은 폭(60 mm)으로
    먼저 대충 옮기고, 좁은 폭(25 mm)으로 한 번 더 맞춘다.
    """
    from vision.transforms import invert_T

    T_cb = invert_T(T_base_cam)
    T_now = np.asarray(T, float).copy()
    total = 0.0
    for max_gap_mm in gaps_mm:
        T_now, dz = _depth_shift_once(T_now, cad, depth_mm, K, T_base_cam, T_cb, max_gap_mm, dist)
        total += dz
    return T_now, total


def _depth_shift_once(T, cad, depth_mm, K, T_base_cam, T_cb, max_gap_mm, dist):
    """depth_scale 한 번: 허용폭 max_gap_mm 안의 점들로 깊이 차 중앙값만큼 시선 방향으로 민다."""
    from vision.transforms import apply_T
    from yolo.pose.register import visible_xyz

    vis_base = visible_xyz(apply_T(cad, T), T_base_cam[:3, 3])
    pc = apply_T(vis_base, T_cb)
    if dist is None:
        uf = pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2]
        vf = pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]
    else:
        import cv2

        uv = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
        uf, vf = uv[:, 0], uv[:, 1]
    u = np.round(uf).astype(int)
    v = np.round(vf).astype(int)
    h, w = depth_mm.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    meas = np.zeros(len(pc))
    meas[inside] = depth_mm[v[inside], u[inside]]
    gap = meas - pc[:, 2]
    good = inside & (meas > 0) & (np.abs(gap) < max_gap_mm)
    if int(good.sum()) < 20:
        return np.asarray(T, float).copy(), 0.0
    dz = float(np.median(gap[good]))
    T_co = T_cb @ np.asarray(T, float)
    c = np.median(pc[good], axis=0)
    # 물체 중심이 시선을 따라 깊이 dz 만큼 움직이도록 전체를 같은 비율로 민다.
    T_co[:3, 3] += c * (dz / float(c[2]))
    return T_base_cam @ T_co, dz


def snap_stable(T: np.ndarray, center: np.ndarray, *, tol_deg: float = 15.0) -> tuple[np.ndarray, int, float]:
    """책상 위에 설 수 있는 자세(세우기 · 눕히기 · 비스듬히, 대칭 포함)로 기울기만 붙인다. yaw 는 그대로.

    (새 T, 자세 클래스 또는 −1, 붙이기 전 기울기 차 도). tol_deg 보다 멀면 그대로 둔다.
    """
    from vision.sim.symmetry import cad_symmetries
    from vision.transforms import rotation_aligning
    from yolo.pose.register import _T_from_base_rpy, class_base_rpys

    up = np.array([0.0, 0.0, 1.0])
    R = np.asarray(T, float)[:3, :3]
    u = R.T @ up
    best = (-1, 180.0, None)
    for cid in (0, 1, 2):
        for rpy in class_base_rpys(cid):
            r0 = _T_from_base_rpy(rpy)[:3, :3].T @ up
            for S in cad_symmetries():
                r = S[:3, :3] @ r0
                ang = float(np.degrees(np.arccos(np.clip(u @ r, -1.0, 1.0))))
                if ang < best[1]:
                    best = (cid, ang, r)
    cid, ang, r = best
    if r is None or ang > tol_deg:
        return np.asarray(T, float).copy(), -1, ang
    A = rotation_aligning(r, u)
    R2 = R @ A
    out = np.asarray(T, float).copy()
    world_c = R @ center + out[:3, 3]
    out[:3, :3] = R2
    out[:3, 3] = world_c - R2 @ center
    return out, cid, ang


def perturb_keypoints(det: dict, rng: np.random.Generator, noise_px: float, outlier_p: float) -> dict:
    """검출 키포인트를 일부러 흔든다: 모두 가우시안 noise_px, outlier_p 확률로 박스 안 아무 곳으로."""
    kxy = np.asarray(det["kxy"], float).copy()
    if noise_px > 0:
        kxy += rng.normal(0.0, noise_px, kxy.shape)
    if outlier_p > 0:
        x0, y0, x1, y1 = det["xyxy"]
        bad = rng.random(len(kxy)) < outlier_p
        kxy[bad, 0] = rng.uniform(x0, x1, int(bad.sum()))
        kxy[bad, 1] = rng.uniform(y0, y1, int(bad.sum()))
    return dict(det, kxy=kxy)


def _eval_scene_pnp(
    name: str, scene: Scene, depth: np.ndarray, settings: Settings, center: np.ndarray, cad: np.ndarray, seed: int
) -> list[dict]:
    from yolo.pose.register import register_pose

    bgr = np.ascontiguousarray(scene.rgb[..., ::-1])
    t0 = time.perf_counter()
    dets = detect_keypoints(bgr, settings.weights)
    t_det = time.perf_counter() - t0
    matched = match_detections(scene, dets)
    if settings.kpt_noise_px > 0 or settings.kpt_outlier_p > 0:
        kr = np.random.default_rng(seed + 5_000_000)
        matched = {
            oid: perturb_keypoints(d, kr, settings.kpt_noise_px, settings.kpt_outlier_p) for oid, d in matched.items()
        }
    dist = REAL_DIST if settings.pnp_dist else None
    clouds = {}
    desk_base = np.array([0.0, 0.0, 1.0, 0.0])
    if settings.method in ("pnp_depth", "pnp_icp") and matched:
        pairs, desk_base, _pm = scene_clouds(
            scene, depth, "none", boxes={oid: d["xyxy"] for oid, d in matched.items()}
        )
        clouds = {o["id"]: inst for o, inst in pairs}
    rows = []
    for o in scene.objects:
        T_gt = np.asarray(o["T_base_obj"], float)
        row = {"scene": name, "obj": o["id"], "pose_class": o["pose_class"], "t_det_s": round(t_det, 4),
               # 정답과 짝이 안 맞은 검출 수 (장면 단위, 장면의 첫 물체에만 적는다).
               "false_det": len(dets) - len(matched) if o is scene.objects[0] else 0}
        det = matched.get(o["id"])
        if det is None:
            row.update(ok=False, aligned_ok=False, pos_mm=None, rot_deg=None, time_s=round(t_det, 3), note="미검출")
            rows.append(row)
            continue
        t1 = time.perf_counter()
        pnp = pnp_pose(det, scene.K, scene.T_base_cam, settings.kpt_conf, dist=dist)
        t_pnp = time.perf_counter() - t1
        if pnp is None:
            row.update(ok=False, aligned_ok=False, pos_mm=None, rot_deg=None, time_s=round(t_det + t_pnp, 3),
                       note="PnP 실패")
            rows.append(row)
            continue
        T = pnp["T"]
        pnp_pos, pnp_rot = pose_error(T, T_gt, center)
        row.update(n_kpt=pnp["n_kpt"], n_inlier=pnp["n_inlier"], reproj_px=round(pnp["reproj_px"], 2),
                   pnp_pos_mm=round(pnp_pos, 3), pnp_rot_deg=round(pnp_rot, 3), t_pnp_s=round(t_pnp, 4))
        aligned = True
        t_icp = 0.0
        if settings.method in ("pnp_depth", "pnp_icp"):
            from yolo.pose.register import seat_cad_on_plane

            t2 = time.perf_counter()
            T, dz = depth_scale(T, cad, depth, scene.K, scene.T_base_cam, dist=dist)
            T, snap_cls, snap_ang = snap_stable(T, center)
            T = seat_cad_on_plane(T, cad, desk_base, np.array([0.0, 0.0, 1.0]))
            t_icp += time.perf_counter() - t2
            dpos, drot = pose_error(T, T_gt, center)
            row.update(depth_dz_mm=round(dz, 2), snap_class=snap_cls, snap_deg=round(snap_ang, 2),
                       depth_pos_mm=round(dpos, 3), depth_rot_deg=round(drot, 3))
        inst = clouds.get(o["id"])
        if settings.method == "pnp_icp" and inst is not None and len(inst.xyz) >= 20:
            t2 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                pose = register_pose(
                    np.asarray(inst.xyz, dtype=np.float64),
                    T_base_cam=None,
                    voxel_mm=settings.voxel_mm,
                    flip=True,
                    tries=TRIES,
                    T_init=T,
                    camera_origin=scene.T_base_cam[:3, 3],
                    desk_plane=desk_base,
                )
            t_icp += time.perf_counter() - t2
            T = np.asarray(pose["T"], float)
            aligned = bool(pose["aligned_ok"])
            row.update(scene_frac=pose.get("scene_frac"), scene_med=pose.get("scene_med"),
                       vis_frac=pose.get("vis_frac"), t_icp_s=round(t_icp, 3))
        pos, rot = pose_error(T, T_gt, center)
        row.update(
            ok=bool(pos <= SUCCESS_POS_MM and rot <= SUCCESS_ROT_DEG),
            aligned_ok=aligned,
            pos_mm=round(pos, 3),
            rot_deg=round(rot, 3),
            time_s=round(t_det + t_pnp + t_icp, 3),
        )
        rows.append(row)
    return rows


# ---- 여러 장면 ----


def _worker(job):
    folder, settings, seed = job
    return eval_scene(folder, settings, seed)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    ok = sum(r["ok"] for r in rows)
    said_ok = sum(r["aligned_ok"] for r in rows)
    confident_wrong = sum(r["aligned_ok"] and not r["ok"] for r in rows)
    missed_right = sum(r["ok"] and not r["aligned_ok"] for r in rows)
    good = [r for r in rows if r["ok"]]
    times = [r["time_s"] for r in rows if r["pos_mm"] is not None]
    rot_all = [r["rot_deg"] for r in rows if r["rot_deg"] is not None]

    def pct(vals, q):
        return round(float(np.percentile(vals, q)), 2) if vals else None

    return {
        "n": n,
        "success": ok,
        "success_rate": round(ok / n, 3) if n else None,
        "aligned_ok": said_ok,
        "confident_wrong": confident_wrong,
        "missed_right": missed_right,
        "pos_mm_median_ok": pct([r["pos_mm"] for r in good], 50),
        "rot_deg_median_ok": pct([r["rot_deg"] for r in good], 50),
        "flip_90": sum(1 for v in rot_all if 60 <= v <= 120),
        "flip_180": sum(1 for v in rot_all if v > 150),
        "time_median_s": pct(times, 50),
        "time_p90_s": pct(times, 90),
        "no_detect": sum(1 for r in rows if r.get("note") == "미검출"),
        "false_det": sum(int(r.get("false_det", 0)) for r in rows),
        "pnp_fail": sum(1 for r in rows if r.get("note") == "PnP 실패"),
        "pnp_only_success": sum(
            1 for r in rows
            if r.get("pnp_pos_mm") is not None and r["pnp_pos_mm"] <= SUCCESS_POS_MM and r["pnp_rot_deg"] <= SUCCESS_ROT_DEG
        ),
        "pnp_only_pos_mm_median": pct([r["pnp_pos_mm"] for r in rows if r.get("pnp_pos_mm") is not None], 50),
        "pnp_only_rot_deg_median": pct([r["pnp_rot_deg"] for r in rows if r.get("pnp_rot_deg") is not None], 50),
        "depth_shift_success": sum(
            1 for r in rows
            if r.get("depth_pos_mm") is not None and r["depth_pos_mm"] <= SUCCESS_POS_MM and r["depth_rot_deg"] <= SUCCESS_ROT_DEG
        ),
        "depth_shift_pos_mm_median": pct([r["depth_pos_mm"] for r in rows if r.get("depth_pos_mm") is not None], 50),
        "by_class": {
            str(c): f"{sum(r['ok'] for r in rows if r['pose_class'] == c)}/{sum(1 for r in rows if r['pose_class'] == c)}"
            for c in sorted({r["pose_class"] for r in rows})
        },
    }


def run_eval(folders: list[Path], settings: Settings, *, workers: int = 1, verbose: bool = False) -> tuple[list[dict], dict]:
    """장면 폴더들을 평가해 (물체마다 결과, 요약) 을 돌려준다."""
    # 노이즈 난수는 장면 번호로 고정한다. 설정을 바꿔도 같은 장면엔 같은 노이즈가 들어간다.
    jobs = [(f, settings, 1000 + k) for k, f in enumerate(folders)]
    t0 = time.perf_counter()
    rows: list[dict] = []
    if workers > 1:
        import multiprocessing as mp

        # 프로세스마다 open3d 스레드를 나눠 쓴다.
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // workers)))
        with mp.get_context("spawn").Pool(workers) as pool:
            for k, part in enumerate(pool.imap(_worker, jobs)):
                rows.extend(part)
                if verbose:
                    print(f"\r{k + 1}/{len(jobs)}", end="", flush=True)
        if verbose:
            print()
    else:
        for job in jobs:
            rows.extend(_worker(job))
            r = rows[-1]
            if verbose:
                print(
                    f"{r['scene']} 클래스 {r['pose_class']}  {'성공' if r['ok'] else '실패'}"
                    f"  코드판단 {'ok' if r['aligned_ok'] else 'miss'}  위치 {r['pos_mm']}mm  회전 {r['rot_deg']}°  {r['time_s']}s",
                    flush=True,
                )
    summary = summarize(rows)
    summary["wall_s"] = round(time.perf_counter() - t0, 1)
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="시뮬 장면으로 정합 평가")
    ap.add_argument("--scenes", type=Path, default=OUT_DIR / "scenes")
    ap.add_argument("--n", type=int, default=20, help="앞에서부터 장면 수")
    ap.add_argument("--noise", action="store_true", help="Orbbec 흉내 노이즈 켬")
    ap.add_argument(
        "--noise-set", action="append", default=[], metavar="KEY=VAL",
        help="노이즈 설정 바꾸기 (예: --noise-set edge_px=1 --noise-set sigma_mm=1.0). NoiseConfig 필드 이름",
    )
    ap.add_argument("--cls", choices=("gt", "none"), default="gt", help="자세 클래스: 정답(gt) / 없음(none)")
    ap.add_argument("--voxel", type=float, default=VOXEL_MM)
    ap.add_argument("--method", choices=("icp", "pnp", "pnp_depth", "pnp_icp"), default="icp",
                    help="icp = 지금 방식, pnp = YOLO-pose 키포인트 + PnP, "
                         "pnp_depth = PnP + 뎁스로 깊이 맞춤 + 설 수 있는 자세로 붙임, pnp_icp = pnp_depth 후 ICP")
    ap.add_argument("--weights", default=None, help="YOLO-pose 가중치 (pnp 계열에 필요)")
    ap.add_argument("--kpt-conf", type=float, default=0.5)
    ap.add_argument("--rgb", default="clean", choices=sorted(PRESETS), help="컬러 영상 열화 조건 (rgb_noise.PRESETS)")
    ap.add_argument("--pnp-dist", action="store_true", help="PnP 에 실제 렌즈 왜곡 계수를 넘긴다")
    ap.add_argument("--kpt-noise", type=float, default=0.0, help="키포인트에 더할 오차 (픽셀)")
    ap.add_argument("--kpt-outlier", type=float, default=0.0, help="키포인트가 엉뚱한 곳으로 튈 확률")
    ap.add_argument("--he-mm", type=float, default=0.0, help="손눈 오차: 위치 (mm)")
    ap.add_argument("--he-deg", type=float, default=0.0, help="손눈 오차: 회전 (도)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default=None, help="결과 파일 이름 (기본: 설정으로 만듦)")
    args = ap.parse_args()

    folders = sorted(p for p in args.scenes.iterdir() if (p / "gt.json").is_file())[: args.n]
    noise = NO_NOISE
    if args.noise:
        overrides = {}
        for item in args.noise_set:
            key, val = item.split("=", 1)
            overrides[key.strip()] = type(getattr(NoiseConfig, key.strip()))(val)
        noise = NoiseConfig(**overrides)
    if args.method != "icp" and not args.weights:
        ap.error("--method pnp / pnp_icp 에는 --weights 가 필요합니다")
    settings = Settings(noise=noise, cls_mode=args.cls, voxel_mm=args.voxel, method=args.method,
                        weights=args.weights, kpt_conf=args.kpt_conf, rgb=PRESETS[args.rgb], pnp_dist=args.pnp_dist,
                        kpt_noise_px=args.kpt_noise, kpt_outlier_p=args.kpt_outlier,
                        he_mm=args.he_mm, he_deg=args.he_deg)
    rows, summary = run_eval(folders, settings, workers=args.workers, verbose=True)
    tag = args.tag or (
        f"n{len(folders)}_{'noise' if args.noise else 'clean'}_cls-{args.cls}_v{args.voxel:g}"
        if args.method == "icp"
        else f"n{len(folders)}_{'noise' if args.noise else 'clean'}_{args.method}"
    )
    res_dir = OUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{tag}.json").write_text(
        json.dumps({"settings": vars(args) | {"scenes": str(args.scenes)}, "summary": summary, "rows": rows}, indent=1, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"저장: {res_dir / (tag + '.json')}")


if __name__ == "__main__":
    main()
