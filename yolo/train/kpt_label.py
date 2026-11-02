#!/usr/bin/env python3
"""브라켓 키포인트 10개를 찍어 YOLO-pose 라벨을 만든다 (실사진 미세조정용).

  python yolo/train/kpt_label.py --images yolo/datasets/seat/images --split train
  python yolo/train/kpt_label.py --images 폴더 --weights 모델.pt

찍는 법
  - 모델(기본 v4n)이 먼저 점을 찍어 둔다. 틀린 점만 끌어서 옮긴다.
  - 빈 곳 클릭 = 지금 번호 점을 그 자리에. 다음 안 찍은 번호로 넘어간다.
  - 점 근처 클릭 + 끌기 = 그 점 옮기기 (다른 브라켓 점이면 그 브라켓이 선택된다).
  - 안 보이는 점은 h (자리는 자세에서 자동).
  - 자동 점 (mixed 는 8 · 9 = 안쪽 접힌 곳) 은 사람이 찍지 않는다. 눈으로 찾을 표시가 없어서 자세에서 CAD 로 채운다.
  - 책상 붙이기 (g, 기본 켜짐): 브라켓이 책상 위에 세우기 · 눕히기 · 비스듬히 중 하나로 놓였다고 보고
    위치 x · y 와 돌아간 각도만 찍은 점에 맞춘다. 보이는 점 2개 이상이면 된다. 끄면 보통 PnP (4개 이상).
  - 하늘색 선 = 그 자세의 CAD 윤곽. 실물 테두리와 맞는지만 보면 된다.

번호는 대칭 신경 쓰지 말고 아무 쪽으로 찍어도 된다. 브라켓은 대각선 축 180° 대칭이라 번호를 매기는 법이 두 가지인데,
저장할 때 자세로 시뮬 라벨과 같은 규칙(canonical_T)에 맞게 번호를 자동으로 다시 매긴다.

키: 0~9 번호 고르기 · h 가려짐 · x 지우기 · g 책상 붙이기 켜기/끄기 · b 화면 밝게 (끔 → 밝게 → 어두운 곳 늘리기, 보기만)
    f 테두리 자동 맞춤 (지금 자세에서 출발해 CAD 윤곽선이 사진 테두리선에 붙도록 위치 · 각도를 다듬음. 자세 종류는 안 바꿈)
    a 브라켓 추가 · Tab 다음 브라켓 · D 브라켓 삭제
    u/우클릭 되돌리기 · s 저장 · n/p 다음/이전 사진(저장) · q 저장 후 끝

카메라: 사진 폴더 옆 camera.json (K, T_base_cam, dist) → 사진 폴더 옆 calib/ (intrinsics.json, eye_to_hand.json, 180° 회전 영상)
       → 이 PC 의 vision/calib_data 순서로 찾는다. 책상 붙이기는 손눈 값으로 책상 높이(베이스 z = 0)를 안다고 본다.
학습/시험 나누기: 사진 폴더 옆 split.json ({"train": [...], "test": [...]}) 이 있으면 --split 으로 고른다.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import ROOT, quiet_gtk

quiet_gtk()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vision.sim.keypoints import KEYPOINTS_CAD_MM, KPT_SET, N_KPT, canonical_T  # noqa: E402
from vision.text import put_text  # noqa: E402

KPT_DIR = ROOT / "datasets" / "kpt"
DEFAULT_WEIGHTS = ROOT.parent / "vision" / "sim" / "out" / "pose_runs" / "v4n" / "weights" / "best.pt"
WIN = "keypoint label"
# 미리 찍기 모델의 박스 신뢰도 문턱. 0.25 면 로봇 서보 · 나사 머리도 잡혔다.
MODEL_CONF = 0.5
# 화면에서 점을 잡는 거리 (표시 픽셀).
GRAB_PX = 12
# 재투영 오차가 이보다 크면 빨간 글씨로 알린다 (원본 픽셀).
WARN_PX = 4.0
# 자유 PnP 와 책상 자세의 기울기 차가 이보다 크면 알린다 (도).
WARN_TILT_DEG = 8.0
# 자세가 이 깊이 범위 밖이면 이상한 자세로 본다 (mm).
DEPTH_OK_MM = (150.0, 1200.0)
PANEL_W = 320
# 사진 위 글씨 칸 높이 (사진을 가리지 않게 따로 둔다)
BAR_H = 190
COLORS = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
          (255, 0, 0), (255, 0, 255), (128, 0, 255), (255, 255, 255), (0, 0, 0)]
OUTLINE_BGR = (255, 200, 0)

UNSET, HIDDEN, VISIBLE = 0, 1, 2
# 번호마다 어디인지 (안내 글).
KPT_NAMES = {
    "mixed": ["바닥 판 끝 · 아래 모서리", "바닥 판 끝 · 아래 모서리 (0 의 반대쪽)", "세운 판 꼭대기 · 바깥 모서리",
              "세운 판 꼭대기 · 바깥 모서리 (2 의 반대쪽)", "뒤꿈치 바깥 모서리 (접힌 곳 바깥)",
              "뒤꿈치 바깥 모서리 (5 의 반대쪽)", "바닥 판 구멍 가운데", "세운 판 구멍 가운데",
              "안쪽 접힌 곳 끝 (자동)", "안쪽 접힌 곳 끝 (자동)"],
    "corners": ["바닥 판 끝 · 아래 모서리", "바닥 판 끝 · 아래 모서리 (0 의 반대쪽)", "세운 판 꼭대기 · 바깥 모서리",
                "세운 판 꼭대기 · 바깥 모서리 (2 의 반대쪽)", "뒤꿈치 바깥 모서리",
                "뒤꿈치 바깥 모서리 (4 의 반대쪽)", "옆벽 곡면 시작 (자동)", "옆벽 곡면 시작 (자동)",
                "옆벽 곡면 끝 (자동)", "옆벽 곡면 끝 (자동)"],
}[KPT_SET]
# 사람이 찍지 않고 자세에서 채우는 점. 대칭으로 서로 바뀌는 짝끼리 묶여 있어야 한다.
AUTO_IDX = {"mixed": (8, 9), "corners": (6, 7, 8, 9)}[KPT_SET]
IS_AUTO = np.zeros(N_KPT, bool)
IS_AUTO[list(AUTO_IDX)] = True
# 키포인트 앞 이만큼(mm) 넘게 떨어진 CAD 면에 막히면 가려진 점. 시뮬 라벨의 vis_tol_mm (4) 과 같게.
SELF_HIDE_MM = 4.0
# 자세를 맞출 때 점의 무게: 모델이 미리 찍은 점 / 저장해서 확인된 점 / 이번에 사람이 찍거나 옮긴 점.
# 모델 점은 실물보다 5~7 px 바깥으로 치우쳐서, 같은 무게면 사람이 고친 점이 저장 때 원래 자리로 끌려간다.
SRC_MODEL, SRC_SAVED, SRC_EDITED = 0, 1, 2
SRC_WEIGHT = np.array([1.0, 3.0, 10.0])
# 책상 붙이기에 필요한 보이는 점 수 (모르는 값이 x, y, 각도 셋).
DESK_MIN_PTS = 2


# ---------------------------------------------------------------- 카메라 · CAD


@dataclass
class Camera:
    K: np.ndarray
    dist: np.ndarray
    T_base_cam: np.ndarray
    source: str


def load_camera(images_dir: Path, path: Path | None = None) -> Camera:
    """camera.json → 사진 폴더 옆 calib/ → 이 PC 캘리브 순서. calib 은 180° 회전 영상 기준으로 K 를 바꾼다."""
    for cand in ([path] if path else [images_dir.parent / "camera.json", images_dir / "camera.json"]):
        if cand is not None and Path(cand).is_file() and Path(cand).suffix == ".json":
            d = json.loads(Path(cand).read_text(encoding="utf-8"))
            dist = np.asarray(d.get("dist", [0, 0, 0, 0, 0]), float)
            return Camera(np.asarray(d["K"], float), dist, np.asarray(d["T_base_cam"], float), str(cand))
    calib = images_dir.parent / "calib"
    if (calib / "intrinsics.json").is_file() and (calib / "eye_to_hand.json").is_file():
        return calib_camera(calib)
    return calib_camera(None)


def calib_camera(calib_dir: Path | None) -> Camera:
    """캘리브 폴더 (intrinsics.json, eye_to_hand.json) → Camera. None 이면 이 보드의 vision/calib_data.

    카메라 영상은 180° 돌려 쓰므로 K · 왜곡도 돌린 영상 기준으로 바꾼다.
    """
    from vision.calib import intrinsics_for_rotate180, load_intrinsics, load_T_base_cam
    from vision.camera import FRAME_HEIGHT, FRAME_WIDTH

    if calib_dir is not None:
        K_raw, dist_raw, src = load_intrinsics(Path(calib_dir) / "intrinsics.json")
        T, _ = load_T_base_cam(Path(calib_dir) / "eye_to_hand.json")
    else:
        K_raw, dist_raw, src = load_intrinsics()
        T, _ = load_T_base_cam()
    K, dist = intrinsics_for_rotate180(K_raw, dist_raw, FRAME_WIDTH, FRAME_HEIGHT)
    return Camera(K, np.asarray(dist, float), T, str(Path(src).parent))


_MESH: tuple[np.ndarray, np.ndarray] | None = None
_SYMS: list[np.ndarray] | None = None
_RAYS = None
_DESK: list[tuple[int, np.ndarray, float]] | None = None


def mesh_mm() -> tuple[np.ndarray, np.ndarray]:
    """CAD 메쉬 (정점 mm, 삼각형)."""
    global _MESH
    if _MESH is None:
        from cad.model import load_mesh_m

        v, f = load_mesh_m()
        _MESH = (v * 1000.0, np.asarray(f, np.int32))
    return _MESH


def symmetries() -> list[np.ndarray]:
    global _SYMS
    if _SYMS is None:
        from vision.sim.symmetry import cad_symmetries

        _SYMS = cad_symmetries()
    return _SYMS


def _ray_scene():
    """CAD 메쉬 (mm) 광선 검사용 장면 (한 번만)."""
    global _RAYS
    if _RAYS is None:
        import open3d as o3d

        v, f = mesh_mm()
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(v.astype(np.float32))
        mesh.triangle.indices = o3d.core.Tensor(f)
        _RAYS = o3d.t.geometry.RaycastingScene()
        _RAYS.add_triangles(mesh)
    return _RAYS


def hidden_from(origins_obj: np.ndarray) -> np.ndarray:
    """CAD 프레임에서 각 키포인트를 origins_obj (N,3) 에서 바라볼 때 브라켓 몸에 막히면 True."""
    import open3d as o3d

    d = KEYPOINTS_CAD_MM - origins_obj
    dist = np.linalg.norm(d, axis=1)
    rays = np.hstack([origins_obj, d / dist[:, None]]).astype(np.float32)
    t = _ray_scene().cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return t < dist - SELF_HIDE_MM


def self_hidden(T_cam_obj: np.ndarray) -> np.ndarray:
    """카메라에서 볼 때 브라켓 자기 몸에 가려진 키포인트 (다른 물체에 가려진 건 모른다)."""
    cam_in_obj = np.linalg.inv(T_cam_obj)[:3, 3]
    return hidden_from(np.tile(cam_in_obj, (N_KPT, 1)))


def symmetry_perm(S: np.ndarray) -> np.ndarray:
    """대칭 S 로 돌렸을 때 CAD 키포인트 i 가 놓이는 자리의 원래 번호. perm[i] = j 이면 S·p_i ≈ p_j."""
    p = KEYPOINTS_CAD_MM
    q = p @ S[:3, :3].T + S[:3, 3]
    d = np.linalg.norm(q[:, None, :] - p[None, :, :], axis=2)
    perm = d.argmin(axis=1)
    if len(set(perm.tolist())) != len(perm) or d.min(axis=1).max() > 1.0:
        raise ValueError("키포인트가 대칭에 닫혀 있지 않다")
    return perm


def desk_candidates() -> list[tuple[int, np.ndarray, float]]:
    """책상 위에 놓일 수 있는 방향 (자세 클래스, 각도 0 일 때 CAD→베이스 4×4, 책상에 닿는 높이 z).

    세우기 1 · 눕히기 2 (Rx ±90) · 비스듬히 1, 각각 대칭 두 가지 (번호를 어느 쪽으로 찍었든 맞게).
    실제 자세 = 평행이동(x, y, z) · Rz(각도) · A.
    """
    global _DESK
    if _DESK is None:
        from motion.robot_kinematics import rpy_deg_to_rotmat

        v, _ = mesh_mm()
        out = []
        for cid, rps in ((0, [(0.0, 0.0)]), (1, [(90.0, 0.0), (-90.0, 0.0)]), (2, [(180.0, 45.0)])):
            for r, p in rps:
                A0 = np.eye(4)
                A0[:3, :3] = rpy_deg_to_rotmat(r, p, 0.0)
                for S in symmetries():
                    A = A0 @ S
                    z = -float((v @ A[:3, :3].T + A[:3, 3])[:, 2].min())
                    out.append((cid, A, z))
        _DESK = out
    return _DESK


def _Rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def desk_pose(A: np.ndarray, z: float, x: float, y: float, yaw: float) -> np.ndarray:
    T = _Rz(yaw) @ A
    T[:3, 3] += [x, y, z]
    return T


def project_base(T_base_obj: np.ndarray, pts: np.ndarray, cam: Camera) -> np.ndarray:
    T = np.linalg.inv(cam.T_base_cam) @ T_base_obj
    rvec = cv2.Rodrigues(T[:3, :3])[0]
    return cv2.projectPoints(pts.astype(np.float64), rvec, T[:3, 3], cam.K, cam.dist)[0].reshape(-1, 2)


def _desk_ray_point(uv: np.ndarray, cam: Camera, height_mm: float) -> np.ndarray:
    """화면 점 → 베이스 높이 height_mm 평면과 만나는 점."""
    n = cv2.undistortPoints(uv.reshape(1, 1, 2).astype(np.float64), cam.K, cam.dist).reshape(2)
    d = cam.T_base_cam[:3, :3] @ np.array([n[0], n[1], 1.0])
    o = cam.T_base_cam[:3, 3]
    s = (height_mm - o[2]) / d[2]
    return o + s * d


def fit_desk(pts: np.ndarray, img: np.ndarray, cam: Camera, weights: np.ndarray | None = None,
             init_T_base: np.ndarray | None = None) -> tuple[np.ndarray, int, float] | None:
    """책상 위 자세로 맞추기. (T_base_obj, 자세 클래스, 재투영 오차 평균 px). 모르는 값은 x, y, 각도.

    모든 후보 방향 × 각도 10° 로 대충 놓아 본 뒤 좋은 것 몇 개를 가우스-뉴턴으로 다듬는다.
    weights 는 점마다 무게 (사람이 고친 점을 더 믿게). init_T_base (직전 자세) 를 주면 그 자리에서도 꼭 다듬어 본다.
    무게 준 오차로만 후보를 고르면 고친 점 하나만 맞는 엉뚱한 자세로 넘어갈 수 있어, 무게 없는 오차로도 후보를 고른다.
    """
    if len(pts) < DESK_MIN_PTS:
        return None
    wt = np.ones(len(pts)) if weights is None else np.asarray(weights, float)
    sw = np.repeat(np.sqrt(wt), 2)
    center_img = (img * wt[:, None]).sum(axis=0) / wt.sum()
    trials = []
    for ci, (cid, A, z) in enumerate(desk_candidates()):
        q = pts @ A[:3, :3].T + A[:3, 3]
        p0 = _desk_ray_point(center_img, cam, z + float(q[:, 2].mean()))
        for yaw in np.deg2rad(np.arange(0.0, 360.0, 10.0)):
            qc = (_Rz(yaw)[:3, :3] @ q.mean(axis=0))
            x, y = p0[0] - qc[0], p0[1] - qc[1]
            d = np.linalg.norm(project_base(desk_pose(A, z, x, y, yaw), pts, cam) - img, axis=1)
            trials.append(((d * wt).sum() / wt.sum(), d.mean(), ci, np.array([x, y, yaw])))
    starts = sorted(trials, key=lambda t: t[0])[:6] + sorted(trials, key=lambda t: t[1])[:6]
    if init_T_base is not None:
        # 직전 자세와 같은 방향인 후보에서, 직전 위치 · 각도로 출발
        for ci, (_cid, A, z) in enumerate(desk_candidates()):
            M = init_T_base[:3, :3] @ A[:3, :3].T
            yaw = float(np.arctan2(M[1, 0], M[0, 0]))
            if _rot_deg(_Rz(yaw)[:3, :3], M) < 5.0:
                t = init_T_base[:3, 3] - (_Rz(yaw) @ A)[:3, 3]
                starts.append((0.0, 0.0, ci, np.array([t[0], t[1], yaw])))
    best = None
    for _e, _e2, ci, prm in starts:
        cid, A, z = desk_candidates()[ci]
        prm = prm.copy()

        def resid(p):
            return (project_base(desk_pose(A, z, p[0], p[1], p[2]), pts, cam) - img).ravel() * sw

        lam = 1e-3
        r = resid(prm)
        for _ in range(30):
            J = np.empty((len(r), 3))
            for j, eps in enumerate((0.05, 0.05, 1e-3)):
                dp = np.zeros(3)
                dp[j] = eps
                J[:, j] = (resid(prm + dp) - r) / eps
            H = J.T @ J
            step = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), J.T @ r)
            r_new = resid(prm + step)
            if r_new @ r_new < r @ r:
                prm, r, lam = prm + step, r_new, lam * 0.3
                if np.abs(step).max() < 1e-4:
                    break
            else:
                lam *= 10.0
        e = float((np.linalg.norm((r / sw).reshape(-1, 2), axis=1) * wt).sum() / wt.sum())
        if best is None or e < best[2]:
            best = (desk_pose(A, z, *prm), cid, e)
    return best


def _rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# 테두리 자동 맞춤: 윤곽선 점에서 가장 가까운 사진 테두리선까지 거리 (이보다 먼 건 이 값으로 자른다)
EDGE_CAP_PX = 6.0
# 맞춘 뒤 평균 거리가 이보다 크면 불확실하다고 알린다
EDGE_WARN_PX = 1.5


def _silhouette(T_base: np.ndarray, cam: Camera, shape: tuple[int, int], off: tuple[int, int]) -> np.ndarray:
    """그 자세의 CAD 를 사진(잘라낸 부분)에 칠한 마스크 (겉모양 안쪽 전부).

    삼각형을 fillPoly 한 번에 넘기면 겹친 곳이 서로 지워져 안에 구멍이 생긴다. 바깥 테두리로 다시 채운다.
    """
    v, f = mesh_mm()
    pv = project_base(T_base, v, cam) - off
    mk = np.zeros(shape, np.uint8)
    cv2.fillPoly(mk, list(pv[f].astype(np.int32)), 1)
    cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mk, cs, -1, 1, -1)
    return mk


def fit_edges(bgr: np.ndarray, T_base: np.ndarray, cam: Camera, margin: int = 25) -> tuple[np.ndarray, float, float] | None:
    """책상 위 자세 T_base 에서 출발해 CAD 윤곽선이 사진 테두리선(Canny)에 붙도록 x, y, 각도를 다듬는다.

    (새 T_base, 맞춘 뒤 평균 거리 px, 움직인 거리 px). 자세 종류(방향 후보)는 출발 자세에 가장 가까운 것으로 둔다.
    흰 책상 위 흰 브라켓도 테두리선은 잡혀서 색과 상관없이 쓴다.
    """
    best = None
    for cid, A, z in desk_candidates():
        M = T_base[:3, :3] @ A[:3, :3].T
        yaw = float(np.arctan2(M[1, 0], M[0, 0]))
        d = _rot_deg(_Rz(yaw)[:3, :3], M)
        if best is None or d < best[0]:
            t = T_base[:3, 3] - (_Rz(yaw) @ A)[:3, 3]
            best = (d, A, z, np.array([t[0], t[1], yaw]))
    _d, A, z, prm = best
    v, _ = mesh_mm()
    pv = project_base(T_base, v, cam)
    H, W = bgr.shape[:2]
    x0, y0 = np.maximum((pv.min(axis=0) - margin).astype(int), 0)
    x1, y1 = np.minimum((pv.max(axis=0) + margin).astype(int), [W, H])
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    g = cv2.GaussianBlur(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(g, 10, 30)
    dt = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)

    def cost(q):
        cs, _ = cv2.findContours(_silhouette(desk_pose(A, z, *q), cam, g.shape, (x0, y0)),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return 1e9
        c = np.vstack(cs).reshape(-1, 2)
        return float(np.minimum(dt[c[:, 1], c[:, 0]], EDGE_CAP_PX).mean())

    start = prm.copy()
    cur = cost(prm)
    for st in ((1.0, 1.0, np.radians(1.0)), (0.5, 0.5, np.radians(0.5)), (0.25, 0.25, np.radians(0.25))):
        better = True
        while better:
            better = False
            for d in range(3):
                for sg in (-1.0, 1.0):
                    q = prm.copy()
                    q[d] += sg * st[d]
                    c2 = cost(q)
                    if c2 < cur:
                        cur, prm, better = c2, q, True
    T_new = desk_pose(A, z, *prm)
    moved = float(np.linalg.norm(project_base(T_new, KEYPOINTS_CAD_MM, cam)
                                 - project_base(desk_pose(A, z, *start), KEYPOINTS_CAD_MM, cam), axis=1).max())
    return T_new, cur, moved


def set_pose(inst: Instance, T_base: np.ndarray, cam: Camera, src: int) -> None:
    """자세 하나로 점을 다시 놓는다: 자리는 투영, 보임/가려짐은 자기 몸 가림, 자동 점은 비워 둠 (fit 이 채움)."""
    T_cam = np.linalg.inv(cam.T_base_cam) @ T_base
    inst.uv = project_base(T_base, KEYPOINTS_CAD_MM, cam)
    hid = self_hidden(T_cam)
    inst.vis = np.where(IS_AUTO, UNSET, np.where(hid, HIDDEN, VISIBLE))
    inst.src = np.where(IS_AUTO, SRC_MODEL, src)


# ---------------------------------------------------------------- 브라켓 하나


@dataclass
class Instance:
    uv: np.ndarray = field(default_factory=lambda: np.zeros((N_KPT, 2)))
    vis: np.ndarray = field(default_factory=lambda: np.zeros(N_KPT, int))
    desk: bool = True
    # 점마다 출처 (SRC_MODEL / SRC_SAVED / SRC_EDITED). 자세 맞출 때 무게가 다르다.
    src: np.ndarray = field(default_factory=lambda: np.zeros(N_KPT, int))
    # 자세 결과 (fit 이 채움)
    T_cam_obj: np.ndarray | None = None
    proj: np.ndarray | None = None
    err: np.ndarray | None = None
    box: tuple[float, float, float, float] | None = None
    hidden_self: np.ndarray | None = None
    outline: list | None = None
    cls: int = -1
    free_tilt: float | None = None
    problem: str = ""


CLASS_KO = {0: "세우기", 1: "눕히기", 2: "비스듬히", -1: "-"}


def fit(inst: Instance, cam: Camera, width: int, height: int) -> None:
    """찍은 점으로 자세를 구해 가려진 점 · 자동 점 · 박스 · 윤곽 · 재투영 오차를 채운다. 실패하면 problem 에 이유."""
    prev_T = None if inst.T_cam_obj is None else cam.T_base_cam @ inst.T_cam_obj
    inst.T_cam_obj = inst.proj = inst.err = inst.box = inst.hidden_self = inst.outline = None
    inst.cls, inst.free_tilt, inst.problem = -1, None, ""
    seen = (inst.vis == VISIBLE) & ~IS_AUTO
    inst.vis[IS_AUTO] = UNSET
    obj = KEYPOINTS_CAD_MM[seen].astype(np.float64)
    img = inst.uv[seen].astype(np.float64)
    n = int(seen.sum())
    wts = SRC_WEIGHT[inst.src[seen]]
    T_free = None
    if n >= 4:
        # 무게는 점을 그만큼 되풀이해서 넣는 것으로 (OpenCV PnP 에 무게 인자가 없다)
        rep_n = np.round(wts).astype(int)
        obj_w, img_w = np.repeat(obj, rep_n, axis=0), np.repeat(img, rep_n, axis=0)
        ok, rvec, tvec = cv2.solvePnP(obj_w, img_w, cam.K, cam.dist, flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            rvec, tvec = cv2.solvePnPRefineLM(obj_w, img_w, cam.K, cam.dist, rvec, tvec)
            T_free = np.eye(4)
            T_free[:3, :3] = cv2.Rodrigues(rvec)[0]
            T_free[:3, 3] = tvec.ravel()
    if inst.desk:
        if n < DESK_MIN_PTS:
            inst.problem = f"보이는 점 {n}개 (책상 붙이기는 {DESK_MIN_PTS}개 이상)"
            return
        res = fit_desk(obj, img, cam, wts, prev_T)
        if res is None:
            inst.problem = "책상 붙이기 실패"
            return
        T_base, inst.cls, _e = res
        T = np.linalg.inv(cam.T_base_cam) @ T_base
        if T_free is not None:
            inst.free_tilt = _rot_deg(T_free[:3, :3], T[:3, :3])
    else:
        if T_free is None:
            inst.problem = f"보이는 점 {n}개 (책상 붙이기 끄면 4개 이상)"
            return
        T = T_free
    z = float(T[2, 3])
    if not (DEPTH_OK_MM[0] <= z <= DEPTH_OK_MM[1]):
        inst.problem = f"자세가 이상함 (깊이 {z:.0f} mm). 점 번호 · 위치 확인"
        return
    rvec = cv2.Rodrigues(T[:3, :3])[0]
    tvec = T[:3, 3]
    proj = cv2.projectPoints(KEYPOINTS_CAD_MM, rvec, tvec, cam.K, cam.dist)[0].reshape(-1, 2)
    inst.T_cam_obj, inst.proj = T, proj
    inst.hidden_self = self_hidden(T)
    inst.err = np.where(seen, np.linalg.norm(proj - inst.uv, axis=1), np.nan)
    # 가려진 점은 자세 자리로, 자동 점은 자리와 보임/가려짐 모두 자세에서
    hid = (inst.vis == HIDDEN) & ~IS_AUTO
    inst.uv[hid] = proj[hid]
    inst.uv[IS_AUTO] = proj[IS_AUTO]
    inst.vis[IS_AUTO] = np.where(inst.hidden_self[IS_AUTO], HIDDEN, VISIBLE)
    # CAD 윤곽 (원본 픽셀): 삼각형을 다 칠한 마스크의 테두리
    v, f = mesh_mm()
    pv = cv2.projectPoints(v, rvec, tvec, cam.K, cam.dist)[0].reshape(-1, 2)
    x0, y0 = np.clip(pv.min(axis=0), 0, [width, height])
    x1, y1 = np.clip(pv.max(axis=0), 0, [width, height])
    if x1 - x0 < 4 or y1 - y0 < 4:
        inst.problem = "브라켓이 화면 밖"
        return
    inst.box = (float(x0), float(y0), float(x1), float(y1))
    sc = 4.0  # 윤곽을 부드럽게 (4배 해상도로 칠함)
    ox, oy = int(x0) - 2, int(y0) - 2
    mw, mh = int((x1 - ox + 4) * sc), int((y1 - oy + 4) * sc)
    mask = np.zeros((mh, mw), np.uint8)
    polys = ((pv[f] - [ox, oy]) * sc).astype(np.int32)
    cv2.fillPoly(mask, list(polys), 255)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inst.outline = [c.reshape(-1, 2) / sc + [ox, oy] for c in cs]


def canonicalize(inst: Instance, cam: Camera) -> bool:
    """대칭 두 자세 중 시뮬 라벨 규칙(canonical_T)의 쪽으로 번호를 다시 매긴다. 바꿨으면 True."""
    if inst.T_cam_obj is None:
        return False
    T_base_obj = cam.T_base_cam @ inst.T_cam_obj
    G = canonical_T(T_base_obj, cam.T_base_cam, symmetries())
    S = np.linalg.inv(T_base_obj) @ G
    if np.allclose(S[:3, :3], np.eye(3), atol=1e-6):
        return False
    perm = symmetry_perm(S)
    inst.uv = inst.uv[perm].copy()
    inst.vis = inst.vis[perm].copy()
    inst.src = inst.src[perm].copy()
    inst.T_cam_obj = inst.T_cam_obj @ S
    if inst.proj is not None:
        inst.proj = inst.proj[perm].copy()
    if inst.err is not None:
        inst.err = inst.err[perm].copy()
    return True


# ---------------------------------------------------------------- 파일


def label_line(inst: Instance, width: int, height: int) -> str:
    """YOLO pose 한 줄: 0 cx cy w h (x y v)×N, 0~1 값. 화면 밖 점은 0 0 0 (시뮬 라벨과 같다).

    점 자리는 찍은 자리가 아니라 맞춘 자세에서 CAD 로 다시 투영한 자리다 (시뮬 라벨과 같은 방식).
    키포인트는 둥근 모서리 바깥의 가상 꼭짓점이라 사람이 실물 모서리에 찍으면 1~3 px 안쪽이 되는데, 이것도 같이 맞춰진다.
    """
    x0, y0, x1, y1 = inst.box
    parts = ["0", f"{(x0 + x1) / 2 / width:.6f}", f"{(y0 + y1) / 2 / height:.6f}",
             f"{(x1 - x0) / width:.6f}", f"{(y1 - y0) / height:.6f}"]
    uv = inst.proj if inst.proj is not None else inst.uv
    for (u, v), vis in zip(uv, inst.vis):
        inside = 0 <= u < width and 0 <= v < height
        if vis == UNSET or not inside:
            parts += ["0", "0", "0"]
        else:
            parts += [f"{u / width:.6f}", f"{v / height:.6f}", str(int(vis))]
    return " ".join(parts)


def save_labels(path: Path, insts: list[Instance], cam: Camera, width: int, height: int) -> list[str]:
    """맞춘 브라켓만 저장. 저장 못 한 브라켓의 이유 목록을 돌려준다."""
    lines, skipped = [], []
    for k, inst in enumerate(insts):
        fit(inst, cam, width, height)
        if inst.box is None:
            skipped.append(f"브라켓 {k + 1}: {inst.problem}")
            continue
        if canonicalize(inst, cam):
            fit(inst, cam, width, height)
        lines.append(label_line(inst, width, height))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    # 다시 열 때 책상 붙이기 상태를 살리려고, 그리고 나중에 자세 판단을 채점하려고 옆에 적어 둔다 (학습에는 안 쓰임).
    kept = [i for i in insts if i.box is not None]
    meta = {"desk": [bool(i.desk) for i in kept], "pose_class": [int(i.cls) for i in kept],
            "pose_class_ko": [CLASS_KO[int(i.cls)] for i in kept]}
    path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return skipped


def load_labels(path: Path, width: int, height: int) -> list[Instance]:
    out = []
    meta = path.with_suffix(".meta.json")
    desk = json.loads(meta.read_text(encoding="utf-8")).get("desk", []) if meta.is_file() else []
    for line in path.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) < 5 + 3 * N_KPT:
            continue
        kp = np.asarray(v[5:5 + 3 * N_KPT], float).reshape(N_KPT, 3)
        inst = Instance()
        inst.uv = kp[:, :2] * [width, height]
        inst.vis = kp[:, 2].astype(int)
        inst.src = np.where(IS_AUTO, SRC_MODEL, SRC_SAVED)  # 저장된 라벨은 사람이 확인한 점
        if len(out) < len(desk):
            inst.desk = bool(desk[len(out)])
        out.append(inst)
    return out


def model_guess(model, bgr: np.ndarray, cam: Camera) -> list[Instance]:
    """모델이 찾은 브라켓 → 미리 찍힌 점. 자동 점은 빼고, 자세로 볼 때 브라켓 몸에 가려진 점은 '가려짐'.
    모델은 가려진 점도 높은 신뢰도로 찍어서 신뢰도로는 못 가린다."""
    r = model.predict(bgr, conf=MODEL_CONF, verbose=False)[0]
    out = []
    if r.keypoints is None or r.boxes is None:
        return out
    h, w = bgr.shape[:2]
    for pts in r.keypoints.xy.cpu().numpy():
        inst = Instance()
        inst.uv = pts.astype(float)
        inst.vis = np.where(IS_AUTO, UNSET, VISIBLE)
        fit(inst, cam, w, h)
        if inst.hidden_self is not None:
            hide = inst.hidden_self & ~IS_AUTO
            if hide.any() and int(((inst.vis == VISIBLE) & ~hide & ~IS_AUTO).sum()) >= DESK_MIN_PTS:
                inst.vis[hide] = HIDDEN
                fit(inst, cam, w, h)
        out.append(inst)
    return out


# ---------------------------------------------------------------- 안내 그림 · 확대경

GUIDE_H = 300
ZOOM = 4
ZOOM_SRC = 60  # 확대경이 보는 원본 범위 (px)
_GUIDE: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def render_cad(eye, width: int, height: int, *, bg: int = 40, fill: float = 0.75,
               extra: np.ndarray | None = None) -> tuple[np.ndarray, "callable"]:
    """CAD 를 eye 방향에서 본 그림 (정사영, 광선으로 그려 가림이 정확하다). (그림, CAD mm 점 → 픽셀 함수).

    extra 를 주면 그 점들(mm)도 화면 안에 들어오게 크기를 정한다.
    """
    import open3d as o3d

    eye = np.asarray(eye, float)
    eye = eye / np.linalg.norm(eye)
    right = np.cross([0.0, 0.0, 1.0], eye)
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    v, _ = mesh_mm()
    pts = v if extra is None else np.vstack([v, extra])
    a, b = pts @ right, pts @ up
    sc = fill * min(width / (a.max() - a.min()), height / (b.max() - b.min()))
    a0, b0 = (a.max() + a.min()) / 2, (b.max() + b.min()) / 2
    xs = (np.arange(width) - width / 2) / sc + a0
    ys = -(np.arange(height) - height / 2) / sc + b0
    X, Y = np.meshgrid(xs, ys)
    org = X[..., None] * right + Y[..., None] * up + 300.0 * eye
    rays = np.concatenate([org, np.broadcast_to(-eye, org.shape)], axis=-1).astype(np.float32)
    hit = _ray_scene().cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))
    t = hit["t_hit"].numpy().reshape(height, width)
    n = hit["primitive_normals"].numpy().reshape(height, width, 3)
    light = np.array([0.4, -0.5, 0.75]) + eye
    light /= np.linalg.norm(light)
    shade = 0.3 + 0.7 * np.abs(n @ light)
    img = np.full((height, width, 3), bg, np.uint8)
    ok = np.isfinite(t)
    g = (60 + 175 * shade[ok]).astype(np.uint8)
    img[ok] = g[:, None]

    def to_px(p_mm: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p_mm)
        return np.stack([(p @ right - a0) * sc + width / 2, -(p @ up - b0) * sc + height / 2], axis=1)

    return np.ascontiguousarray(img), to_px


def _guide_base() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """세운 브라켓을 앞 · 오른쪽 · 위에서 본 그림 (한 번만 그림). (그림, 키포인트 픽셀, 가려진 점)."""
    global _GUIDE
    if _GUIDE is not None:
        return _GUIDE
    eye = np.array([1.0, -0.9, 0.8])
    eye /= np.linalg.norm(eye)
    img, to_px = render_cad(eye, PANEL_W, GUIDE_H, extra=KEYPOINTS_CAD_MM)
    hid = hidden_from(KEYPOINTS_CAD_MM + 300.0 * eye)
    _GUIDE = (img, to_px(KEYPOINTS_CAD_MM), hid)
    return _GUIDE


def guide_panel(current: int, height: int, zoom: np.ndarray | None) -> np.ndarray:
    """오른쪽: 위 = 브라켓 그림 + 번호 (지금 번호 크게), 가운데 = 안내 글, 아래 = 확대경."""
    panel = np.full((height, PANEL_W, 3), 40, np.uint8)
    base, kp_px, hid = _guide_base()
    panel[:GUIDE_H] = base
    # 가려진 점을 먼저, 보이는 점을 위에 그린다. 가려진 점 번호는 괄호. 자동 점은 네모.
    for i in sorted(range(N_KPT), key=lambda k: (not hid[k], k == current)):
        p = (int(kp_px[i][0]), int(kp_px[i][1]))
        on = i == current
        r = 9 if on else 5
        if IS_AUTO[i]:
            cv2.rectangle(panel, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), COLORS[i], 1)
        else:
            cv2.circle(panel, p, r, COLORS[i], 2 if hid[i] else -1)
        if on:
            cv2.circle(panel, p, 13, (255, 255, 255), 2)
        txt = f"({i})" if hid[i] else str(i)
        dx = -30 if hid[i] else 9
        put_text(panel, txt, (p[0] + dx, p[1] + (12 if hid[i] else -6)), 0.7 if on else 0.45,
                 COLORS[i] if i != 9 else (230, 230, 230), 2 if on else 1)
    y = GUIDE_H + 18
    for line in (f"지금 번호 {current}: {KPT_NAMES[current]}", "보이는 점만 찍기. 안 보이면 h",
                 f"네모({', '.join(map(str, AUTO_IDX))}) = 자동, 안 찍음", "번호는 대칭 상관없이 (저장 때 자동 정리)",
                 "하늘색 선 = CAD 윤곽. 실물과 맞는지 보기", "+ = 저장될 자리 (자세에서 다시 투영)"):
        put_text(panel, line, (8, y), 0.45, (220, 220, 220), 1)
        y += 20
    if zoom is not None:
        zh, zw = zoom.shape[:2]
        y0 = height - zh - 8
        x0 = (PANEL_W - zw) // 2
        if y0 > y:
            panel[y0:y0 + zh, x0:x0 + zw] = zoom
            cv2.rectangle(panel, (x0 - 1, y0 - 1), (x0 + zw, y0 + zh), (200, 200, 200), 1)
    return panel


def zoom_view(bgr: np.ndarray, insts: list["Instance"], sel: int, cur: int, center: tuple[float, float]) -> np.ndarray:
    """마우스 주변 ZOOM_SRC px 를 ZOOM 배로. 점 · 윤곽 · 십자선을 같이 그린다."""
    h, w = bgr.shape[:2]
    half = ZOOM_SRC // 2
    cx = int(min(max(center[0], half), w - half))
    cy = int(min(max(center[1], half), h - half))
    crop = bgr[cy - half:cy + half, cx - half:cx + half]
    big = cv2.resize(crop, (ZOOM_SRC * ZOOM, ZOOM_SRC * ZOOM), interpolation=cv2.INTER_NEAREST)
    off = np.array([cx - half, cy - half])
    for k, inst in enumerate(insts):
        if k == sel and inst.outline:
            for c in inst.outline:
                cv2.polylines(big, [((c - off) * ZOOM).astype(np.int32)], True, OUTLINE_BGR, 1, cv2.LINE_AA)
        for i, ((u, v), vv) in enumerate(zip(inst.uv, inst.vis)):
            if vv == UNSET:
                continue
            p = (int((u - off[0]) * ZOOM), int((v - off[1]) * ZOOM))
            if 0 <= p[0] < big.shape[1] and 0 <= p[1] < big.shape[0]:
                r = 6 if k == sel else 4
                if IS_AUTO[i]:
                    cv2.rectangle(big, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), COLORS[i], 1)
                else:
                    cv2.circle(big, p, r, COLORS[i], -1 if vv == VISIBLE else 2)
                if k == sel and i == cur:
                    cv2.circle(big, p, 11, (255, 255, 255), 1)
    m = (int((center[0] - off[0]) * ZOOM), int((center[1] - off[1]) * ZOOM))
    cv2.line(big, (m[0], 0), (m[0], big.shape[0]), (0, 255, 255), 1)
    cv2.line(big, (0, m[1]), (big.shape[1], m[1]), (0, 255, 255), 1)
    return big


# ---------------------------------------------------------------- 창

BRIGHT_NAMES = ("", "화면 밝게", "어두운 곳 늘리기")


def enhance(bgr: np.ndarray, mode: int) -> np.ndarray:
    """화면에 보일 영상만 바꾼다 (사진 · 라벨은 그대로).

    1 = 어두운 곳 밝게 (감마), 2 = 어두운 곳 늘리기 (0~70 → 0~255, 검은 브라켓 면 구분용. 책상은 하얗게 날아감).
    """
    if mode == 1:
        lut = (255.0 * (np.arange(256) / 255.0) ** 0.45).astype(np.uint8)
        return cv2.LUT(bgr, lut)
    if mode == 2:
        lut = np.clip(np.arange(256) * (255.0 / 70.0), 0, 255).astype(np.uint8)
        return cv2.LUT(bgr, lut)
    return bgr


class Session:
    def __init__(self, cam: Camera, scale: float) -> None:
        self.cam = cam
        self.scale = scale
        self.insts: list[Instance] = []
        self.sel = 0
        self.cur = 0
        self.drag: int | None = None
        self.history: list[tuple[list[Instance], int, int]] = []
        self.w = self.h = 0
        self.note = ""
        self.hover: tuple[float, float] | None = None
        self.bright = 0  # 사진을 바꿔도 유지
        self.img: np.ndarray | None = None  # 지금 사진 (자동 맞춤용)

    # 상태
    def push(self) -> None:
        self.history.append((copy.deepcopy(self.insts), self.sel, self.cur))
        self.history = self.history[-50:]

    def undo(self) -> None:
        if self.history:
            self.insts, self.sel, self.cur = self.history.pop()
            self.refit()

    def refit(self) -> None:
        for inst in self.insts:
            fit(inst, self.cam, self.w, self.h)

    def selected(self) -> Instance | None:
        return self.insts[self.sel] if 0 <= self.sel < len(self.insts) else None

    def next_unset(self) -> None:
        """다음 안 찍은 번호 (자동 점은 건너뛴다)."""
        inst = self.selected()
        if inst is None:
            return
        for k in range(1, N_KPT + 1):
            i = (self.cur + k) % N_KPT
            if inst.vis[i] == UNSET and not IS_AUTO[i]:
                self.cur = i
                return

    # 마우스
    def _grab(self, x: float, y: float) -> tuple[int, int] | None:
        """(브라켓, 점) 가까운 것. 선택된 브라켓을 먼저 본다. 자동 점은 잡지 않는다."""
        order = [self.sel] + [k for k in range(len(self.insts)) if k != self.sel]
        lim = GRAB_PX / self.scale
        for k in order:
            if not (0 <= k < len(self.insts)):
                continue
            inst = self.insts[k]
            d = np.linalg.norm(inst.uv - [x, y], axis=1)
            d[(inst.vis == UNSET) | IS_AUTO] = np.inf
            i = int(d.argmin())
            if d[i] <= lim:
                return k, i
        return None

    def on_mouse(self, event, xd, yd, _flags, _param) -> None:
        yd -= BAR_H  # 위 글씨 칸
        if yd < 0 and event != cv2.EVENT_MOUSEMOVE:
            return
        x, y = xd / self.scale, yd / self.scale
        if event == cv2.EVENT_MOUSEMOVE and x < self.w:
            self.hover = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= self.w:
                return
            hit = self._grab(x, y)
            if hit is not None:
                self.push()
                self.sel, self.cur = hit
                self.drag = self.cur
                inst = self.insts[self.sel]
                inst.uv[self.cur] = (x, y)
                inst.vis[self.cur] = VISIBLE
                inst.src[self.cur] = SRC_EDITED
                return
            if IS_AUTO[self.cur]:
                self.note = f"{self.cur}번은 자동 점이라 찍지 않습니다"
                return
            self.push()
            if not self.insts:
                self.insts.append(Instance())
                self.sel, self.cur = 0, 0
            inst = self.insts[self.sel]
            inst.uv[self.cur] = (x, y)
            inst.vis[self.cur] = VISIBLE
            inst.src[self.cur] = SRC_EDITED
            self.refit()
            self.next_unset()
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            inst = self.insts[self.sel]
            inst.uv[self.drag] = (min(max(x, 0), self.w - 1), min(max(y, 0), self.h - 1))
        elif event == cv2.EVENT_LBUTTONUP and self.drag is not None:
            self.drag = None
            self.refit()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo()

    # 키
    def key(self, k: int) -> None:
        inst = self.selected()
        if ord("0") <= k <= ord("9") and k - ord("0") < N_KPT:
            self.cur = k - ord("0")
            self.note = f"{self.cur}번은 자동 점 (자세에서 채움)" if IS_AUTO[self.cur] else ""
        elif k == ord("h") and inst is not None and not IS_AUTO[self.cur]:
            self.push()
            inst.vis[self.cur] = HIDDEN
            inst.src[self.cur] = SRC_EDITED
            self.refit()
            self.next_unset()
        elif k == ord("x") and inst is not None and not IS_AUTO[self.cur]:
            self.push()
            inst.vis[self.cur] = UNSET
            self.refit()
        elif k == ord("f") and inst is not None:
            if inst.T_cam_obj is None or not inst.desk or self.img is None:
                self.note = "자동 맞춤은 윤곽이 먼저 있어야 하고 책상 붙이기가 켜져 있어야 합니다"
                return
            res = fit_edges(self.img, self.cam.T_base_cam @ inst.T_cam_obj, self.cam)
            if res is None:
                self.note = "자동 맞춤 실패 (브라켓이 화면 가장자리)"
                return
            self.push()
            T_new, dist, moved = res
            set_pose(inst, T_new, self.cam, SRC_EDITED)
            self.refit()
            self.note = f"자동 맞춤: 테두리와 평균 {dist:.1f} px, {moved:.1f} px 움직임"
            if dist > EDGE_WARN_PX:
                self.note += "  (불확실: 윤곽 확인, 아니면 u)"
        elif k == ord("g") and inst is not None:
            self.push()
            inst.desk = not inst.desk
            self.refit()
            self.note = "책상 붙이기 켬" if inst.desk else "책상 붙이기 끔 (보통 PnP, 보이는 점 4개 이상)"
        elif k == ord("b"):
            self.bright = (self.bright + 1) % len(BRIGHT_NAMES)
        elif k == ord("a"):
            self.push()
            self.insts.append(Instance())
            self.sel, self.cur = len(self.insts) - 1, 0
        elif k == 9 and self.insts:  # Tab
            self.sel = (self.sel + 1) % len(self.insts)
        elif k == ord("D") and inst is not None:
            self.push()
            self.insts.pop(self.sel)
            self.sel = max(0, min(self.sel, len(self.insts) - 1))
        elif k == ord("u"):
            self.undo()

    # 그리기
    def draw(self, bgr: np.ndarray, name: str, index: int, total: int) -> np.ndarray:
        s = self.scale
        bgr = enhance(bgr, self.bright)
        vis = cv2.resize(bgr, (int(self.w * s), int(self.h * s)), interpolation=cv2.INTER_LINEAR)
        for k, inst in enumerate(self.insts):
            on = k == self.sel
            if inst.box is not None and not on:
                x0, y0, x1, y1 = (np.array(inst.box) * s).astype(int)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (160, 160, 160), 1)
            if on and inst.outline:
                for c in inst.outline:
                    cv2.polylines(vis, [(c * s).astype(np.int32)], True, OUTLINE_BGR, 1, cv2.LINE_AA)
            if on and inst.proj is not None:
                # 저장될 자리 (자세에서 다시 투영). 찍은 점과 멀면 그 점이 틀렸거나 자세 클래스가 틀림.
                for i, (u, v) in enumerate(inst.proj * s):
                    if inst.vis[i] != UNSET:
                        cv2.drawMarker(vis, (int(u), int(v)), COLORS[i], cv2.MARKER_CROSS, 9, 1)
            for i, ((u, v), vv) in enumerate(zip(inst.uv * s, inst.vis)):
                if vv == UNSET:
                    continue
                p = (int(u), int(v))
                r = 5 if on else 3
                if IS_AUTO[i]:
                    cv2.rectangle(vis, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), COLORS[i], 1)
                else:
                    cv2.circle(vis, p, r, COLORS[i], -1 if vv == VISIBLE else 1)
                if on:
                    if i == self.cur:
                        cv2.circle(vis, p, 10, (255, 255, 255), 1)
                    put_text(vis, str(i), (p[0] + 6, p[1] - 6), 0.45, COLORS[i] if i != 9 else (230, 230, 230), 1)
        big = ""
        top = [f"{index + 1}/{total} {name}  브라켓 {self.sel + 1 if self.insts else 0}/{len(self.insts)}  지금 번호 {self.cur}"
               + (f"  [{BRIGHT_NAMES[self.bright]}]" if self.bright else "")]
        inst = self.selected()
        warn = []
        if inst is not None:
            if inst.problem:
                warn.append(inst.problem)
            elif inst.err is not None:
                pose = f"자세: {CLASS_KO[inst.cls]}" if inst.desk else "자세: - (책상 붙이기 끔)"
                detail = f"재투영 평균 {np.nanmean(inst.err):.1f} px, 최대 {np.nanmax(inst.err):.1f} px"
                if inst.free_tilt is not None:
                    detail += f"   자유 PnP 와 {inst.free_tilt:.0f}°"
                big = pose
                top.append(detail)
                e = np.nan_to_num(inst.err, nan=0.0)
                bad = [i for i in range(N_KPT) if e[i] > WARN_PX]
                if bad:
                    warn.append(f"어긋난 점: {bad} (윤곽과 안 맞으면 그 점을 옮기기)")
                if inst.desk and inst.free_tilt is not None and inst.free_tilt > WARN_TILT_DEG:
                    warn.append("자유 PnP 와 많이 다름: 자세 클래스가 맞는지 윤곽 확인")
        # 글씨는 사진 위 따로 둔 칸에 (사진을 가리지 않게).
        reds = (warn + ([self.note] if self.note else []))[:3]
        bar = np.full((BAR_H, vis.shape[1] + PANEL_W, 3), 30, np.uint8)
        y = 32
        put_text(bar, big or "자세: -", (10, y), 1.0, (0, 255, 0), 2)
        y += 40
        for line in top:
            put_text(bar, line, (10, y), 0.65, (120, 255, 120), 2)
            y += 28
        for line in reds:
            put_text(bar, line, (10, y), 0.65, (80, 80, 255), 2)
            y += 28
        help_lines = ["클릭 = 지금 번호 찍기,  끌기 = 옮기기", "0~9 번호   h 가려짐   x 지우기",
                      "f 테두리 자동 맞춤   g 책상 붙이기   b 밝게", "a 브라켓 추가   Tab 다음   D 삭제",
                      "u / 우클릭 되돌리기", "s 저장   n / p 다음 · 이전 사진   q 끝"]
        x_help = bar.shape[1] - 420
        for k, line in enumerate(help_lines):
            put_text(bar, line, (x_help, 26 + 27 * k), 0.58, (230, 230, 230), 1)
        zoom = zoom_view(bgr, self.insts, self.sel, self.cur, self.hover) if self.hover is not None else None
        panel = guide_panel(self.cur, vis.shape[0], zoom)
        return np.vstack([bar, np.hstack([vis, panel])])


def _image_paths(images_dir: Path, split: str) -> list[Path]:
    paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    sp = images_dir.parent / "split.json"
    if split != "all":
        if not sp.is_file():
            raise SystemExit(f"--split {split} 인데 {sp} 가 없습니다")
        names = set(json.loads(sp.read_text(encoding="utf-8"))[split])
        paths = [p for p in paths if p.name in names]
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="브라켓 키포인트 라벨 (YOLO-pose)")
    ap.add_argument("--images", type=Path, default=KPT_DIR / "images")
    ap.add_argument("--labels", type=Path, default=None, help="기본: 사진 폴더 옆 labels")
    ap.add_argument("--camera", type=Path, default=None, help="camera.json (K, T_base_cam, dist). 기본: 폴더 옆 → calib/ → 이 PC")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="미리 찍을 모델. '' 이면 안 씀")
    ap.add_argument("--split", choices=("all", "train", "test"), default="all", help="split.json 의 어느 쪽만")
    ap.add_argument("--scale", type=float, default=1.5, help="화면 확대 (정밀하게 찍기)")
    args = ap.parse_args()

    images = _image_paths(args.images, args.split)
    if not images:
        raise SystemExit(f"사진이 없습니다: {args.images}")
    labels_dir = args.labels or args.images.parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    # 키포인트 묶음이 다른 라벨과 섞이지 않게 폴더에 적어 둔다.
    tag = labels_dir / "kpt_set.txt"
    if tag.is_file() and tag.read_text().strip() != KPT_SET:
        raise SystemExit(f"이 폴더는 '{tag.read_text().strip()}' 키포인트로 라벨됨. 지금은 '{KPT_SET}' (SIM_KPT_SET 확인)")
    tag.write_text(KPT_SET + "\n")
    cam = load_camera(args.images, args.camera)
    print(f"사진 {len(images)}장 ({args.split}), 카메라: {cam.source}, 키포인트: {KPT_SET}, 자동 점: {AUTO_IDX}")

    model = None
    if args.weights and Path(args.weights).is_file():
        from ultralytics import YOLO

        model = YOLO(args.weights)
        print(f"미리 찍기 모델: {args.weights}")

    sess = Session(cam, args.scale)
    idx = 0

    def open_image(i: int) -> np.ndarray:
        bgr = cv2.imread(str(images[i]))
        sess.h, sess.w = bgr.shape[:2]
        sess.img = bgr
        lab = labels_dir / f"{images[i].stem}.txt"
        if lab.is_file():
            sess.insts = load_labels(lab, sess.w, sess.h)
            sess.note = ""
        elif model is not None:
            sess.insts = model_guess(model, bgr, cam)
            sess.note = f"모델이 미리 찍음 ({len(sess.insts)}개). 윤곽 확인 후 s 또는 n"
        else:
            sess.insts = []
            sess.note = ""
        sess.sel, sess.cur, sess.history, sess.drag = 0, 0, [], None
        sess.refit()
        return bgr

    def save(i: int) -> None:
        skipped = save_labels(labels_dir / f"{images[i].stem}.txt", sess.insts, cam, sess.w, sess.h)
        msg = f"저장 {images[i].name}: {len(sess.insts) - len(skipped)}개"
        if skipped:
            msg += "  (못 넣음: " + "; ".join(skipped) + ")"
        print(msg)
        sess.note = msg

    bgr = open_image(idx)
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, sess.on_mouse)
    try:
        while True:
            cv2.imshow(WIN, sess.draw(bgr, images[idx].name, idx, len(images)))
            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue
            if k == ord("q"):
                save(idx)
                break
            if k == ord("s"):
                save(idx)
                bgr = open_image(idx)  # 정리된 번호로 다시 보여 준다
                continue
            if k in (ord("n"), ord("p")):
                save(idx)
                idx = max(0, min(idx + (1 if k == ord("n") else -1), len(images) - 1))
                bgr = open_image(idx)
                continue
            sess.key(k)
    finally:
        cv2.destroyAllWindows()
    done = len(list(labels_dir.glob("*.txt"))) - 1
    print(f"라벨 {done}장 → {labels_dir}")


if __name__ == "__main__":
    main()
