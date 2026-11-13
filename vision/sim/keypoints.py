"""YOLO-pose 용 브라켓 키포인트 (CAD 프레임 mm) 와 시뮬 자동 라벨.

CAD 프레임: 바닥 판은 z −2~2 (x −2~40), 세운 판은 x −2~2 (z −2~40), 폭 y ±17.5.

  0, 1  바닥 판 끝 아래 모서리   (40, ±17.5, −2)
  2, 3  세운 판 끝 바깥 모서리   (−2, ±17.5, 40)
  4, 5  뒤꿈치 바깥 모서리       (−2, ±17.5, −2)
  6     바닥 판 구멍 중심 (윗면)  (22, 0, 2)
  7     세운 판 구멍 중심 (안쪽면) (2, 0, 22)
  8, 9  안쪽 모서리             (2, ±17.5, 2)

대각선 축 180° 대칭 S 로 돌리면 0↔3, 1↔2, 4↔5, 6↔7, 8↔9 가 서로 바뀐다. 그림만 보고는 둘을 구분할 수 없으므로
라벨은 정해진 규칙으로 둘 중 하나(대표 자세)를 골라 단다 (canonical_T).
"""

from __future__ import annotations

import os

import numpy as np

_MIXED = np.array(
    [
        [40.0, 17.5, -2.0],
        [40.0, -17.5, -2.0],
        [-2.0, 17.5, 40.0],
        [-2.0, -17.5, 40.0],
        [-2.0, 17.5, -2.0],
        [-2.0, -17.5, -2.0],
        [22.0, 0.0, 2.0],
        [2.0, 0.0, 22.0],
        [2.0, 17.5, 2.0],
        [2.0, -17.5, 2.0],
    ]
)
# 모서리만 10개 (v5 비교용). 옆벽(y ±17.5)은 판 끝에서 높이 6 mm 로 시작해 곡면으로 올라가 꼭대기 x 6 mm 에서 끝난다.
#   0, 1 바닥 판 끝 아래 (40, ±17.5, −2)    2, 3 세운 판 꼭대기 바깥 (−2, ±17.5, 40)
#   4, 5 뒤꿈치 바깥 (−2, ±17.5, −2)       6, 7 바닥 판 끝 위, 곡면 시작 (40, ±17.5, 6)
#   8, 9 세운 판 꼭대기 안쪽, 곡면 끝 (6, ±17.5, 40)
# 대칭 S 로 0↔3, 1↔2, 4↔5, 6↔9, 7↔8 이 바뀐다.
_CORNERS = np.array(
    [
        [40.0, 17.5, -2.0],
        [40.0, -17.5, -2.0],
        [-2.0, 17.5, 40.0],
        [-2.0, -17.5, 40.0],
        [-2.0, 17.5, -2.0],
        [-2.0, -17.5, -2.0],
        [40.0, 17.5, 6.0],
        [40.0, -17.5, 6.0],
        [6.0, 17.5, 40.0],
        [6.0, -17.5, 40.0],
    ]
)
# 어느 키포인트 묶음을 쓸지: 환경 변수 SIM_KPT_SET = mixed (기본) / corners.
# 데이터 만들기 · 학습 · 평가가 모두 같은 값을 봐야 한다 (여러 프로세스로 나뉘어도 환경 변수는 이어진다).
KPT_SET = os.environ.get("SIM_KPT_SET", "mixed")
KEYPOINTS_CAD_MM = {"mixed": _MIXED, "corners": _CORNERS}[KPT_SET]
N_KPT = len(KEYPOINTS_CAD_MM)
# 영상을 좌우로 뒤집으면 브라켓은 y 거울상이 된다. 브라켓이 y=0 면에 거울 대칭이라 y± 짝만 바뀐다.
FLIP_IDX = {"mixed": [1, 0, 3, 2, 5, 4, 6, 7, 9, 8], "corners": [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]}[KPT_SET]

# 대칭 S 가 바꾸는 방향: d = (1, 0, −1)/√2 는 S 로 −d 가 된다.
_D_CAD = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
# 베이스 z 성분이 이보다 크면 위아래로 대표 자세를 고르고, 아니면 영상 좌우로 고른다.
_UP_DECIDE = 0.3


def canonical_T(T_base_obj: np.ndarray, T_base_cam: np.ndarray, symmetries: list[np.ndarray]) -> np.ndarray:
    """대칭으로 같은 자세 중 라벨에 쓸 하나를 고른다.

    1. d 가 책상 쪽(아래)을 확실히 향하는 쪽 → 세운 브라켓이면 '책상에 닿은 판 = 바닥 판'.
    2. d 가 거의 수평이면 (눕히기 · 비스듬히) 카메라 영상에서 d 가 오른쪽(+u)을 향하는 쪽.
    """
    T = np.asarray(T_base_obj, float)
    R_cam_base = np.asarray(T_base_cam, float)[:3, :3].T
    best, best_key = T, None
    for S in symmetries:
        G = T @ S
        d_base = G[:3, :3] @ _D_CAD
        if abs(float(d_base[2])) > _UP_DECIDE:
            key = (1, -float(d_base[2]))
        else:
            key = (0, float((R_cam_base @ d_base)[0]))
        if best_key is None or key > best_key:
            best, best_key = G, key
    return best


def project(
    K: np.ndarray, T_base_cam: np.ndarray, xyz_base: np.ndarray, dist: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """베이스 mm 점 → (픽셀 uv (N,2), 카메라 z mm (N,)). dist 를 주면 렌즈 왜곡까지 넣는다."""
    T_cam_base = np.linalg.inv(np.asarray(T_base_cam, float))
    pc = xyz_base @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    if dist is None:
        uv = (pc[:, :2] / pc[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    else:
        import cv2

        uv = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.asarray(dist, float))[0].reshape(-1, 2)
    return uv, pc[:, 2]


def keypoint_labels(
    T_label: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    depth_mm: np.ndarray,
    *,
    width: int,
    height: int,
    vis_tol_mm: float = 4.0,
    dist: np.ndarray | None = None,
) -> np.ndarray:
    """(N_KPT, 3) = u, v, 보임(2 보임 / 1 가려짐 / 0 화면 밖). 보임은 그 픽셀의 렌더 뎁스와 키포인트 깊이를 비교한다.

    영상을 렌즈 왜곡했으면 dist 를 같이 줘서 키포인트도 왜곡된 자리에 둔다 (depth_mm 도 왜곡된 것).
    """
    pts = KEYPOINTS_CAD_MM @ T_label[:3, :3].T + T_label[:3, 3]
    uv, z = project(K, T_base_cam, pts, dist)
    out = np.zeros((N_KPT, 3))
    out[:, :2] = uv
    for i, ((u, v), zi) in enumerate(zip(uv, z)):
        if not (0 <= u < width and 0 <= v < height) or zi <= 0:
            continue
        # 그 픽셀에 키포인트보다 vis_tol_mm 넘게 앞에 있는 면이 없으면 보임. 구멍 중심처럼 뒤가 비어도 보임.
        d = float(depth_mm[int(v), int(u)])
        seen = d <= 0 or d >= zi - vis_tol_mm
        out[i, 2] = 2 if seen else 1
    return out
