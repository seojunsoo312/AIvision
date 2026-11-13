"""YOLO 인스턴스마다 점군을 모으고 최상단을 고른다.

포즈(xyzrpy)는 여기 없다. 정본은 register.py (CAD). 로컬 면 PCA는 pose.debug.local_plane.
카메라·YOLO·시리얼은 열지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision.transforms import apply_T
from vision.depth_cloud import (
    Z_MAX_MM,
    Z_MIN_MM,
    depth_edge_mask,
    largest_component,
    plane_foreground_mask,
    points_from_mask,
)


@dataclass
class InstanceCloud:
    xyxy: list[float]
    roi: np.ndarray
    xyz: np.ndarray
    rgb: np.ndarray
    height_score: float
    n_zok: int = 0
    quad: np.ndarray | None = None  # (4, 2) 가로세로 박스 꼭짓점
    class_id: int | None = None


def height_above_plane(xyz: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """책상 평면 ax+by+cz+d=0. 카메라 쪽 물체는 부호 거리가 음수다."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    signed = pts @ np.asarray(plane[:3], dtype=np.float64) + float(plane[3])
    return -signed


def _height_score(xyz: np.ndarray, plane: np.ndarray | None, *, in_base: bool) -> float:
    if len(xyz) == 0:
        return float("-inf")
    if in_base:
        return float(np.percentile(xyz[:, 2], 95))
    if plane is not None:
        return float(np.percentile(height_above_plane(xyz, plane), 95))
    return float(-np.median(xyz[:, 2]))


class _Box:
    """YOLO 검출처럼 쓰는 상자 (xyxy 만, 클래스 없음)."""

    def __init__(self, xyxy):
        self.xyxy = [np.asarray(xyxy, dtype=np.float64)]
        self.cls = None


# 브라켓은 책상 위에 놓여 있다 (CAD 책상 위 세 자세 · 키포인트와 같은 전제). 가장 긴 변이 42 mm 라
# 어느 자세로 놓여도 이보다 높은 점은 브라켓일 수 없다: 흰 면에서 10~30 cm 가깝게 튀는 뎁스, 지나가는 팔 · 손.
BRACKET_MAX_HEIGHT_MM = 60.0

# YOLO 없이 뎁스로 찾는 물체: 이보다 작거나 (중앙 높이가) 높으면 브라켓이 아니다.
BLOB_MIN_PIXELS = 300
BLOB_MAX_HEIGHT_MM = BRACKET_MAX_HEIGHT_MM
_BLOB_BORDER_PX = 5
_BLOB_SCALE = 2
# 덩어리 둘레 띠(원래 픽셀로 8~20px) 중 책상 바닥 비율.
# 책상 위 물체는 0.9 안팎, 책상 프레임은 0.5 안팎 (그림자·가장자리 뎁스 때문에 1 은 안 된다).
BLOB_DESK_RING = 0.7
_RING_IN_PX = 4  # 절반 해상도 픽셀
_RING_OUT_PX = 10


def _desk_ring_frac(depth, K, plane, plane_mm, mask, stat) -> float:
    """덩어리 바로 바깥 띠에서 책상 높이(±plane_mm)인 픽셀 비율."""
    import cv2

    x, y, bw, bh = stat
    h, w = depth.shape[:2]
    m = _RING_OUT_PX
    xa, ya, xb, yb = max(0, x - m), max(0, y - m), min(w, x + bw + m), min(h, y + bh + m)
    blob = mask[ya:yb, xa:xb].astype(np.uint8)
    outer = cv2.dilate(blob, np.ones((2 * m + 1, 2 * m + 1), np.uint8))
    inner = cv2.dilate(blob, np.ones((2 * _RING_IN_PX + 1, 2 * _RING_IN_PX + 1), np.uint8))
    ring = (outer > 0) & (inner == 0)
    if not ring.any():
        return 0.0
    z = depth[ya:yb, xa:xb].astype(np.float64)
    v, u = np.mgrid[ya:yb, xa:xb]
    pts = np.dstack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    height = height_above_plane(pts.reshape(-1, 3), plane).reshape(z.shape)
    desk = (z > Z_MIN_MM) & (z < Z_MAX_MM) & (np.abs(height) <= float(plane_mm))
    return float(desk[ring].mean())


def _box_overlap(a, b) -> float:
    """작은 상자 쪽에서 본 겹침 비율 (0~1)."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return 0.0 if small <= 0 else iw * ih / small


def missed_blob_boxes(blobs: list[list[float]], yolo_xyxys: list[list[float]], overlap: float = 0.3) -> list[list[float]]:
    """뎁스 덩어리 중 YOLO 상자와 거의 겹치지 않는 것 (YOLO 가 놓친 물체)."""
    return [b for b in blobs if all(_box_overlap(b, y) < overlap for y in yolo_xyxys)]


def depth_blob_boxes(
    depth: np.ndarray,
    K: np.ndarray,
    plane: np.ndarray,
    plane_mm: float,
    T_base_cam: np.ndarray | None,
) -> list[list[float]]:
    """YOLO 가 못 찾을 때: 책상 위로 솟은 덩어리의 상자.

    화면 가장자리에 닿거나, 둘레가 책상 바닥이 아니거나(로봇팔·책상 프레임), 너무 높은 것(팔)은 뺀다.
    """
    import cv2

    # 상자만 찾으면 되니 절반 해상도로 (프레임마다 돌아서 4배 빠르게).
    f = _BLOB_SCALE
    depth = depth[::f, ::f]
    K = np.asarray(K, dtype=np.float64).copy()
    K[:2] /= f  # 절반 해상도의 픽셀 i = 원래 픽셀 f·i
    # 가장자리 튀는 뎁스는 여기서 빼지 않는다. 절반 해상도에선 구멍 둘레까지 깎여 상자가 쪼개진다.
    # 점군을 만들 때(collect_instances) 원래 해상도로 뺀다.
    fg = plane_foreground_mask(depth, K, plane, height_mm=float(plane_mm))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    # 컬러에 맞춘 뎁스는 화면 끝까지 차지 않는다. 뎁스가 있는 영역의 끝을 가장자리로 본다.
    valid = (depth > Z_MIN_MM) & (depth < Z_MAX_MM)
    cols = np.flatnonzero(valid.mean(axis=0) > 0.5)
    rows = np.flatnonzero(valid.mean(axis=1) > 0.5)
    if len(cols) == 0 or len(rows) == 0:
        return []
    b = max(1, _BLOB_BORDER_PX // f)
    x_lo, x_hi = int(cols[0]) + b, int(cols[-1]) - b
    y_lo, y_hi = int(rows[0]) + b, int(rows[-1]) - b
    out: list[list[float]] = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area * f * f < BLOB_MIN_PIXELS:
            continue
        if x <= x_lo or y <= y_lo or x + bw - 1 >= x_hi or y + bh - 1 >= y_hi:
            continue
        if _desk_ring_frac(depth, K, plane, plane_mm, labels == i, (x, y, bw, bh)) < BLOB_DESK_RING:
            continue
        xyz, _ = points_from_mask(depth, K, labels == i, stride=1)
        if len(xyz) == 0:
            continue
        # 중앙값: 튀는 뎁스 몇 %가 섞여도 브라켓을 팔로 오인하지 않게.
        if T_base_cam is not None:
            mid = float(np.median(apply_T(xyz, T_base_cam)[:, 2]))
        else:
            mid = float(np.median(height_above_plane(xyz, plane)))
        if mid > BLOB_MAX_HEIGHT_MM:
            continue
        out.append([float(x * f), float(y * f), float((x + bw) * f), float((y + bh) * f)])
    return out


def collect_instances(
    *,
    depth,
    bgr,
    K: np.ndarray,
    result,
    plane: np.ndarray | None,
    pad: int,
    plane_mm: float,
    stride: int,
    T_base_cam: np.ndarray | None,
    drop_edges: bool = True,
    xyxys: list[list[float]] | None = None,
    extra_xyxys: list[list[float]] | None = None,
) -> list[InstanceCloud]:
    """YOLO 상자마다 점군 하나, 책상 바탕과 가장자리의 튀는 뎁스는 뺀다.

    xyxys 를 주면 YOLO 결과 대신 그 상자들을 쓴다 (클래스 없음).
    extra_xyxys 는 YOLO 상자 뒤에 더한다 (YOLO 가 놓친 뎁스 덩어리, 클래스 없음).
    """
    h, w = bgr.shape[:2]
    out: list[InstanceCloud] = []
    if xyxys is not None:
        dets = [_Box(b) for b in xyxys]
    else:
        obb = getattr(result, "obb", None)
        boxes = getattr(result, "boxes", None)
        dets = boxes if boxes is not None and len(boxes) else obb
        dets = [] if dets is None else list(dets)
    dets = list(dets) + [_Box(b) for b in extra_xyxys or []]
    if not dets:
        return out
    in_base = T_base_cam is not None
    # 이미지 전체 마스크는 프레임마다 한 번만 만들고, 브라켓마다 상자 안만 잘라 쓴다
    # (브라켓마다 640×480 을 다시 계산하던 것이 카메라 창 한 프레임의 절반을 차지했다).
    keep = ~depth_edge_mask(depth) if drop_edges else None
    fg = None if plane is None else plane_foreground_mask(depth, K, plane, height_mm=float(plane_mm))
    zok = (depth > Z_MIN_MM) & (depth < Z_MAX_MM)
    stride = max(1, int(stride))
    for i, det in enumerate(dets):
        raw = det.xyxy[0]
        if hasattr(raw, "cpu"):
            raw = raw.cpu().numpy()
        xyxy = [float(v) for v in np.asarray(raw).reshape(-1)[:4]]
        xa, ya, xb, yb = xyxy
        quad = np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]], dtype=np.float32)
        class_id = None
        cls = getattr(det, "cls", None)
        if cls is not None:
            if hasattr(cls, "cpu"):
                cls = cls.cpu().numpy()
            cls_arr = np.asarray(cls).reshape(-1)
            if len(cls_arr):
                class_id = int(cls_arr[0])
        # mask_from_xyxy 와 같은 상자 (pad 포함, 화면 안으로 자름)
        xa = max(0, int(np.floor(min(xa, xb))) - pad)
        xe = min(w, int(np.ceil(max(xyxy[0], xyxy[2]))) + pad)
        ya = max(0, int(np.floor(min(ya, yb))) - pad)
        ye = min(h, int(np.ceil(max(xyxy[1], xyxy[3]))) + pad)
        roi = np.zeros((h, w), dtype=bool)
        if xe > xa and ye > ya:
            box = (slice(ya, ye), slice(xa, xe))
            n_zok = int(np.count_nonzero(zok[box]))
            sub = np.ones((ye - ya, xe - xa), dtype=bool)
            # 책상 평면에서 plane_mm 이내 점은 제거한다.
            if fg is not None:
                sub &= fg[box]
            if keep is not None:
                sub &= keep[box]
            sub = largest_component(sub)
            roi[box] = sub
            # 점은 전체 이미지 기준 stride 격자에서 뽑는다 (상자 시작을 격자에 맞춘다)
            x0 = -(-xa // stride) * stride
            y0 = -(-ya // stride) * stride
            Kc = np.asarray(K, dtype=np.float64).copy()
            Kc[0, 2] -= x0
            Kc[1, 2] -= y0
            xyz, rgb = points_from_mask(depth[y0:ye, x0:xe], Kc, sub[y0 - ya:, x0 - xa:],
                                        bgr=None if bgr is None else bgr[y0:ye, x0:xe], stride=stride)
        else:
            n_zok = 0
            xyz, rgb = np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        if in_base and len(xyz):
            xyz = apply_T(xyz, T_base_cam)
        if len(xyz):
            # 책상 위 전제: BRACKET_MAX_HEIGHT_MM 보다 높은 점을 버린다 (맨 위 고르기 · 정합 모두 이 점군을 쓴다).
            height = xyz[:, 2] if in_base else (height_above_plane(xyz, plane) if plane is not None else None)
            if height is not None:
                keep_h = height <= BRACKET_MAX_HEIGHT_MM
                xyz, rgb = xyz[keep_h], rgb[keep_h]
        score = _height_score(xyz, None if in_base else plane, in_base=in_base)
        out.append(
            InstanceCloud(
                xyxy=xyxy,
                roi=roi,
                xyz=xyz,
                rgb=rgb,
                height_score=score,
                n_zok=n_zok,
                quad=quad,
                class_id=class_id,
            )
        )
    return out


def _excluded(xyz: np.ndarray, exclude) -> bool:
    """점군 가운데 (x, y) 가 제외 원 (x, y, r) 중 하나 안에 있으면 True."""
    if not exclude:
        return False
    cx, cy = _center_xy(xyz)
    return any((cx - x) ** 2 + (cy - y) ** 2 <= r * r for x, y, r in exclude)


def _center_xy(xyz: np.ndarray) -> tuple[float, float]:
    cx, cy = np.median(np.asarray(xyz, dtype=np.float64)[:, :2], axis=0)
    return float(cx), float(cy)


def ordered_targets(instances: list[InstanceCloud], *, exclude=None, region=None, nearest: bool = False):
    """고를 수 있는 물체 [(번호, x, y)] 를 고르는 순서대로. 점이 로봇 베이스 mm 일 때 쓴다.

    exclude=[(x, y, r)] 원 안은 빼고, region=((x0, x1), (y0, y1)) 이면 가운데가 그 안인 것만.
    nearest 면 베이스 원점에서 가까운 순, 아니면 select_topmost 와 같은 높은 순.
    """
    out = []
    for i, inst in enumerate(instances):
        if not len(inst.xyz) or _excluded(inst.xyz, exclude):
            continue
        cx, cy = _center_xy(inst.xyz)
        if region is not None:
            (x0, x1), (y0, y1) = region
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
        out.append((i, cx, cy))
    if nearest:
        out.sort(key=lambda t: t[1] ** 2 + t[2] ** 2)
    else:
        out.sort(key=lambda t: (instances[t[0]].height_score, len(instances[t[0]].xyz)), reverse=True)
    return out


def select_topmost(instances: list[InstanceCloud], exclude=None) -> int | None:
    """가장 높은 물체. exclude=[(x, y, r)] 를 주면 가운데가 그 원 안인 물체는 빼고 고른다
    (점이 로봇 베이스 mm 일 때만 뜻이 있다)."""
    scored = [
        (i, inst.height_score, len(inst.xyz))
        for i, inst in enumerate(instances)
        if len(inst.xyz) and not _excluded(inst.xyz, exclude)
    ]
    if not scored:
        return None
    scored.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return int(scored[0][0])
