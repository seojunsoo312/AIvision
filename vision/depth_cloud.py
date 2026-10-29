"""정렬된 RGB-D → 2D ROI 안의 카메라 좌표 점군."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vision.rgbd import OrbbecV1

Z_MIN_MM = 80.0
Z_MAX_MM = 1800.0
# (Z_MIN, Z_MAX) 열린 구간을 cv2.inRange(닫힌 구간)로 쓰기 위한 경계
_Z_MIN_OPEN = float(np.nextafter(np.float32(Z_MIN_MM), np.float32(np.inf)))
_Z_MAX_OPEN = float(np.nextafter(np.float32(Z_MAX_MM), np.float32(-np.inf)))


def open_orbbec(width: int = 640, height: int = 480, **kwargs):
    return OrbbecV1(width, height, **kwargs)


def rotate180(bgr, depth_mm):
    if bgr is not None:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    if depth_mm is not None:
        depth_mm = cv2.rotate(depth_mm, cv2.ROTATE_180)
    return bgr, depth_mm


def colorize_depth(
    depth_mm: np.ndarray, center_span_mm: float = 60.0
) -> np.ndarray:
    """깊이를 보기 좋게 자동 확대한다. 점군 계산에는 영향을 주지 않는다.

    화면용이라 매 프레임 돈다. numpy 대신 cv2 연산으로 14ms → 2.5ms.
    """
    depth = depth_mm if depth_mm.dtype == np.float32 else depth_mm.astype(np.float32)
    valid = cv2.inRange(depth, _Z_MIN_OPEN, _Z_MAX_OPEN)
    # 확대 중심은 1/16 픽셀의 중앙값으로 충분하다.
    sub = depth[::4, ::4]
    values = sub[(sub > Z_MIN_MM) & (sub < Z_MAX_MM)]
    if values.size == 0:
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)

    center = float(np.median(values))
    # 작업대 깊이 주변을 확대한다. 극단값과 프레임 가장자리 노이즈는 무시한다.
    low = max(Z_MIN_MM, center - center_span_mm / 2.0)
    high = min(Z_MAX_MM, center + center_span_mm / 2.0)
    scale = 255.0 / max(high - low, 1.0)
    clipped = cv2.max(cv2.min(depth, high), low)
    # -0.5: convertScaleAbs 는 반올림하므로 예전 astype(uint8) 처럼 버림이 되게 한다.
    norm = cv2.convertScaleAbs(clipped, alpha=scale, beta=-low * scale - 0.5)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    vis = cv2.bitwise_and(color, color, mask=valid)
    cv2.putText(
        vis,
        f"depth {low:.0f}-{high:.0f} mm",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def mask_from_xyxy(
    h: int, w: int, xyxy: list[float], pad: int = 2
) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    xa = max(0, int(np.floor(min(x1, x2))) - pad)
    xb = min(w, int(np.ceil(max(x1, x2))) + pad)
    ya = max(0, int(np.floor(min(y1, y2))) - pad)
    yb = min(h, int(np.ceil(max(y1, y2))) + pad)
    mask = np.zeros((h, w), dtype=bool)
    if xb > xa and yb > ya:
        mask[ya:yb, xa:xb] = True
    return mask


def fit_plane_ransac(
    xyz: np.ndarray,
    threshold_mm: float = 3.0,
    iterations: int = 200,
    min_abs_nz: float = 0.7,
) -> tuple[np.ndarray, int]:
    """가장 큰 전방 평면 ax+by+cz+d=0을 찾는다."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 100:
        raise ValueError("평면 추정 점이 부족합니다.")
    if len(pts) > 12000:
        rng = np.random.default_rng(42)
        pts = pts[rng.choice(len(pts), 12000, replace=False)]
    rng = np.random.default_rng(42)
    best_plane = None
    best_inliers = None
    best_count = 0
    for _ in range(iterations):
        p0, p1, p2 = pts[rng.choice(len(pts), 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        length = float(np.linalg.norm(normal))
        if length < 1e-6:
            continue
        normal /= length
        if abs(float(normal[2])) < min_abs_nz:
            continue
        d = -float(np.dot(normal, p0))
        distances = np.abs(pts @ normal + d)
        inliers = distances <= threshold_mm
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_plane = np.r_[normal, d]
            best_inliers = inliers
    if best_plane is None or best_inliers is None or best_count < 100:
        raise ValueError("책상 평면을 찾지 못했습니다.")

    # RANSAC inlier 전체로 평면을 다시 맞춘다.
    fit = pts[best_inliers]
    center = fit.mean(axis=0)
    _, _, vh = np.linalg.svd(fit - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    d = -float(np.dot(normal, center))
    # 카메라 원점이 음의 쪽, 책상 위 물체도 음의 쪽이 되도록 통일한다.
    if d > 0:
        normal = -normal
        d = -d
    plane = np.r_[normal, d].astype(np.float64)
    count = int(np.count_nonzero(np.abs(pts @ normal + d) <= threshold_mm))
    return plane, count


def plane_foreground_mask(
    depth_mm: np.ndarray,
    K: np.ndarray,
    plane: np.ndarray,
    height_mm: float = 4.0,
) -> np.ndarray:
    """책상 평면보다 카메라 쪽으로 height_mm 이상 돌출된 픽셀만 남긴다."""
    h, w = depth_mm.shape[:2]
    vv, uu = np.indices((h, w), dtype=np.float64)
    z = depth_mm.astype(np.float64)
    x = (uu - float(K[0, 2])) * z / float(K[0, 0])
    y = (vv - float(K[1, 2])) * z / float(K[1, 1])
    signed = plane[0] * x + plane[1] * y + plane[2] * z + plane[3]
    valid = (z > Z_MIN_MM) & (z < Z_MAX_MM)
    return valid & (signed < -abs(height_mm))


PLANE_HEIGHT_MIN_MM = 3.0
PLANE_HEIGHT_MAX_MM = 10.0
PLANE_HEIGHT_MARGIN_MM = 1.5


def plane_noise_mm(
    xyz: np.ndarray, plane: np.ndarray, *, band_mm: float = 15.0, pct: float = 99.0
) -> float | None:
    """책상 뎁스 노이즈(mm): 평면 '뒤쪽'(카메라에서 먼 쪽)으로 튄 점들의 pct 백분위.

    책상 아래에는 물체가 없으니 뒤쪽 점은 모두 노이즈다. 노이즈는 앞뒤로 비슷하게 퍼지므로
    물체에 섞이지 않고 앞쪽 노이즈 크기를 잴 수 있다. 점이 모자라면 None.
    """
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    signed = pts @ np.asarray(plane[:3], dtype=np.float64) + float(plane[3])
    # 물체는 음수 쪽. 양수 = 책상보다 먼 쪽.
    behind = signed[(signed > 0.0) & (signed < band_mm)]
    if len(behind) < 200:
        return None
    return float(np.percentile(behind, pct))


def auto_plane_height_mm(noise_mm: float | None) -> float:
    """책상 노이즈 + 여유를 [MIN, MAX] 로 묶은 책상 제거 높이."""
    if noise_mm is None:
        return PLANE_HEIGHT_MIN_MM
    return float(np.clip(noise_mm + PLANE_HEIGHT_MARGIN_MM, PLANE_HEIGHT_MIN_MM, PLANE_HEIGHT_MAX_MM))


def depth_edge_mask(
    depth_mm: np.ndarray, *, jump_mm: float = 10.0, mid_mm: float = 3.0, ksize: int = 5
) -> np.ndarray:
    """물체 가장자리의 튀는 픽셀(앞 물체와 뒤 책상 사이 중간 깊이)을 True 로 표시한다.

    ksize 창 안에 jump_mm 이상 깊이 차가 있고, 픽셀이 창의 가장 가까운 값과 가장 먼 값
    양쪽에서 mid_mm 넘게 떨어져 있으면 중간에 뜬 점으로 본다. 물체 윗면(가장 가까운 값)과
    책상(가장 먼 값)은 남는다.
    """
    z = depth_mm.astype(np.float32)
    valid = (z > Z_MIN_MM) & (z < Z_MAX_MM)
    kernel = np.ones((ksize, ksize), np.uint8)
    zmax = cv2.dilate(np.where(valid, z, 0.0).astype(np.float32), kernel)
    zmin = cv2.erode(np.where(valid, z, np.float32(1e6)).astype(np.float32), kernel)
    jump = (zmax - zmin) >= jump_mm
    middle = ((zmax - z) > mid_mm) & ((z - zmin) > mid_mm)
    return valid & jump & middle


def largest_component(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """고립된 뎁스 노이즈를 버리고 가장 큰 연결 성분만 남긴다."""
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    index = int(np.argmax(areas)) + 1
    if int(stats[index, cv2.CC_STAT_AREA]) < min_area:
        return np.zeros_like(mask, dtype=bool)
    return labels == index


def pca_obb(xyz: np.ndarray) -> dict[str, np.ndarray] | None:
    """점군의 PCA 중심, 주축, 축 방향 크기를 계산한다."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 3:
        return None
    center = np.median(pts, axis=0)
    centered = pts - center
    cov = np.cov(centered, rowvar=False)
    values, axes = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = values[order]
    axes = axes[:, order]
    projected = centered @ axes
    low = np.percentile(projected, 2, axis=0)
    high = np.percentile(projected, 98, axis=0)
    return {
        "center": center.astype(np.float32),
        "axes": axes.astype(np.float32),
        "extents": (high - low).astype(np.float32),
        "eigenvalues": values.astype(np.float32),
    }


def points_from_mask(
    depth_mm: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    bgr=None,
    stride: int = 2,
    z_min: float = Z_MIN_MM,
    z_max: float = Z_MAX_MM,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_mm.shape[:2]
    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    uu, vv = np.meshgrid(us, vs)
    keep = mask[vv, uu]
    z = depth_mm[vv, uu]
    keep = keep & (z > z_min) & (z < z_max)
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    u = uu[keep].astype(np.float64)
    v = vv[keep].astype(np.float64)
    z = z[keep].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    if bgr is None:
        rgb = np.zeros((len(xyz), 3), dtype=np.uint8)
    else:
        rgb = bgr[vv[keep], uu[keep]][:, ::-1].copy()
    return xyz, rgb


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if rgb is None:
            for p in xyz:
                f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")
        else:
            for p, c in zip(xyz, rgb):
                f.write(
                    f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
                )
