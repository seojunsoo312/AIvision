"""깨끗한 시뮬 뎁스를 Orbbec 처럼 지저분하게 만든다. 단계마다 켜고 끌 수 있다.

1. 거리 노이즈: 가우시안. 45 cm 에서 sigma_mm, 거리 제곱에 비례.
2. 가장자리 튀는 점: 깊이가 크게 바뀌는 경계 근처 픽셀을 앞뒤 깊이 사이 값으로 바꾼다.
3. 구멍(0): 경계와, 시선에 많이 기운 면에 확률적으로.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class NoiseConfig:
    sigma_mm: float = 1.5  # 기준 거리에서 거리 노이즈 표준편차
    ref_mm: float = 450.0  # sigma_mm 을 잰 거리
    edge_jump_mm: float = 8.0  # 이 이상 깊이 차가 나는 곳을 경계로 본다
    edge_px: int = 2  # 경계에서 이 픽셀 안쪽까지 튀는 점 후보
    edge_flying_p: float = 0.5  # 경계 후보가 중간 깊이로 튈 확률
    edge_hole_p: float = 0.15  # 경계 후보가 0 이 될 확률
    grazing_deg: float = 70.0  # 면 법선과 시선 사이 각이 이보다 크면 구멍 후보
    grazing_hole_p: float = 0.6
    enabled: bool = True


NO_NOISE = NoiseConfig(enabled=False)


def _normals_angle_deg(depth_mm: np.ndarray, K: np.ndarray) -> np.ndarray:
    """픽셀마다 면 법선과 시선 사이 각(도). 뎁스를 점으로 바꿔 이웃 차분으로 법선을 구한다."""
    h, w = depth_mm.shape
    vv, uu = np.indices((h, w), dtype=np.float64)
    z = depth_mm.astype(np.float64)
    x = (uu - K[0, 2]) * z / K[0, 0]
    y = (vv - K[1, 2]) * z / K[1, 1]
    P = np.stack([x, y, z], axis=-1)
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    ray = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-12)
    cos = np.abs(np.sum(n * ray, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def add_noise(
    depth_mm: np.ndarray, K: np.ndarray, rng: np.random.Generator, cfg: NoiseConfig | None = None
) -> np.ndarray:
    """깨끗한 뎁스(mm) → 노이즈를 넣은 뎁스. 입력은 바꾸지 않는다. cfg 를 안 주면 기본 노이즈."""
    cfg = NoiseConfig() if cfg is None else cfg
    clean = np.asarray(depth_mm, dtype=np.float32)
    if not cfg.enabled:
        return clean.copy()
    valid = clean > 0
    out = clean.copy()

    # 2. 가장자리 튀는 점. 경계 판단은 깨끗한 뎁스로 한다.
    k = 2 * int(cfg.edge_px) + 1
    kernel = np.ones((k, k), np.uint8)
    zmax = cv2.dilate(np.where(valid, clean, 0.0).astype(np.float32), kernel)
    zmin = cv2.erode(np.where(valid, clean, np.float32(1e6)).astype(np.float32), kernel)
    edge = valid & ((zmax - zmin) >= cfg.edge_jump_mm)
    r = rng.random(clean.shape)
    flying = edge & (r < cfg.edge_flying_p)
    mix = rng.uniform(0.15, 0.85, clean.shape).astype(np.float32)
    out[flying] = (zmin + mix * (zmax - zmin))[flying]

    # 1. 거리 노이즈.
    sigma = cfg.sigma_mm * (clean / cfg.ref_mm) ** 2
    out = out + (rng.standard_normal(clean.shape).astype(np.float32) * sigma.astype(np.float32))

    # 3. 구멍.
    holes = edge & (~flying) & (rng.random(clean.shape) < cfg.edge_hole_p / max(1e-6, 1.0 - cfg.edge_flying_p))
    ang = _normals_angle_deg(clean, K)
    holes |= valid & (ang > cfg.grazing_deg) & (rng.random(clean.shape) < cfg.grazing_hole_p)
    out[holes | ~valid] = 0.0
    return out.astype(np.float32)
