"""깨끗한 시뮬 컬러 영상을 실제 카메라처럼 망가뜨린다. 렌즈 왜곡은 컬러 · 뎁스 · 분할에 똑같이 넣는다.

실제 카메라 창(roi_cloud.py)은 왜곡을 풀지 않고 K 만 쓴다. 그래서 왜곡 케이스는 영상만 왜곡하고 처리 코드는 K 만 쓰면
실제와 같은 조건이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import cv2
import numpy as np

# 실제 Orbbec 공장 왜곡 (intrinsics.json, 180° 돌린 영상 기준으로 p1 · p2 부호를 뒤집은 값). k1 k2 p1 p2 k3.
REAL_DIST = np.array([0.04763977, -0.06503717, 7.43489e-06, 0.000164143, 0.0])


@dataclass(frozen=True)
class RgbConfig:
    noise_sd: float = 0.0  # 센서 노이즈 표준편차 (0~255). 어두울수록 크게 보이도록 밝기 제곱근에 비례
    blur_sd: float = 0.0  # 가우시안 흐림 (픽셀)
    gain: float = 1.0  # 노출 (밝기 배율). 1 보다 크면 흰 브라켓이 하얗게 날아간다
    gamma: float = 1.0
    jpeg: int = 0  # JPEG 품질 (0 = 압축 안 함)
    distort: bool = False  # 실제 렌즈 왜곡 (컬러 · 뎁스 · 분할 모두)


CLEAN = RgbConfig()

# 시험용 이름 붙은 조건.
PRESETS: dict[str, RgbConfig] = {
    "clean": CLEAN,
    "noise4": RgbConfig(noise_sd=4.0),
    "noise8": RgbConfig(noise_sd=8.0),
    "noise15": RgbConfig(noise_sd=15.0),
    "blur1": RgbConfig(blur_sd=1.0),
    "blur2": RgbConfig(blur_sd=2.0),
    "dark": RgbConfig(gain=0.55),
    "bright": RgbConfig(gain=1.35),
    "overexp": RgbConfig(gain=1.7),
    "gamma": RgbConfig(gamma=1.4),
    "jpeg50": RgbConfig(jpeg=50),
    "jpeg25": RgbConfig(jpeg=25),
    "distort": RgbConfig(distort=True),
    # 여러 가지를 조금씩 섞은 "실제 같은" 조건.
    "real_mild": RgbConfig(noise_sd=4.0, blur_sd=0.7, gain=1.1, jpeg=85, distort=True),
    "real_hard": RgbConfig(noise_sd=8.0, blur_sd=1.2, gain=1.3, gamma=1.15, jpeg=60, distort=True),
}


def random_config(rng: np.random.Generator) -> RgbConfig:
    """학습 데이터용 무작위 열화. 시험 조건보다 조금 넓게 덮는다."""
    return RgbConfig(
        noise_sd=float(rng.uniform(0.0, 12.0)),
        blur_sd=float(rng.choice([0.0, rng.uniform(0.3, 1.8)])),
        gain=float(np.exp(rng.uniform(np.log(0.5), np.log(1.6)))),
        gamma=float(np.exp(rng.uniform(np.log(0.8), np.log(1.3)))),
        jpeg=int(rng.choice([0, int(rng.uniform(35, 95))])),
        distort=bool(rng.random() < 0.5),
    )


@lru_cache(maxsize=4)
def _distort_maps(K_key: tuple, dist_key: tuple, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """왜곡된 출력 픽셀마다 '왜곡 없는 원본'의 어느 픽셀을 가져올지."""
    K = np.asarray(K_key, float).reshape(3, 3)
    dist = np.asarray(dist_key, float)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([uu.ravel(), vv.ravel()], axis=1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(h, w, 2)
    return und[..., 0].astype(np.float32), und[..., 1].astype(np.float32)


def distort(img: np.ndarray, K: np.ndarray, *, nearest: bool, dist: np.ndarray = REAL_DIST) -> np.ndarray:
    """왜곡 없는 영상 → 렌즈 왜곡이 있는 영상. 뎁스 · 분할은 nearest=True (값을 섞지 않는다)."""
    h, w = img.shape[:2]
    mx, my = _distort_maps(tuple(np.asarray(K, float).ravel()), tuple(np.asarray(dist, float).ravel()), w, h)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def degrade_rgb(rgb: np.ndarray, cfg: RgbConfig, rng: np.random.Generator, K: np.ndarray | None = None) -> np.ndarray:
    """RGB uint8 → 열화한 RGB uint8. 순서: 왜곡 → 흐림 → 노출 · 감마 → 센서 노이즈 → JPEG."""
    img = np.asarray(rgb)
    if cfg.distort:
        if K is None:
            raise ValueError("왜곡에는 K 가 필요합니다")
        img = distort(img, K, nearest=False)
    x = img.astype(np.float32) / 255.0
    if cfg.blur_sd > 0:
        x = cv2.GaussianBlur(x, (0, 0), cfg.blur_sd)
    x = np.clip(x * cfg.gain, 0.0, 1.0) ** (1.0 / cfg.gamma)
    if cfg.noise_sd > 0:
        sd = cfg.noise_sd / 255.0 * np.sqrt(np.clip(x, 0.02, 1.0))
        x = x + rng.standard_normal(x.shape).astype(np.float32) * sd
    out = (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if cfg.jpeg > 0:
        ok, buf = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(cfg.jpeg)])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy()
    return out


def with_distort(cfg: RgbConfig, on: bool) -> RgbConfig:
    return replace(cfg, distort=on)
