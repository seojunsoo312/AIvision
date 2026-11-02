"""시뮬 장면 파일 읽기 · 쓰기. MuJoCo 없이도 부를 수 있다 (저장된 장면만 평가할 때).

장면 폴더: rgb.png · depth_mm.npy · seg.npy · gt.json · camera.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
OUT_DIR = SIM_DIR / "out"

WIDTH, HEIGHT = 640, 480


@dataclass
class Scene:
    """렌더 결과 한 장면. 길이는 mm."""

    rgb: np.ndarray  # (H, W, 3) uint8, RGB
    depth_mm: np.ndarray  # (H, W) float32, z 깊이. 값 없음 = 0
    seg: np.ndarray  # (H, W) uint8. 0 = 배경, i+1 = i 번째 브라켓
    K: np.ndarray
    T_base_cam: np.ndarray
    objects: list[dict]


def save_scene(scene: Scene, out: Path) -> None:
    """장면 폴더에 rgb.png · depth_mm.npy · seg.npy · gt.json · camera.json 을 쓴다."""
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "rgb.png"), cv2.cvtColor(scene.rgb, cv2.COLOR_RGB2BGR))
    np.save(out / "depth_mm.npy", scene.depth_mm)
    np.save(out / "seg.npy", scene.seg)
    (out / "gt.json").write_text(json.dumps({"objects": scene.objects}, indent=2), encoding="utf-8")
    cam = {"K": scene.K.tolist(), "T_base_cam": scene.T_base_cam.tolist(), "width": WIDTH, "height": HEIGHT}
    (out / "camera.json").write_text(json.dumps(cam, indent=2), encoding="utf-8")


def load_scene(folder: Path) -> Scene:
    import cv2

    folder = Path(folder)
    rgb = cv2.cvtColor(cv2.imread(str(folder / "rgb.png")), cv2.COLOR_BGR2RGB)
    cam = json.loads((folder / "camera.json").read_text(encoding="utf-8"))
    gt = json.loads((folder / "gt.json").read_text(encoding="utf-8"))
    return Scene(
        rgb,
        np.load(folder / "depth_mm.npy"),
        np.load(folder / "seg.npy"),
        np.asarray(cam["K"], float),
        np.asarray(cam["T_base_cam"], float),
        gt["objects"],
    )
