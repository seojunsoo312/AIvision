"""시뮬로 YOLO-pose 학습 데이터를 만든다. 라벨(박스 + 키포인트 10개)은 정답 자세에서 자동으로 나온다.

장면마다 브라켓 1~3개, 책상 · 브라켓 색과 조명을 무작위로, 카메라 자세도 손눈 오차만큼 조금씩 흔든다.
출력은 ultralytics pose 형식: images/{train,val}/*.jpg, labels/{train,val}/*.txt, data.yaml

    python -m vision.sim.pose_data --train 3000 --val 300 --workers 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vision.sim.data import HEIGHT, OUT_DIR, WIDTH
from vision.sim.keypoints import FLIP_IDX, N_KPT, canonical_T, keypoint_labels

DATA_DIR = OUT_DIR / "pose_data"
# 화면에 이만큼도 안 보이는 브라켓은 라벨에서 뺀다 (YOLO 도 못 찾는 크기).
MIN_VISIBLE_PX = 150
N_OBJECTS_P = {1: 0.4, 2: 0.35, 3: 0.25}


def label_lines(scene, symmetries, dist: np.ndarray | None = None) -> list[str]:
    """한 장면의 YOLO pose 라벨 줄들. cls cx cy w h (x y v)×N. 좌표는 화면 크기로 나눈 0~1 값.

    박스는 분할 마스크의 외접 사각형 (평가 때 YOLO 박스 대신 쓰는 것과 같은 규칙).
    영상을 렌즈 왜곡했으면 scene 의 뎁스 · 분할도 왜곡된 것이어야 하고, dist 로 키포인트도 왜곡한다.
    """
    lines = []
    for o in scene.objects:
        if o["visible_px"] < MIN_VISIBLE_PX:
            continue
        ys, xs = np.nonzero(scene.seg == o["id"])
        if len(xs) < MIN_VISIBLE_PX // 2:
            continue
        x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
        T_lab = canonical_T(np.asarray(o["T_base_obj"], float), scene.T_base_cam, symmetries)
        kp = keypoint_labels(T_lab, scene.K, scene.T_base_cam, scene.depth_mm, width=WIDTH, height=HEIGHT, dist=dist)
        parts = ["0", f"{(x0 + x1) / 2 / WIDTH:.6f}", f"{(y0 + y1) / 2 / HEIGHT:.6f}",
                 f"{(x1 - x0) / WIDTH:.6f}", f"{(y1 - y0) / HEIGHT:.6f}"]
        for u, v, vis in kp:
            if vis == 0:
                parts += ["0", "0", "0"]
            else:
                parts += [f"{u / WIDTH:.6f}", f"{v / HEIGHT:.6f}", str(int(vis))]
        lines.append(" ".join(parts))
    return lines


def _make_split(job) -> int:
    """한 작업자: 같은 split 의 이미지 번호 범위를 만든다."""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from dataclasses import replace

    import cv2

    from vision.sim.rgb_noise import REAL_DIST, degrade_rgb, distort, random_config
    from vision.sim.scene import SceneMaker, default_T_base_cam, jitter_camera
    from vision.sim.symmetry import cad_symmetries

    split, start, count, seed, out, degrade, n_distract, robot = job
    rng = np.random.default_rng(seed)
    maker = SceneMaker(max(N_OBJECTS_P), n_distractors=n_distract, robot=robot)
    syms = cad_symmetries()
    T0 = default_T_base_cam()
    ks = np.array(list(N_OBJECTS_P))
    ps = np.array(list(N_OBJECTS_P.values()))
    n_obj = 0
    for k in range(start, start + count):
        maker.active = int(rng.choice(ks, p=ps / ps.sum()))
        if n_distract:
            maker.distract_active = int(rng.integers(0, n_distract + 1))
        if robot:
            # 실제 화면에는 거의 늘 로봇이 있다. 10% 는 빼서 로봇 없이도 되게.
            maker.robot_joints_deg = maker.random_robot_joints(rng) if rng.random() < 0.9 else None
        maker.randomize_look(rng)
        maker.set_camera(jitter_camera(T0, rng))
        scene = maker.make(rng)
        dist = None
        if degrade:
            # 실제 카메라 같은 열화를 무작위로. 왜곡이면 뎁스 · 분할 · 키포인트도 같이 왜곡한다.
            cfg = random_config(rng)
            if cfg.distort:
                dist = REAL_DIST
                scene = replace(
                    scene,
                    depth_mm=distort(scene.depth_mm, scene.K, nearest=True),
                    seg=distort(scene.seg, scene.K, nearest=True),
                )
            scene = replace(scene, rgb=degrade_rgb(scene.rgb, cfg, rng, scene.K))
        name = f"{split}_{k:05d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), cv2.cvtColor(scene.rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = label_lines(scene, syms, dist)
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_obj += len(lines)
    return n_obj


def write_yaml(out: Path) -> Path:
    path = out / "data.yaml"
    text = (
        f"path: {Path(out).resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"kpt_shape: [{N_KPT}, 3]\n"
        f"flip_idx: {FLIP_IDX}\n"
        "names:\n  0: bracket\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="시뮬 YOLO-pose 데이터 만들기")
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--degrade", action="store_true", help="컬러 노이즈 · 흐림 · 노출 · JPEG · 렌즈 왜곡을 무작위로 넣는다")
    ap.add_argument("--distractors", type=int, default=0, help="장면마다 방해물 0~N 개 (상자 · 원통 · 케이블 등)")
    ap.add_argument("--robot", action="store_true", help="흰 로봇 팔을 무작위 자세로 넣는다")
    args = ap.parse_args()

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    jobs = []
    for s_i, (split, total) in enumerate((("train", args.train), ("val", args.val))):
        per = int(np.ceil(total / args.workers))
        for w in range(args.workers):
            start = w * per
            count = min(per, total - start)
            if count > 0:
                jobs.append((split, start, count, args.seed * 1000 + s_i * 100 + w, args.out, args.degrade,
                             args.distractors, args.robot))
    t0 = time.perf_counter()
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(args.workers) as pool:
        n_obj = sum(pool.map(_make_split, jobs))
    yaml_path = write_yaml(args.out)
    info = {"train": args.train, "val": args.val, "objects": n_obj, "seconds": round(time.perf_counter() - t0, 1)}
    (args.out / "info.json").write_text(json.dumps(info), encoding="utf-8")
    print(f"완료: {info}  →  {yaml_path}")


if __name__ == "__main__":
    main()
