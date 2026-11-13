"""학생 자리 실사진에서 시뮬로 학습한 YOLO-pose 를 확인한다. 실험용 (sim-pose 브랜치).

정답 라벨 없이 볼 수 있는 것만 본다.
  - 모델마다 찾은 브라켓 수 (사진 모음 그림으로 사람이 놓친 것 · 엉뚱한 것을 확인)
  - 키포인트로 PnP 를 풀었을 때 재투영 오차 (키포인트가 CAD 모양과 맞는지)
  - 그 자세가 책상 위에 설 수 있는 자세(세우기 · 눕히기 · 비스듬히)와 몇 도 다른지
  - 이 자리 손눈 캘리브와 시뮬 카메라(10/29) 의 차이

    python -m yolo.pose.sim_seat_check                    # yolo/datasets/seat/{images,calib}
결과: 바탕화면 sim_학생자리_시험/<모델>_전체.jpg, 결과.json, 표 출력
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from vision.text import put_text
from yolo.config import ROOT

SEAT_DIR = ROOT / "datasets" / "seat"
RUNS = ROOT.parent / "vision" / "sim" / "out" / "pose_runs"
OUT_DIR = Path.home() / "Desktop" / "sim_학생자리_시험"
# 모델 이름 → 키포인트 묶음 (학습 때 쓴 SIM_KPT_SET)
MODELS = {"v4n": "mixed", "v5n": "corners"}
CONF = 0.5
PNP_REPROJ_PX = 4.0
COLORS = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
          (255, 0, 0), (255, 0, 255), (128, 0, 255), (255, 255, 255), (0, 0, 0)]
CLASS_KO = {0: "세우기", 1: "눕히기", 2: "비스듬히", -1: "안 맞음"}


def load_seat_camera(calib_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """이 자리 캘리브 → (K, dist, T_base_cam). 사진은 180° 회전 저장이라 K 도 회전에 맞춘다."""
    from vision.calib import intrinsics_for_rotate180, load_intrinsics, load_T_base_cam
    from vision.camera import FRAME_HEIGHT, FRAME_WIDTH

    K_raw, dist_raw, _ = load_intrinsics(calib_dir / "intrinsics.json")
    K, dist = intrinsics_for_rotate180(K_raw, dist_raw, FRAME_WIDTH, FRAME_HEIGHT)
    T, _ = load_T_base_cam(calib_dir / "eye_to_hand.json")
    return K, np.asarray(dist, float), T


def camera_diff(T_a: np.ndarray, T_b: np.ndarray) -> dict:
    """두 T_base_cam 의 위치 차 (mm, 축별) 와 방향 차 (도)."""
    d = T_a[:3, 3] - T_b[:3, 3]
    R = T_a[:3, :3].T @ T_b[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    return {"dx": round(float(d[0]), 1), "dy": round(float(d[1]), 1), "dz": round(float(d[2]), 1),
            "dist_mm": round(float(np.linalg.norm(d)), 1), "rot_deg": round(ang, 2)}


def check_model(name: str, kset: str, images: list[Path], K, dist, T_base_cam) -> tuple[list[dict], list[np.ndarray]]:
    """한 모델: 사진마다 검출 · PnP · 자세 붙이기 결과와 그림."""
    from ultralytics import YOLO

    from vision.sim.keypoints import _CORNERS, _MIXED
    from yolo.pose.sim_eval import snap_stable

    kp_cad = {"mixed": _MIXED, "corners": _CORNERS}[kset]
    model = YOLO(str(RUNS / name / "weights" / "best.pt"))
    rows, tiles = [], []
    for path in images:
        img = cv2.imread(str(path))
        r = model.predict(img, conf=CONF, verbose=False)[0]
        vis = img.copy()
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        kxy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else np.zeros((0, 10, 2))
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        for b, k, c in zip(boxes, kxy, confs):
            row = {"image": path.name, "box": [round(float(v), 1) for v in b], "conf": round(float(c), 3)}
            ok, rvec, tvec, inl = cv2.solvePnPRansac(kp_cad, k.astype(np.float64), K, dist,
                                                     flags=cv2.SOLVEPNP_SQPNP, reprojectionError=PNP_REPROJ_PX)
            label, col = "PnP 실패", (0, 0, 255)
            if ok and inl is not None and len(inl) >= 4:
                inl = inl.ravel()
                rvec, tvec = cv2.solvePnPRefineLM(kp_cad[inl], k[inl].astype(np.float64), K, dist, rvec, tvec)
                proj = cv2.projectPoints(kp_cad, rvec, tvec, K, dist)[0].reshape(-1, 2)
                err = np.linalg.norm(proj - k, axis=1)
                T = np.eye(4)
                T[:3, :3] = cv2.Rodrigues(rvec)[0]
                T[:3, 3] = tvec.ravel()
                T_base = T_base_cam @ T
                _, cls, tilt = snap_stable(T_base, kp_cad.mean(axis=0))
                # PnP 깊이는 약하지만, 여러 브라켓에서 한쪽으로 치우치면 카메라 높이 차이일 수 있다.
                low_z = float((kp_cad @ T_base[:3, :3].T + T_base[:3, 3])[:, 2].min())
                row.update(inliers=len(inl), reproj_med=round(float(np.median(err[inl])), 2),
                           reproj_max=round(float(err.max()), 2), cls=int(cls), tilt_deg=round(float(tilt), 1),
                           lowest_z_mm=round(low_z, 1), cam_depth_mm=round(float(tvec[2, 0]), 1))
                label = f"{CLASS_KO[int(cls)]} {tilt:.0f}deg {np.median(err[inl]):.1f}px"
                col = (0, 200, 0) if tilt <= 5 else ((0, 200, 255) if tilt <= 15 else (0, 0, 255))
            rows.append(row)
            x0, y0, x1, y1 = b.astype(int)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
            for i, (x, y) in enumerate(k):
                cv2.circle(vis, (int(x), int(y)), 3, COLORS[i], -1)
            put_text(vis, label, (x0, max(14, y0 - 6)), 0.5, col, 1)
        put_text(vis, f"{path.name}  {len(boxes)}개", (6, 470), 0.5, (0, 0, 255), 1)
        tiles.append(cv2.resize(vis, (320, 240)))
    return rows, tiles


def summarize(rows: list[dict], n_images: int) -> dict:
    fitted = [r for r in rows if "tilt_deg" in r]
    tilts = np.array([r["tilt_deg"] for r in fitted]) if fitted else np.zeros(0)
    out = {"images": n_images, "detections": len(rows), "pnp_ok": len(fitted),
           "tilt_le5": int((tilts <= 5).sum()), "tilt_le15": int((tilts <= 15).sum()),
           "reproj_med_px": round(float(np.median([r["reproj_med"] for r in fitted])), 2) if fitted else None,
           "lowest_z_med_mm": round(float(np.median([r["lowest_z_mm"] for r in fitted])), 1) if fitted else None,
           "by_class": {CLASS_KO[c]: sum(1 for r in fitted if r["cls"] == c) for c in (0, 1, 2, -1)}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="학생 자리 실사진 확인")
    ap.add_argument("--images", type=Path, default=SEAT_DIR / "images")
    ap.add_argument("--calib", type=Path, default=SEAT_DIR / "calib")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()

    images = sorted(list(args.images.glob("*.jpg")) + list(args.images.glob("*.png")))
    if not images:
        raise SystemExit(f"사진이 없습니다: {args.images}")
    K, dist, T_base_cam = load_seat_camera(args.calib)
    from vision.sim.scene import T_BASE_CAM as SIM_T

    cam = camera_diff(T_base_cam, SIM_T)
    print(f"사진 {len(images)}장. 이 자리 카메라 - 시뮬 카메라: {cam}")
    print(f"이 자리 K: fx {K[0, 0]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}, 카메라 높이 {T_base_cam[2, 3]:.1f} mm")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {"camera_vs_sim": cam, "models": {}}
    for name in args.models:
        rows, tiles = check_model(name, MODELS[name], images, K, dist, T_base_cam)
        sm = summarize(rows, len(images))
        result["models"][name] = {"summary": sm, "rows": rows}
        print(name, json.dumps(sm, ensure_ascii=False))
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
        cv2.imwrite(str(OUT_DIR / f"{name}_전체.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (OUT_DIR / "결과.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
