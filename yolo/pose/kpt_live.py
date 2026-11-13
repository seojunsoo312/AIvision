#!/usr/bin/env python3
"""키포인트 자세 · 불량 검사를 사진 폴더로 보는 창.

카메라는 통합 카메라 창으로 넘어간다 (자세 방식만 키포인트, 검사 · 펜던트는 CAD 정합과 같다):
  python -m yolo.pose.kpt_live          → python yolo/pose/roi_cloud.py --pose kpt 와 같다
사진 폴더 (카메라 없이 시험):
  python -m yolo.pose.kpt_live --images yolo/datasets/seat/images --calib yolo/datasets/seat/calib

평소: 사진 → YOLO-pose 키포인트 → 책상 붙이기 (세우기 · 눕히기 · 비스듬히 중 하나로, 손눈의 책상 높이로 깊이).
      화면에 CAD 윤곽 · 자세 종류 · 베이스 위치(mm) · 돌아간 각도.
c   : 정밀 분석. 테두리 자동 맞춤 → 흰 브라켓 불량 판정 (마커). 화면을 멈추고 터미널에 표 (펜던트에 넣을 xyz · RPY 포함).
      윤곽 색: 초록 양품 · 빨강 불량 · 보라 검사 안 함 (검은 브라켓) · 주황 브라켓 아님 (윤곽이 사진 테두리와 안 맞음).
      아무 키나 누르면 다시 실시간.
n/p : 다음 / 이전 사진
q   : 끝

뎁스는 쓰지 않는다. 브라켓이 책상 위에 놓여 있다고 가정한다 (겹치거나 기대 있으면 틀린다).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import quiet_gtk

quiet_gtk()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vision.text import put_text  # noqa: E402

WIN = "keypoint live"
OUTLINE_BGR = (255, 200, 0)


def label(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color, thick: int = 1) -> None:
    """밝은 책상 위에서도 읽히게 글씨 뒤에 어두운 띠를 깐다."""
    x, y = org
    w = int(sum(24 if ord(ch) > 127 else 12 for ch in text) * scale) + 8  # 한글은 영문보다 두 배쯤 넓다
    h = int(22 * scale) + 4
    x0, y0 = max(0, x - 3), max(0, y - h + 4)
    x1, y1 = min(img.shape[1], x0 + w), min(img.shape[0], y + 6)
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.3).astype(np.uint8)
    put_text(img, text, org, scale, color, thick)


def _yaw_deg(T_base: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(T_base[1, 0], T_base[0, 0])))


def live_pose(frame: np.ndarray, model, cam) -> list:
    """실시간용: 키포인트 → 책상 붙이기. kpt_label.Instance 목록 (윤곽 · 자세 클래스 포함)."""
    from yolo.train import kpt_label as KL

    return [i for i in KL.model_guess(model, frame, cam) if i.T_cam_obj is not None]


def draw_live(frame: np.ndarray, insts: list, cam, fps: float, note: str) -> np.ndarray:
    from yolo.train import kpt_label as KL

    vis = frame.copy()
    for k, inst in enumerate(insts):
        for c in inst.outline or []:
            cv2.polylines(vis, [c.astype(np.int32)], True, OUTLINE_BGR, 1, cv2.LINE_AA)
        for i, ((u, v), vv) in enumerate(zip(inst.uv, inst.vis)):
            if vv:
                cv2.circle(vis, (int(u), int(v)), 2, KL.COLORS[i], -1)
        T = cam.T_base_cam @ inst.T_cam_obj
        x0, y0 = int(inst.box[0]), int(inst.box[1])
        label(vis, f"{k} {KL.CLASS_KO[inst.cls]}", (x0, max(14, y0 - 22)), 0.5, (0, 255, 0), 1)
        label(vis, f"x{T[0, 3]:.0f} y{T[1, 3]:.0f} {_yaw_deg(T):.0f}°", (x0, max(30, y0 - 5)), 0.45, (0, 255, 0), 1)
    label(vis, f"브라켓 {len(insts)}  {fps:.1f} fps   c 정밀 분석 · n/p 사진 · q 끝", (8, 22), 0.55, (0, 255, 0), 1)
    if note:
        label(vis, note, (8, 46), 0.55, (80, 80, 255), 1)
    return vis


def analyze(frame: np.ndarray, model, cam) -> tuple[np.ndarray, list[dict]]:
    """정밀 분석: 테두리 자동 맞춤 + 불량 판정. (그림, 결과 목록). 통합 카메라 창과 같은 코드를 쓴다."""
    from yolo.pose.pose_methods import draw_judged, judge_all

    poses = [cam.T_base_cam @ inst.T_cam_obj for inst in live_pose(frame, model, cam)]
    results = judge_all(frame, poses, cam)
    return draw_judged(frame, results, "정밀 분석 (멈춤). 아무 키나 → 실시간"), results


def print_results(results: list[dict], title: str) -> None:
    from yolo.pose.pose_methods import print_judged

    print_judged(results, title)


def main() -> None:
    ap = argparse.ArgumentParser(description="실시간 키포인트 자세 · 불량 검사")
    ap.add_argument("--weights", default=None, help="기본: kpt_best.pt → kpt_base.pt (없으면 pose_runs 의 v4n_ft2 등)")
    ap.add_argument("--images", type=Path, default=None, help="카메라 대신 이 폴더의 사진")
    ap.add_argument("--calib", type=Path, default=None,
                    help="캘리브 폴더 (intrinsics.json, eye_to_hand.json). 기본: 이 보드의 vision/calib_data")
    args = ap.parse_args()

    if args.images is None:
        # 카메라는 통합 카메라 창에서 (자세 방식 = 키포인트). k 로 CAD 정합과 바꿔 볼 수 있다.
        import os

        cmd = [sys.executable, str(Path(__file__).resolve().parent / "roi_cloud.py"), "--pose", "kpt"]
        if args.weights:
            cmd += ["--kpt-weights", str(args.weights)]
        print("카메라는 통합 카메라 창으로 엽니다: " + " ".join(cmd[1:]))
        os.execv(sys.executable, cmd)

    from ultralytics import YOLO

    from yolo.train import kpt_label as KL

    weights = Path(args.weights) if args.weights else KL.pick_weights()
    if not weights.is_file():
        raise SystemExit(f"가중치가 없습니다: {weights}\n키포인트 기본 모델 yolo/weights/kpt_base.pt 를 USB 로 복사하세요 (노트북의 v4n_ft2 best.pt).")
    model = YOLO(str(weights))
    cam = KL.calib_camera(args.calib)
    print(f"가중치 {weights}\n캘리브 {cam.source}  (카메라 높이 {cam.T_base_cam[2, 3]:.0f} mm)")

    images = sorted(list(args.images.glob("*.jpg")) + list(args.images.glob("*.png")))
    if not images:
        raise SystemExit(f"사진이 없습니다: {args.images}")
    idx = 0
    note, fps, t_prev = "", 0.0, time.perf_counter()
    try:
        while True:
            frame = cv2.imread(str(images[idx]))
            if frame is None:
                print(f"사진을 읽을 수 없습니다: {images[idx]}")
                break
            insts = live_pose(frame, model, cam)
            now = time.perf_counter()
            fps = 0.8 * fps + 0.2 / max(now - t_prev, 1e-3)
            t_prev = now
            name = images[idx].name
            cv2.imshow(WIN, draw_live(frame, insts, cam, fps, note or name))
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                break
            if k == ord("c"):
                vis, results = analyze(frame, model, cam)
                print_results(results, name)
                cv2.imshow(WIN, vis)
                cv2.waitKey(0)
                note = ""
            elif k in (ord("n"), ord("p")):
                idx = max(0, min(idx + (1 if k == ord("n") else -1), len(images) - 1))
                note = ""
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
