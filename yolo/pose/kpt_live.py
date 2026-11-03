#!/usr/bin/env python3
"""실시간 키포인트 자세 · 불량 검사 창 (sim-pose 브랜치 실험). 기존 roi_cloud 는 그대로 두고 따로 돈다.

  python -m yolo.pose.kpt_live                                    # 카메라 (이 보드 캘리브)
  python -m yolo.pose.kpt_live --images yolo/datasets/seat/images --calib yolo/datasets/seat/calib   # 사진으로 시험

평소: 카메라 → YOLO-pose 키포인트 → 책상 붙이기 (세우기 · 눕히기 · 비스듬히 중 하나로, 손눈의 책상 높이로 깊이).
      화면에 CAD 윤곽 · 자세 종류 · 베이스 위치(mm) · 돌아간 각도.
c   : 정밀 분석. 테두리 자동 맞춤 → 흰 브라켓 불량 판정 (마커). 화면을 멈추고 터미널에 표 (펜던트에 넣을 xyz · RPY 포함).
      윤곽 색: 초록 양품 · 빨강 불량 · 보라 검사 안 함 (검은 브라켓) · 주황 브라켓 아님 (윤곽이 사진 테두리와 안 맞음).
      아무 키나 누르면 다시 실시간.
s   : 지금 영상 저장 (yolo/datasets/live/)
n/p : (--images 일 때) 다음 / 이전 사진
q   : 끝

뎁스는 쓰지 않는다. 브라켓이 책상 위에 놓여 있다고 가정한다 (겹치거나 기대 있으면 틀린다).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import ROOT, quiet_gtk

quiet_gtk()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vision.text import put_text  # noqa: E402

LIVE_DIR = ROOT / "datasets" / "live"
WIN = "keypoint live"
OUTLINE_BGR = (255, 200, 0)
# 판정 색: 초록 양품, 빨강 불량, 보라 검사 안 함 (검은 브라켓. 회색은 화면에서 안 보였다), 주황 브라켓 아님
VERDICT_BGR = {"양품": (0, 200, 0), "불량": (0, 0, 255), "검사 안 함": (255, 0, 200), "브라켓 아님": (0, 140, 255)}


def label(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color, thick: int = 1) -> None:
    """밝은 책상 위에서도 읽히게 글씨 뒤에 어두운 띠를 깐다."""
    x, y = org
    w = int(sum(24 if ord(ch) > 127 else 12 for ch in text) * scale) + 8  # 한글은 영문보다 두 배쯤 넓다
    h = int(22 * scale) + 4
    x0, y0 = max(0, x - 3), max(0, y - h + 4)
    x1, y1 = min(img.shape[1], x0 + w), min(img.shape[0], y + 6)
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.3).astype(np.uint8)
    put_text(img, text, org, scale, color, thick)


def pendant_numbers(T_base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """베이스 mm 4×4 → 펜던트 칸 숫자 (xyz mm, 베이스 RPY, 물체축 RPY). roi_cloud 와 같은 표기."""
    from motion.grasp import canonicalize_extrinsic_rpy, rpy_extrinsic_to_body_xyz
    from yolo.pose.register import T_to_xyzrpy

    xyz, rpy_raw = T_to_xyzrpy(T_base)
    rpy_base = canonicalize_extrinsic_rpy(rpy_raw)
    rpy_body = rpy_extrinsic_to_body_xyz(rpy_base)
    return np.asarray(xyz, float), np.asarray(rpy_base, float), np.asarray(rpy_body, float)


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
    label(vis, f"브라켓 {len(insts)}  {fps:.1f} fps   c 정밀 분석 · s 저장 · q 끝", (8, 22), 0.55, (0, 255, 0), 1)
    if note:
        label(vis, note, (8, 46), 0.55, (80, 80, 255), 1)
    return vis


def analyze(frame: np.ndarray, model, cam) -> tuple[np.ndarray, list[dict]]:
    """정밀 분석: 테두리 자동 맞춤 + 불량 판정. (그림, 결과 목록). 터미널 출력은 부르는 쪽."""
    from yolo.pose.sim_defect_check import inspect
    from yolo.train import kpt_label as KL

    results = inspect(frame, model, cam)
    vis = frame.copy()
    for k, r in enumerate(results):
        inst = KL.Instance()
        KL.set_pose(inst, r["T_base"], cam, KL.SRC_SAVED)
        KL.fit(inst, cam, *frame.shape[1::-1])
        r["inst"] = inst
        r["pendant"] = pendant_numbers(r["T_base"])
        col = VERDICT_BGR[r["verdict"]]
        if r["white"]:
            vis[r["marks"]] = (0, 0, 255)
        for c in inst.outline or []:
            cv2.polylines(vis, [c.astype(np.int32)], True, col, 2, cv2.LINE_AA)
        x0, y0 = int(inst.box[0]), int(inst.box[1])
        label(vis, f"{k} {r['pose']} {r['verdict']}", (x0, max(16, y0 - 6)), 0.6, col, 2)
    label(vis, "정밀 분석 (멈춤). 아무 키나 → 실시간", (8, 22), 0.6, (0, 255, 255), 2)
    return vis, results


def print_results(results: list[dict], title: str) -> None:
    print(f"\n== {title}  브라켓 {len(results)}개")
    if not results:
        return
    print(f"{'#':>2s} {'자세':6s} {'색':3s} {'판정':8s} {'어두운px':>7s} {'테두리':>6s}   x mm    y mm    z mm |  베이스 Rx  Ry  Rz |  물체축 Rx  Ry  Rz")
    for k, r in enumerate(results):
        xyz, rb, ro = r["pendant"]
        print(f"{k:>2d} {r['pose']:6s} {'흰' if r['white'] else '검':3s} {r['verdict']:8s} {r['dark_px']:>7d} {r['edge_px']:6.2f}  "
              f"{xyz[0]:6.1f}  {xyz[1]:6.1f}  {xyz[2]:6.1f} | {rb[0]:6.1f} {rb[1]:6.1f} {rb[2]:6.1f} | "
              f"{ro[0]:6.1f} {ro[1]:6.1f} {ro[2]:6.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="실시간 키포인트 자세 · 불량 검사")
    ap.add_argument("--weights", default=None, help="기본: pose_runs 의 v4n_ft2 → v4n_ft → v4n 중 있는 것")
    ap.add_argument("--images", type=Path, default=None, help="카메라 대신 이 폴더의 사진")
    ap.add_argument("--calib", type=Path, default=None,
                    help="캘리브 폴더 (intrinsics.json, eye_to_hand.json). 기본: 이 보드의 vision/calib_data")
    args = ap.parse_args()

    from ultralytics import YOLO

    from yolo.train import kpt_label as KL

    weights = Path(args.weights) if args.weights else KL.pick_weights()
    if not weights.is_file():
        raise SystemExit(f"가중치가 없습니다: {weights}\n노트북의 vision/sim/out/pose_runs/v4n_ft2/weights/best.pt 를 같은 자리에 복사하세요.")
    model = YOLO(str(weights))
    cam = KL.calib_camera(args.calib)
    print(f"가중치 {weights}\n캘리브 {cam.source}  (카메라 높이 {cam.T_base_cam[2, 3]:.0f} mm)")

    cap, images, idx = None, [], 0
    if args.images:
        images = sorted(list(args.images.glob("*.jpg")) + list(args.images.glob("*.png")))
        if not images:
            raise SystemExit(f"사진이 없습니다: {args.images}")
    else:
        from vision.camera import open_camera

        cap = open_camera()

    def grab():
        if images:
            return cv2.imread(str(images[idx]))
        from vision.camera import grab_bgr

        return grab_bgr(cap)

    note, fps, t_prev = "", 0.0, time.perf_counter()
    try:
        while True:
            frame = grab()
            if frame is None:
                print("프레임을 읽을 수 없습니다.")
                break
            insts = live_pose(frame, model, cam)
            now = time.perf_counter()
            fps = 0.8 * fps + 0.2 / max(now - t_prev, 1e-3)
            t_prev = now
            name = images[idx].name if images else ""
            cv2.imshow(WIN, draw_live(frame, insts, cam, fps, note or name))
            k = cv2.waitKey(1 if cap is not None else 30) & 0xFF
            if k == ord("q"):
                break
            if k == ord("c"):
                vis, results = analyze(frame, model, cam)
                print_results(results, name or datetime.now().strftime("%H:%M:%S"))
                cv2.imshow(WIN, vis)
                cv2.waitKey(0)
                note = ""
            elif k == ord("s"):
                LIVE_DIR.mkdir(parents=True, exist_ok=True)
                path = LIVE_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
                cv2.imwrite(str(path), frame)
                note = f"저장 {path.name}"
                print(note)
            elif images and k in (ord("n"), ord("p")):
                idx = max(0, min(idx + (1 if k == ord("n") else -1), len(images) - 1))
                note = ""
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
