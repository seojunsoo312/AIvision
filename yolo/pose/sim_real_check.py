"""시뮬로 학습한 YOLO-pose 를 실사진(yolo/datasets/raw)에 돌려 박스 라벨과 비교한다. 실험용.

키포인트 정답은 없으므로 박스만 센다: 찾은 브라켓 수, 놓친 수, 엉뚱한 검출 수.
'작업 구역' 은 라벨된 브라켓 중심들을 감싸는 사각형을 조금 넓힌 것 (영상 기준). 구역 밖 검출을 버리면 얼마나 주는지 본다.

    python -m yolo.pose.sim_real_check vision/sim/out/pose_runs/v3n/weights/best.pt --name v3n
결과: 바탕화면 sim_실사진_시험/<name>_*.jpg 와 표 한 줄

마스킹 (박스 중심이 가려진 곳이면 검출을 버림):
  --desk-mask   알루미늄 프로파일 안쪽(x DESK_X0~DESK_X1)만 남김
  --robot-mask base   움직이지 않는 로봇 베이스 사각형을 가림
  --robot-mask color  사진마다 보라색 로봇을 색으로 따서 가림 (실제 시스템이면 관절값으로 로봇을 투영해 대신함)
  --robot-mask both   베이스 사각형 + 색 (케이블은 베이스 사각형이, 움직이는 팔은 색이 가림)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

from yolo.config import ROOT

RAW_DIR = ROOT / "datasets" / "raw"
OUT_DIR = Path.home() / "Desktop" / "sim_실사진_시험"
ZONE_MARGIN_PX = 30
MATCH_IOU = 0.5
# 옛 사진 기준 알루미늄 프로파일 안쪽 경계 (px)
DESK_X0, DESK_X1 = 45, 530
# 움직이지 않는 로봇 베이스 (x0, y0, x1, y1) px
ROBOT_BASE_BOX = (215, 335, 355, 480)
# 보라색 로봇 색 범위 (HSV, OpenCV H 0~179)
ROBOT_HSV_LO, ROBOT_HSV_HI = (115, 80, 40), (150, 255, 255)
ROBOT_DILATE_PX = 8
COLORS = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
          (255, 0, 0), (255, 0, 255), (128, 0, 255), (255, 255, 255), (0, 0, 0)]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _labels(img_path: str) -> list[list[float]]:
    lab = Path(img_path.replace("/images/", "/labels/")).with_suffix(".txt")
    out = []
    if lab.is_file():
        for line in lab.read_text().splitlines():
            v = [float(x) for x in line.split()[:5]]
            if len(v) == 5:
                cx, cy, w, h = v[1] * 640, v[2] * 480, v[3] * 640, v[4] * 480
                out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out


def keep_mask(img: np.ndarray, desk: bool, robot: str) -> np.ndarray:
    """검출을 남길 픽셀 = True. 마스킹을 안 하면 전부 True."""
    h, w = img.shape[:2]
    keep = np.ones((h, w), bool)
    if desk:
        keep[:, :DESK_X0] = False
        keep[:, DESK_X1:] = False
    if robot in ("base", "both"):
        x0, y0, x1, y1 = ROBOT_BASE_BOX
        keep[y0:y1, x0:x1] = False
    if robot in ("color", "both"):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, ROBOT_HSV_LO, ROBOT_HSV_HI)
        # 작은 점은 버리고 로봇 덩어리만, 테두리를 조금 넓힌다.
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.dilate(m, np.ones((2 * ROBOT_DILATE_PX + 1,) * 2, np.uint8))
        keep &= m == 0
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description="시뮬 YOLO-pose 를 실사진에서 시험")
    ap.add_argument("weights")
    ap.add_argument("--name", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--desk-mask", action="store_true", help="알루미늄 프로파일 안쪽만 남김")
    ap.add_argument("--robot-mask", choices=("none", "base", "color", "both"), default="none")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    files = sorted(glob.glob(str(RAW_DIR / "images" / "*.jpg")))
    gts = {f: _labels(f) for f in files}
    centers = np.array([[(g[0] + g[2]) / 2, (g[1] + g[3]) / 2] for gl in gts.values() for g in gl])
    zone = [*(centers.min(axis=0) - ZONE_MARGIN_PX), *(centers.max(axis=0) + ZONE_MARGIN_PX)]
    tp = fn = fp = fp_zone = gt_masked = 0
    tiles = []
    for k, f in enumerate(files):
        img = cv2.imread(f)
        r = model.predict(img, conf=args.conf, verbose=False, device=args.device)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        keep = keep_mask(img, args.desk_mask, args.robot_mask)
        cxy = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1).astype(int) if len(boxes) else np.zeros((0, 2), int)
        cxy = np.clip(cxy, 0, [img.shape[1] - 1, img.shape[0] - 1])
        sel = np.array([bool(keep[y, x]) for x, y in cxy], bool)
        if r.boxes is not None and len(boxes):
            r = r[sel]
        boxes = boxes[sel]
        # 정답 브라켓 중심이 가려진 곳에 있으면 마스킹 때문에 절대 못 찾는다 (따로 센다).
        for g in gts[f]:
            gx, gy = int((g[0] + g[2]) / 2), int((g[1] + g[3]) / 2)
            if not keep[min(gy, img.shape[0] - 1), min(gx, img.shape[1] - 1)]:
                gt_masked += 1
        used = set()
        for g in gts[f]:
            best = max(((_iou(g, b), j) for j, b in enumerate(boxes)), default=(0.0, -1))
            if best[0] >= MATCH_IOU:
                tp += 1
                used.add(best[1])
            else:
                fn += 1
        for j, b in enumerate(boxes):
            if j in used:
                continue
            fp += 1
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]:
                fp_zone += 1
        if k % 8 == 0 and len(tiles) < 16:
            vis = img.copy()
            vis[~keep] = (vis[~keep] * 0.4).astype(np.uint8)
            cv2.rectangle(vis, (int(zone[0]), int(zone[1])), (int(zone[2]), int(zone[3])), (255, 255, 0), 1)
            for g in gts[f]:
                cv2.rectangle(vis, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (255, 255, 255), 1)
            if r.boxes is not None and len(boxes):
                kxy = r.keypoints.xy.cpu().numpy()
                kc = r.keypoints.conf.cpu().numpy()
                for j, b in enumerate(boxes):
                    col = (0, 255, 0) if j in used else (0, 0, 255)
                    cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 1)
                    for i, ((x, y), c) in enumerate(zip(kxy[j], kc[j])):
                        if c >= 0.5:
                            cv2.circle(vis, (int(x), int(y)), 2, COLORS[i], -1)
            tiles.append(cv2.resize(vis, (480, 360)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 16, 4)]
    cv2.imwrite(str(OUT_DIR / f"{args.name}_전체16장.jpg"), np.vstack(rows))
    res = {"name": args.name, "conf": args.conf, "brackets": tp + fn, "found": tp, "missed": fn,
           "false": fp, "false_in_zone": fp_zone, "images": len(files),
           "desk_mask": args.desk_mask, "robot_mask": args.robot_mask, "gt_in_mask": gt_masked}
    (OUT_DIR / f"{args.name}.json").write_text(json.dumps(res, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
