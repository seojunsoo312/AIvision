"""시뮬 데이터(vision/sim/pose_data.py)로 YOLO-pose (브라켓 키포인트 10개) 를 학습한다. 실험용.

    python -m yolo.pose.sim_pose_train --epochs 80
결과: vision/sim/out/pose_runs/<name>/weights/best.pt
"""

from __future__ import annotations

import argparse

from vision.sim.data import OUT_DIR
from vision.sim.pose_data import DATA_DIR
from yolo.config import WEIGHTS_DIR

RUNS_DIR = OUT_DIR / "pose_runs"


def main() -> None:
    ap = argparse.ArgumentParser(description="시뮬 YOLO-pose 학습")
    ap.add_argument("--data", default=str(DATA_DIR / "data.yaml"))
    start = WEIGHTS_DIR / "yolo11n-pose.pt"
    ap.add_argument(
        "--model",
        default=str(start) if start.is_file() else "yolo11n-pose.pt",
        help="시작 가중치. yolo/weights 에 없으면 ultralytics 가 지금 폴더에 내려받는다",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default="v2")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        workers=4,
        # 뒤집기는 쓰지 않는다. 눕히기 · 비스듬히는 대칭 때문에 "영상 오른쪽 판 끝 = 0번" 규칙으로 라벨을 정했는데,
        # 좌우로 뒤집으면 이 규칙이 반대가 되어 서로 모순된 라벨을 배운다 (flip_idx 로는 못 고친다).
        flipud=0.0,
        fliplr=0.0,
        degrees=10.0,
        plots=True,
    )


if __name__ == "__main__":
    main()
