"""YOLO paths and pose-class names.

자세 클래스는 POSE_CLASSES 가 정본이다. 라벨 txt 의 맨 앞 숫자가 그 인덱스다.
기존 파일의 0 은 지우지 않는다. 라벨 UI에서 눌러 고치기 전까지 세우기로 읽힌다.
"""

from __future__ import annotations

import os
from pathlib import Path

from cad.model import class_name
from vision.gui_env import quiet_gtk  # noqa: F401  (yolo 스크립트들이 여기서 가져다 쓴다)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

CLASS_ID = 0
# (인덱스, data.yaml 이름, 라벨 화면). 베이스 자세: 세우기 / 눕히기 / 비스듬히.
POSE_CLASSES: tuple[tuple[int, str, str], ...] = (
    (0, "세우기", "세우기"),
    (1, "눕히기", "눕히기"),
    (2, "비스듬히", "비스듬히"),
)

RAW_IMAGES = ROOT / "datasets" / "raw" / "images"
RAW_LABELS = ROOT / "datasets" / "raw" / "labels"
SPLITS_DIR = ROOT / "datasets" / "splits"
TRAIN_TXT = SPLITS_DIR / "train.txt"
VAL_TXT = SPLITS_DIR / "val.txt"
DATA_YAML = ROOT / "data.yaml"
WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"
# 등록 요청 파일(runs/roi/)과 대기 시간은 motion/register_link.py.
BEST_PT = WEIGHTS_DIR / "best.pt"

# 부품 이름(class)은 cad/model.yaml 에만 둔다. 읽기는 cad.model.
CLASS_NAME = class_name()


def pose_name(class_id: int, *, korean: bool = False) -> str:
    for cid, name, ko in POSE_CLASSES:
        if cid == int(class_id):
            return ko if korean else name
    return str(class_id)


def pose_names_yaml() -> str:
    lines = [f"nc: {len(POSE_CLASSES)}", "names:"]
    for cid, name, _ko in POSE_CLASSES:
        lines.append(f"  {cid}: {name}")
    return "\n".join(lines) + "\n"


def add_class_argument(parser) -> None:
    parser.add_argument(
        "--class-name",
        default=None,
        metavar="NAME",
        help=f"YOLO 1클래스 표시 이름. 생략하면 cad/model.yaml ({CLASS_NAME})",
    )


def class_from_args(args) -> str:
    raw = getattr(args, "class_name", None)
    if raw is None:
        return CLASS_NAME
    name = str(raw).strip()
    return name or CLASS_NAME


VAL_RATIO = 0.2
SPLIT_SEED = 42

# Orin Nano 8GB. 학습이 OOM 나면 batch를 2로.
TRAIN_IMGSZ = 640
TRAIN_BATCH = 4
TRAIN_EPOCHS = 50
# 0 이면 CPU 한 코어가 사진 준비를 혼자 해 GPU 가 절반을 논다. Orin Nano Super 실측(280장):
# workers 0 → 20.1s/epoch, workers 2 → 14.6s/epoch, RAM 최대 6.3GB. batch 8 은 2.4배 빠르지만
# RAM 6.8GB·스왑 1.2GB 에 과전류 스로틀이 계속 걸려 4 로 둔다.
TRAIN_WORKERS = 2
DETECT_CONF = 0.5


def default_start_pt() -> str:
    """전이학습 시작 가중치. weights/ 에 있으면 그 경로, 없으면 파일명만 (ultralytics 가 받음)."""
    name = "yolo11n.pt"
    local = WEIGHTS_DIR / name
    return str(local) if local.is_file() else name


