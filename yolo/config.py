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
# 가중치 이름 (모두 yolo/weights/, git 에 없음):
#   box_base.pt  박스 기본 모델 (강사가 미리 넣어 둠, box_base.py 가 만든다)   → 5차시 학습의 출발점
#   box_best.pt  학생이 5차시에 학습한 박스 모델 (train.py 가 만든다)
#   kpt_base.pt  키포인트 기본 모델 (강사가 미리 넣어 둠, 노트북의 v4n_ft2)   → 6차시 학습의 출발점
#   kpt_best.pt  학생이 6차시에 학습한 키포인트 모델 (kpt_train.py 가 만든다)
# 초기화 (reset_class.py) 는 *_best.pt 만 지운다.
BEST_PT = WEIGHTS_DIR / "box_best.pt"
BOX_BASE_PT = WEIGHTS_DIR / "box_base.pt"
KPT_BASE_PT = WEIGHTS_DIR / "kpt_base.pt"
KPT_BEST_PT = WEIGHTS_DIR / "kpt_best.pt"
# 예전 이름. 이름을 바꾸기 전의 보드는 이 파일만 있다 → box_best_pt() · box_base_pt() 가 대신 쓴다.
OLD_BEST_PT = WEIGHTS_DIR / "best.pt"
OLD_BOX_BASE_PT = WEIGHTS_DIR / "bracket_base.pt"


def box_best_pt():
    """박스 모델 경로. box_best.pt 가 없고 예전 이름 best.pt 만 있으면 그것을 쓴다."""
    if not BEST_PT.exists() and OLD_BEST_PT.exists():
        print(f"박스 모델: 예전 이름 {OLD_BEST_PT.name} 을 씁니다 (새 이름은 {BEST_PT.name}).")
        return OLD_BEST_PT
    return BEST_PT


def box_base_pt():
    """박스 기본 모델 경로. box_base.pt 가 없고 예전 이름 bracket_base.pt 만 있으면 그것을 쓴다."""
    if not BOX_BASE_PT.exists() and OLD_BOX_BASE_PT.exists():
        return OLD_BOX_BASE_PT
    return BOX_BASE_PT

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


