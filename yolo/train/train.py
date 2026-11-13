#!/usr/bin/env python3
"""박스 모델 (자세 3클래스: 세우기 / 눕히기 / 비스듬히) 을 이 보드에서 학습한다.

  python yolo/train/train.py                 # 기본: 박스 기본 모델 box_base.pt 에서 짧게 (학생 사진 20장이면 됨)
  python yolo/train/train.py --no-base       # 예전 방식: yolo11n 에서 처음부터 (사진 수백 장 필요)

기본 모델 (yolo/weights/box_base.pt) 이 있으면 키포인트 학습 (kpt_train.py) 과 같은 방법으로 학습한다:
학생 사진을 10번 반복하고 박스 섞음 세트 (yolo/datasets/box_base/) 300장을 섞어 15 에폭, 결과는 마지막 에폭 모델.
  - 기본 모델 없이 사진 20장으로 처음부터 하면 절반도 못 찾는다 (노트북 시험: mAP50 0.47, 기본 모델에서 0.995).
  - 반복은 학생 사진을 충분히 배우게, 섞음은 그동안 원래 실력을 잊지 않게 한다. 섞지 않고 10번 반복하면
    원래 실력을 잊었다 (학습에 안 쓴 실사진 0.995 → 0.884). 노트북 RTX 약 5분.
--no-mix 는 섞음 · 반복 없이 기본 모델에서 가볍게 (원래 실력은 유지하지만 새로 배우는 것이 적다).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import (
    BEST_PT,
    DATA_YAML,
    POSE_CLASSES,
    RUNS_DIR,
    TRAIN_BATCH,
    TRAIN_EPOCHS,
    TRAIN_IMGSZ,
    TRAIN_TXT,
    TRAIN_WORKERS,
    WEIGHTS_DIR,
    box_base_pt,
    default_start_pt,
    pose_names_yaml,
    VAL_TXT,
)

BASE_PT = box_base_pt()  # box_base.pt (없으면 예전 이름 bracket_base.pt)
BOX_BASE = Path(__file__).resolve().parents[1] / "datasets" / "box_base"


def _mix_lists(own_train: list[str], own_val: list[str], repeat: int, n_base: int, seed: int = 0) -> Path:
    """학생 목록 × repeat + 섞음 세트 → splits/train_mix.txt · val_mix.txt 와 runs/train_mix.yaml."""
    import random

    base = sorted(str(p.resolve()) for p in (BOX_BASE / "images").iterdir())
    rng = random.Random(seed)
    rng.shuffle(base)
    train = [p for p in own_train for _ in range(repeat)] + base[:n_base]
    val = own_val + base[n_base:n_base + 30]
    mix_train, mix_val = TRAIN_TXT.with_name("train_mix.txt"), VAL_TXT.with_name("val_mix.txt")
    mix_train.write_text("".join(p + "\n" for p in train), encoding="utf-8")
    mix_val.write_text("".join(p + "\n" for p in val), encoding="utf-8")
    yaml = RUNS_DIR / "train_mix.yaml"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    yaml.write_text(f"path: {RUNS_DIR}\ntrain: {mix_train}\nval: {mix_val}\n{pose_names_yaml()}", encoding="utf-8")
    print(f"학생 사진 {len(own_train)}장 (×{repeat}) + 섞음 {min(n_base, len(base))}장 → 학습 {len(train)} · 검증 {len(val)}")
    return yaml


def _device() -> str | int:
    import torch

    if torch.cuda.is_available():
        return 0
    print("경고: CUDA 없음. CPU 학습은 Orin Nano에서 사실상 불가능에 가깝다.")
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="3-pose YOLO11n train")
    parser.add_argument("--epochs", type=int, default=None, help="기본: 기본 모델에서면 15, 처음부터면 50")
    parser.add_argument("--batch", type=int, default=TRAIN_BATCH)
    parser.add_argument("--imgsz", type=int, default=TRAIN_IMGSZ)
    parser.add_argument("--model", default=None, help="시작 가중치. 기본: box_base.pt (있으면)")
    parser.add_argument("--no-base", action="store_true", help="기본 모델 · 섞음 세트 없이 yolo11n 에서 처음부터")
    parser.add_argument("--no-mix", action="store_true", help="섞음 세트 · 반복 없이 기본 모델에서 가볍게")
    parser.add_argument("--repeat", type=int, default=10, help="학생 사진을 몇 번 반복해 넣을지")
    parser.add_argument("--base-n", type=int, default=300, help="섞음 세트에서 학습에 넣을 장 수")
    args = parser.parse_args()
    batch = args.batch
    from_base = not args.no_base and args.model is None and BASE_PT.is_file()
    start_model = args.model or (str(BASE_PT) if from_base else default_start_pt())
    if args.model is None and not args.no_base and not BASE_PT.is_file():
        print(f"경고: 박스 기본 모델이 없습니다 ({BASE_PT}). yolo11n 에서 처음부터 학습합니다 (사진이 많이 필요).")
    epochs = args.epochs if args.epochs is not None else (15 if from_base else TRAIN_EPOCHS)
    run_name = "train"
    dest = BEST_PT

    n_train = 0
    if TRAIN_TXT.is_file():
        n_train = sum(1 for line in TRAIN_TXT.read_text(encoding="utf-8").splitlines() if line.strip())
    if n_train < 8:
        raise SystemExit("데이터셋이 없습니다. python yolo/train/prepare.py 를 먼저 실행하세요.")
    if not DATA_YAML.exists():
        raise SystemExit(f"없음: {DATA_YAML}")

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        raise SystemExit(
            "ultralytics 가 이 파이썬에 없습니다.\n"
            f"지금 실행: {sys.executable}\n"
            "conda activate AIvision 한 뒤, /bin/python 말고:\n"
            "  python yolo/train/train.py"
        )

    data_yaml = DATA_YAML
    mixing = from_base and not args.no_mix
    if mixing and not (BOX_BASE / "images").is_dir():
        print(f"경고: 박스 섞음 세트가 없습니다 ({BOX_BASE}). 섞지 않고 가볍게 학습합니다 "
              "(노트북에서 python yolo/train/box_base.py --make-data 로 만들어 보드에 복사).")
        mixing = False
    if mixing:
        own_train = [ln for ln in TRAIN_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
        own_val = [ln for ln in VAL_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
        data_yaml = _mix_lists(own_train, own_val, args.repeat, args.base_n)

    device = _device()
    print(
        f"classes={[name for _cid, name, _ko in POSE_CLASSES]}  "
        f"device={device}  batch={batch}  "
        f"epochs={epochs}  model={start_model}"
    )
    model = YOLO(start_model)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        workers=TRAIN_WORKERS,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        cache="ram",  # 사진 수백 장이면 0.2GB 남짓. 매 epoch JPEG 를 다시 풀지 않는다.
        patience=30,
        plots=True,
    )
    # 기본 모델에서 짧게 학습할 때는 마지막 에폭 모델 (검증 세트 대부분이 섞음 세트라 첫 에폭이 "가장 좋음" 으로 뽑힌다)
    run_best = RUNS_DIR / run_name / "weights" / ("last.pt" if mixing else "best.pt")
    if not run_best.exists():
        raise SystemExit(f"학습 결과 없음: {run_best}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_best, dest)
    print(f"복사 {run_best} → {dest}")
    print("다음: python yolo/train/detect.py")


if __name__ == "__main__":
    main()
