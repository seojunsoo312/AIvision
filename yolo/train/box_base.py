#!/usr/bin/env python3
"""(강사, 노트북) 박스 기본 모델 box_base.pt 를 만든다.

① CAD 정합 방식의 박스 모델 (박스 + 자세 종류 0 세우기 · 1 눕히기 · 2 비스듬히) 의 출발점이다.
학생은 5차시에 사진 20장을 라벨해 이 모델 위에 학습한다. train.py 가 기본으로 이 모델에서 시작하고,
기본 모델 학습 데이터 (yolo/datasets/box_base/) 를 섞음 세트로 섞는다 (키포인트 kpt_train.py 와 같은 방법).

  python yolo/train/box_base.py --make-data     # 섞음 세트 yolo/datasets/box_base/ (실사진 + 시뮬, 실제 파일)
  python yolo/train/box_base.py --train         # 그것으로 학습 → yolo/weights/box_base.pt
  python yolo/train/box_base.py --make-eval     # 평가 세트 (학습에 안 쓴 실사진 · 시뮬) yolo/runs/box_eval/

라벨은 새로 손으로 달지 않는다.
  실사진: 키포인트 라벨 (박스 칸 + 옆 .meta.json 의 자세 종류) 을 박스 라벨로 바꾼다 (학생 자리 학습분 30 · 젯슨 2차 42).
  시뮬: 장면의 분할 마스크로 박스, 정답 자세로 자세 종류 (책상 위 세 자세가 아닌 것 · 거의 가려진 것은 뺀다).
평가용으로 빼 두는 것: 학생 자리 시험 사진 25장, 시뮬 scenes3 (브라켓 3개씩), scenes_robot 뒤 50장.
보드에는 box_base.pt 와 yolo/datasets/box_base/ (25 MB) 를 같은 경로로 복사한다 (reset_class.py 가 지우지 않는다).
노트북 결과 (2026-11-09): 학습에 안 쓴 평가 세트 (실사진 25 + 시뮬 100) mAP50 0.99, 80 에폭 9분.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import BOX_BASE_PT, ROOT, RUNS_DIR, TRAIN_WORKERS, WEIGHTS_DIR, pose_names_yaml  # noqa: E402

BOX_BASE = ROOT / "datasets" / "box_base"
BASE_PT = BOX_BASE_PT
EVAL_DIR = RUNS_DIR / "box_eval"
SIM_OUT = ROOT.parent / "vision" / "sim" / "out"
# 시뮬 장면 묶음 → 섞음 세트에 넣을 장 수. scenes3 · scenes_robot 뒤 절반은 평가용
SIM_TRAIN_SETS = {
    "scenes": 60, "scenes_random": 60, "scenes_distract": 60, "scenes_dim": 40, "scenes_harsh": 40,
    "scenes_lowcontrast": 40, "scenes_silver": 40, "scenes3_random": 50, "scenes_robot": 50,
}
MIN_VISIBLE_PX = 200  # 이보다 적게 보이는 브라켓은 라벨에서 뺀다 (거의 다 가려짐)


def kpt_to_box_lines(kpt_txt: Path) -> list[str] | None:
    """키포인트 라벨 → 박스 라벨 줄 ("자세종류 cx cy w h"). 자세 종류 (meta) 가 없거나 모르면 None."""
    meta = kpt_txt.with_suffix(".meta.json")
    if not meta.is_file():
        return None
    classes = json.loads(meta.read_text(encoding="utf-8")).get("pose_class", [])
    rows = [ln.split() for ln in kpt_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(classes) != len(rows):
        return None
    out = []
    for cls, v in zip(classes, rows):
        if int(cls) not in (0, 1, 2):
            return None
        out.append(f"{int(cls)} {v[1]} {v[2]} {v[3]} {v[4]}")
    return out


def sim_box_lines(folder: Path) -> list[str]:
    """시뮬 장면 → 박스 라벨 줄. 분할 마스크의 외접 사각형, 정답 자세 종류."""
    from vision.sim.data import load_scene
    from yolo.pose.sim_eval import seg_box

    scene = load_scene(folder)
    h, w = scene.seg.shape
    out = []
    for o in scene.objects:
        if o["pose_class"] not in (0, 1, 2) or o["visible_px"] < MIN_VISIBLE_PX:
            continue
        box = seg_box(scene.seg, o["id"])
        if box is None:
            continue
        x1, y1, x2, y2 = box
        out.append(f"{o['pose_class']} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return out


def _put(out: Path, name: str, img: Path, lines: list[str]) -> None:
    shutil.copy2(img.resolve(), out / "images" / f"{name}{img.suffix.lower()}")
    (out / "labels" / f"{name}.txt").write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")


def _real_sources(which: str) -> list[tuple[str, Path, Path]]:
    """(이름, 사진, 키포인트 라벨). which = train (섞음 세트) / eval (학생 자리 시험)."""
    out = []
    seat = ROOT / "datasets" / "seat"
    split = json.loads((seat / "split.json").read_text(encoding="utf-8"))
    if which == "train":
        for n in split["train"]:
            out.append((f"seat_{Path(n).stem}", seat / "images" / n, seat / "labels" / f"{Path(n).stem}.txt"))
        j2 = ROOT / "datasets" / "jetson2"
        for img in sorted((j2 / "images").iterdir()):
            out.append((f"jetson2_{img.stem}", img, j2 / "labels" / f"{img.stem}.txt"))
    else:
        for n in split["test"]:
            out.append((f"seat_{Path(n).stem}", seat / "images" / n, seat / "labels_test_ref" / f"{Path(n).stem}.txt"))
    return out


def make_data(out: Path, real: str, sim_sets: dict[str, tuple[int, int]]) -> None:
    """실사진 + 시뮬 장면 → out/images, out/labels (실제 파일). sim_sets: 묶음 → (시작 번호, 장 수)."""
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    n_real = n_skip = 0
    for name, img, lab in _real_sources(real):
        if not (img.is_file() and lab.is_file()):
            continue
        lines = kpt_to_box_lines(lab)
        if lines is None:
            n_skip += 1
            continue
        _put(out, name, img, lines)
        n_real += 1
    n_sim = n_obj = 0
    for set_name, (start, count) in sim_sets.items():
        folders = sorted(p for p in (SIM_OUT / set_name).iterdir() if (p / "rgb.png").is_file())[start:start + count]
        for f in folders:
            lines = sim_box_lines(f)
            _put(out, f"sim_{set_name}_{f.name}", f / "rgb.png", lines)
            n_sim += 1
            n_obj += len(lines)
    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"{out}: 실사진 {n_real}장 (자세 종류 없어 뺀 것 {n_skip}) + 시뮬 {n_sim}장 (브라켓 {n_obj}개), {size_mb:.0f} MB")


def _write_yaml(path: Path, train: Path, val: Path) -> None:
    path.write_text(f"path: {path.parent}\ntrain: {train}\nval: {val}\n{pose_names_yaml()}", encoding="utf-8")


def train_base(epochs: int, batch: int) -> None:
    """섞음 세트 전체로 yolo11n 에서 학습 → box_base.pt. 검증은 평가 세트 (학습에 안 쓴 것)."""
    if not (BOX_BASE / "images").is_dir():
        raise SystemExit("먼저 --make-data")
    if not (EVAL_DIR / "images").is_dir():
        raise SystemExit("먼저 --make-eval")
    from ultralytics import YOLO

    from yolo.config import default_start_pt
    from yolo.train.train import _device

    work = RUNS_DIR / "box_base"
    work.mkdir(parents=True, exist_ok=True)
    yaml = work / "data.yaml"
    _write_yaml(yaml, BOX_BASE / "images", EVAL_DIR / "images")
    model = YOLO(default_start_pt())
    model.train(data=str(yaml), epochs=epochs, imgsz=640, batch=batch, device=_device(), workers=TRAIN_WORKERS,
                project=str(work), name="train", exist_ok=True, plots=True)
    best = work / "train" / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, BASE_PT)
    print(f"복사 {best} → {BASE_PT}")


def main() -> None:
    ap = argparse.ArgumentParser(description="(강사) 박스 기본 모델과 섞음 세트")
    ap.add_argument("--make-data", action="store_true", help="섞음 세트 yolo/datasets/box_base 만들기")
    ap.add_argument("--make-eval", action="store_true", help="평가 세트 yolo/runs/box_eval 만들기")
    ap.add_argument("--train", action="store_true", help="box_base.pt 학습")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    if not (args.make_data or args.make_eval or args.train):
        ap.error("--make-data / --make-eval / --train 중 하나 이상")
    if args.make_data:
        make_data(BOX_BASE, "train", {k: (0, v) for k, v in SIM_TRAIN_SETS.items()})
    if args.make_eval:
        make_data(EVAL_DIR, "eval", {"scenes3": (0, 50), "scenes_robot": (50, 50)})
    if args.train:
        train_base(args.epochs, args.batch)


if __name__ == "__main__":
    main()
