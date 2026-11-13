#!/usr/bin/env python3
"""키포인트 모델 미세조정: 학생이 라벨한 사진 몇 장 + 기본 섞음 세트 → yolo/weights/kpt_best.pt

수업 (6차시):
  python yolo/train/kpt_train.py
    사진 yolo/datasets/raw/images, 라벨 yolo/datasets/raw/kpt_labels (kpt_label.py --labels 로 단 것)
    기본 모델 (kpt_base.pt = v4n_ft2) 에서 시작해 짧게 학습. 끝나면 카메라 창 (roi_cloud.py --pose kpt, k) 과
    키포인트 라벨 도구의 자동 찍기가 이 모델을 쓴다. reset_class.py 가 지운다.

강사 (노트북에서 한 번, 보드에 복사):
  python yolo/train/kpt_train.py --make-base
    시뮬 사진 일부 + 학생 자리 · 젯슨 실사진을 yolo/datasets/kpt_base/ 로 모은다 (보드에는 시뮬 사진이 없다).
    학생 사진 20장만으로 학습하면 원래 잘 하던 것을 잊는다. 이 세트를 섞어 막는다.

학생 사진은 여러 번 반복해 넣는다 (기본 10번). 몇 장이라 한 번씩만 넣으면 섞음 세트에 묻힌다.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import KPT_BEST_PT, ROOT, RUNS_DIR, TRAIN_WORKERS, WEIGHTS_DIR  # noqa: E402

RAW_IMAGES = ROOT / "datasets" / "raw" / "images"
RAW_KPT_LABELS = ROOT / "datasets" / "raw" / "kpt_labels"
BASE_DIR = ROOT / "datasets" / "kpt_base"
KPT_RUNS = RUNS_DIR / "kpt"
OUT_PT = KPT_BEST_PT
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
# --make-base 에서 가져오는 곳 (노트북에만 있다): v4n_ft2 를 학습한 데이터. 이름 앞이 seat_ · jetson2_ 면 실사진, sim_ 이면 시뮬
FT2_DATA = ROOT.parent / "vision" / "sim" / "out" / "pose_data_ft2"


def _labeled_images(images: Path, labels: Path) -> list[tuple[Path, Path]]:
    """라벨이 있는 사진만 (라벨 도구에서 저장한 것). 빈 라벨 파일 = 브라켓 없는 사진으로 쓴다."""
    out = []
    for img in sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        lab = labels / f"{img.stem}.txt"
        if lab.is_file():
            out.append((img, lab))
    return out


def _link(src: Path, dst: Path) -> None:
    """사진은 복사하지 않고 링크로 (용량). 링크가 안 되는 곳이면 복사."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _check_kpt_set(labels: Path) -> None:
    """라벨 폴더의 키포인트 묶음이 지금 코드와 같은지 (다르면 학습이 엉뚱해진다)."""
    from vision.sim.keypoints import KPT_SET

    tag = labels / "kpt_set.txt"
    if tag.is_file() and tag.read_text().strip() != KPT_SET:
        raise SystemExit(f"{labels} 는 '{tag.read_text().strip()}' 키포인트 라벨입니다. 지금 코드는 '{KPT_SET}'.")


def write_yaml(data_dir: Path) -> Path:
    from vision.sim.keypoints import FLIP_IDX, N_KPT

    path = data_dir / "data.yaml"
    path.write_text(
        f"path: {data_dir}\ntrain: images/train\nval: images/val\n"
        f"kpt_shape: [{N_KPT}, 3]\nflip_idx: {FLIP_IDX}\nnames:\n  0: bracket\n",
        encoding="utf-8",
    )
    return path


def build_dataset(pairs: list[tuple[Path, Path]], base: list[tuple[Path, Path]], data_dir: Path,
                  repeat: int, n_base: int, seed: int = 0) -> tuple[int, int]:
    """학습 폴더를 새로 만든다. (학습 사진 수, 검증 사진 수).

    학생 사진: 5장에 1장꼴로 검증에 두고 (최소 1장), 나머지를 repeat 번 넣는다.
    섞음 세트: 앞 n_base 장은 학습, 그다음 최대 30장은 검증 (잊어버리는지 보려고).
    """
    if data_dir.exists():
        shutil.rmtree(data_dir)
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    n_val = max(1, len(pairs) // 5)
    val_own, train_own = pairs[:n_val], pairs[n_val:]
    base = list(base)
    rng.shuffle(base)
    train_base, val_base = base[:n_base], base[n_base:n_base + 30]

    def put(split: str, img: Path, lab: Path, name: str) -> None:
        _link(img, data_dir / "images" / split / f"{name}{img.suffix.lower()}")
        dst = data_dir / "labels" / split / f"{name}.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lab, dst)

    n_train = 0
    for img, lab in train_own:
        for r in range(repeat):
            put("train", img, lab, f"own_{img.stem}_r{r}")
            n_train += 1
    for img, lab in train_base:
        put("train", img, lab, f"base_{img.stem}")
        n_train += 1
    for img, lab in val_own:
        put("val", img, lab, f"own_{img.stem}")
    for img, lab in val_base:
        put("val", img, lab, f"base_{img.stem}")
    write_yaml(data_dir)
    return n_train, len(val_own) + len(val_base)


def make_base(out: Path, n_sim: int, seed: int = 0) -> None:
    """(노트북) v4n_ft2 학습 데이터에서 실사진 전부 + 시뮬 n_sim 장을 실제 파일로 복사한다 (보드로 옮기기 위해)."""
    src_img, src_lab = FT2_DATA / "images", FT2_DATA / "labels"
    if not src_img.is_dir():
        raise SystemExit(f"없음: {src_img}\n--make-base 는 시뮬 데이터가 있는 노트북에서 실행합니다.")
    real, sim = {}, []
    for split in ("train", "val"):
        for img in sorted((src_img / split).iterdir()):
            lab = src_lab / split / f"{img.stem}.txt"
            if not lab.is_file():
                continue
            if img.name.startswith("sim_"):
                sim.append((img, lab))
            else:
                # 실사진은 여러 번 넣으려고 _r0, _r1 … 로 겹쳐 있다. 하나만.
                key = img.stem.rsplit("_r", 1)[0]
                real.setdefault(key, (img, lab, key))
    rng = random.Random(seed)
    rng.shuffle(sim)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    for img, lab, key in real.values():
        shutil.copy2(img.resolve(), out / "images" / f"{key}{img.suffix}")
        shutil.copy2(lab, out / "labels" / f"{key}.txt")
    for img, lab in sim[:n_sim]:
        shutil.copy2(img.resolve(), out / "images" / img.name)
        shutil.copy2(lab, out / "labels" / f"{img.stem}.txt")
    from vision.sim.keypoints import KPT_SET

    (out / "labels" / "kpt_set.txt").write_text(KPT_SET + "\n")
    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"기본 섞음 세트: 실사진 {len(real)}장 + 시뮬 {min(n_sim, len(sim))}장 → {out} ({size_mb:.0f} MB)")
    print("보드마다 같은 경로 (~/AIvision/yolo/datasets/kpt_base/) 로 복사합니다.")


def main() -> None:
    ap = argparse.ArgumentParser(description="키포인트 모델 미세조정 (학생 사진 + 기본 섞음 세트)")
    ap.add_argument("--images", type=Path, default=RAW_IMAGES, help="학생 사진 폴더")
    ap.add_argument("--labels", type=Path, default=RAW_KPT_LABELS, help="키포인트 라벨 폴더 (kpt_label.py --labels)")
    ap.add_argument("--base", type=Path, default=BASE_DIR, help="기본 섞음 세트 (--make-base 로 만든 것)")
    ap.add_argument("--repeat", type=int, default=10, help="학생 사진을 몇 번 반복해 넣을지")
    ap.add_argument("--base-n", type=int, default=300, help="섞음 세트에서 학습에 넣을 장 수")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", type=Path, default=None, help="시작 모델. 기본: kpt_base.pt (학생 모델이 아니라 기본 모델)")
    ap.add_argument("--make-base", action="store_true", help="(강사, 노트북) 기본 섞음 세트 만들기")
    ap.add_argument("--base-sim", type=int, default=400, help="--make-base 때 넣을 시뮬 장 수")
    args = ap.parse_args()

    if args.make_base:
        make_base(args.base, args.base_sim)
        return

    from yolo.train.kpt_label import base_weights

    start = args.model or base_weights()
    if not Path(start).is_file():
        raise SystemExit(f"시작 모델이 없습니다: {start}\n키포인트 기본 모델 yolo/weights/kpt_base.pt 를 USB 로 복사하세요 (노트북의 v4n_ft2 best.pt).")
    if not args.labels.is_dir():
        raise SystemExit(f"키포인트 라벨이 없습니다: {args.labels}\n"
                         f"먼저: python yolo/train/kpt_label.py --images {args.images} --labels {args.labels}")
    _check_kpt_set(args.labels)
    pairs = _labeled_images(args.images, args.labels)
    if len(pairs) < 4:
        raise SystemExit(f"라벨한 사진이 {len(pairs)}장입니다. 4장 이상 라벨하세요 (s 로 저장).")
    base = _labeled_images(args.base / "images", args.base / "labels") if (args.base / "images").is_dir() else []
    if not base:
        print(f"경고: 기본 섞음 세트가 없습니다 ({args.base}). 학생 사진만으로 학습하면 원래 실력을 잊을 수 있습니다.")

    data_dir = KPT_RUNS / "data"
    n_train, n_val = build_dataset(pairs, base, data_dir, args.repeat, args.base_n)
    print(f"학생 사진 {len(pairs)}장 (×{args.repeat}) + 섞음 {min(args.base_n, len(base))}장 → 학습 {n_train} · 검증 {n_val}")
    print(f"시작 모델 {start}")

    from ultralytics import YOLO

    from yolo.train.train import _device

    model = YOLO(str(start))
    model.train(
        data=str(data_dir / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=_device(),
        workers=TRAIN_WORKERS,
        project=str(KPT_RUNS),
        name="train",
        exist_ok=True,
        patience=30,
        plots=True,
        # 뒤집기는 쓰지 않는다 (시뮬 학습 sim_pose_train 과 같은 설정). 눕히기 · 비스듬히는 대칭 때문에
        # "영상 오른쪽 판 끝 = 0번" 규칙으로 라벨을 정해서, 좌우로 뒤집으면 서로 모순된 라벨을 배운다.
        flipud=0.0,
        fliplr=0.0,
        degrees=10.0,
    )
    # best.pt 가 아니라 마지막 것: 검증 세트 대부분이 섞음 세트라 첫 에폭 (거의 안 배운 모델) 이 "가장 좋음" 으로 뽑힌다.
    # 짧은 미세조정이라 끝까지 배운 모델을 쓴다 (노트북 시험에서 섞음 세트 · 시뮬 성능은 그대로였다).
    run_best = KPT_RUNS / "train" / "weights" / "last.pt"
    if not run_best.is_file():
        raise SystemExit(f"학습 결과 없음: {run_best}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_best, OUT_PT)
    print(f"복사 {run_best} → {OUT_PT}")
    print("카메라 창 (roi_cloud.py --pose kpt 또는 k) 과 키포인트 라벨 도구가 이제 이 모델을 씁니다.")


if __name__ == "__main__":
    main()
