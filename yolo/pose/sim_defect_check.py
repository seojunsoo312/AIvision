"""흰 브라켓 마커 불량 검사 (규칙 기반) 합성 시험. 실험용 (sim-pose 브랜치).

규칙: 자세로 구한 CAD 윤곽 안쪽(테두리에서 ERODE_PX 줄임) 에서 구멍 자리를 빼고, 아주 어두운 픽셀(회색값 < DARK_LEVEL) 이
MIN_DARK_PX 개 이상이면 불량. 구멍 안쪽 그림자는 마커만큼 어두워서 CAD 로 구멍 자리를 계산해 뺀다.
값은 학생 자리 양품 흰 브라켓 30개 (어두운 픽셀 최대 4) 와 실제 마커 불량 5장 (최소 11), 합성 중간 진하기 마커로 정했다 (2026-11-02).
합성: 학생 자리 사진의 흰 브라켓 위에 마커 선 · 점을 그려 불량 사진을 만든다 (굵기 · 길이 · 잉크 진하기를 바꿔 가며).

    python -m yolo.pose.sim_defect_check                                       # 합성 시험
    python -m yolo.pose.sim_defect_check --real yolo/datasets/seat_defect/defect   # 실제 사진 판정
결과: 터미널에 표 (사진 · 브라켓마다 판정) + 바탕화면 sim_불량검사_합성.jpg / sim_불량검사_<폴더>.jpg
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from yolo.config import ROOT

SEAT = ROOT / "datasets" / "seat"
DARK_LEVEL = 90
ERODE_PX = 3
MIN_DARK_PX = 7
# 합성 마커: (굵기 px, 길이 px). 길이 0 = 점 (지름 = 굵기). 이 카메라에서 1 px ≈ 1 mm.
# 실제 마커 사진은 굵기 2~3 px, 길이 10~20 px, 작은 점도 있었다. 그보다 작고 짧은 것까지 본다.
STROKES = [(1.5, 5), (1.5, 8), (2.0, 5), (2.0, 8), (2.0, 12), (3.0, 0), (4.0, 0), (5.0, 0)]
# 잉크 진하기 (칠한 곳 회색값 범위). 진함 = 실제 마커 사진에서 잰 값 (0~40). 덜 칠했거나 번들거리면 더 연하다.
INKS = {"진함": (20, 40), "중간": (50, 70), "연함": (80, 100)}
# 4 배로 그려 줄여서 1 px 보다 가는 굵기도 표현한다
_SS = 4


# 구멍 자리 (CAD mm): 두 판 모두 길이 17 (가운데 22), 폭 8.5, 판 두께 4. 둘레 HOLE_MARGIN_MM 까지 뺀다.
# 비스듬히 놓이면 구멍 안을 들여다보게 되어 안쪽 그림자가 마커만큼 어둡다.
HOLE_MARGIN_MM = 1.0
# 자세가 1 px 쯤 어긋나도 구멍 가장자리 그림자가 새지 않게, 뺄 구멍 자리를 화면에서 이만큼 더 넓힌다
HOLE_GROW_PX = 1
_SLOT_HALF_LEN, _SLOT_HALF_W, _PLATE_HALF_T = 8.5, 4.25, 2.0


def _in_hole_zone(p: np.ndarray) -> np.ndarray:
    m = HOLE_MARGIN_MM
    near_y = np.abs(p[:, 1]) <= _SLOT_HALF_W + m
    bottom = (np.abs(p[:, 0] - 22.0) <= _SLOT_HALF_LEN + m) & (np.abs(p[:, 2]) <= _PLATE_HALF_T + 1.0)
    upright = (np.abs(p[:, 2] - 22.0) <= _SLOT_HALF_LEN + m) & (np.abs(p[:, 0]) <= _PLATE_HALF_T + 1.0)
    return near_y & (bottom | upright)


def bracket_region(T_base: np.ndarray, cam, shape: tuple[int, int]) -> np.ndarray:
    """검사할 픽셀: 자세의 CAD 윤곽 안쪽 (테두리 ERODE_PX 줄임) 에서 구멍 자리를 뺀 것.

    픽셀마다 카메라에서 광선을 쏴 CAD 에 처음 닿는 곳이 구멍 벽 · 구멍 둘레이거나, 구멍을 지나 아무것도 안 닿으면 뺀다.
    """
    import open3d as o3d

    from yolo.train import kpt_label as KL

    # 바깥 테두리만 줄인다 (구멍 쪽은 줄이지 않는다: 구멍 바로 옆 마커도 봐야 한다)
    mk = cv2.erode(KL._silhouette(T_base, cam, shape, (0, 0)), np.ones((2 * ERODE_PX + 1,) * 2, np.uint8))
    ys, xs = np.nonzero(mk)
    if len(xs):
        n = cv2.undistortPoints(np.stack([xs, ys], 1).astype(np.float64).reshape(-1, 1, 2), cam.K, cam.dist).reshape(-1, 2)
        T_obj_cam = np.linalg.inv(T_base) @ cam.T_base_cam
        d = np.hstack([n, np.ones((len(n), 1))]) @ T_obj_cam[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        o = np.broadcast_to(T_obj_cam[:3, 3], d.shape)
        t = KL._ray_scene().cast_rays(o3d.core.Tensor(np.hstack([o, d]).astype(np.float32)))["t_hit"].numpy()
        hit = np.isfinite(t)
        p = o + d * np.where(hit, t, 0.0)[:, None]
        drop = ~hit | _in_hole_zone(p)
        hole = np.zeros_like(mk)
        hole[ys[drop], xs[drop]] = 1
        if HOLE_GROW_PX > 0:
            hole = cv2.dilate(hole, np.ones((2 * HOLE_GROW_PX + 1,) * 2, np.uint8))
        mk[hole > 0] = 0
    return mk


def dark_count(bgr: np.ndarray, region: np.ndarray) -> int:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return int(((g < DARK_LEVEL) & (region > 0)).sum())


def draw_marker(bgr: np.ndarray, region: np.ndarray, width: float, length: float, rng,
                ink_range: tuple[float, float] = INKS["진함"]) -> np.ndarray | None:
    """영역 안에 마커 선 (약간 휨) 또는 점 하나. 4 배로 그려 줄여 가는 선도 부드럽게. 영역을 대부분 벗어나면 None."""
    ys, xs = np.nonzero(region)
    if len(xs) < 30:
        return None
    for _ in range(30):
        k = rng.integers(len(xs))
        c = np.array([xs[k], ys[k]], float)
        a = rng.uniform(0, np.pi)
        d = np.array([np.cos(a), np.sin(a)])
        n = np.array([-d[1], d[0]])
        bend = rng.uniform(-0.15, 0.15) * length
        t = np.linspace(-0.5, 0.5, 12)[:, None]
        pts = c + t * length * d + (1 - 4 * t ** 2) * bend * n
        h, w = region.shape
        big = np.zeros((h * _SS, w * _SS), np.uint8)
        if length <= 0:
            cv2.circle(big, tuple((c * _SS).astype(int)), max(1, int(round(width * _SS / 2))), 255, -1, cv2.LINE_AA)
        else:
            cv2.polylines(big, [(pts * _SS).astype(np.int32)], False, 255, max(1, int(round(width * _SS))), cv2.LINE_AA)
        a = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(float) / 255.0
        cover = a > 0.3
        if cover.sum() and (cover & (region > 0)).sum() >= 0.8 * cover.sum():
            ink = rng.uniform(*ink_range) + rng.normal(0, 3, 3)
            alpha = (a * rng.uniform(0.9, 0.97))[..., None]
            return (bgr * (1 - alpha) + ink * alpha).astype(np.uint8)
    return None


POSE_WEIGHTS = ROOT.parent / "vision" / "sim" / "out" / "pose_runs" / "v4n_ft" / "weights" / "best.pt"
# 흰 브라켓으로 볼 검사 영역 밝기 (중앙값). 검은 브라켓은 검사하지 않는다.
WHITE_MEDIAN = 150


def inspect(bgr: np.ndarray, model, cam) -> list[dict]:
    """한 사진의 브라켓마다 판정. 찾기 → 책상 붙이기 → 테두리 자동 맞춤 → 규칙.

    돌려주는 것: [{"box", "pose", "white", "dark_px", "verdict" ("양품" / "불량" / "검사 안 함"), "region", "T_base"}]
    """
    from yolo.train import kpt_label as KL

    H, W = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = []
    for inst in KL.model_guess(model, bgr, cam):
        if inst.T_cam_obj is None:
            continue
        res = KL.fit_edges(bgr, cam.T_base_cam @ inst.T_cam_obj, cam)
        T = res[0] if res is not None else cam.T_base_cam @ inst.T_cam_obj
        KL.set_pose(inst, T, cam, KL.SRC_SAVED)
        KL.fit(inst, cam, W, H)
        reg = bracket_region(T, cam, (H, W))
        white = bool(reg.sum() > 50 and np.median(g[reg > 0]) > WHITE_MEDIAN)
        n = dark_count(bgr, reg)
        verdict = ("불량" if n >= MIN_DARK_PX else "양품") if white else "검사 안 함"
        out.append({"box": inst.box, "pose": KL.CLASS_KO[inst.cls], "white": white, "dark_px": n,
                    "verdict": verdict, "region": reg, "T_base": T})
    return out


def run_real(folder: Path) -> None:
    """폴더 안 사진마다 판정해 터미널에 표로, 바탕화면에 그림으로."""
    from ultralytics import YOLO

    from vision.text import put_text
    from yolo.train import kpt_label as KL

    cam = KL.load_camera(SEAT / "images")
    model = YOLO(str(POSE_WEIGHTS))
    images = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
    print(f"사진 {len(images)}장 ({folder})   규칙: 회색 < {DARK_LEVEL} 픽셀 {MIN_DARK_PX} 개 이상이면 불량 (흰 브라켓만)")
    print(f"{'사진':28s} {'브라켓':>4s} {'자세':6s} {'색':3s} {'어두운 px':>8s}  판정")
    tiles, count = [], {"양품": 0, "불량": 0, "검사 안 함": 0}
    for p in images:
        bgr = cv2.imread(str(p))
        found = inspect(bgr, model, cam)
        if not found:
            print(f"{p.name:28s}    - 브라켓 못 찾음")
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for k, r in enumerate(found):
            count[r["verdict"]] += 1
            print(f"{p.name:28s} {k:>4d} {r['pose']:6s} {'흰' if r['white'] else '검':3s} {r['dark_px']:>8d}  {r['verdict']}")
            reg = r["region"]
            sil = KL._silhouette(r["T_base"], cam, reg.shape, (0, 0))
            ys, xs = np.nonzero(sil)
            if not len(xs):
                continue
            x0, y0 = max(0, xs.min() - 18), max(0, ys.min() - 18)
            x1, y1 = xs.max() + 18, ys.max() + 18
            a = bgr[y0:y1, x0:x1].copy()
            ov = a.copy()
            ov[reg[y0:y1, x0:x1] > 0] = (0, 200, 0)
            b = cv2.addWeighted(a, 0.75, ov, 0.25, 0)
            b[((g < DARK_LEVEL) & (reg > 0))[y0:y1, x0:x1]] = (0, 0, 255)
            A = cv2.resize(a, (240, 240), interpolation=cv2.INTER_CUBIC)
            B = cv2.resize(b, (240, 240), interpolation=cv2.INTER_NEAREST)
            put_text(B, f"{r['dark_px']}px → {r['verdict']}", (4, 20), 0.6, (0, 0, 255), 2)
            tiles.append(np.hstack([A, B]))
    print("합계: " + ", ".join(f"{k} {v}" for k, v in count.items()))
    if tiles:
        sheet = np.vstack(tiles)
        put_text(sheet, "초록 = 검사 영역, 빨강 = 마커로 본 픽셀", (4, sheet.shape[0] - 8), 0.5, (0, 0, 255), 1)
        dest = Path.home() / "Desktop" / f"sim_불량검사_{folder.parent.name}_{folder.name}.jpg"
        cv2.imwrite(str(dest), sheet)
        print(f"그림: {dest}")


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="흰 브라켓 마커 불량 검사 (규칙 기반)")
    ap.add_argument("--real", type=Path, default=None, help="이 폴더의 실제 사진을 판정 (없으면 합성 시험)")
    args = ap.parse_args()
    if args.real is not None:
        run_real(args.real)
        return

    from vision.text import put_text
    from yolo.train import kpt_label as KL

    cam = KL.load_camera(SEAT / "images")
    rng = np.random.default_rng(0)
    # 흰 브라켓 모으기: 학습 라벨 + 시험 기준 답
    whites = []
    for lab_dir in (SEAT / "labels", SEAT / "labels_test_ref"):
        for lab in sorted(lab_dir.glob("*.txt")):
            if lab.name == "kpt_set.txt":
                continue
            img = SEAT / "images" / f"{lab.stem}.jpg"
            bgr = cv2.imread(str(img))
            H, W = bgr.shape[:2]
            for inst in KL.load_labels(lab, W, H):
                KL.fit(inst, cam, W, H)
                if inst.T_cam_obj is None:
                    continue
                T = cam.T_base_cam @ inst.T_cam_obj
                reg = bracket_region(T, cam, (H, W))
                g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                if reg.sum() > 50 and np.median(g[reg > 0]) > 150:  # 흰 브라켓만
                    whites.append((img.name, bgr, reg))
    good = [dark_count(b, r) for _, b, r in whites]
    # 그림: 1줄 양품, 2~4줄 선 2x8 mm 진함 · 중간 · 연함, 5줄 점 4 mm 중간
    show = {"양품": [], ("진함", 2.0, 8): [], ("중간", 2.0, 8): [], ("연함", 2.0, 8): [], ("중간", 4.0, 0): []}

    def tile(img, reg, text):
        ys, xs = np.nonzero(reg)
        x0, y0 = max(0, xs.min() - 15), max(0, ys.min() - 15)
        crop = cv2.resize(img[y0:ys.max() + 15, x0:xs.max() + 15], (200, 200), interpolation=cv2.INTER_CUBIC)
        put_text(crop, text, (4, 20), 0.5, (0, 0, 255), 1)
        return crop

    for name, bgr, reg in whites[::5][:6]:
        show["양품"].append(tile(bgr, reg, f"양품: {dark_count(bgr, reg)}"))
    rows = [f"흰 브라켓 {len(whites)}개.  양품(원본) 어두운 픽셀 수: 최대 {max(good)}, 중앙값 {int(np.median(good))}",
            "잡은 수 / 시도 (흰 브라켓 30개 × 3번)   | " + " | ".join(f"{k} (회색 {a}~{b})" for k, (a, b) in INKS.items())]
    result = {"good_max": int(max(good)), "strokes": {}}
    for w, L in STROKES:
        what = f"점 지름 {w:g}mm" if L == 0 else f"선 {w:g} × {L:g}mm"
        cells = []
        for ink_name, ink in INKS.items():
            counts = []
            for name, bgr, reg in whites:
                for rep_i in range(3):
                    d = draw_marker(bgr, reg, w, L, rng, ink)
                    if d is None:
                        continue
                    counts.append(dark_count(d, reg))
                    key = (ink_name, w, L)
                    if key in show and len(show[key]) < 6 and rep_i == 0:
                        verdict = "불량" if counts[-1] >= MIN_DARK_PX else "양품"
                        show[key].append(tile(d, reg, f"{ink_name} {what}: {counts[-1]} → {verdict}"))
            c = np.array(counts)
            caught = int((c >= MIN_DARK_PX).sum())
            cells.append(f"{caught}/{len(c)}")
            result["strokes"][f"{ink_name}_{w}x{L}"] = {"caught": caught, "n": len(c)}
        rows.append(f"{what:16s} | " + " | ".join(cells))
    false_alarm = sum(g >= MIN_DARK_PX for g in good)
    rows.append(f"양품을 불량으로 잘못 판정: {false_alarm}/{len(good)}  (문턱: 회색 < {DARK_LEVEL}, {MIN_DARK_PX} px 이상, 테두리 {ERODE_PX} px 제외)")
    print("\n".join(rows))
    sheet = np.vstack([np.hstack(v + [np.zeros((200, 200, 3), np.uint8)] * (6 - len(v))) for v in show.values()])
    cv2.imwrite(str(Path.home() / "Desktop" / "sim_불량검사_합성.jpg"), sheet)
    (ROOT.parent / "vision" / "sim" / "out" / "results" / "defect_rule_synth.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
