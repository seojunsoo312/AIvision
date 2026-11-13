"""쌓인 더미 (빈 피킹) 시뮬 장면에서 자세 방식들을 정답과 비교한다. 실험용 (sim-pose 브랜치).

장면 만들기:  python -m vision.sim.scene --count 30 --objects 8 --pile --seed 100 --out vision/sim/out/scenes_pile
평가:        python -m yolo.pose.sim_pile_eval --scenes vision/sim/out/scenes_pile --noise

방식
  cad_rest   : 카메라 창 CAD 정합 기본 (책상 위 세 자세로 찾고, 안 맞으면 예전 방식)
  cad_free   : CAD 정합, 책상 가정 없이 (예전 방식 전체 검색)
  kpt_snap   : 카메라 창 키포인트 기본 (책상 붙이기)
  kpt_pnp    : 키포인트 PnP 만 (책상 가정 없음)
  kpt_icp    : 키포인트 PnP → 뎁스로 깊이 맞춤 → ICP (책상 가정 없음)
물체 찾기는 모두 같은 조건이 되게, CAD 쪽은 정답 분할 박스, 키포인트 쪽은 키포인트 모델 검출을 쓴다.

정답 쪽 "위가 열림" = 다른 브라켓이 그 위를 덮지 않음 (집을 수 있는 후보).
"고른 것" = 방식마다 가장 높다고 본 브라켓 (CAD 는 카메라 창의 select_topmost, 키포인트는 추정 자세의 꼭대기 높이).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time
from pathlib import Path

import numpy as np

from vision.sim.data import load_scene
from vision.sim.noise import add_noise
from yolo.pose.sim_eval import (
    SUCCESS_POS_MM,
    SUCCESS_ROT_DEG,
    depth_scale,
    detect_keypoints,
    match_detections,
    pnp_pose,
    pose_error,
    scene_clouds,
    seg_box,
)

METHODS = ("cad_rest", "cad_free", "kpt_snap", "kpt_pnp", "kpt_icp")
# 집기에는 이 정도면 된다 (그리퍼 여유). 엄격한 기준 (3 mm / 5°) 도 같이 센다.
PICK_POS_MM = 5.0
PICK_ROT_DEG = 10.0
COVER_FRAC = 0.15  # 위가 이만큼 넘게 덮이면 집기 후보에서 뺀다
MIN_VISIBLE_PX = 300  # 이보다 적게 보이는 브라켓은 평가에서 뺀다 (거의 다 가려짐)


def _cad_points() -> np.ndarray:
    from yolo.train import kpt_label as KL

    v, _f = KL.mesh_mm()
    return np.asarray(v, float)[::7]


def _world(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ np.asarray(T, float)[:3, :3].T + np.asarray(T, float)[:3, 3]


def open_top(scene, pts: np.ndarray) -> dict[int, bool]:
    """정답: 브라켓마다 위가 열려 있는지. 다른 브라켓 점이 xy 6 mm 안, 3 mm 넘게 위에 있는 점이
    COVER_FRAC 넘게 많으면 덮인 것 (옆 브라켓 모서리가 조금 걸친 것은 집는 데 지장 없다고 본다)."""
    W = {o["id"]: _world(o["T_base_obj"], pts) for o in scene.objects}
    out = {}
    for i, wi in W.items():
        under = np.zeros(len(wi), bool)
        for j, wj in W.items():
            if i == j:
                continue
            d = np.linalg.norm(wi[:, None, :2] - wj[None, :, :2], axis=2)
            dz = wj[None, :, 2] - wi[:, None, 2]
            under |= np.any((d < 6.0) & (dz > 3.0), axis=1)
        out[i] = float(under.mean()) <= COVER_FRAC
    return out


def _err_row(T, T_gt, center) -> dict:
    if T is None:
        return {"pos": None, "rot": None, "ok": False, "pick": False}
    pos, rot = pose_error(np.asarray(T, float), T_gt, center)
    return {"pos": pos, "rot": rot, "ok": pos <= SUCCESS_POS_MM and rot <= SUCCESS_ROT_DEG,
            "pick": pos <= PICK_POS_MM and rot <= PICK_ROT_DEG}


def eval_scene(folder: Path, *, noise: bool, weights: str, seed: int, cad, center, pts) -> list[dict]:
    from yolo.pose.instances import select_topmost
    from yolo.pose.register import register_pose
    from yolo.train import kpt_label as KL

    scene = load_scene(folder)
    depth = add_noise(scene.depth_mm, scene.K, np.random.default_rng(seed)) if noise else scene.depth_mm
    gt = {o["id"]: o for o in scene.objects}
    top_open = open_top(scene, pts)
    est: dict[str, dict[int, np.ndarray | None]] = {m: {} for m in METHODS}
    times: dict[str, float] = {m: 0.0 for m in METHODS}

    # --- CAD 정합 (정답 분할 박스 → 카메라 창과 같은 점군) ---
    pairs, desk_base, _pm = scene_clouds(scene, depth, "none")
    clouds = {o["id"]: inst for o, inst in pairs}
    chosen_cad = None
    idx = select_topmost([inst for _o, inst in pairs])
    if idx is not None:
        chosen_cad = pairs[idx][0]["id"]
    for oid, inst in clouds.items():
        if len(inst.xyz) < 20:
            continue
        for m, method in (("cad_rest", "rest"), ("cad_free", "class")):
            t0 = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    pose = register_pose(np.asarray(inst.xyz, np.float64), T_base_cam=None, voxel_mm=1.0, flip=True,
                                         tries=4, camera_origin=scene.T_base_cam[:3, 3], desk_plane=desk_base,
                                         class_id=None, method=method)
                est[m][oid] = np.asarray(pose["T"], float) if pose.get("aligned_ok") else None
            except Exception:
                est[m][oid] = None
            times[m] += time.perf_counter() - t0

    # --- 키포인트 ---
    bgr = np.ascontiguousarray(scene.rgb[..., ::-1])
    kcam = KL.Camera(scene.K, np.zeros(5), scene.T_base_cam, "sim")
    t0 = time.perf_counter()
    model = _kpt_model(weights)
    snap = KL.model_guess(model, bgr, kcam)
    times["kpt_snap"] += time.perf_counter() - t0
    snap_dets = [{"xyxy": list(i.box), "T": kcam.T_base_cam @ i.T_cam_obj} for i in snap
                 if i.T_cam_obj is not None and i.box is not None]
    for oid, d in match_detections(scene, snap_dets).items():
        est["kpt_snap"][oid] = d["T"]
    t0 = time.perf_counter()
    dets = detect_keypoints(bgr, weights)
    matched = match_detections(scene, dets)
    times["kpt_pnp"] += time.perf_counter() - t0
    for oid, det in matched.items():
        t1 = time.perf_counter()
        p = pnp_pose(det, scene.K, scene.T_base_cam, 0.5)
        times["kpt_pnp"] += time.perf_counter() - t1
        est["kpt_pnp"][oid] = None if p is None else p["T"]
        if p is None:
            est["kpt_icp"][oid] = None
            continue
        t1 = time.perf_counter()
        T, _dz = depth_scale(p["T"], cad, depth, scene.K, scene.T_base_cam)
        inst = clouds.get(oid)
        if inst is not None and len(inst.xyz) >= 20:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    pose = register_pose(np.asarray(inst.xyz, np.float64), T_base_cam=None, voxel_mm=1.0, flip=True,
                                         tries=4, T_init=T, camera_origin=scene.T_base_cam[:3, 3],
                                         desk_plane=desk_base, method="class")
                T = np.asarray(pose["T"], float)
            except Exception:
                pass
        est["kpt_icp"][oid] = T
        times["kpt_icp"] += time.perf_counter() - t1

    # --- 정답과 비교 ---
    rows = []
    for oid, o in gt.items():
        if o["visible_px"] < MIN_VISIBLE_PX:
            continue
        T_gt = np.asarray(o["T_base_obj"], float)
        top_z = float(_world(T_gt, pts)[:, 2].max())
        on_desk = o["pose_class"] >= 0 and float(_world(T_gt, pts)[:, 2].min()) < 3.0
        row = {"scene": folder.name, "obj": oid, "open": top_open[oid], "on_desk": on_desk, "top_z": top_z,
               "visible_px": o["visible_px"]}
        for m in METHODS:
            row[m] = _err_row(est[m].get(oid), T_gt, center)
            row[m]["found"] = oid in est[m]
        rows.append(row)

    # 방식마다 고른 브라켓 (가장 높다고 본 것)
    chosen = {"cad_rest": chosen_cad, "cad_free": chosen_cad}
    for m in ("kpt_snap", "kpt_pnp", "kpt_icp"):
        cand = [(float(_world(T, pts)[:, 2].max()), oid) for oid, T in est[m].items() if T is not None]
        chosen[m] = max(cand)[1] if cand else None
    pick = {}
    for m in METHODS:
        oid = chosen[m]
        if oid is None:
            pick[m] = {"chosen": None, "open": False, "pick": False}
            continue
        T = est[m].get(oid)
        e = _err_row(T, np.asarray(gt[oid]["T_base_obj"], float), center)
        pick[m] = {"chosen": oid, "open": top_open[oid], "pick": bool(e["pick"] and top_open[oid])}
    return rows, pick, times


_MODELS: dict = {}


def _kpt_model(weights: str):
    if weights not in _MODELS:
        from ultralytics import YOLO

        _MODELS[weights] = YOLO(weights)
    return _MODELS[weights]


def summarize(all_rows: list[dict], all_picks: list[dict], times: dict, n_scenes: int) -> str:
    def rate(rows, m, key):
        if not rows:
            return "-"
        return f"{sum(r[m][key] for r in rows)}/{len(rows)} ({100 * sum(r[m][key] for r in rows) / len(rows):.0f}%)"

    groups = {
        "모든 브라켓": all_rows,
        "책상 위 (세 자세)": [r for r in all_rows if r["on_desk"]],
        "얹히거나 기댄 것": [r for r in all_rows if not r["on_desk"]],
        "위가 열린 것 (집기 후보)": [r for r in all_rows if r["open"]],
    }
    lines = [f"장면 {n_scenes}개, 평가 브라켓 {len(all_rows)}개 (보이는 픽셀 {MIN_VISIBLE_PX} 이상)", ""]
    lines.append(f"자세 맞음 (집기 기준 {PICK_POS_MM:.0f} mm · {PICK_ROT_DEG:.0f}°)")
    lines.append("| 묶음 | " + " | ".join(METHODS) + " |")
    lines.append("|---" * (len(METHODS) + 1) + "|")
    for name, rows in groups.items():
        lines.append(f"| {name} | " + " | ".join(rate(rows, m, "pick") for m in METHODS) + " |")
    lines.append("")
    lines.append(f"엄격 기준 ({SUCCESS_POS_MM:.0f} mm · {SUCCESS_ROT_DEG:.0f}°), 모든 브라켓: "
                 + ", ".join(f"{m} {rate(all_rows, m, 'ok')}" for m in METHODS))
    lines.append("")
    lines.append("방식마다 '맨 위' 로 고른 브라켓: 위가 열려 있고 자세도 맞음 (= 그대로 집으면 됨)")
    for m in METHODS:
        n_ok = sum(p[m]["pick"] for p in all_picks)
        n_open = sum(p[m]["open"] for p in all_picks)
        lines.append(f"- {m}: {n_ok}/{len(all_picks)} 장면  (고른 것이 위가 열림 {n_open}/{len(all_picks)})")
    lines.append("")
    lines.append("장면당 시간 (노트북): " + ", ".join(f"{m} {times[m] / max(n_scenes, 1):.1f}s" for m in METHODS))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="쌓인 더미 (빈 피킹) 자세 평가")
    ap.add_argument("--scenes", type=Path, default=Path("vision/sim/out/scenes_pile"))
    ap.add_argument("--n", type=int, default=0, help="앞에서 몇 장면만 (0 = 전부)")
    ap.add_argument("--noise", action="store_true", help="뎁스에 실물 같은 노이즈")
    ap.add_argument("--weights", default=None, help="키포인트 모델 (기본 v4n_ft2 → v4n_ft → v4n)")
    ap.add_argument("--out", type=Path, default=None, help="결과 표를 이 파일 (md) 에도")
    args = ap.parse_args()

    from yolo.pose.register import load_cad_xyz
    from yolo.train import kpt_label as KL

    weights = args.weights or str(KL.base_weights())  # 시험은 기본 모델로 (학생 모델과 섞이지 않게)
    with contextlib.redirect_stdout(io.StringIO()):
        cad = load_cad_xyz(voxel_mm=1.0)
    center = cad.mean(axis=0)
    pts = _cad_points()
    folders = sorted(p for p in args.scenes.iterdir() if (p / "rgb.png").is_file())
    if args.n:
        folders = folders[: args.n]
    all_rows, all_picks = [], []
    times = {m: 0.0 for m in METHODS}
    for k, f in enumerate(folders):
        rows, pick, t = eval_scene(f, noise=args.noise, weights=weights, seed=1000 + k, cad=cad, center=center, pts=pts)
        all_rows += rows
        all_picks.append(pick)
        for m in METHODS:
            times[m] += t[m]
        print(f"{f.name}: " + "  ".join(f"{m} {sum(r[m]['pick'] for r in rows)}/{len(rows)}" for m in METHODS), flush=True)
    text = summarize(all_rows, all_picks, times, len(folders))
    head = f"뎁스 노이즈 {'켬' if args.noise else '끔'}, 키포인트 {weights}"
    print("\n" + head + "\n" + text)
    if args.out:
        args.out.write_text(f"# 쌓인 더미 평가\n\n{head}\n\n{text}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
