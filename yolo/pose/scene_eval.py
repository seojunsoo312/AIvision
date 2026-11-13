"""저장한 실물 장면 (카메라 창 s) 으로 CAD 정합을 키포인트와 비교한다. 고칠 때마다 전후를 이 표로 본다.

장면마다
  기준: 키포인트 (+테두리 맞춤) 로 찾은 브라켓들과 흰 / 검 판정.
  CAD: 카메라 창과 같은 순서 (YOLO 상자 + 놓친 것은 뎁스 덩어리 → 점군 → 책상 위 세 자세 정합).
       맨 위는 느린 예전 방식까지, 나머지는 빠른 정합만 (카메라 창과 같다).
브라켓마다 CAD 성공 여부, 키포인트와 위치 · Rz 차이 (6차시 비교표와 같은 숫자), 걸린 시간을 적고,
흰 / 검으로 나눠 센다. 같은 배치를 여러 장 찍은 장면들은 맨 위 고르기와 결과가 같은지도 본다.

    python -m yolo.pose.scene_eval                      # yolo/runs/roi/scenes/ 전부
    python -m yolo.pose.scene_eval scene_A scene_B -v   # 브라켓마다 한 줄씩
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np

from yolo.config import ROOT

SCENES_DIR = ROOT / "runs" / "roi" / "scenes"
MATCH_MM = 40.0  # CAD 점군 가운데와 키포인트 브라켓 가운데가 이 안이면 같은 브라켓
AGREE_MM = 6.0  # 위치 차이 = x · y 차이 중 큰 값 (부록 8.6)
AGREE_DEG = 7.0
SAME_LAYOUT_MM = 10.0


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _sym_diff(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """대칭을 감안한 실제 차이 (위치 x · y 중 큰 값 mm, 회전 각 °). 이름 (대칭 중 어느 쪽) 이 달라도 같은 자세면 0 에 가깝다."""
    from yolo.train.kpt_label import symmetries

    best = None
    for S in symmetries():
        Ta = np.asarray(T_a, float) @ S
        c = (np.trace(Ta[:3, :3].T @ np.asarray(T_b, float)[:3, :3]) - 1.0) / 2.0
        ang = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
        pos = float(np.max(np.abs(Ta[:2, 3] - np.asarray(T_b, float)[:2, 3])))
        if best is None or ang < best[1]:
            best = (pos, ang)
    return best


def _scene_inputs(folder: Path):
    from vision.depth_cloud import auto_plane_height_mm, fit_plane_ransac, plane_noise_mm, points_from_mask

    bgr = cv2.imread(str(folder / "rgb.png"))
    depth = np.load(folder / "depth_mm.npy")
    cam = json.loads((folder / "camera.json").read_text(encoding="utf-8"))
    K, T_bc = np.asarray(cam["K"], float), np.asarray(cam["T_base_cam"], float)
    table, _ = points_from_mask(depth, K, np.ones(depth.shape[:2], bool), stride=5)
    plane, _ = fit_plane_ransac(table)
    plane_mm = auto_plane_height_mm(plane_noise_mm(table, plane))
    return bgr, depth, K, T_bc, plane, plane_mm


def evaluate(folder: Path, yolo, kpt_weights: Path, cad_xyz: np.ndarray, verbose: bool) -> dict:
    import yolo.pose.roi_cloud as RC
    from yolo.config import DETECT_CONF
    from yolo.pose.instances import collect_instances, depth_blob_boxes, missed_blob_boxes, select_topmost
    from yolo.pose.pose_methods import KptWorker, judge_all, kpt_camera, refine_pose
    from yolo.pose.register import apply_T, register_pose

    bgr, depth, K, T_bc, plane, plane_mm = _scene_inputs(folder)
    RC._CANON_T_BC = T_bc  # 대칭 이름을 카메라 창과 같게
    centroid = cad_xyz.mean(axis=0)

    # 기준: 키포인트
    kcam = kpt_camera(K, T_bc)
    worker = evaluate.kpt_worker
    if worker is None:
        worker = evaluate.kpt_worker = KptWorker(kcam, kpt_weights)
    worker.cam = kcam
    refs = []
    kres = worker.estimate(bgr)
    poses = [refine_pose(bgr, r["T"], kcam) for r in kres]
    for r, T_ref, j in zip(kres, poses, judge_all(bgr, poses, kcam) if poses else []):
        xyz, rpy, _ = RC._place_ui_xyzrpy(T_ref)
        refs.append({"center": apply_T(centroid[None], T_ref)[0], "xyz": xyz, "rpy": rpy, "T": T_ref,
                     "white": bool(j.get("white")), "pose": j.get("pose", "?"), "cad": None})

    # CAD: 카메라 창과 같은 입력
    res = yolo.predict(bgr, conf=DETECT_CONF, device="cpu", verbose=False, imgsz=640)[0]
    yolo_boxes = RC._yolo_xyxys(res)
    extra = missed_blob_boxes(depth_blob_boxes(depth, K, plane, plane_mm, T_bc), yolo_boxes)
    inst = collect_instances(depth=depth, bgr=bgr, K=K, result=res, plane=plane, pad=RC._PAD, plane_mm=plane_mm,
                             stride=RC._STRIDE, T_base_cam=T_bc, extra_xyxys=extra)
    top = select_topmost(inst)
    desk = RC._work_desk_plane(plane, T_bc)
    unmatched = []
    for i, s in enumerate(inst):
        if len(s.xyz) < 20:
            continue
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # 정합 안의 안내 줄은 숨긴다
                    p = register_pose(np.asarray(s.xyz, float), T_base_cam=None, voxel_mm=RC._VOXEL_MM, flip=True,
                                  tries=4, camera_origin=T_bc[:3, 3], desk_plane=desk, class_id=s.class_id,
                                  method="rest", fallback=(i == top))
        except Exception as exc:  # noqa: BLE001
            p = {"aligned_ok": False, "source": f"오류 {exc}", "T": None}
        dt = time.perf_counter() - t0
        center = np.median(s.xyz, axis=0)
        cad = {"ok": bool(p.get("aligned_ok")), "source": str(p.get("source")), "sec": dt, "top": i == top,
               "n": len(s.xyz), "h99": float(np.percentile(s.xyz[:, 2], 99)), "yolo": s.class_id is not None}
        if cad["ok"]:
            cad["xyz"], cad["rpy"], _ = RC._place_ui_xyzrpy(p["T"])
            cad["T"] = np.asarray(p["T"], float)
        d = [float(np.linalg.norm(r["center"][:2] - center[:2])) for r in refs]
        j = int(np.argmin(d)) if d else -1
        if j >= 0 and d[j] <= MATCH_MM and refs[j]["cad"] is None:
            refs[j]["cad"] = cad
        else:
            unmatched.append((center, cad))

    rows = []
    for r in refs:
        c = r["cad"]
        row = {"scene": folder.name, "white": r["white"], "pose": r["pose"], "found": c is not None,
               "ok": bool(c and c["ok"]), "top": bool(c and c["top"]), "sec": c["sec"] if c else 0.0,
               "xy": r["xyz"][:2], "agree": None, "dpos": None, "drz": None, "tilt": None, "cad": c}
        if c and c["ok"]:
            # 실제 차이 (대칭 감안) 로 맞음 · 틀림을 가린다. 화면 숫자 (이름) 차이는 따로 적는다.
            row["dpos"], row["drot"] = _sym_diff(c["T"], r["T"])
            row["drz"] = row["drot"]
            row["ui_dpos"] = float(np.max(np.abs(c["xyz"][:2] - r["xyz"][:2])))
            row["ui_drz"] = abs(_wrap(c["rpy"][2] - r["rpy"][2]))
            row["name_differs"] = row["ui_drz"] > row["drot"] + 20.0
            row["agree"] = row["dpos"] <= AGREE_MM and row["drot"] <= AGREE_DEG
        rows.append(row)
        if verbose:
            col = "흰" if r["white"] else "검"
            if c is None:
                state = "CAD 상자 없음"
            elif not c["ok"]:
                state = f"CAD miss ({c['source']}, {c['sec']:.1f}s, n={c['n']} 높이{c['h99']:.0f})"
            else:
                state = (f"CAD ok {c['sec']:.1f}s  차이 {row['dpos']:.1f} mm · {row['drot']:.1f}°"
                         + ("" if row["agree"] else "  ← 안 맞음")
                         + (f"  (화면 숫자는 이름이 달라 Rz {row['ui_drz']:.0f}° 차이)" if row["name_differs"] else ""))
            print(f"  {col} {r['pose']:<4} ({r['xyz'][0]:6.1f},{r['xyz'][1]:6.1f}){' [맨위]' if row['top'] else ''}  {state}")
    for center, c in unmatched:
        if verbose:
            print(f"  ?  키포인트에 없는 CAD 물체 ({center[0]:.0f},{center[1]:.0f}) n={c['n']} {'ok' if c['ok'] else 'miss'}")
    top_xy = next((r["xy"] for r in rows if r["top"]), None)
    return {"rows": rows, "extra": len(unmatched), "extra_ok": sum(c["ok"] for _, c in unmatched), "top_xy": top_xy,
            "layout": [r["xy"] for r in rows]}


evaluate.kpt_worker = None


def _same_layout(a: list, b: list) -> bool:
    if len(a) != len(b) or not a:
        return False
    return all(min(float(np.linalg.norm(p - q)) for q in b) <= SAME_LAYOUT_MM for p in a)


def summarize(results: list[dict]) -> None:
    rows = [r for res in results for r in res["rows"]]
    for white, name in ((True, "흰"), (False, "검")):
        rs = [r for r in rows if r["white"] == white]
        if not rs:
            continue
        ok = [r for r in rs if r["ok"]]
        agree = [r for r in ok if r["agree"]]
        secs = [r["sec"] for r in rs if r["found"]]
        line = (f"{name} {len(rs):3d}개: CAD 상자 {sum(r['found'] for r in rs)} · 성공 {len(ok)} · "
                f"키포인트와 맞음 {len(agree)} (≤{AGREE_MM:g} mm · {AGREE_DEG:g}°)")
        if ok:
            line += (f" · 차이 중앙 {np.median([r['dpos'] for r in ok]):.1f} mm / {np.median([r['drz'] for r in ok]):.1f}°"
                     f" 최대 {max(r['dpos'] for r in ok):.1f} mm / {max(r['drz'] for r in ok):.1f}°")
        if secs:
            line += f" · 시간 중앙 {np.median(secs):.1f}s 최대 {max(secs):.1f}s"
        print(line)
        wrong = [r for r in ok if not r["agree"]]
        for r in wrong:
            print(f"   틀린 성공: {r['scene']} {r['pose']} ({r['xy'][0]:.0f},{r['xy'][1]:.0f}) "
                  f"차이 {r['dpos']:.1f} mm · {r['drz']:.1f}°")
    names = [r for r in rows if r.get("name_differs")]
    if names:
        print(f"같은 자세인데 화면 숫자 이름 (대칭) 이 다른 것: {len(names)}개 — 6차시 비교표에서 Rz 가 크게 달라 보인다")
    print(f"키포인트에 없는 CAD 물체: {sum(r['extra'] for r in results)}개 (그중 성공 {sum(r['extra_ok'] for r in results)})")
    # 같은 배치를 여러 장: 맨 위 고르기 · 성공 여부가 같은가
    groups: list[list[dict]] = []
    for res in results:
        for g in groups:
            if _same_layout(g[0]["layout"], res["layout"]):
                g.append(res)
                break
        else:
            groups.append([res])
    multi = [g for g in groups if len(g) > 1]
    for g in multi:
        tops = [r["top_xy"] for r in g]
        same_top = all(t is not None and tops[0] is not None and np.linalg.norm(t - tops[0]) <= SAME_LAYOUT_MM
                       for t in tops)
        oks = [tuple(sorted((round(float(x[0])), round(float(x[1]))) for x, ok in
                            ((rr["xy"], rr["ok"]) for rr in r["rows"]) if ok)) for r in g]
        print(f"같은 배치 {len(g)}장 ({g[0]['rows'][0]['scene'] if g[0]['rows'] else '?'}…): "
              f"맨 위 {'같음' if same_top else '달라짐'} · CAD 성공 목록 {'같음' if len(set(oks)) == 1 else '달라짐'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="저장한 장면으로 CAD 정합을 키포인트와 비교")
    ap.add_argument("scenes", nargs="*", type=Path, help=f"장면 폴더 (기본: {SCENES_DIR} 전부)")
    ap.add_argument("-v", "--verbose", action="store_true", help="브라켓마다 한 줄")
    ap.add_argument("--kpt-weights", type=Path, default=ROOT / "weights" / "kpt_base.pt")
    args = ap.parse_args()
    folders = args.scenes or sorted(p for p in SCENES_DIR.iterdir() if (p / "rgb.png").is_file())
    folders = [f if f.is_dir() else SCENES_DIR / f for f in folders]

    from ultralytics import YOLO

    from yolo.config import box_best_pt
    from yolo.pose.register import load_cad_xyz

    yolo = YOLO(str(box_best_pt()))
    cad_xyz = load_cad_xyz(voxel_mm=1.0)
    results = []
    try:
        for f in folders:
            if args.verbose:
                print(f"## {f.name}")
            results.append(evaluate(f, yolo, args.kpt_weights, cad_xyz, args.verbose))
    finally:
        if evaluate.kpt_worker is not None:
            evaluate.kpt_worker.close()
    print(f"== 장면 {len(results)}개")
    summarize(results)


if __name__ == "__main__":
    main()
