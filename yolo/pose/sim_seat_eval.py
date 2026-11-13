"""학생 자리 시험용 사진에서 모델끼리 채점한다 (기준 답이 있을 때). 실험용 (sim-pose 브랜치).

기준 답: yolo/datasets/seat/labels_test_ref/ (모델 점에서 출발해 테두리 자동 맞춤 → 사람이 그림으로 확인).
채점:
  - 검출: 기준 박스와 IoU 0.5 이상이면 찾음. 남는 검출은 엉뚱한 검출.
  - 키포인트: 0~7 번 중 기준에서 보이는 점의 픽셀 차 (대칭으로 번호가 바뀐 경우도 같은 답으로 본다).
  - 자세: 모델 키포인트로 책상 붙이기 → 기준 자세와 비교. 위치 = CAD 중심 mm, 회전 = 대칭 고려 °.
    뎁스 없이 책상 높이(손눈)로 깊이를 정하므로 실제 파이프라인(PnP + 뎁스 + 붙이기) 과 비슷한 조건.

    python -m yolo.pose.sim_seat_eval v4n v4n_ft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from yolo.config import ROOT

SEAT = ROOT / "datasets" / "seat"
RUNS = ROOT.parent / "vision" / "sim" / "out" / "pose_runs"
CONF = 0.5
POS_OK_MM, ROT_OK_DEG = 3.0, 5.0


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description="학생 자리 시험용 사진 채점")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from ultralytics import YOLO

    from yolo.pose.sim_eval import pose_error
    from yolo.train import kpt_label as KL

    names = set(json.loads((SEAT / "split.json").read_text(encoding="utf-8"))[args.split])
    images = sorted(p for p in (SEAT / "images").glob("*.jpg") if p.name in names)
    cam = KL.load_camera(SEAT / "images")
    ref_dir = SEAT / "labels_test_ref"
    S = [s for s in KL.symmetries() if not np.allclose(s, np.eye(4))][0]
    perm = KL.symmetry_perm(S)
    center = KL.mesh_mm()[0].mean(axis=0)
    # 기준 답 (자세 · 박스 · 점)
    refs = {}
    for p in images:
        bgr = cv2.imread(str(p))
        H, W = bgr.shape[:2]
        insts = KL.load_labels(ref_dir / f"{p.stem}.txt", W, H)
        for i in insts:
            KL.fit(i, cam, W, H)
        refs[p.name] = (bgr, insts)
    out = {}
    for name in args.models:
        model = YOLO(str(RUNS / name / "weights" / "best.pt"))
        found = missed = false = 0
        kerr, pos, rot, ok, cls_ok, n_pose = [], [], [], 0, 0, 0
        for p in images:
            bgr, ref = refs[p.name]
            H, W = bgr.shape[:2]
            r = model.predict(bgr, conf=CONF, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            kxy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else np.zeros((0, 10, 2))
            used = set()
            for g in ref:
                cand = [(_iou(g.box, b), j) for j, b in enumerate(boxes) if j not in used]
                best = max(cand, default=(0.0, -1))
                if best[0] < 0.5:
                    missed += 1
                    continue
                j = best[1]
                used.add(j)
                found += 1
                # 키포인트 오차: 기준에서 보이는 0~7 번 (대칭 번호도 같은 답)
                vis = (g.vis == KL.VISIBLE) & ~KL.IS_AUTO
                pred = kxy[j]
                e1 = np.linalg.norm(pred[vis] - g.uv[vis], axis=1)
                e2 = np.linalg.norm(pred[perm][vis] - g.uv[vis], axis=1)
                kerr.extend((e1 if e1.mean() <= e2.mean() else e2).tolist())
                # 자세: 모델 점 전부로 책상 붙이기 (모델 점은 사람이 고치지 않은 상태)
                inst = KL.Instance()
                inst.uv = pred.astype(float)
                inst.vis = np.where(KL.IS_AUTO, KL.UNSET, KL.VISIBLE)
                KL.fit(inst, cam, W, H)
                if inst.T_cam_obj is None:
                    continue
                n_pose += 1
                T_est = cam.T_base_cam @ inst.T_cam_obj
                T_ref = cam.T_base_cam @ g.T_cam_obj
                pe, re = pose_error(T_est, T_ref, center)
                pos.append(pe)
                rot.append(re)
                ok += int(pe <= POS_OK_MM and re <= ROT_OK_DEG)
                cls_ok += int(inst.cls == g.cls)
            false += len(boxes) - len(used)
        n_ref = found + missed
        sm = {"brackets": n_ref, "found": found, "missed": missed, "false": false,
              "kpt_err_med_px": round(float(np.median(kerr)), 2) if kerr else None,
              "kpt_err_p90_px": round(float(np.percentile(kerr, 90)), 2) if kerr else None,
              "class_ok": f"{cls_ok}/{n_pose}", "success_3mm_5deg": f"{ok}/{n_ref}",
              "pos_med_mm": round(float(np.median(pos)), 2) if pos else None,
              "rot_med_deg": round(float(np.median(rot)), 2) if rot else None,
              "pos_p90_mm": round(float(np.percentile(pos, 90)), 2) if pos else None,
              "rot_p90_deg": round(float(np.percentile(rot, 90)), 2) if rot else None}
        out[name] = sm
        print(name, json.dumps(sm, ensure_ascii=False))
    dest = ROOT.parent / "vision" / "sim" / "out" / "results" / f"seat_eval_{args.split}_{'_'.join(args.models)}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {dest}")


if __name__ == "__main__":
    main()
