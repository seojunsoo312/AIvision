"""손눈 사진으로 팔 관절의 0° 오차를 찾아 팔로워 캘리브와 손눈 결과를 함께 고친다.

  python vision/handeye/compute.py --fix-zero

왜 필요한가
  팔 캘리브는 첫 Enter 때의 자세(L자)를 관절 0° 로 쓴다 (motion/hw_controller.py).
  L자를 눈으로 맞추면 몇 도씩 틀어지고, 그러면 팔이 보고하는 손끝 자세가 틀려서
  손눈 결과가 몇 mm · 몇 도 어긋나고 (책상이 로봇 XY 와 기울어 보임) 팔도 그만큼 빗나간다.

어떻게 찾나
  보드는 손끝에 고정돼 있으므로, 맞는 답이면 모든 사진에서 「손끝에서 본 보드」가 같아야 한다.
  관절마다 몇 도를 더하면 사진들이 가장 잘 맞는지를 카메라 위치 · 보드 위치와 함께 푼다.
  S1(베이스)과 S6(손목 돌리기)는 카메라 · 보드 위치와 구분되지 않아 풀지 않는다.

안전장치
  - 사진을 짝수 · 홀수 번째로 나눠 따로 푼 값이 FIT_AGREE_DEG 넘게 다르면 적용하지 않는다.
  - 보정해도 오차가 충분히 줄지 않거나, 값이 MAX_FIX_DEG 보다 크면 적용하지 않는다.
  - 팔로워를 다른 프로그램(펜던트 등)이 쓰고 있으면 멈춘다.
  - 캘리브 파일과 손눈 사진 기록은 고치기 전에 백업한다.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from motion.base_frame import USER_BASE_FRAME, tcp_pose_user_from_urdf
from motion.hw_controller import port_users, shift_follower_zero
from motion.robot_kinematics import DEFAULT_URDF, RobotKinematics

FIT_JOINTS = ("S2", "S3", "S4", "S5")
# 회전 잔차를 mm 와 같은 무게로 보려고 곱하는 길이 (mm/rad).
ROT_WEIGHT_MM = 100.0
# 이보다 작으면 고칠 것이 없다고 본다 (도).
MIN_FIX_DEG = 0.3
# 이보다 크면 캘리브가 크게 잘못된 것이라 자동으로 고치지 않는다 (도).
MAX_FIX_DEG = 15.0
# 짝수 · 홀수 사진으로 따로 푼 값의 차이 한도 (도).
FIT_AGREE_DEG = 1.0
# 보정 뒤 평균 위치 잔차가 보정 전의 이 비율보다 작아야 적용한다.
MIN_IMPROVE = 0.6
# 한 사진의 잔차가 중앙값의 이 배수와 OUTLIER_MM 를 둘 다 넘으면 빼고 다시 푼다.
OUTLIER_X = 3.0
OUTLIER_MM = 3.0
MIN_SAMPLES = 10
# 지금 오차가 이보다 작으면 이미 잘 맞는 것으로 본다 (평균 mm, 평균 도).
GOOD_MM = 1.5
GOOD_DEG = 0.8


def _T(rv, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rv, dtype=float).reshape(3))[0]
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def _rv(T) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(T, dtype=float)[:3, :3])[0].ravel()


class ZeroFit:
    """관절 0° 오차 dq, 카메라 자세 T_base_cam, 손끝→보드 T_tcp_board 를 함께 푼다."""

    def __init__(self, kin: RobotKinematics, joints: list[dict], T_cam_board: list[np.ndarray]) -> None:
        self.kin = kin
        self.joints = [dict(j) for j in joints]
        self.T_cb = [np.asarray(T, dtype=float) for T in T_cam_board]

    def fk(self, joints: dict, dq: np.ndarray) -> np.ndarray:
        j = dict(joints)
        for name, d in zip(FIT_JOINTS, dq):
            j[name] = float(j[name]) + float(d)
        p = tcp_pose_user_from_urdf(self.kin.forward_tcp(self.kin.q_from_deg(j)))
        T = np.eye(4)
        T[:3, :3] = p.rotation
        T[:3, 3] = p.xyz_mm
        return T

    def residual(self, p: np.ndarray, idx, fit_joints: bool) -> np.ndarray:
        dq = p[:4] if fit_joints else np.zeros(4)
        T_bc, T_tb = _T(p[4:7], p[7:10]), _T(p[10:13], p[13:16])
        out = []
        for i in idx:
            A = self.fk(self.joints[i], dq) @ T_tb
            B = T_bc @ self.T_cb[i]
            out.extend(A[:3, 3] - B[:3, 3])
            out.extend(cv2.Rodrigues(B[:3, :3].T @ A[:3, :3])[0].ravel() * ROT_WEIGHT_MM)
        return np.asarray(out)

    def per_sample(self, p, idx, fit_joints: bool) -> tuple[np.ndarray, np.ndarray]:
        r = self.residual(p, idx, fit_joints).reshape(-1, 6)
        return np.linalg.norm(r[:, :3], axis=1), np.degrees(np.linalg.norm(r[:, 3:] / ROT_WEIGHT_MM, axis=1))

    def solve(self, p0: np.ndarray, idx, fit_joints: bool, iters: int = 40) -> np.ndarray:
        """Levenberg-Marquardt (수치 미분)."""
        p = np.asarray(p0, dtype=float).copy()
        active = list(range(16)) if fit_joints else list(range(4, 16))
        steps = np.array([1e-3] * 4 + [1e-5] * 3 + [1e-3] * 3 + [1e-5] * 3 + [1e-3] * 3)
        lam = 1e-2
        r = self.residual(p, idx, fit_joints)
        cost = float(r @ r)
        for _ in range(iters):
            J = np.zeros((len(r), len(active)))
            for c, i in enumerate(active):
                pp = p.copy()
                pp[i] += steps[i]
                J[:, c] = (self.residual(pp, idx, fit_joints) - r) / steps[i]
            A, g = J.T @ J, J.T @ r
            while True:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
                pn = p.copy()
                pn[active] += step
                rn = self.residual(pn, idx, fit_joints)
                cn = float(rn @ rn)
                if cn < cost:
                    done = cost - cn < 1e-9 * max(cost, 1.0)
                    p, r, cost = pn, rn, cn
                    lam = max(lam * 0.3, 1e-9)
                    break
                lam *= 10.0
                if lam > 1e8:
                    return p
            if done:
                break
        return p

    def initial(self, T_base_cam: np.ndarray, T_tcp_board: np.ndarray) -> np.ndarray:
        return np.r_[np.zeros(4), _rv(T_base_cam), T_base_cam[:3, 3], _rv(T_tcp_board), T_tcp_board[:3, 3]]


def estimate(samples, T_base_cam: np.ndarray, T_tcp_board: np.ndarray, kin: RobotKinematics | None = None) -> dict:
    """사진들로 관절 0° 오차를 푼다. 결과와 판단(ok, 이유)을 dict 로 돌려준다."""
    kin = kin or RobotKinematics(DEFAULT_URDF)
    joints = [s.meta.get("joints_deg") for s in samples]
    if any(j is None for j in joints):
        return {"ok": False, "reason": "관절 각도가 없는 예전 사진이 섞여 있습니다. 사진을 다시 찍으세요."}
    fit = ZeroFit(kin, joints, [s.T_cam_board for s in samples])
    idx = list(range(len(samples)))
    if len(idx) < MIN_SAMPLES:
        return {"ok": False, "reason": f"사진 {len(idx)}장. 최소 {MIN_SAMPLES}장이 필요합니다."}

    p0 = fit.solve(fit.initial(T_base_cam, T_tcp_board), idx, fit_joints=False)
    t0, r0 = fit.per_sample(p0, idx, False)
    p1 = fit.solve(p0, idx, fit_joints=True)
    t1, _ = fit.per_sample(p1, idx, True)
    # 보정해도 혼자 크게 어긋나는 사진은 빼고 다시 푼다 (보드가 흔들렸거나 팔이 덜 멈춘 사진).
    limit = max(OUTLIER_X * float(np.median(t1)), OUTLIER_MM)
    keep = [i for i, e in zip(idx, t1) if e <= limit]
    dropped = [samples[i].name for i in idx if i not in keep]
    if dropped and len(keep) >= MIN_SAMPLES:
        p0 = fit.solve(p0, keep, fit_joints=False)
        t0, r0 = fit.per_sample(p0, keep, False)
        p1 = fit.solve(p0, keep, fit_joints=True)
    else:
        keep, dropped = idx, []
    t1, r1 = fit.per_sample(p1, keep, True)

    halves = []
    for part in (keep[0::2], keep[1::2]):
        ph = fit.solve(fit.solve(p0, part, fit_joints=False), part, fit_joints=True)
        halves.append(ph[:4])
    agree = np.abs(halves[0] - halves[1])

    dq = {j: float(v) for j, v in zip(FIT_JOINTS, p1[:4])}
    res = {
        "dq_deg": dq,
        "halves_deg": [{j: float(v) for j, v in zip(FIT_JOINTS, h)} for h in halves],
        "before_mm": (float(t0.mean()), float(t0.max())),
        "before_deg": (float(r0.mean()), float(r0.max())),
        "after_mm": (float(t1.mean()), float(t1.max())),
        "after_deg": (float(r1.mean()), float(r1.max())),
        "n_used": len(keep),
        "dropped": dropped,
        "ok": True,
        "reason": "",
    }
    big = max(abs(v) for v in dq.values())
    if res["before_mm"][0] <= GOOD_MM and res["before_deg"][0] <= GOOD_DEG:
        res.update(ok=False, reason=(
            f"이미 잘 맞습니다 (평균 {res['before_mm'][0]:.2f} mm, {res['before_deg'][0]:.2f}°). 고칠 것이 없습니다."
        ), nothing=True)
    elif big < MIN_FIX_DEG:
        res.update(ok=False, reason=f"관절 0° 오차가 모두 {MIN_FIX_DEG}° 미만입니다. 고칠 것이 없습니다.", nothing=True)
    elif big > MAX_FIX_DEG:
        res.update(ok=False, reason=f"오차가 {big:.1f}° 로 너무 큽니다. 자동으로 고치지 않습니다. 팔 캘리브를 다시 하세요.")
    elif float(agree.max()) > FIT_AGREE_DEG:
        res.update(ok=False, reason=(
            f"사진 절반씩 푼 값이 최대 {agree.max():.1f}° 다릅니다 (한도 {FIT_AGREE_DEG}°). "
            "자세를 더 다양하게 (S1 좌우 · 손목 꺾기 · 높낮이) 20장 이상 찍고 다시 하세요."
        ))
    elif res["after_mm"][0] > MIN_IMPROVE * res["before_mm"][0]:
        res.update(ok=False, reason="보정해도 오차가 충분히 줄지 않습니다. 관절 0° 문제가 아닌 것 같습니다 (보드 · 카메라 고정 확인).")
    return res


def print_estimate(res: dict) -> None:
    if "dq_deg" not in res:
        return
    print("\n== 관절 0° 오차 (손눈 사진으로 추정)")
    print(f"{'관절':4} {'오차':>8} {'절반 A':>8} {'절반 B':>8}")
    for j in FIT_JOINTS:
        a, b = res["halves_deg"][0][j], res["halves_deg"][1][j]
        print(f"{j:4} {res['dq_deg'][j]:+7.2f}° {a:+7.2f}° {b:+7.2f}°")
    print(f"손끝-보드 오차  지금    mm {res['before_mm'][0]:.2f}/{res['before_mm'][1]:.2f}  "
          f"deg {res['before_deg'][0]:.2f}/{res['before_deg'][1]:.2f}  (평균/최대)")
    print(f"               고치면  mm {res['after_mm'][0]:.2f}/{res['after_mm'][1]:.2f}  "
          f"deg {res['after_deg'][0]:.2f}/{res['after_deg'][1]:.2f}")
    print(f"사용 {res['n_used']}장" + (f", 뺀 사진 {res['dropped']}" if res["dropped"] else ""))


def rewrite_samples(samples, applied: dict[str, float], kin: RobotKinematics | None = None) -> Path:
    """사진 기록(json)의 관절 각도에 applied 를 더하고 손끝 자세를 다시 계산한다. 원본 폴더는 백업한다."""
    kin = kin or RobotKinematics(DEFAULT_URDF)
    folder = samples[0].json_path.parent
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = folder.with_name(f"{folder.name}_before_zero_fix_{stamp}")
    n = 1
    while backup.exists():
        n += 1
        backup = folder.with_name(f"{folder.name}_before_zero_fix_{stamp}_{n}")
    shutil.copytree(folder, backup)
    for js in sorted(folder.glob("*.json")):
        if not js.with_suffix(".png").is_file():
            continue
        meta = json.loads(js.read_text(encoding="utf-8"))
        if "joints_deg" not in meta:
            continue
        joints = {k: float(v) for k, v in meta["joints_deg"].items()}
        for j, v in applied.items():
            joints[j] += v
        p = tcp_pose_user_from_urdf(kin.forward_tcp(kin.q_from_deg(joints)))
        T = np.eye(4)
        T[:3, :3] = p.rotation
        T[:3, 3] = p.xyz_mm
        done = dict(meta.get("zero_fix_deg") or {})
        for j, v in applied.items():
            done[j] = float(done.get(j, 0.0)) + float(v)
        meta.update(
            joints_deg=joints,
            T_base_tcp=T.tolist(),
            tcp_xyz_mm=[float(x) for x in p.xyz_mm],
            tcp_rpy_deg=[float(x) for x in p.rpy_deg],
            base_frame=USER_BASE_FRAME,
            zero_fix_deg=done,
        )
        js.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return backup


def run(samples_dir: Path, *, yes: bool = False) -> None:
    """--fix-zero: 추정 → 확인 → 팔로워 캘리브 고치기 → 사진 기록 고치기 → 손눈 다시 계산."""
    from vision.handeye import compute as C

    busy = port_users()
    if busy:
        raise SystemExit("팔로워를 다른 프로그램이 쓰고 있습니다. 펜던트 등을 먼저 닫으세요:\n  " + "\n  ".join(busy))
    samples = C.load_samples(samples_dir)
    if len(samples) < MIN_SAMPLES:
        raise SystemExit(f"쓸 수 있는 사진 {len(samples)}장. 최소 {MIN_SAMPLES}장. 먼저 capture.py 로 찍으세요.")
    T_bc = C.solve_eye_to_hand([s.T_base_tcp for s in samples], [s.T_cam_board for s in samples])
    T_tb = C.mean_rigid([C.invert_T(s.T_base_tcp) @ T_bc @ s.T_cam_board for s in samples])
    print("\n관절 오차를 푸는 중 (1~2분)...")
    kin = RobotKinematics(DEFAULT_URDF)
    res = estimate(samples, T_bc, T_tb, kin)
    print_estimate(res)
    if not res["ok"]:
        print(f"\n적용하지 않음: {res['reason']}")
        return
    if not yes:
        ans = input("\n팔로워 캘리브와 손눈 결과를 이 값으로 고칠까요? 팔은 움직이지 않습니다. (y/N): ")
        if ans.strip().lower() not in ("y", "yes", "ㅛ"):
            print("취소했습니다. 아무것도 바꾸지 않았습니다.")
            return
    dq = {j: v for j, v in res["dq_deg"].items() if abs(v) >= MIN_FIX_DEG / 3}
    applied, calib_backup = shift_follower_zero(dq)
    print("\n팔로워 캘리브 고침: " + ", ".join(f"{j} {v:+.2f}°" for j, v in applied.items()))
    print(f"  원래 파일 백업: {calib_backup}")
    samples_backup = rewrite_samples(samples, applied, kin)
    print(f"손눈 사진 기록 고침. 원본 백업: {samples_backup}")
    print("\n손눈 다시 계산:")
    C.compute(samples_dir)
    print(
        "\n확인할 것\n"
        "  1. 펜던트 Connect → HOME: 반듯한 L자인지\n"
        "  2. 카메라 창(roi_cloud)을 껐다 켜고 r: 「vs 로봇 XY」 가 1° 안쪽인지\n"
        "되돌리기: 위 백업 파일을 follower.json 으로 복사한 뒤 calibrate_arms.sh follower 에서 Enter,\n"
        "          사진 폴더는 백업 폴더로 바꾸고 compute.py 를 다시 실행"
    )
