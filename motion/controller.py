"""가상 / 실기 조그와 go-to. 관절 소프트 한계는 clamp_q, 메시 충돌이면 그 스텝을 막는다."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from motion.base_frame import (
    R_URDF_FROM_USER,
    USER_BASE_FRAME,
    ee_target_urdf_from_user,
    tcp_pose_user_from_urdf,
    xyz_user_from_urdf,
)
from motion.hw_controller import Hardware
from motion.pose_server import PoseServer
from motion.robot_kinematics import (
    ARM_JOINT_NAMES,
    DEG2RAD,
    GRIPPER_CLOSED_CAD_DEG,
    GRIPPER_JOINT,
    GRIPPER_OPEN_CAD_DEG,
    HOME_JOINTS_DEG,
    INIT_POSE_JOINTS_DEG,
    JOINT_SIGN,
    TCP_FRAME,
    URDF_JOINT_NAMES,
    RobotKinematics,
    TcpPose,
)

FPS = 30
JOINT_VEL_DEG_S = 45.0  # 화면 표시 100%
JOINT_VEL_MIN_DEG_S = 8.0  # 화면 표시 10% (예전 20°/s 의 40%)
GRIPPER_VEL_UNIT_S = 40.0
JOG_VEL_MPS = 0.025  # XYZ 조그 기본 속도. GUI 슬라이더는 5–45 mm/s
JOG_VEL_MIN_MPS = 0.005
JOG_VEL_MAX_MPS = 0.045
JOG_ROT_MIN_DEG_S = 5.0
JOG_ROT_MAX_DEG_S = 45.0
JOG_ROT_DEG_S = (JOG_ROT_MIN_DEG_S + JOG_ROT_MAX_DEG_S) / 2.0
JOG_ROT_RAD_S = JOG_ROT_DEG_S * DEG2RAD
DEFAULT_ARM_TORQUE_PCT = 100.0
DEFAULT_GRIPPER_TORQUE_PCT = 50.0
SPEED_PCT_MIN = 10.0
SPEED_PCT_MAX = 100.0
DEFAULT_SPEED_PCT = 50.0
# 화면 10% == 예전 40% 오버라이드, 화면 100% == 예전 150%.
SPEED_SCALE_AT_MIN = 0.40
SPEED_SCALE_AT_MAX = 1.50
SPEED_PRESETS = (("느림", 10.0), ("보통", 50.0), ("빠름", 100.0))
# 관절 go-to 목표에 이미 있을 때의 fault 문구. 오류가 아니다 (러너·Arm 은 성공으로 본다).
ALREADY_AT_TARGET = "이미 목표 관절에 있음"
MAX_TARGET_LEAD_M = 0.012  # 명령 TCP 와 실측 TCP 차이 12 mm 까지 (실기). 1틱보다 작은 목표는 STS3215 가 움직이지 않는다.
# q_send = (1-β) q_prev + β q_ik. β 가 작을수록 덜 떨리지만 더 늦게 따라간다.
CART_IK_BLEND = 0.28
GOTO_DURATION_S = 2.5
GOTO_MIN_S = 0.2
GOTO_MAX_S = 30.0
GOTO_SETTLE_POS_M = 0.0008  # 0.8 mm — EE go-to 는 TCP 가 목표에 닿을 때까지 IK 를 계속한다
GOTO_SETTLE_MAX_S = 2.5
GOTO_SETTLE_IK_LOOPS = 4
# EE go-to: TCP 를 직선으로 옮기면 기둥(L1 = J1–J2)에 부딪힐 때 먼저 위로 올린다.
# 높이는 길이 열리는 가장 낮은 경유 높이. 막히면 MAX 쪽으로 한 단계씩 올린다.
GOTO_LIFT_CLEAR_M = 0.025
GOTO_LIFT_STEP_M = 0.010
GOTO_LIFT_MAX_Z_M = 0.16
GOTO_LIFT_SKIP_M = 0.005
GOTO_PATH_CHECK_N = 10
# 관절 공간 EE go-to: servo_toward 한 번에 최대 5°/프레임.
# 먼 집기 → 놓기 (S1 ~120° 이상)는 고정 24 프레임으로 모자라다.
GOTO_JOINTS_IK_FRAMES = 200
GOTO_JOINTS_IK_POS_M = 0.015

CART_AXES = ("x", "y", "z", "wx", "wy", "wz")
ROT_FRAME_BASE = "base"
ROT_FRAME_TCP = "tcp"


def _speed_pct_t(pct: float) -> float:
    span = SPEED_PCT_MAX - SPEED_PCT_MIN
    return (float(np.clip(pct, SPEED_PCT_MIN, SPEED_PCT_MAX)) - SPEED_PCT_MIN) / span


def speed_pct_to_scale(pct: float) -> float:
    t = _speed_pct_t(pct)
    return SPEED_SCALE_AT_MIN + t * (SPEED_SCALE_AT_MAX - SPEED_SCALE_AT_MIN)


def speed_pct_to_joint_vel(pct: float) -> float:
    t = _speed_pct_t(pct)
    return JOINT_VEL_MIN_DEG_S + t * (JOINT_VEL_DEG_S - JOINT_VEL_MIN_DEG_S)


@dataclass(frozen=True)
class PendantState:
    q: np.ndarray
    joints_deg: dict[str, float]
    pose: TcpPose
    fault: str
    mode: str
    rot_frame: str
    xyz_frame: str
    connected: bool
    torque: bool
    ee_err_mm: float | None
    err_xyz_mm: np.ndarray | None


class Controller:
    def __init__(self, kinematics: RobotKinematics, hardware: Hardware | None = None) -> None:
        self._kin = kinematics
        self._hw = hardware
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._joint_jog = dict.fromkeys(URDF_JOINT_NAMES, 0)
        self._cart_jog = dict.fromkeys(CART_AXES, 0)
        self._q = kinematics.q_home()
        self._pose = kinematics.forward_tcp(self._q)
        self._target_xyz: np.ndarray | None = None
        self._target_rot: np.ndarray | None = None
        self._fault = ""
        self._goto: dict | None = None
        self._interrupt_id = 0
        self._mode = "virtual"
        self._rot_frame = ROT_FRAME_BASE
        self._xyz_frame = ROT_FRAME_BASE
        self._q_meas: np.ndarray | None = None
        self._pose_meas: TcpPose | None = None
        self._ee_err_mm: float | None = None
        self._err_xyz_mm: np.ndarray | None = None
        self._hw_hold = False
        self._pose_server: PoseServer | None = None
        self._jog_vel_mps = float(JOG_VEL_MPS)
        self._jog_rot_rad_s = float(JOG_ROT_RAD_S)
        self._speed_pct = float(DEFAULT_SPEED_PCT)
        self._torque_pct = {
            name: (
                DEFAULT_GRIPPER_TORQUE_PCT
                if name == GRIPPER_JOINT
                else DEFAULT_ARM_TORQUE_PCT
            )
            for name in URDF_JOINT_NAMES
        }

    def start(self, *, pose_server: bool = True) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if pose_server and self._pose_server is None:
            self._pose_server = PoseServer(self.handeye_payload)
            try:
                self._pose_server.start()
            except OSError as exc:
                self._pose_server = None
                print(f"hand-eye pose server failed ({exc})")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._pose_server is not None:
            self._pose_server.stop()
            self._pose_server = None
        self.disconnect()

    def set_joint_jog(self, joint: str, sign: int) -> None:
        if joint not in self._joint_jog:
            raise KeyError(joint)
        with self._lock:
            if sign != 0 and self._frozen_unlocked():
                return
            self._goto = None
            if sign != 0:
                self._interrupt_id += 1
            self._joint_jog[joint] = int(np.sign(sign))
            if sign != 0:
                self._reset_cart_targets()

    def set_tcp_jog_mm_s(self, mm_s: float) -> None:
        lo = JOG_VEL_MIN_MPS * 1000.0
        hi = JOG_VEL_MAX_MPS * 1000.0
        vel = float(np.clip(mm_s, lo, hi)) / 1000.0
        with self._lock:
            self._jog_vel_mps = vel

    def set_rpy_jog_deg_s(self, deg_s: float) -> None:
        vel = float(np.clip(deg_s, JOG_ROT_MIN_DEG_S, JOG_ROT_MAX_DEG_S))
        with self._lock:
            self._jog_rot_rad_s = vel * DEG2RAD

    def set_speed_pct(self, pct: float) -> None:
        value = float(np.clip(pct, SPEED_PCT_MIN, SPEED_PCT_MAX))
        with self._lock:
            self._speed_pct = value

    def _speed_scale_unlocked(self) -> float:
        return speed_pct_to_scale(self._speed_pct)

    def _apply_torque_pcts_to_hw(self) -> None:
        hw = self._hw
        if hw is None or not hw.is_connected:
            return
        with self._lock:
            pcts = dict(self._torque_pct)
        try:
            hw.apply_torque_pcts(pcts)
        except Exception as exc:
            raise RuntimeError(f"torque limit write failed ({exc})") from exc

    @staticmethod
    def _normalize_jog_frame(frame: str) -> str:
        key = str(frame).strip().lower()
        if key in {"base", "world"}:
            return ROT_FRAME_BASE
        if key in {"tcp", "tool", "local"}:
            return ROT_FRAME_TCP
        raise ValueError(f"jog frame must be base|tcp, got {frame!r}")

    def set_rot_frame(self, frame: str) -> None:
        """RPY 조그 기준: 'base' (월드 고정) 또는 'tcp' (공구 기준)."""
        key = self._normalize_jog_frame(frame)
        with self._lock:
            self._rot_frame = key
            self._reset_cart_targets()

    def set_xyz_frame(self, frame: str) -> None:
        """XYZ 조그 기준: 'base' (프로젝트 +X 앞) 또는 'tcp' (공구 기준). Go-to 는 항상 베이스."""
        key = self._normalize_jog_frame(frame)
        with self._lock:
            self._xyz_frame = key
            self._reset_cart_targets()

    def xyz_frame(self) -> str:
        with self._lock:
            return self._xyz_frame

    def rot_frame(self) -> str:
        with self._lock:
            return self._rot_frame

    def set_cart_jog(self, axis: str, sign: int) -> None:
        if axis not in self._cart_jog:
            raise KeyError(axis)
        with self._lock:
            if sign != 0 and self._frozen_unlocked():
                return
            self._goto = None
            if sign != 0:
                self._interrupt_id += 1
            self._cart_jog[axis] = int(np.sign(sign))
            if not any(self._cart_jog.values()):
                self._reset_cart_targets()

    def clear_jog(self) -> None:
        """누르고 있는 조그만 멈춘다. 절대 go-to 는 계속 간다."""
        with self._lock:
            for name in self._joint_jog:
                self._joint_jog[name] = 0
            for name in self._cart_jog:
                self._cart_jog[name] = 0
            self._reset_cart_targets()

    def interrupt_id(self) -> int:
        with self._lock:
            return int(self._interrupt_id)

    def stop_all(self) -> None:
        with self._lock:
            self._interrupt_id += 1
            self._clear_motion_unlocked()
            self._fault = ""

    def goto_active(self) -> bool:
        with self._lock:
            return self._goto is not None

    def home(self) -> None:
        self.start_joint_goto(dict(HOME_JOINTS_DEG))

    def init_pose(self) -> None:
        """작업 시작 자세 (펜던트 초기자세 버튼)."""
        self.start_joint_goto(dict(INIT_POSE_JOINTS_DEG))

    def snap_to_joints(self, joints_deg: dict[str, float]) -> bool:
        """관절을 보간 없이 목표로 즉시 설정 (가상 전용; 실기 write 경로도 타지만 호출부에서 막음)."""
        with self._lock:
            if self._frozen_unlocked():
                return False
            self._clear_motion_unlocked()
        q = self._kin.q_from_deg(joints_deg)
        return self._commit(q)

    def snap_init_pose(self) -> str:
        """티칭 '초기자세로 이동': 가상은 즉시 스냅, 실기 연결 시 관절 go-to.

        Returns:
            ``"snap"`` | ``"goto"`` | ``""`` (실패/동결).
        """
        hw = self._hw
        connected = bool(hw is not None and hw.is_connected)
        with self._lock:
            mode = self._mode
            frozen = self._frozen_unlocked()
        if frozen:
            return ""
        if connected and mode == "real":
            self.init_pose()
            return "goto"
        if not self.snap_to_joints(dict(INIT_POSE_JOINTS_DEG)):
            return ""
        return "snap"

    def connect(self) -> None:
        if self._hw is None:
            raise RuntimeError("no hardware configured")
        with self._lock:
            self._clear_motion_unlocked()
            self._hw_hold = True
        try:
            q = self._hw.connect(self._kin)
            pose = self._kin.forward_tcp(q)
            with self._lock:
                self._q = q
                self._pose = pose
                self._q_meas = q.copy()
                self._pose_meas = pose
                self._ee_err_mm = 0.0
                self._err_xyz_mm = np.zeros(3)
                self._mode = "real"
                self._hw_hold = False
                self._fault = ""
            self._apply_torque_pcts_to_hw()
        except Exception:
            with self._lock:
                self._hw_hold = False
                self._mode = "virtual"
            if self._hw is not None and self._hw.is_connected:
                self._hw.disconnect()
            raise

    def disconnect(self) -> None:
        hw = self._hw
        if hw is not None and hw.is_connected:
            hw.disconnect()
        with self._lock:
            self._clear_motion_unlocked()
            if self._q_meas is not None:
                self._q = self._q_meas.copy()
                if self._pose_meas is not None:
                    self._pose = self._pose_meas
            self._mode = "virtual"
            self._q_meas = None
            self._pose_meas = None
            self._ee_err_mm = None
            self._err_xyz_mm = None
            self._fault = ""
            self._hw_hold = False

    def estop(self, reason: str = "E-stop — 토크 꺼짐") -> None:
        with self._lock:
            self._interrupt_id += 1
            self._clear_motion_unlocked()
            self._fault = reason
        if self._hw is not None and self._hw.is_connected:
            try:
                self._hw.disable_torque()
            except Exception as exc:
                with self._lock:
                    self._fault = f"{reason} ({exc})"

    def start_ee_goto(
        self,
        xyz_mm: np.ndarray,
        rpy_deg: np.ndarray,
        *,
        duration_s: float | None = None,
        lin_mps: float | None = None,
        rot_deg_s: float | None = None,
    ) -> float:
        """Go-to TCP. xyz/rpy are project base (+X forward), not raw URDF.

        기본 시간은 XYZ/RPY 조그 속도 × 속도%. duration_s 를 주면 그 시간을 쓴다.
        lin_mps / rot_deg_s 를 주면 조그 속도 대신 그 값으로 시간을 잡는다.
        """
        xyz1, R1 = ee_target_urdf_from_user(xyz_mm, rpy_deg)
        with self._lock:
            if self._frozen_unlocked():
                return 0.0
            self._interrupt_id += 1
            self._clear_motion_unlocked()
            xyz0 = self._pose.xyz_m.copy()
            R0 = self._pose.rotation.copy()
            q = self._q.copy()
            lift = self._ee_straight_hits_unlocked(q, xyz0, R0, xyz1, R1)
            legs = self._ee_legs_unlocked(
                xyz0,
                R0,
                xyz1,
                R1,
                q=q,
                lift=lift,
                lin_mps=lin_mps,
                rot_deg_s=rot_deg_s,
            )
            first, rest = legs[0], legs[1:]
            duration_s_sum = float(sum(leg["duration_s"] for leg in legs))
            if duration_s is None:
                duration_s = duration_s_sum
            duration_s = float(np.clip(duration_s, GOTO_MIN_S, GOTO_MAX_S))
            if duration_s_sum > 1e-9 and abs(duration_s - duration_s_sum) > 1e-9:
                scale = duration_s / duration_s_sum
                for leg in legs:
                    leg["duration_s"] = float(leg["duration_s"]) * scale
                    leg["n"] = max(int(leg["duration_s"] * FPS), 1)
                first, rest = legs[0], legs[1:]
            self._goto = self._ee_goto_from_leg(
                first,
                queue=rest,
                final_xyz=xyz1,
                final_R=R1,
                lifted=lift,
            )
            self._fault = ""
        return duration_s

    def start_joint_goto(
        self,
        joints_deg: dict[str, float],
        *,
        duration_s: float | None = None,
    ) -> float:
        """관절 go-to. 기본 시간은 조그 각속도 × 속도%."""
        with self._lock:
            if self._frozen_unlocked():
                return 0.0
            self._interrupt_id += 1
            self._clear_motion_unlocked()
            q1 = self._kin.q_from_deg(joints_deg)
            q0 = self._q.copy()
            if float(np.max(np.abs(q1 - q0))) < 1e-6:
                self._goto = None
                self._fault = ALREADY_AT_TARGET
                return 0.0
            if duration_s is None:
                duration_s = self._joint_goto_duration_unlocked(q0, q1)
            duration_s = float(np.clip(duration_s, GOTO_MIN_S, GOTO_MAX_S))
            self._goto = {
                "kind": "joint",
                "i": 0,
                "n": max(int(duration_s * FPS), 1),
                "q0": q0,
                "q1": q1,
            }
            self._fault = ""
        return duration_s

    def start_ee_goto_joints(
        self,
        xyz_mm: np.ndarray,
        rpy_deg: np.ndarray,
        *,
        duration_s: float | None = None,
        speed_deg_s: float | None = None,
    ) -> float:
        """TCP 목표까지 IK 한 뒤 관절 보간. TCP 직선은 쓰지 않는다.

        드롭 XY처럼 같은 자세로 가로지를 때 그리퍼가 팔 링크와 겹치는 경로를 피한다.
        """
        xyz1, R1 = ee_target_urdf_from_user(xyz_mm, rpy_deg)
        with self._lock:
            if self._frozen_unlocked():
                return 0.0
            q = self._q.copy()
        q0 = q.copy()
        err = float("inf")
        for _ in range(GOTO_JOINTS_IK_FRAMES):
            q = self._kin.clamp_q(
                self._kin.servo_toward(q, xyz1, R_ref=R1, ori_weight=1.0)
            )
            pose = self._kin.forward_tcp(q)
            err = float(np.linalg.norm(pose.xyz_m - xyz1))
            if err <= GOTO_JOINTS_IK_POS_M:
                break
        if err > GOTO_JOINTS_IK_POS_M:
            with self._lock:
                self._interrupt_id += 1
                self._clear_motion_unlocked()
                self._fault = f"IK 실패 ({err * 1000.0:.0f} mm)"
            return 0.0
        joints = self._kin.joints_deg(q)
        if duration_s is None and speed_deg_s is not None:
            j0 = self._kin.joints_deg(q0)
            peak = max(abs(float(joints[name]) - float(j0[name])) for name in ARM_JOINT_NAMES)
            duration_s = None if peak < 1e-3 else peak / float(speed_deg_s)
        return self.start_joint_goto(joints, duration_s=duration_s)

    def _ee_goto_duration_unlocked(
        self,
        xyz0: np.ndarray,
        R0: np.ndarray,
        xyz1: np.ndarray,
        R1: np.ndarray,
        *,
        lin_mps: float | None = None,
        rot_deg_s: float | None = None,
    ) -> float:
        lin = float(self._jog_vel_mps if lin_mps is None else lin_mps)
        rot = float(self._jog_rot_rad_s if rot_deg_s is None else (rot_deg_s * DEG2RAD))
        dist = float(np.linalg.norm(np.asarray(xyz1, dtype=float) - np.asarray(xyz0, dtype=float)))
        t_lin = dist / max(lin, 1e-6)
        ang = float(np.linalg.norm(pin.log3(np.asarray(R0, dtype=float).T @ np.asarray(R1, dtype=float))))
        t_rot = ang / max(rot, 1e-9)
        return max(t_lin, t_rot, GOTO_MIN_S)

    def _ee_straight_hits_unlocked(
        self,
        q: np.ndarray,
        xyz0: np.ndarray,
        R0: np.ndarray,
        xyz1: np.ndarray,
        R1: np.ndarray,
    ) -> bool:
        hit, _ = self._ee_follow_unlocked(q, xyz0, R0, xyz1, R1)
        return hit

    def _ee_follow_unlocked(
        self,
        q: np.ndarray,
        xyz0: np.ndarray,
        R0: np.ndarray,
        xyz1: np.ndarray,
        R1: np.ndarray,
    ) -> tuple[bool, np.ndarray]:
        qk = np.asarray(q, dtype=float).copy()
        n = GOTO_PATH_CHECK_N
        for i in range(1, n + 1):
            a = i / n
            xyz = (1.0 - a) * xyz0 + a * xyz1
            R = R0 @ pin.exp3(a * pin.log3(R0.T @ R1))
            qk = self._kin.clamp_q(
                self._kin.servo_toward(qk, xyz, R_ref=R, ori_weight=1.0)
            )
            if self._kin.in_collision(qk):
                return True, qk
        return False, qk

    def _ee_via_z_unlocked(
        self,
        q: np.ndarray,
        xyz0: np.ndarray,
        R0: np.ndarray,
        xyz1: np.ndarray,
        R1: np.ndarray,
    ) -> float:
        z0 = float(xyz0[2])
        z1 = float(xyz1[2])
        z = max(z0, z1) + GOTO_LIFT_CLEAR_M
        z_cap = max(z, GOTO_LIFT_MAX_Z_M)
        while z <= z_cap + 1e-9:
            via0 = np.array([float(xyz0[0]), float(xyz0[1]), z], dtype=float)
            via1 = np.array([float(xyz1[0]), float(xyz1[1]), z], dtype=float)
            qk = np.asarray(q, dtype=float)
            hit, qk = self._ee_follow_unlocked(qk, xyz0, R0, via0, R0)
            if not hit:
                hit, qk = self._ee_follow_unlocked(qk, via0, R0, via1, R1)
            if not hit:
                hit, qk = self._ee_follow_unlocked(qk, via1, R1, xyz1, R1)
            if not hit:
                return z
            z += GOTO_LIFT_STEP_M
        return z_cap

    def _ee_legs_unlocked(
        self,
        xyz0: np.ndarray,
        R0: np.ndarray,
        xyz1: np.ndarray,
        R1: np.ndarray,
        *,
        q: np.ndarray,
        lift: bool,
        lin_mps: float | None,
        rot_deg_s: float | None,
    ) -> list[dict]:
        if not lift:
            return [
                self._ee_leg_unlocked(
                    xyz0, R0, xyz1, R1, lin_mps=lin_mps, rot_deg_s=rot_deg_s
                )
            ]
        z_via = self._ee_via_z_unlocked(q, xyz0, R0, xyz1, R1)
        via0 = np.array([float(xyz0[0]), float(xyz0[1]), z_via], dtype=float)
        via1 = np.array([float(xyz1[0]), float(xyz1[1]), z_via], dtype=float)
        pts = [
            (xyz0, R0, False),
            (via0, R0, True),
            (via1, R1, False),
            (xyz1, R1, False),
        ]
        legs: list[dict] = []
        for (p0, r0, _s0), (p1, r1, lift_seg) in zip(pts, pts[1:]):
            dp = float(np.linalg.norm(p1 - p0))
            dang = float(np.linalg.norm(pin.log3(r0.T @ r1)))
            if dp < GOTO_LIFT_SKIP_M and dang < 0.02:
                continue
            leg = self._ee_leg_unlocked(
                p0, r0, p1, r1, lin_mps=lin_mps, rot_deg_s=rot_deg_s, lift_seg=lift_seg
            )
            legs.append(leg)
        if not legs:
            return [
                self._ee_leg_unlocked(
                    xyz0, R0, xyz1, R1, lin_mps=lin_mps, rot_deg_s=rot_deg_s
                )
            ]
        return legs

    def _ee_leg_unlocked(
        self,
        xyz0: np.ndarray,
        R0: np.ndarray,
        xyz1: np.ndarray,
        R1: np.ndarray,
        *,
        lin_mps: float | None,
        rot_deg_s: float | None,
        lift_seg: bool = False,
    ) -> dict:
        duration_s = self._ee_goto_duration_unlocked(
            xyz0, R0, xyz1, R1, lin_mps=lin_mps, rot_deg_s=rot_deg_s
        )
        duration_s = float(np.clip(duration_s, GOTO_MIN_S, GOTO_MAX_S))
        return {
            "xyz0": np.asarray(xyz0, dtype=float).copy(),
            "R0": np.asarray(R0, dtype=float).copy(),
            "xyz1": np.asarray(xyz1, dtype=float).copy(),
            "R1": np.asarray(R1, dtype=float).copy(),
            "n": max(int(duration_s * FPS), 1),
            "duration_s": duration_s,
            "lift_seg": bool(lift_seg),
        }

    def _ee_goto_from_leg(
        self,
        leg: dict,
        *,
        queue: list[dict],
        final_xyz: np.ndarray,
        final_R: np.ndarray,
        lifted: bool,
    ) -> dict:
        return {
            "kind": "ee",
            "i": 0,
            "n": int(leg["n"]),
            "xyz0": leg["xyz0"],
            "R0": leg["R0"],
            "xyz1": leg["xyz1"],
            "R1": leg["R1"],
            "queue": list(queue),
            "final_xyz": np.asarray(final_xyz, dtype=float).copy(),
            "final_R": np.asarray(final_R, dtype=float).copy(),
            "lifted": bool(lifted),
            "lift_seg": bool(leg.get("lift_seg")),
        }

    def _joint_goto_duration_unlocked(self, q0: np.ndarray, q1: np.ndarray) -> float:
        t_max = GOTO_MIN_S
        grip_span = abs(GRIPPER_OPEN_CAD_DEG - GRIPPER_CLOSED_CAD_DEG)
        for name in URDF_JOINT_NAMES:
            i = self._kin.q_index(name)
            dq_deg = abs(float(q1[i] - q0[i])) / DEG2RAD
            if dq_deg < 1e-6:
                continue
            if name == GRIPPER_JOINT:
                vel_deg_s = (grip_span / 100.0) * GRIPPER_VEL_UNIT_S
                vel = max(vel_deg_s * self._speed_scale_unlocked(), 1e-3)
            else:
                vel = max(speed_pct_to_joint_vel(self._speed_pct), 1e-3)
            t_max = max(t_max, dq_deg / vel)
        return t_max

    def snapshot(self) -> PendantState:
        with self._lock:
            q = self._q.copy()
            pose = tcp_pose_user_from_urdf(self._pose)
            fault = self._fault
            mode = self._mode
            rot_frame = self._rot_frame
            xyz_frame = self._xyz_frame
            err = self._ee_err_mm
            err_xyz = None if self._err_xyz_mm is None else xyz_user_from_urdf(self._err_xyz_mm)
        hw = self._hw
        connected = bool(hw is not None and hw.is_connected)
        torque = bool(hw is not None and hw.torque_enabled)
        return PendantState(
            q=q,
            joints_deg=self._kin.joints_deg(q),
            pose=pose,
            fault=fault,
            mode=mode,
            rot_frame=rot_frame,
            xyz_frame=xyz_frame,
            connected=connected,
            torque=torque,
            ee_err_mm=err,
            err_xyz_mm=err_xyz,
        )

    def handeye_payload(self) -> dict:
        """프로젝트 베이스(+X 앞)에서 잰 실측 TCP. 없으면 명령 자세를 쓴다."""
        with self._lock:
            q = self._q_meas.copy() if self._q_meas is not None else self._q.copy()
            pose_u = self._pose_meas if self._pose_meas is not None else self._pose
            pose = tcp_pose_user_from_urdf(pose_u)
            mode = self._mode
            fault = self._fault
            # 펜던트 오차 표시와 같은 기준: 명령 − 실측, 프로젝트 베이스 mm.
            ee_err = self._ee_err_mm
            err_xyz = None if self._err_xyz_mm is None else xyz_user_from_urdf(self._err_xyz_mm)
        hw = self._hw
        connected = bool(hw is not None and hw.is_connected)
        torque = bool(hw is not None and hw.torque_enabled)
        T = np.eye(4)
        T[:3, :3] = np.asarray(pose.rotation, dtype=float)
        T[:3, 3] = np.asarray(pose.xyz_mm, dtype=float)
        return {
            "ok": bool(connected and torque and mode == "real"),
            "connected": connected,
            "torque": torque,
            "mode": mode,
            "fault": fault,
            "gripper_frame": TCP_FRAME,
            "base_frame": USER_BASE_FRAME,
            "joints_deg": {k: float(v) for k, v in self._kin.joints_deg(q).items()},
            "tcp_xyz_mm": np.asarray(pose.xyz_mm, dtype=float).tolist(),
            "tcp_rpy_deg": np.asarray(pose.rpy_deg, dtype=float).tolist(),
            "T_base_tcp": T.tolist(),
            "ee_err_mm": None if ee_err is None else float(ee_err),
            "err_xyz_mm": None if err_xyz is None else np.asarray(err_xyz, dtype=float).tolist(),
        }

    def _frozen_unlocked(self) -> bool:
        """연결 중이거나 비상 정지 뒤(버스는 켜짐, 토크는 꺼짐)면 True."""
        if self._hw_hold:
            return True
        hw = self._hw
        return bool(self._mode == "real" and hw is not None and hw.is_connected and not hw.torque_enabled)

    def _clear_motion_unlocked(self) -> None:
        for name in self._joint_jog:
            self._joint_jog[name] = 0
        for name in self._cart_jog:
            self._cart_jog[name] = 0
        self._reset_cart_targets()
        self._goto = None

    def _reset_cart_targets(self) -> None:
        self._target_xyz = None
        self._target_rot = None

    def _commit(self, q: np.ndarray, *, rewind_cart: bool = False, allow_lift_escape: bool = True) -> bool:
        with self._lock:
            frozen = self._frozen_unlocked()
            q_now = self._q.copy()
            z_now = float(self._pose.xyz_m[2])
        if frozen:
            return False
        q = self._kin.clamp_q(q)
        pose = self._kin.forward_tcp(q)
        # 이미 접힌 자세는 L1/L2 와 L4 가 겹친다. 조그는 진짜 +Z 방향이면 새로 충돌해도
        # 빠져나가게 둔다. Go-to 에는 이 예외를 쓰지 않는다 —
        # 쓰면 EE 경로가 그리퍼·손목을 기둥 속으로 통과시킬 수 있다.
        entering = self._kin.in_collision(q) and not self._kin.in_collision(q_now)
        lifting = float(pose.xyz_m[2]) >= z_now + 1e-4
        if entering and not (allow_lift_escape and lifting):
            with self._lock:
                self._fault = "충돌 — 이번 조그는 무시"
                if rewind_cart:
                    self._reset_cart_targets()
            return False
        with self._lock:
            self._q = q
            self._pose = pose
            self._fault = ""
        hw = self._hw
        if hw is not None and hw.torque_enabled:
            try:
                hw.write_q(self._kin, q)
            except Exception as exc:
                self.estop(f"E-stop — send failed ({exc})")
                return False
        return True

    def _refresh_meas(self) -> None:
        hw = self._hw
        with self._lock:
            mode = self._mode
            hold = self._hw_hold
        if hold or mode != "real" or hw is None or not hw.is_connected:
            return
        try:
            q_meas = hw.read_q(self._kin)
        except Exception as exc:
            self.estop(f"E-stop — read failed ({exc})")
            return
        pose_meas = self._kin.forward_tcp(q_meas)
        with self._lock:
            self._q_meas = q_meas
            self._pose_meas = pose_meas
            delta = (self._pose.xyz_m - pose_meas.xyz_m) * 1000.0
            self._err_xyz_mm = delta
            self._ee_err_mm = float(np.linalg.norm(delta))

    def _loop(self) -> None:
        dt = 1.0 / FPS
        while not self._stop.is_set():
            t0 = time.perf_counter()
            self._step(dt)
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    def _step(self, dt: float) -> None:
        with self._lock:
            goto = self._goto
            joint_jog = dict(self._joint_jog)
            cart_jog = dict(self._cart_jog)
            q = self._q
        if goto is not None:
            self._goto_step()
        elif any(joint_jog.values()):
            self._joint_step(dt, q, joint_jog)
        elif any(cart_jog.values()):
            self._cart_step(dt, q, cart_jog)
        self._refresh_meas()

    def _goto_step(self) -> None:
        with self._lock:
            g = self._goto
            q = self._q
            if g is None:
                return
            i = g["i"] + 1
            a = min(i / g["n"], 1.0)
            kind = g["kind"]
            n = g["n"]
            q0 = g.get("q0")
            q1 = g.get("q1")
            xyz0 = g.get("xyz0")
            xyz1 = g.get("xyz1")
            R0 = g.get("R0")
            R1 = g.get("R1")
            lift_seg = bool(g.get("lift_seg"))
            lifted = bool(g.get("lifted"))
            queue = list(g.get("queue") or [])
            final_xyz = g.get("final_xyz", xyz1)
            final_R = g.get("final_R", R1)
        if kind == "joint":
            q_next = (1.0 - a) * q0 + a * q1
        else:
            xyz = (1.0 - a) * xyz0 + a * xyz1
            R = R0 @ pin.exp3(a * pin.log3(R0.T @ R1))
            s6_sat = self._kin.s6_at_limit(q)
            if a >= 1.0:
                ori_w = 0.0 if s6_sat else 0.35
            else:
                ori_w = 0.15 if s6_sat else 1.0
            loops = 1 if a < 1.0 else GOTO_SETTLE_IK_LOOPS
            q_next = q
            for _ in range(loops):
                q_next = self._kin.servo_toward(q_next, xyz, R_ref=R, ori_weight=ori_w)
        ok = self._commit(q_next, allow_lift_escape=lift_seg)
        with self._lock:
            if self._goto is None:
                return
            if not ok:
                if kind == "ee" and not lifted:
                    pose = self._pose
                    legs = self._ee_legs_unlocked(
                        pose.xyz_m,
                        pose.rotation,
                        np.asarray(final_xyz, dtype=float),
                        np.asarray(final_R, dtype=float),
                        q=self._q,
                        lift=True,
                        lin_mps=None,
                        rot_deg_s=None,
                    )
                    self._goto = self._ee_goto_from_leg(
                        legs[0],
                        queue=legs[1:],
                        final_xyz=final_xyz,
                        final_R=final_R,
                        lifted=True,
                    )
                    self._fault = ""
                    return
                self._goto = None
                if not self._fault:
                    self._fault = "이동 막힘 (경로가 충돌함)"
                return
            if kind == "ee":
                is_last = len(queue) == 0
                if not is_last:
                    if i >= n:
                        nxt = queue[0]
                        pose = self._pose
                        nxt = dict(nxt)
                        nxt["xyz0"] = pose.xyz_m.copy()
                        nxt["R0"] = pose.rotation.copy()
                        self._goto = self._ee_goto_from_leg(
                            nxt,
                            queue=queue[1:],
                            final_xyz=final_xyz,
                            final_R=final_R,
                            lifted=lifted,
                        )
                    else:
                        self._goto["i"] = i
                    return
                settled, err_mm = self._ee_at_goal_unlocked(final_xyz, final_R)
                max_i = n + int(GOTO_SETTLE_MAX_S * FPS)
                if settled:
                    self._goto = None
                    self._fault = ""
                elif i >= max_i:
                    self._goto = None
                    self._fault = f"목표까지 {err_mm:.1f} mm 남음 (도달 못함)"
                else:
                    self._goto["i"] = i
                return
            if i >= n:
                self._goto = None
                self._fault = ""
            else:
                self._goto["i"] = i

    def _ee_at_goal_unlocked(
        self, xyz1: np.ndarray, R1: np.ndarray
    ) -> tuple[bool, float]:
        pose = self._kin.forward_tcp(self._q)
        err_p = float(np.linalg.norm(pose.xyz_m - np.asarray(xyz1, dtype=float)))
        return bool(err_p < GOTO_SETTLE_POS_M), err_p * 1000.0

    def _joint_step(self, dt: float, q: np.ndarray, joint_jog: dict[str, int]) -> None:
        with self._lock:
            scale = self._speed_scale_unlocked()
            joint_vel = speed_pct_to_joint_vel(self._speed_pct)
        dq = np.zeros_like(q)
        for name, sign in joint_jog.items():
            if sign == 0:
                continue
            if name == GRIPPER_JOINT:
                span = abs(GRIPPER_OPEN_CAD_DEG - GRIPPER_CLOSED_CAD_DEG)
                vel = (span / 100.0) * GRIPPER_VEL_UNIT_S * scale
            else:
                vel = joint_vel
            dq[self._kin.q_index(name)] = JOINT_SIGN[name] * sign * vel * DEG2RAD * dt
        self._commit(q + dq)

    def _cart_step(self, dt: float, q: np.ndarray, cart_jog: dict[str, int]) -> None:
        pose = self._kin.forward_tcp(q)
        translating = any(cart_jog[k] for k in ("x", "y", "z"))
        rotating = any(cart_jog[k] for k in ("wx", "wy", "wz"))
        with self._lock:
            jog_vel = float(self._jog_vel_mps)
            jog_rot = float(self._jog_rot_rad_s)
            meas_xyz = (
                None
                if self._mode != "real" or self._pose_meas is None
                else self._pose_meas.xyz_m.copy()
            )
            if self._target_xyz is None or self._target_rot is None:
                self._target_xyz = pose.xyz_m.copy()
                self._target_rot = pose.rotation.copy()
            # XYZ 조그: base = 프로젝트 축 (+X 앞). TCP = 그리퍼 축.
            # 누적 목표는 URDF 좌표로 둔다. Go-to 숫자는 베이스 좌표 그대로.
            v_user = np.array([cart_jog["x"], cart_jog["y"], cart_jog["z"]], dtype=float)
            xyz_frame = self._xyz_frame
            if xyz_frame == ROT_FRAME_TCP:
                delta_urdf = self._target_rot @ v_user
            else:
                delta_urdf = R_URDF_FROM_USER @ v_user
            self._target_xyz = self._target_xyz + jog_vel * dt * delta_urdf
            # 앞서 나가는 한도는 q_cmd 가 아니라 실제 팔 기준. IK 는 q_cmd 에 계속 누적해야 한다.
            # 아니면 프레임마다 관절 목표가 Feetech 1틱보다 작아서 아무것도 안 움직인다.
            ref = meas_xyz if meas_xyz is not None else pose.xyz_m
            lead = self._target_xyz - ref
            cmd_user = np.array(
                [cart_jog["x"] != 0, cart_jog["y"] != 0, cart_jog["z"] != 0],
                dtype=bool,
            )
            if translating and xyz_frame == ROT_FRAME_TCP:
                n = float(np.linalg.norm(delta_urdf))
                if n > 1e-9:
                    axis = delta_urdf / n
                    along = float(np.dot(lead, axis))
                    if along > MAX_TARGET_LEAD_M:
                        self._target_xyz -= axis * (along - MAX_TARGET_LEAD_M)
            else:
                # 지금 움직이는 URDF 축 (앞서 나가기 한도용).
                cmd = np.abs(R_URDF_FROM_USER) @ cmd_user.astype(float) > 0.5
                if translating and np.any(cmd):
                    lead_cmd = np.where(cmd, lead, 0.0)
                    nlead = float(np.linalg.norm(lead_cmd))
                    if nlead > MAX_TARGET_LEAD_M:
                        self._target_xyz[cmd] = ref[cmd] + lead[cmd] * (MAX_TARGET_LEAD_M / nlead)
            if rotating:
                w_user = jog_rot * dt * np.array(
                    [cart_jog["wx"], cart_jog["wy"], cart_jog["wz"]], dtype=float
                )
                if self._rot_frame == ROT_FRAME_TCP:
                    # 몸체 / TCP 기준: Roll→TCP X, Pitch→TCP Y, Yaw→TCP Z.
                    self._target_rot = self._target_rot @ pin.exp3(w_user)
                else:
                    # 프로젝트 베이스 축 (URDF Rz(-90°) 의 +X 앞).
                    w = R_URDF_FROM_USER @ w_user
                    self._target_rot = pin.exp3(w) @ self._target_rot
            xyz = self._target_xyz.copy()
            rot = self._target_rot.copy()
            tcp_xyz = xyz_frame == ROT_FRAME_TCP

        # TCP 조그는 그리퍼 방향으로 움직이므로 위치 오차가 베이스 한 축에만
        # 생기지 않는다. 누적 점을 세 축 모두 따라간다.
        cmd_mask = (
            None
            if (not translating or tcp_xyz)
            else np.array([bool(cart_jog["x"]), bool(cart_jog["y"]), bool(cart_jog["z"])], dtype=bool)
        )
        q_ik = self._kin.servo_toward(
            q,
            xyz,
            R_ref=rot if rotating else None,
            cmd_mask=cmd_mask,
            ori_weight=1.0 if rotating else 0.0,
        )
        # 사양: 명령을 부드럽게. IK 영공간 그대로면 (X 조그 때 S1↔S4) 버스가 틱 단위로 떨린다.
        q_next = (1.0 - CART_IK_BLEND) * q + CART_IK_BLEND * q_ik
        if not self._commit(q_next, rewind_cart=True):
            return
