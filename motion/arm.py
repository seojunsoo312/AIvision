"""교육용 팔 API. 내부는 Controller. 단위는 펜던트와 같다 (mm, deg, 베이스 +X 앞)."""

from __future__ import annotations

import time

import numpy as np

from motion.controller import (
    DEFAULT_SPEED_PCT,
    FPS,
    GOTO_MAX_S,
    GOTO_SETTLE_MAX_S,
    JOG_ROT_DEG_S,
    JOG_ROT_MAX_DEG_S,
    JOG_ROT_MIN_DEG_S,
    JOG_VEL_MAX_MPS,
    JOG_VEL_MIN_MPS,
    JOINT_VEL_DEG_S,
    JOINT_VEL_MIN_DEG_S,
    Controller,
    speed_pct_to_joint_vel,
)
from motion.hw_controller import Hardware, grip_100_to_user, grip_user_to_100
from motion.robot_kinematics import (
    ARM_JOINT_NAMES,
    DEFAULT_URDF,
    GRIPPER_JOINT,
    HOME_JOINTS_DEG,
    INIT_POSE_JOINTS_DEG,
    RobotKinematics,
)

# 관절 go-to 가 이미 목표에 있을 때 컨트롤러가 내는 문구. 실패가 아니라 성공으로 본다.
_ALREADY = "already at joint target"
_MODES = ("virtual", "real")
_MOVE_MODES = ("abs", "rel")
# moveL 직선 속도 기본값과 범위 (mm/s). 범위는 컨트롤러 조그 한계와 같다.
_LIN_MM_S = 25.0
_LIN_MIN_MM_S = JOG_VEL_MIN_MPS * 1000.0
_LIN_MAX_MM_S = JOG_VEL_MAX_MPS * 1000.0
# 관절 이동 기본 속도 (deg/s). 펜던트 기본 속도 %를 환산한 값 (약 24).
_JOINT_DEG_S = round(speed_pct_to_joint_vel(DEFAULT_SPEED_PCT))


class ArmError(RuntimeError):
    """학생용 이동·연결 실패."""


class Arm:
    """가상/실기 팔. Meshcat이 같이 뜬다.

    arm = Arm()                       # mode="virtual"
    arm = Arm(mode="real")
    arm.moveL([30, 0, 0], "rel")
    arm.moveJ([0, 10, 0, 0, 0, 0], "rel")
    """

    def __init__(self, mode: str = "virtual", *, open_browser: bool = True) -> None:
        """팔을 시작한다.

        mode: "virtual" (로봇 없이 Meshcat 만) 또는 "real" (실기 시리얼 연결).
        open_browser: False 면 Meshcat 주소만 출력하고 브라우저는 열지 않는다.
        실패하면 ArmError.
        """
        key = str(mode).strip().lower()
        if key not in _MODES:
            raise ArmError("mode는 'virtual' 또는 'real' 이어야 합니다")
        self._mode = key
        self._ctrl: Controller | None = None
        self._viz = None
        # 이 Arm 이 컨트롤러를 직접 만들었으면 True. close() 때 함께 끈다.
        self._owned = True
        # 펜던트에 붙은 Arm 이면 True. 펜던트에서 누른 정지를 감지해 이동을 중단한다.
        self._watch_stop = False
        # 마지막으로 확인한 컨트롤러 중단 번호. 바뀌었으면 누군가 정지를 누른 것이다.
        self._motion_epoch = 0
        kin = RobotKinematics(DEFAULT_URDF)
        hardware = Hardware() if key == "real" else None
        try:
            self._ctrl = Controller(kin, hardware=hardware)
            self._ctrl.start(pose_server=False)
            if key == "real":
                self._ctrl.connect()
            self._viz = self._open_visualizer(kin, open_browser=open_browser)
            self._show()
            url = getattr(self._viz, "url", "") or ""
            print(f"Arm      mode={self._mode}")
            if url:
                print(f"Meshcat  {url}")
        except ArmError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            if key == "real":
                raise ArmError(f"실기 연결 실패 ({exc})") from exc
            raise ArmError(str(exc)) from exc

    @classmethod
    def attach(cls, controller: Controller) -> Arm:
        """펜던트가 이미 돌리는 컨트롤러에 붙인다. Meshcat·시리얼은 새로 열지 않는다."""
        arm = cls.__new__(cls)
        arm._mode = "attached"
        arm._ctrl = controller
        arm._viz = None
        arm._owned = False
        arm._watch_stop = True
        arm._motion_epoch = controller.interrupt_id()
        return arm

    def where(self) -> dict[str, list[float] | float]:
        """지금 상태를 돌려준다.

        반환: {"joints_deg": [j1..j6] (도), "gripper": 0–100 (0=닫힘),
               "xyz_mm": [x, y, z] (베이스 기준 mm), "rpy_deg": [roll, pitch, yaw] (도)}
        값은 소수 둘째 자리까지.
        """
        st = self._snapshot()
        self._show(st)
        joints = st.joints_deg
        return {
            "joints_deg": [_round2(joints[name]) for name in ARM_JOINT_NAMES],
            "gripper": _round2(grip_user_to_100(float(joints[GRIPPER_JOINT]))),
            "xyz_mm": [_round2(v) for v in st.pose.xyz_mm],
            "rpy_deg": [_round2(v) for v in st.pose.rpy_deg],
        }

    def moveL(
        self, xyz, mode: str, rpy=None, speed=None, rpy_speed=None, path: str = "line"
    ) -> dict[str, list[float] | float]:
        """그리퍼 끝(TCP)을 좌표로 움직인다. 도착할 때까지 기다린다.

        xyz: mm 3개. mode="rel" 이면 지금 위치에 더하는 값, "abs" 면 베이스 기준 목표 위치.
        rpy: 목표 자세 [roll, pitch, yaw] (도). 생략하면 지금 자세 유지. "rel" 에서도 변위가 아니라 목표다.
        path: "line" 이면 TCP 가 직선으로 간다 (막히면 위로 돌아감).
              "joint" 면 목표를 관절각으로 바꿔 moveJ 처럼 간다.
        speed: line 은 mm/s (기본 25, 5–45), joint 는 deg/s (기본 24, 8–45).
        rpy_speed: 자세 속도 deg/s (기본 25, 5–45). line 에서만 쓴다.
        반환: 도착 후 where() 와 같은 딕셔너리. 닿지 못하면 ArmError.
        """
        how = str(mode).strip().lower()
        if how not in _MOVE_MODES:
            raise ArmError("moveL의 두 번째 인자는 'abs' 또는 'rel' 이어야 합니다")
        route = str(path).strip().lower()
        if route not in ("line", "joint"):
            raise ArmError("path는 'line' 또는 'joint' 이어야 합니다")
        delta_or_abs = _vec3(xyz, "xyz는 mm 3개입니다")
        ctrl = self._require_ctrl()
        self._reject_external_stop(ctrl)
        st = ctrl.snapshot()
        # 목표 위치와 자세를 베이스 기준 절대값으로 만든다.
        if how == "rel":
            target_xyz = np.asarray(st.pose.xyz_mm, dtype=float) + delta_or_abs
        else:
            target_xyz = delta_or_abs
        if rpy is None:
            target_rpy = np.asarray(st.pose.rpy_deg, dtype=float)
        else:
            target_rpy = _vec3(rpy, "rpy는 deg 3개입니다")
        # path="joint": 목표 TCP 를 관절각으로 풀어서 관절 보간으로 간다.
        if route == "joint":
            if rpy_speed is not None:
                raise ArmError("path가 joint이면 rpy_speed는 쓰지 않습니다")
            joint_deg_s = _bounded(
                speed,
                _JOINT_DEG_S,
                JOINT_VEL_MIN_DEG_S,
                JOINT_VEL_DEG_S,
                "path가 joint이면 speed는 8–45 deg/s 입니다",
            )
            dur = ctrl.start_ee_goto_joints(
                target_xyz, target_rpy, speed_deg_s=joint_deg_s
            )
            self._mark_motion(ctrl)
            self._wait_started(dur, joint=True)
            return self.where()
        # path="line": TCP 를 직선으로 옮긴다.
        lin_mm_s = _bounded(
            speed, _LIN_MM_S, _LIN_MIN_MM_S, _LIN_MAX_MM_S, "speed는 5–45 mm/s 입니다"
        )
        rot_deg_s = _bounded(
            rpy_speed,
            JOG_ROT_DEG_S,
            JOG_ROT_MIN_DEG_S,
            JOG_ROT_MAX_DEG_S,
            "rpy_speed는 5–45 deg/s 입니다",
        )
        dur = ctrl.start_ee_goto(
            target_xyz,
            target_rpy,
            lin_mps=lin_mm_s / 1000.0,
            rot_deg_s=rot_deg_s,
        )
        self._mark_motion(ctrl)
        self._wait_started(dur, joint=False)
        return self.where()

    def moveJ(self, joints, mode: str, gripper=None, speed=None) -> dict[str, list[float] | float]:
        """관절 6개를 각도로 움직인다. TCP 직선 이동이 아니다. 도착할 때까지 기다린다.

        joints: S1–S6 각도 6개 (도). mode="rel" 이면 지금 각도에 더하고, "abs" 면 목표 각도.
        gripper: 생략하면 그대로. 주면 같은 mode 를 따른다 (abs 는 0–100 값, rel 은 더할 값).
        speed: deg/s (기본 24, 8–45). 가장 많이 도는 관절 기준으로 시간을 잡는다.
        반환: 도착 후 where() 와 같은 딕셔너리.
        """
        how = str(mode).strip().lower()
        if how not in _MOVE_MODES:
            raise ArmError("moveJ의 두 번째 인자는 'abs' 또는 'rel' 이어야 합니다")
        delta_or_abs = _vec6(joints, "joints는 deg 6개입니다")
        joint_deg_s = _bounded(
            speed,
            _JOINT_DEG_S,
            JOINT_VEL_MIN_DEG_S,
            JOINT_VEL_DEG_S,
            "speed는 8–45 deg/s 입니다",
        )
        ctrl = self._require_ctrl()
        self._reject_external_stop(ctrl)
        current = ctrl.snapshot().joints_deg
        if how == "rel":
            target = {
                name: float(current[name]) + float(delta_or_abs[i])
                for i, name in enumerate(ARM_JOINT_NAMES)
            }
        else:
            target = {
                name: float(delta_or_abs[i]) for i, name in enumerate(ARM_JOINT_NAMES)
            }
        target[GRIPPER_JOINT] = _gripper_user(current[GRIPPER_JOINT], gripper, how)
        # 가장 많이 도는 관절이 speed 로 돌도록 이동 시간을 정한다. 거의 안 움직이면 기본 시간.
        peak = max(abs(target[name] - float(current[name])) for name in ARM_JOINT_NAMES)
        duration_s = None if peak < 1e-3 else peak / joint_deg_s
        dur = ctrl.start_joint_goto(target, duration_s=duration_s)
        self._mark_motion(ctrl)
        self._wait_started(dur, joint=True)
        return self.where()

    def grip(self, value: float) -> None:
        """그리퍼만 움직인다. 0=닫힘, 100=완전히 열림. 범위 밖 값은 0–100 으로 자른다."""
        ctrl = self._require_ctrl()
        self._reject_external_stop(ctrl)
        joints = dict(ctrl.snapshot().joints_deg)
        joints[GRIPPER_JOINT] = grip_100_to_user(float(value))
        dur = ctrl.start_joint_goto(joints)
        self._mark_motion(ctrl)
        self._wait_started(dur, joint=True)

    def home(self) -> dict[str, list[float] | float]:
        """홈 자세 (관절각 모두 0, 위팔이 서고 아래팔이 앞을 향한 L자) 로 관절 이동."""
        ctrl = self._require_ctrl()
        self._reject_external_stop(ctrl)
        dur = ctrl.start_joint_goto(dict(HOME_JOINTS_DEG))
        self._mark_motion(ctrl)
        self._wait_started(dur, joint=True)
        return self.where()

    def initial(self) -> dict[str, list[float] | float]:
        """펜던트 초기자세로 관절 이동. 작업 시작용이며 그리퍼는 닫힌다."""
        ctrl = self._require_ctrl()
        self._reject_external_stop(ctrl)
        dur = ctrl.start_joint_goto(dict(INIT_POSE_JOINTS_DEG))
        self._mark_motion(ctrl)
        self._wait_started(dur, joint=True)
        return self.where()

    def stop(self) -> None:
        """진행 중인 이동을 멈춘다. 바로 돌아온다."""
        self._require_ctrl().stop_all()
        self._show()

    def close(self) -> None:
        """제어 루프와 Meshcat 을 끄고, 실기면 연결을 끊는다. 이후 다른 메서드는 ArmError."""
        if not self._owned:
            self._ctrl = None
            self._viz = None
            return
        ctrl = self._ctrl
        viz = self._viz
        self._ctrl = None
        self._viz = None
        if viz is not None:
            try:
                viz.close()
            except Exception:
                pass
        if ctrl is not None:
            ctrl.stop()

    def _reject_external_stop(self, ctrl: Controller) -> None:
        """펜던트에 붙은 상태에서 그사이 정지가 눌렸으면 새 이동을 시작하지 않는다."""
        if self._watch_stop and ctrl.interrupt_id() != self._motion_epoch:
            raise ArmError("중단")

    def _mark_motion(self, ctrl: Controller) -> None:
        """방금 시작한 이동의 중단 번호를 기억한다."""
        if self._watch_stop:
            self._motion_epoch = ctrl.interrupt_id()

    def _require_ctrl(self) -> Controller:
        if self._ctrl is None:
            raise ArmError("이미 종료된 Arm 입니다")
        return self._ctrl

    def _snapshot(self):
        return self._require_ctrl().snapshot()

    def _show(self, st=None) -> None:
        """Meshcat 화면을 지금 관절 상태로 다시 그린다."""
        viz = self._viz
        if viz is None:
            return
        if st is None:
            if self._ctrl is None:
                return
            st = self._ctrl.snapshot()
        viz.display(st.q)

    def _wait_started(self, duration_s: float, *, joint: bool) -> None:
        """시작한 이동이 끝날 때까지 기다린다. 실패·중단·시간 초과면 ArmError."""
        ctrl = self._require_ctrl()
        st = ctrl.snapshot()
        fault = str(st.fault or "")
        if joint and fault == _ALREADY:
            self._show(st)
            return
        # 이동이 시작조차 안 됐으면 이유를 찾아 알려 준다.
        if not ctrl.goto_active():
            if self._watch_stop and ctrl.interrupt_id() != self._motion_epoch:
                raise ArmError("중단")
            if fault:
                raise ArmError(fault)
            if duration_s <= 0.0:
                raise ArmError("동작 불가 (연결 중 또는 E-stop)")
            self._show(st)
            return
        # 예상 시간 + 여유를 넘기면 멈추고 시간 초과로 본다.
        deadline = time.monotonic() + GOTO_MAX_S + GOTO_SETTLE_MAX_S + max(float(duration_s), 0.0) + 2.0
        while True:
            st = ctrl.snapshot()
            self._show(st)
            fault = str(st.fault or "")
            if fault and not (joint and fault == _ALREADY):
                raise ArmError(fault)
            if self._watch_stop and ctrl.interrupt_id() != self._motion_epoch:
                raise ArmError("중단")
            if not ctrl.goto_active():
                return
            if time.monotonic() > deadline:
                ctrl.stop_all()
                raise ArmError("이동 시간 초과")
            time.sleep(1.0 / FPS)

    @staticmethod
    def _open_visualizer(kin: RobotKinematics, *, open_browser: bool):
        from motion.visualizer import Visualizer

        return Visualizer(kin, open_browser=open_browser)


def _round2(value) -> float:
    return round(float(value), 2)


def _vecn(value, n: int, err: str) -> np.ndarray:
    """숫자 n 개짜리 배열로 바꾼다. 개수가 틀리거나 숫자가 아니면 err 로 ArmError."""
    try:
        arr = np.asarray(value, dtype=float).reshape(n)
    except (TypeError, ValueError):
        raise ArmError(err) from None
    if not np.all(np.isfinite(arr)):
        raise ArmError(err)
    return arr


def _vec3(value, err: str) -> np.ndarray:
    return _vecn(value, 3, err)


def _vec6(value, err: str) -> np.ndarray:
    return _vecn(value, 6, err)


def _bounded(value, default: float, lo: float, hi: float, err: str) -> float:
    """속도 인자 검사. None 이면 기본값, lo–hi 밖이면 err 로 ArmError."""
    if value is None:
        return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ArmError(err) from None
    if not np.isfinite(number) or number < lo or number > hi:
        raise ArmError(err)
    return number


def _gripper_user(current_user_deg: float, gripper, how: str) -> float:
    """moveJ 의 gripper 인자 (0–100, abs/rel) → 내부 그리퍼 각도. None 이면 지금 값 유지."""
    if gripper is None:
        return float(current_user_deg)
    try:
        asked = float(gripper)
    except (TypeError, ValueError):
        raise ArmError("gripper는 0–100 숫자입니다") from None
    if not np.isfinite(asked):
        raise ArmError("gripper는 0–100 숫자입니다")
    if how == "rel":
        asked = grip_user_to_100(float(current_user_deg)) + asked
    return grip_100_to_user(asked)
