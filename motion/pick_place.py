"""책상 위 픽앤플레이스 순서. GUI 없음, YOLO 없음.

집기는 grasp TCP RPY 를 쓴다. 내려놓을 때는 TCP Rx ∥ 프로젝트 베이스 +X 가 되게 yaw 를 돌린다.
놓는 Z = 집은 Z.
상태는 idle/running/done/fault 와 단계 번호뿐이다 — 행동 트리(BT)가 아니다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from motion import runlog
from motion.controller import ALREADY_AT_TARGET, Controller
from motion.hw_controller import grip_100_to_user
from motion.robot_kinematics import (
    GRIPPER_JOINT,
    INIT_POSE_JOINTS_DEG,
    rotmat_to_rpy_deg,
    rpy_deg_to_rotmat,
)

PICK_LIFT_MM = 30.0
PLACE_APPROACH_MM = 50.0
# 픽앤플레이스 그리퍼 (0=닫힘, 100=완전 열림). 여닫힘 폭은 여기만 고친다.
GRIPPER_OPEN_100 = 30.0
GRIPPER_CLOSE_100 = 0.0
# 스텝과 스텝 사이 대기 (초). 0 이면 바로 다음.
STEP_PAUSE_S = 1.0
# 물체를 쥔 채 실패하면 그 자리 아래로 이 속도로 천천히 내려 놓는다 (m/s).
SAFE_LOWER_MPS = 0.010
# 내려놓은 뒤 들어 올리는 높이 (mm).
SAFE_LIFT_MM = 30.0
# 이 문구로 시작하는 실패는 내려놓기를 하지 않고 쥔 채 멈춘다 (통신·제어가 불안할 때 더 움직이지 않는다).
# Stop(중단)도 사람이 멈춘 것이므로 그대로 둔다.
_NO_RECOVER = ("통신 오류", "E-stop", "제어 오류", "중단")
# 관절 go-to 가 이미 도착해 있으면 이 문구를 낸다 — 성공으로 본다.
_ALREADY = ALREADY_AT_TARGET
_log = runlog.get("pick_place")


@dataclass(frozen=True)
class PickPlaceStep:
    name: str
    kind: str  # ee | ee_joints | gripper | joints
    xyz_mm: np.ndarray | None = None
    rpy_deg: np.ndarray | None = None
    grip_100: float | None = None
    joints_deg: dict[str, float] | None = None
    # True: 이 단계를 시작하면 "물체 쥠", False: "물체 놓음". None 이면 바꾸지 않는다.
    hold: bool | None = None
    # 손끝 직선 이동 속도 (m/s). None 이면 실행기 기본값.
    lin_mps: float | None = None


@dataclass
class PickPlaceRunner:
    ctrl: Controller
    lin_mps: float | None = None
    rot_deg_s: float | None = None
    _steps: list[PickPlaceStep] = field(default_factory=list)
    _i: int = 0
    _phase: str = "idle"  # idle | running | done | fault
    _interrupt: int = 0
    _dwell_until: float = 0.0
    status: str = ""
    # 쥔 채 실패해서 내려놓는 중이면 그 원인. 아니면 "".
    _recover_cause: str = ""
    # 집은 높이 (mm). 내려놓을 높이로 쓴다.
    _place_z: float | None = None

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def recovering(self) -> bool:
        """쥔 채 실패해서 내려놓는 중이면 True."""
        return bool(self._recover_cause)

    @property
    def step_name(self) -> str:
        if not self._steps or self._i >= len(self._steps):
            return ""
        return self._steps[self._i].name

    def busy(self) -> bool:
        return self._phase == "running"

    def start(self, steps: list[PickPlaceStep]) -> bool:
        if not steps:
            self._phase = "fault"
            self.status = "빈 시퀀스"
            return False
        if self._phase == "running":
            self.status = "이미 실행 중"
            return False
        if self.ctrl.holding():
            # 첫 단계에서 그리퍼를 열기 때문에, 쥔 채로 시작하면 공중에서 물체를 놓친다.
            self._phase = "fault"
            self.status = "물체를 쥐고 있어 시작하지 않습니다 — 먼저 내려놓으세요"
            _log.warning("%s", self.status)
            return False
        self._steps = list(steps)
        self._i = 0
        self._dwell_until = 0.0
        self._phase = "running"
        self._recover_cause = ""
        self._place_z = _grasp_z(self._steps)
        _log.info("시작  %d단계: %s", len(self._steps), " → ".join(_step_text(st) for st in self._steps))
        ok = self._launch()
        self._log_after(0, "idle")  # 첫 단계 (또는 바로 실패) 를 남긴다.
        return ok

    def abort(self, reason: str = "중단") -> None:
        self._phase = "fault"
        self.status = reason
        self.ctrl.stop_all()

    def tick(self) -> str:
        """GUI 루프에서 부른다. idle|running|done|fault 를 돌려준다."""
        i0, phase0 = self._i, self._phase
        phase = self._tick()
        self._log_after(i0, phase0)
        if phase == "fault" and phase0 == "running":
            phase = self._after_fault()
        elif phase == "done" and self._recover_cause:
            # 내려놓기까지 끝났다. 원래 실패이므로 fault 로 끝낸다.
            self._phase = phase = "fault"
            self.status = f"{self._recover_cause} → 물체를 내려놓고 멈춤"
            _log.warning("실패 후 내려놓기 완료: %s", self._recover_cause)
            self._recover_cause = ""
        return phase

    def _after_fault(self) -> str:
        """실패 직후. 물체를 쥐고 있으면 그 자리 아래로 내려놓기를 시작하거나, 쥔 채 멈춘다."""
        if not self.ctrl.holding():
            return self._phase
        cause = self.status
        if self._recover_cause:
            # 내려놓기 도중에 또 실패: 더 움직이지 않고 쥔 채 멈춘다.
            self.status = f"물체를 쥔 채 멈춤 — 손으로 받친 뒤 내려 주세요 ({self._recover_cause} / {cause})"
            _log.warning("%s", self.status)
            self._recover_cause = ""
            return self._phase
        if cause.startswith(_NO_RECOVER) or self._place_z is None:
            self.status = f"물체를 쥔 채 멈춤 — 손으로 받친 뒤 내려 주세요 ({cause})"
            _log.warning("%s", self.status)
            return self._phase
        pose = self.ctrl.snapshot().pose
        xyz = np.asarray(pose.xyz_mm, dtype=float)
        rpy = np.asarray(pose.rpy_deg, dtype=float)
        steps = build_safe_putdown_steps(xyz, rpy, place_z_mm=float(self._place_z))
        _log.warning("물체를 쥔 채 실패 (%s) → 그 자리 아래로 내려놓기", cause)
        self._recover_cause = cause
        self._steps = steps
        self._i = 0
        self._dwell_until = 0.0
        self._phase = "running"
        self._launch()
        self._log_after(0, "idle")
        if self._phase == "fault":
            return self._after_fault()
        return self._phase

    def _log_after(self, i0: int, phase0: str) -> None:
        """단계가 넘어가거나 끝·실패가 되면 한 줄 남긴다."""
        if self._phase == "running" and (self._i != i0 or phase0 != "running"):
            _log.info("단계 %d/%d  %s", self._i + 1, len(self._steps), self.step_name)
        elif self._phase != phase0 and self._phase == "done":
            _log.info("완료")
        elif self._phase != phase0 and self._phase == "fault":
            _log.warning("실패  단계 %d/%d %s: %s", self._i + 1, len(self._steps), self.step_name, self.status)

    def _tick(self) -> str:
        if self._phase != "running":
            return self._phase
        if int(self.ctrl.interrupt_id()) != int(self._interrupt):
            self._phase = "fault"
            self.status = "중단 (Stop/조그)"
            return self._phase
        st = self.ctrl.snapshot()
        fault = str(st.fault or "")
        if fault and fault != _ALREADY:
            self._phase = "fault"
            self.status = fault
            return self._phase
        if self.ctrl.goto_active():
            self._dwell_until = 0.0
            self.status = self.step_name
            return self._phase
        if self._i + 1 >= len(self._steps):
            self._phase = "done"
            self.status = "완료"
            return self._phase
        pause = float(STEP_PAUSE_S)
        now = time.monotonic()
        if pause > 1e-9:
            if self._dwell_until <= 0.0:
                self._dwell_until = now + pause
                self.status = f"{self.step_name} 대기 {pause:.1f}s"
                return self._phase
            if now < self._dwell_until:
                self.status = f"{self.step_name} 대기 {pause:.1f}s"
                return self._phase
        self._dwell_until = 0.0
        self._i += 1
        if not self._launch():
            return self._phase
        return self._phase

    def _launch(self) -> bool:
        step = self._steps[self._i]
        self.status = step.name
        if step.hold is not None:
            self.ctrl.set_holding(step.hold)
        dur = 0.0
        if step.kind == "ee":
            dur = self.ctrl.start_ee_goto(
                np.asarray(step.xyz_mm, dtype=float),
                np.asarray(step.rpy_deg, dtype=float),
                lin_mps=step.lin_mps if step.lin_mps is not None else self.lin_mps,
                rot_deg_s=self.rot_deg_s,
            )
        elif step.kind == "ee_joints":
            dur = self.ctrl.start_ee_goto_joints(
                np.asarray(step.xyz_mm, dtype=float),
                np.asarray(step.rpy_deg, dtype=float),
            )
        elif step.kind == "gripper":
            joints = dict(self.ctrl.snapshot().joints_deg)
            joints[GRIPPER_JOINT] = grip_100_to_user(float(step.grip_100))
            dur = self.ctrl.start_joint_goto(joints)
        elif step.kind == "joints":
            dur = self.ctrl.start_joint_goto(dict(step.joints_deg or INIT_POSE_JOINTS_DEG))
        else:
            self._phase = "fault"
            self.status = f"알 수 없는 단계 {step.kind}"
            return False
        self._interrupt = int(self.ctrl.interrupt_id())
        st = self.ctrl.snapshot()
        fault = str(st.fault or "")
        if self.ctrl.goto_active():
            return True
        if dur <= 0.0 and (not fault or fault == _ALREADY):
            return True
        if fault and fault != _ALREADY:
            self._phase = "fault"
            self.status = fault
            return False
        if dur <= 0.0:
            self._phase = "fault"
            self.status = "go-to 시작 실패"
            return False
        return True


def rpy_tcp_rx_parallel_base_x(rpy_deg: np.ndarray) -> np.ndarray:
    """grasp 의 pitch 는 최대한 유지하고, TCP X 를 프로젝트 베이스 X 와 정확히 나란하게 한다.

    TCP X 의 부호 (와 X 축 180° 회전)는 ``rpy_deg`` 에 더 가까운 쪽을 고른다.
    """
    R0 = rpy_deg_to_rotmat(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2]))
    y0 = np.asarray(R0[:, 1], dtype=float)
    best_R = R0
    best_err = None
    for sx in (1.0, -1.0):
        x = np.array([sx, 0.0, 0.0], dtype=float)
        y = y0 - x * float(np.dot(y0, x))
        yn = float(np.linalg.norm(y))
        if yn < 1e-9:
            y = np.array([0.0, 0.0, 1.0], dtype=float)
            y = y - x * float(np.dot(y, x))
            yn = float(np.linalg.norm(y))
        y = y / yn
        z = np.cross(x, y)
        z = z / float(np.linalg.norm(z))
        y = np.cross(z, x)
        for R in (
            np.column_stack([x, y, z]),
            np.column_stack([x, -y, -z]),
        ):
            err = float(np.linalg.norm(R - R0, ord="fro"))
            if best_err is None or err < best_err:
                best_err = err
                best_R = R
    return rotmat_to_rpy_deg(best_R)


def build_pick_place_steps(
    *,
    p_xyz_mm: np.ndarray,
    p_rpy_deg: np.ndarray,
    g_xyz_mm: np.ndarray,
    g_rpy_deg: np.ndarray,
    drop_xy_mm: np.ndarray,
    gripper_close_100: float = GRIPPER_CLOSE_100,
    gripper_open_100: float = GRIPPER_OPEN_100,
    pick_lift_mm: float = PICK_LIFT_MM,
    place_approach_mm: float = PLACE_APPROACH_MM,
) -> list[PickPlaceStep]:
    """직선 픽앤플레이스. 놓는 Z 는 집기 TCP 에서, 놓는 RPY 는 TCP Rx ∥ 베이스 X."""
    p_xyz = np.asarray(p_xyz_mm, dtype=float).reshape(3)
    g_xyz = np.asarray(g_xyz_mm, dtype=float).reshape(3)
    p_rpy = np.asarray(p_rpy_deg, dtype=float).reshape(3)
    g_rpy = np.asarray(g_rpy_deg, dtype=float).reshape(3)
    drop_rpy = rpy_tcp_rx_parallel_base_x(g_rpy)
    drop_xy = np.asarray(drop_xy_mm, dtype=float).reshape(2)
    z_g = float(g_xyz[2])
    z_lift = z_g + float(pick_lift_mm)
    z_hi = z_g + float(place_approach_mm)
    z_xy = max(z_lift, z_hi)
    close_100 = float(np.clip(gripper_close_100, 0.0, 100.0))
    open_100 = float(np.clip(gripper_open_100, 0.0, 100.0))

    def ee(name: str, xyz: np.ndarray, rpy: np.ndarray, *, joints: bool = False) -> PickPlaceStep:
        return PickPlaceStep(
            name=name,
            kind="ee_joints" if joints else "ee",
            xyz_mm=np.asarray(xyz, dtype=float).reshape(3),
            rpy_deg=np.asarray(rpy, dtype=float).reshape(3),
        )

    def grip(name: str, val: float, hold: bool | None = None) -> PickPlaceStep:
        return PickPlaceStep(name=name, kind="gripper", grip_100=float(val), hold=hold)

    gx, gy = float(g_xyz[0]), float(g_xyz[1])
    dx, dy = float(drop_xy[0]), float(drop_xy[1])
    steps = [
        ee("P", p_xyz, p_rpy),
        grip("그리퍼 열기", open_100),
        ee("G", g_xyz, g_rpy),
        grip("그리퍼 닫기", close_100, hold=True),
        ee("집기 리프트", np.array([gx, gy, z_lift]), g_rpy),
        ee("드롭 XY", np.array([dx, dy, z_xy]), drop_rpy, joints=True),
    ]
    if z_xy > z_hi + 1e-6:
        steps.append(ee("드롭 위", np.array([dx, dy, z_hi]), drop_rpy))
    steps.extend(
        [
            ee("하강", np.array([dx, dy, z_g]), drop_rpy),
            grip("그리퍼 열기", open_100, hold=False),
            ee("놓기 리프트", np.array([dx, dy, z_hi]), drop_rpy),
            PickPlaceStep(
                name="초기자세",
                kind="joints",
                joints_deg={k: float(v) for k, v in INIT_POSE_JOINTS_DEG.items()},
            ),
        ]
    )
    return steps


def _step_text(step: PickPlaceStep) -> str:
    """로그용 단계 한 마디: 이름과 목표 (mm 또는 그리퍼 0–100)."""
    if step.xyz_mm is not None:
        x, y, z = (float(v) for v in step.xyz_mm)
        return f"{step.name}({x:.0f},{y:.0f},{z:.0f})"
    if step.grip_100 is not None:
        return f"{step.name}({step.grip_100:.0f})"
    return step.name


def build_safe_putdown_steps(
    xyz_mm: np.ndarray, rpy_deg: np.ndarray, *, place_z_mm: float
) -> list[PickPlaceStep]:
    """쥔 채 실패했을 때: 지금 x, y 에서 집은 높이까지 천천히 내려가 그리퍼를 열고 조금 든다.

    공중에서 그리퍼를 열지 않는다. 이미 그 높이 이하면 내려가는 단계를 건너뛴다.
    """
    x, y, z = (float(v) for v in np.asarray(xyz_mm, dtype=float).reshape(3))
    rpy = np.asarray(rpy_deg, dtype=float).reshape(3)
    zp = float(place_z_mm)
    steps: list[PickPlaceStep] = []
    if z > zp + 1.0:
        steps.append(
            PickPlaceStep(
                name="안전 하강", kind="ee", xyz_mm=np.array([x, y, zp]), rpy_deg=rpy, lin_mps=SAFE_LOWER_MPS
            )
        )
    steps.append(PickPlaceStep(name="그리퍼 열기 (내려놓기)", kind="gripper", grip_100=GRIPPER_OPEN_100, hold=False))
    steps.append(
        PickPlaceStep(name="안전 리프트", kind="ee", xyz_mm=np.array([x, y, min(z, zp) + SAFE_LIFT_MM]), rpy_deg=rpy)
    )
    return steps


def _grasp_z(steps: list[PickPlaceStep]) -> float | None:
    """물체를 쥐는 단계 바로 앞의 손끝 이동 높이 = 집은 높이 (mm)."""
    last_z = None
    for st in steps:
        if st.hold is True:
            return last_z
        if st.xyz_mm is not None:
            last_z = float(np.asarray(st.xyz_mm, dtype=float)[2])
    return None
