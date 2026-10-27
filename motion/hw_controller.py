"""LeRobot SOFollower 래퍼. IK·GUI 는 모른다. 버스 0° = 캘리브 때 Enter 누른 자세."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from motion.robot_kinematics import (
    ARM_JOINT_NAMES,
    GRIPPER_JOINT,
    LEROBOT_FROM_URDF,
    URDF_JOINT_NAMES,
    RobotKinematics,
)

STS3215_RESOLUTION = 4096
GRIPPER_MID = 50.0
DEFAULT_PORT = "/dev/so101_follower"
DEFAULT_ROBOT_ID = "follower"
SO_FOLLOWER_NAME = "so_follower"
# Feetech Max_Torque_Limit / Torque_Limit: 0–1000 = 0–100%.
TORQUE_LIMIT_UNITS = 1000
def grip_user_to_100(user_deg: float) -> float:
    """펜던트 S7 사용자 각도 (HOME=0) → LeRobot 그리퍼 0–100 (HOME=50)."""
    return float(np.clip(user_deg + GRIPPER_MID, 0.0, 100.0))


def grip_100_to_user(grip_100: float) -> float:
    """LeRobot 그리퍼 0–100 → 펜던트 S7 사용자 각도."""
    return float(np.clip(grip_100, 0.0, 100.0)) - GRIPPER_MID


def hf_calibration_dir() -> Path:
    """lerobot-calibrate 가 so101_follower 에 쓰는 것과 같은 폴더."""
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

    return HF_LEROBOT_CALIBRATION / ROBOTS / SO_FOLLOWER_NAME


def resolve_calibration_dir(explicit: Path | None, _robot_id: str) -> Path:
    """--calibration-dir. 없으면 이 PC 의 lerobot-calibrate 캐시."""
    if explicit is not None:
        return Path(explicit)
    return hf_calibration_dir()


def _enter_offsets_deg(calibration: dict) -> dict[str, float]:
    """LeRobot DEGREES 는 범위 가운데가 0. 캘리브 Enter (2047) 가 0° 가 되게 옮긴다."""
    half_turn = (STS3215_RESOLUTION - 1) // 2
    max_res = STS3215_RESOLUTION - 1
    offsets: dict[str, float] = {}
    for urdf_name in ARM_JOINT_NAMES:
        motor = LEROBOT_FROM_URDF[urdf_name]
        cal = calibration.get(motor)
        if cal is None:
            offsets[urdf_name] = 0.0
            continue
        mid = (float(cal.range_min) + float(cal.range_max)) / 2.0
        offsets[urdf_name] = (mid - half_turn) * 360.0 / max_res
    return offsets


class Hardware:
    """connect() 는 엔코더를 읽어 q 를 맞춘 뒤 토크를 켠다. 가상 자세를 실기에 보내지 않는다."""

    def __init__(
        self,
        *,
        port: str = DEFAULT_PORT,
        robot_id: str = DEFAULT_ROBOT_ID,
        calibration_dir: Path | None = None,
        dof_mode: int = 7,
    ) -> None:
        self.port = port
        self.robot_id = robot_id
        self.calibration_dir = Path(calibration_dir) if calibration_dir is not None else None
        self.dof_mode = int(dof_mode)
        self._lock = threading.Lock()
        self._robot = None
        self._enter_off: dict[str, float] = {}
        self._torque = False

    @property
    def is_connected(self) -> bool:
        robot = self._robot
        return bool(robot is not None and robot.is_connected)

    @property
    def torque_enabled(self) -> bool:
        return bool(self.is_connected and self._torque)

    def connect(self, kin: RobotKinematics) -> np.ndarray:
        """버스 켜기, 토크 끄기, 엔코더로 q 읽기, 그 값을 목표로 쓰기, 토크 켜기. Pinocchio q 를 돌려준다."""
        from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower

        self.disconnect()
        calib_dir = resolve_calibration_dir(self.calibration_dir, self.robot_id)
        kwargs: dict = {
            "port": self.port,
            "id": self.robot_id,
            "use_degrees": True,
            "dof_mode": self.dof_mode,
            "max_relative_target": None,
            "cameras": {},
            "calibration_dir": calib_dir,
        }
        robot = SOFollower(SO101FollowerConfig(**kwargs))
        print(f"calibration {robot.calibration_fpath}")
        if not robot.calibration:
            raise RuntimeError(
                f"팔 캘리브 파일이 없습니다: {robot.calibration_fpath}\n"
                "먼저 캘리브를 하세요: bash ~/AIvision/deploy/calibrate_arms.sh follower"
            )
        missing = [
            LEROBOT_FROM_URDF[n] for n in URDF_JOINT_NAMES if LEROBOT_FROM_URDF[n] not in robot.calibration
        ]
        if missing:
            raise RuntimeError(f"캘리브 파일에 모터가 빠져 있습니다: {missing}. 캘리브를 다시 하세요.")

        try:
            robot.connect(calibrate=False)
        except Exception:
            try:
                robot.disconnect()
            except Exception:
                pass
            raise

        try:
            # configure() 가 끝날 때 토크를 다시 켠다. 예전 목표가 아니라 지금 잰 자세를 유지한다.
            with self._lock:
                robot.bus.disable_torque()
            self._enter_off = _enter_offsets_deg(robot.calibration)
            self._robot = robot
            self._torque = False
            self.hold_present()
            return self.read_q(kin)
        except Exception:
            self._robot = None
            self._torque = False
            self._enter_off = {}
            try:
                robot.bus.disable_torque()
            except Exception:
                pass
            try:
                robot.disconnect()
            except Exception:
                pass
            raise

    def disconnect(self) -> None:
        robot = self._robot
        self._robot = None
        self._torque = False
        self._enter_off = {}
        if robot is None:
            return
        try:
            robot.bus.disable_torque()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass

    def disable_torque(self, *, keep_gripper: bool = False) -> None:
        """토크 끄기. keep_gripper 면 그리퍼 모터만 힘을 유지한다 (쥔 물체가 빠지지 않게)."""
        self._torque = False
        robot = self._robot
        if robot is None:
            return
        with self._lock:
            if keep_gripper:
                arm_motors = [m for m in robot.bus.motors if m != LEROBOT_FROM_URDF[GRIPPER_JOINT]]
                robot.bus.disable_torque(arm_motors)
            else:
                robot.bus.disable_torque()

    def enable_torque(self) -> None:
        """지금 엔코더 자세를 유지한다. 마지막 Goal_Position 으로 토크를 켜지 않는다."""
        if self._robot is None:
            return
        self.hold_present()

    def hold_present(self) -> None:
        """Present_Position 에서 토크 ON.

        STS3215 는 토크가 꺼져 있으면 Goal_Position 을 무시한다. 그래서 목표를 쓰고 켜면
        마지막으로 받아들인 목표로 튄다 (손목이 내려가 있으면 위로 튀어 오름).
        모터마다 토크를 켜고, 곧바로 그 모터의 현재 틱을 목표로 쓴다.
        """
        robot = self._robot
        if robot is None:
            raise RuntimeError("실기가 연결되어 있지 않습니다")
        with self._lock:
            present = robot.bus.sync_read("Present_Position", normalize=False)
            if not self._torque:
                self._widen_position_limits(robot, present)
            for motor, ticks in present.items():
                robot.bus.enable_torque(motor)
                robot.bus.write("Goal_Position", motor, int(ticks), normalize=False)
        self._torque = True

    @staticmethod
    def _widen_position_limits(robot, present: dict[str, int]) -> None:
        """손으로 움직여 기록된 가동 범위를 넘었으면, 펌웨어가 그 틱을 받아들이게 한계를 넓힌다."""
        for motor, ticks in present.items():
            cal = robot.calibration.get(motor) if robot.calibration else None
            if cal is None:
                continue
            t = int(ticks)
            lo, hi = int(cal.range_min), int(cal.range_max)
            if lo <= t <= hi:
                continue
            new_lo = max(0, min(lo, t))
            new_hi = min(STS3215_RESOLUTION - 1, max(hi, t))
            robot.bus.write("Min_Position_Limit", motor, new_lo, normalize=False)
            robot.bus.write("Max_Position_Limit", motor, new_hi, normalize=False)

    def read_q(self, kin: RobotKinematics) -> np.ndarray:
        robot = self._robot
        if robot is None:
            raise RuntimeError("실기가 연결되어 있지 않습니다")
        with self._lock:
            obs = robot.get_observation()
        user = self._bus_to_user(obs)
        return kin.q_from_deg(user)

    def write_q(self, kin: RobotKinematics, q: np.ndarray) -> None:
        robot = self._robot
        if robot is None:
            raise RuntimeError("실기가 연결되어 있지 않습니다")
        action = self._user_to_bus(kin.joints_deg(q))
        with self._lock:
            robot.send_action(action)

    def set_torque_pct(self, urdf_name: str, pct: float) -> None:
        """실시간 토크 상한. 0–100% → Feetech 0–1000. EEPROM 잠금은 그 모터만 푼다."""
        if urdf_name not in LEROBOT_FROM_URDF:
            raise KeyError(urdf_name)
        robot = self._robot
        if robot is None or not self.is_connected:
            return
        ticks = int(np.clip(round(float(pct) * 10.0), 1, TORQUE_LIMIT_UNITS))
        motor = LEROBOT_FROM_URDF[urdf_name]
        with self._lock:
            robot.bus.write("Lock", motor, 0, normalize=False)
            try:
                robot.bus.write("Max_Torque_Limit", motor, ticks, normalize=False)
                robot.bus.write("Torque_Limit", motor, ticks, normalize=False)
            finally:
                robot.bus.write("Lock", motor, 1, normalize=False)

    def apply_torque_pcts(self, pcts: dict[str, float]) -> None:
        for name, pct in pcts.items():
            self.set_torque_pct(name, pct)

    def _bus_to_user(self, obs: dict) -> dict[str, float]:
        user: dict[str, float] = {}
        for urdf_name in ARM_JOINT_NAMES:
            motor = LEROBOT_FROM_URDF[urdf_name]
            bus = float(obs[f"{motor}.pos"])
            user[urdf_name] = bus + self._enter_off.get(urdf_name, 0.0)
        grip = float(obs.get("gripper.pos", GRIPPER_MID))
        user[GRIPPER_JOINT] = grip_100_to_user(grip)
        return user

    def _user_to_bus(self, user: dict[str, float]) -> dict[str, float]:
        action: dict[str, float] = {}
        for urdf_name in ARM_JOINT_NAMES:
            motor = LEROBOT_FROM_URDF[urdf_name]
            action[f"{motor}.pos"] = float(user[urdf_name]) - self._enter_off.get(urdf_name, 0.0)
        action["gripper.pos"] = grip_user_to_100(float(user[GRIPPER_JOINT]))
        return action


# ---------------------------------------------------------------- 관절 0° 옮기기 (손눈 보정)


def port_users(port: str = DEFAULT_PORT) -> list[str]:
    """이 시리얼 포트를 열고 있는 프로세스 (pid 명령줄). 펜던트가 켜져 있는지 볼 때 쓴다."""
    import os

    try:
        target = os.path.realpath(port)
    except OSError:
        return []
    users = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                if os.path.realpath(os.path.join(fd_dir, fd)) == target:
                    cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
                    users.append(f"{pid} {cmd.strip()}")
                    break
        except OSError:
            continue
    return users


def shift_follower_zero(
    dq_deg: dict[str, float],
    *,
    port: str = DEFAULT_PORT,
    robot_id: str = DEFAULT_ROBOT_ID,
    calibration_dir: Path | None = None,
) -> tuple[dict[str, float], Path]:
    """팔로워 관절의 0° 를 dq 만큼 옮긴다: 지금 보고 각도 + dq 가 새 보고 각도가 된다.

    「L자를 그만큼 더 정확히 맞춰 캘리브했다」와 같다. 토크를 끈 채 모터의 Homing_Offset 과
    위치 한계를 쓰고, 다시 읽어 확인한 뒤 캘리브 파일을 저장한다. 원래 파일은 옆에 백업한다.

    dq_deg: URDF 관절 이름(S1..S6) → 도. 실제로 옮긴 각도(정수 카운트 단위)와 백업 경로를 돌려준다.
    """
    import shutil
    import time

    from lerobot.motors import MotorCalibration
    from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower

    busy = port_users(port)
    if busy:
        raise RuntimeError("팔로워를 다른 프로그램이 쓰고 있습니다. 먼저 닫으세요:\n  " + "\n  ".join(busy))
    robot = SOFollower(
        SO101FollowerConfig(
            port=port, id=robot_id, dof_mode=7, cameras={},
            calibration_dir=resolve_calibration_dir(calibration_dir, robot_id),
        )
    )
    if not robot.calibration:
        raise RuntimeError(f"팔 캘리브 파일이 없습니다: {robot.calibration_fpath}")
    k = 360.0 / (STS3215_RESOLUTION - 1)
    max_res = STS3215_RESOLUTION - 1
    counts = {LEROBOT_FROM_URDF[j]: int(round(float(v) / k)) for j, v in dq_deg.items()}
    counts = {m: d for m, d in counts.items() if d != 0}
    if not counts:
        return {}, Path(robot.calibration_fpath)

    new = dict(robot.calibration)
    for motor, d in counts.items():
        c = robot.calibration[motor]
        if c.range_min == 0 and c.range_max == max_res:
            # 끝없이 도는 관절 (elbow_roll): 범위는 0~4095 그대로, 기준만 옮긴다.
            lo, hi = 0, max_res
        else:
            lo, hi = max(c.range_min + d, 0), min(c.range_max + d, max_res)
        off = c.homing_offset - d
        if abs(off) > 2047:
            raise RuntimeError(f"{motor}: 옮긴 기준값 {off} 이 ±2047 을 넘습니다. 그 팔은 캘리브를 다시 하세요.")
        new[motor] = MotorCalibration(id=c.id, drive_mode=c.drive_mode, homing_offset=off, range_min=lo, range_max=hi)

    path = Path(robot.calibration_fpath)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.before_zero_fix.{stamp}")
    n = 1
    while backup.exists():  # 같은 초에 두 번 고쳐도 먼저 백업을 덮어쓰지 않는다.
        n += 1
        backup = path.with_name(f"{path.name}.before_zero_fix.{stamp}_{n}")
    shutil.copy2(path, backup)
    bus = robot.bus
    bus.connect()
    try:
        bus.disable_torque()
        motors = list(counts)
        before = bus.sync_read("Present_Position", motors, normalize=False)
        bus.write_calibration(new)
        time.sleep(0.1)
        after = bus.sync_read("Present_Position", motors, normalize=False)
        stored = bus.read_calibration()
    finally:
        bus.disconnect(disable_torque=True)
    bad = [
        m for m in motors
        if abs((after[m] - before[m]) - counts[m]) > 3
        or (stored[m].homing_offset, stored[m].range_min, stored[m].range_max)
        != (new[m].homing_offset, new[m].range_min, new[m].range_max)
    ]
    if bad:
        raise RuntimeError(
            f"모터에 쓴 값이 맞지 않습니다: {bad}. 캘리브 파일은 바꾸지 않았습니다.\n"
            "  bash ~/AIvision/deploy/calibrate_arms.sh follower 에서 Enter 만 눌러 파일 값을 다시 쓰세요."
        )
    robot.calibration = new
    robot._save_calibration()
    applied = {j: counts[LEROBOT_FROM_URDF[j]] * k for j in dq_deg if LEROBOT_FROM_URDF[j] in counts}
    return applied, backup
