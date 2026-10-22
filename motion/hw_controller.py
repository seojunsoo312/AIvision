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

    def disable_torque(self) -> None:
        self._torque = False
        robot = self._robot
        if robot is None:
            return
        with self._lock:
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
