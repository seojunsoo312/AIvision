"""lerobot-calibrate 와 같지만, 끝없이 도는 관절을 이 7축 팔에 맞게 바꾼다.

lerobot 은 wrist_roll 을 끝없이 도는 관절로 보고 범위를 0~4095 로 고정한다. 이 팔은 반대다.
  - wrist_roll : 턱에 막혀 한 바퀴를 못 돈다 → 다른 관절처럼 턱까지 돌려 MIN/MAX 를 기록한다.
                 기록한 범위는 모터 위치 한계로도 들어가서 턱에 부딪히기 전에 모터가 멈춘다.
  - elbow_roll : 턱 없이 계속 돈다 → 표에 MIN 0 / MAX 4095 로 고정해 보여 주고 그대로 저장한다
                 (Enter 자세 ±180°). 돌리지 않아도 된다.

관절 0° 는 범위와 상관없이 첫 단계에서 Enter 를 누른 자세다 (motion/hw_controller.py
_enter_offsets_deg). 그래서 범위를 기록해도 0° 는 바뀌지 않는다.

리더 그리퍼는 drive_mode=1 로 저장한다 (텔레오퍼레이션에서 팔로워와 같은 방향).

deploy/calibrate_arms.sh 가 부른다. 인자는 lerobot-calibrate 와 같다.
"""

from __future__ import annotations

import logging
import sys

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.scripts.lerobot_calibrate import main
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.utils.utils import enter_pressed, move_cursor_up

logger = logging.getLogger(__name__)

# 리더 그리퍼는 팔로워와 여는 방향이 반대다. drive_mode=1 이면 lerobot 이 0~100 을 뒤집어(100 - 값)
# 읽어서 lerobot-teleoperate 에서 같은 방향으로 따라온다. 소프트웨어 값이라 모터에는 쓰지 않는다.
REVERSED_LEADER_MOTORS = ("gripper",)

FULL_TURN_MOTORS = ("elbow_roll",)  # 턱 없이 계속 도는 관절. 범위를 기록하지 않고 0~4095 로 고정한다.


def _half_turn_homings_checked(self) -> dict:
    """bus.set_half_turn_homings 와 같지만, 쓸 수 없는 값이면 원인을 한국어로 알리고 멈춘다.

    모터는 켜진 뒤 엔코더 경계(0/4095)를 넘으면 -19, 4200 처럼 0~4095 밖의 값을 낸다. 그 자리에서
    Enter 를 누르면 Homing_Offset(±2047)에 담을 수 없어 lerobot 이 'Magnitude … exceeds 2047' 로 죽는다.
    """
    bus = self.bus
    bus.reset_calibration()
    pos = bus.sync_read("Present_Position", normalize=False)
    bad = {m: p for m, p in pos.items() if not 0 <= p <= 4095}
    if bad:
        listed = ", ".join(f"{m}={p}" for m, p in bad.items())
        raise SystemExit(
            f"\n✗ 모터 위치가 0~4095 밖입니다: {listed}\n"
            "  모터가 켜진 뒤 엔코더 경계를 넘어 돌아간 상태입니다. 이 자리에서는 캘리브를 할 수 없습니다.\n"
            f"  → {self.id} 팔을 HOME 자세(L자)로 둔 채 팔 전원 어댑터를 뽑았다 다시 꽂고, 움직이지 말고 다시 실행하세요."
        )
    return bus.set_half_turn_homings()


def _record_ranges(bus) -> tuple[dict, dict]:
    """bus.record_ranges_of_motion 과 같은 표. 끝없이 도는 관절은 표에서도 MIN 0 / MAX 4095 로 고정해
    보여 주고, POS 는 한 바퀴(0~4095) 안의 값으로 보여 준다. 그 관절은 안 돌려도 된다."""
    motors = list(bus.motors)
    start = bus.sync_read("Present_Position", motors, normalize=False)
    mins, maxes = dict(start), dict(start)
    for m in FULL_TURN_MOTORS:
        mins[m], maxes[m] = 0, 4095
    while True:
        pos = bus.sync_read("Present_Position", motors, normalize=False)
        print("\n-------------------------------------------")
        print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
        for m in motors:
            if m in FULL_TURN_MOTORS:
                print(f"{m:<15} | {0:>6} | {pos[m] % 4096:>6} | {4095:>6}   (고정)")
            else:
                mins[m], maxes[m] = min(mins[m], pos[m]), max(maxes[m], pos[m])
                print(f"{m:<15} | {mins[m]:>6} | {pos[m]:>6} | {maxes[m]:>6}")
        if enter_pressed():
            break
        move_cursor_up(len(motors) + 3)
    unmoved = [m for m in motors if m not in FULL_TURN_MOTORS and mins[m] == maxes[m]]
    if unmoved:
        raise SystemExit(f"\n✗ 안 돌린 관절이 있습니다: {', '.join(unmoved)}. 처음부터 다시 하고 그 관절도 돌리세요.")
    return mins, maxes


def _calibrate_all_joints(self) -> None:
    """SOFollower / SOLeader.calibrate 에서 끝없이 도는 관절만 wrist_roll → elbow_roll 로 바꾼 것."""
    if self.calibration:
        user_input = input(
            f"\n저장된 캘리브({self.id})가 있습니다. 그대로 쓰려면 Enter, 새로 하려면 c 입력 후 Enter: "
        )
        if user_input.strip().lower() != "c":
            logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
            self.bus.write_calibration(self.calibration)
            return

    logger.info(f"\nRunning calibration of {self}")
    self.bus.disable_torque()
    for motor in self.bus.motors:
        self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    input(f"\n[1/2] {self.id} 팔을 HOME 자세(L자, 모든 관절 0)로 두고 Enter: ")
    homing_offsets = _half_turn_homings_checked(self)

    print(
        "\n[2/2] 관절 7개를 하나씩 돌리세요. 다 돌렸으면 Enter.\n"
        "      턱이 있는 관절(wrist_roll · 그리퍼 포함)은 양쪽 턱까지 천천히.\n"
        "      elbow_roll 은 턱 없이 도는 관절이라 표에 0~4095 로 고정돼 있습니다. 안 돌려도 됩니다."
    )
    range_mins, range_maxes = _record_ranges(self.bus)

    # 켜진 뒤 엔코더 경계를 넘으면 MIN 이 음수, MAX 가 4095 를 넘는다. 모터 위치 한계(0~4095)에 그대로
    # 쓸 수 없으므로 0~4095 로 자른다. 0° 는 Enter 자세(2047)라 그대로다.
    for motor in self.bus.motors:
        lo, hi = range_mins[motor], range_maxes[motor]
        if lo < 0 or hi > 4095:
            range_mins[motor], range_maxes[motor] = max(lo, 0), min(hi, 4095)
            print(f"   ! {motor}: {lo} ~ {hi} → {range_mins[motor]} ~ {range_maxes[motor]} (0~4095 로 자름)")

    self.calibration = {
        motor: MotorCalibration(
            id=m.id,
            drive_mode=int(isinstance(self, SOLeader) and motor in REVERSED_LEADER_MOTORS),
            homing_offset=homing_offsets[motor],
            range_min=range_mins[motor],
            range_max=range_maxes[motor],
        )
        for motor, m in self.bus.motors.items()
    }
    self.bus.write_calibration(self.calibration)
    self._save_calibration()
    print(f"저장: {self.calibration_fpath}")


SOFollower.calibrate = _calibrate_all_joints
SOLeader.calibrate = _calibrate_all_joints

if __name__ == "__main__":
    sys.exit(main())
