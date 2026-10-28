#!/usr/bin/env python3
"""원격 조종: 리더 팔을 손으로 움직이면 팔로워 팔이 따라 움직인다. 런처가 켜고 끈다.

  python launcher/teleop.py            # 실제 팔 (팔로워 · 리더 자동 연결)
  python launcher/teleop.py --sim      # 가상 팔로 시험 (리더 대신 정해진 움직임)

lerobot-teleoperate 와 다른 점
  - 켤 때 팔로워가 리더 자세까지 천천히 맞춘 뒤 따라간다 (갑자기 튀지 않게).
  - 두 팔 모두 캘리브 때 Enter 누른 L자 자세를 0° 로 맞춘다 (펜던트와 같은 기준).
  - 팔로워가 자기 몸과 부딪히는 자세로는 따라가지 않는다.
  - 끌 때 앉은 자세로 간 뒤 힘을 끈다 (팔이 툭 떨어지지 않게).

런처와 주고받기
  명령 (표준 입력, 한 줄에 하나)
    start  정지중이면 다시 따라가기 (리더 자세로 다시 맞춘 뒤)
    stop   그 자리에 멈춤 (리더를 움직여도 따라가지 않음)
    off    앉은 자세로 간 뒤 끝냄
    estop  그 자리에 멈춤. 그 뒤에는 off 만 받는다.
  상태 (표준 출력, 한 줄에 JSON 하나)
    {"state": ..., "step": 지금 하는 일, "msg": 글}
    state: connecting · aligning · running · stopped · turning_off · off · estop · error
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from common import Warm, close_windows, open_viz_window, prepare, split_geometry  # noqa: E402
from motion import runlog  # noqa: E402
from motion.controller import Controller  # noqa: E402
from motion.hw_controller import _enter_offsets_deg, grip_100_to_user, port_users  # noqa: E402
from motion.robot_kinematics import (  # noqa: E402
    ARM_JOINT_NAMES,
    GRIPPER_JOINT,
    INIT_POSE_JOINTS_DEG,
    LEROBOT_FROM_URDF,
    SHUTDOWN_JOINTS_DEG,
    URDF_JOINT_NAMES,
)

FPS = 30
TICK_S = 1.0 / FPS
LEADER_PORT = "/dev/so101_leader"
LEADER_ID = "leader"
FOLLOWER_PORT = "/dev/so101_follower"
# 켤 때 리더 자세로 맞추는 속도 (도/초) 와 "맞췄다" 로 보는 차이 (도).
ALIGN_DEG_S = 45.0
ALIGNED_DEG = 3.0
# 따라가는 중 한 번에 움직이는 최대 각도 (도). 모터 최고 속도보다 조금 빠르게 둔다.
FOLLOW_MAX_STEP_DEG = 15.0
# 리더 값이 한 번에 이만큼 넘게 바뀌면 잘못 읽은 값으로 보고 버린다 (엔코더 0/4095 경계 등).
LEADER_JUMP_DEG = 90.0
# 리더를 이만큼 연속으로 못 읽으면 멈춘다.
LEADER_FAIL_LIMIT = 5
_log = runlog.get("teleop")
ESTOP_MSG = "E-stop — 팔이 그 자리에 멈췄습니다.\n「끄기」를 누르면 앉은 자세로 간 뒤 끕니다."


class Leader:
    """리더 팔 (lerobot SO101Leader). 값은 펜던트와 같은 관절 각도 (L자 = 0°, 그리퍼 HOME = 0)."""

    def __init__(self, port: str = LEADER_PORT, robot_id: str = LEADER_ID) -> None:
        self.port = port
        self.robot_id = robot_id
        self._dev = None
        self._off: dict[str, float] = {}

    def connect(self) -> None:
        from lerobot.teleoperators.so_leader import SO101LeaderConfig, SOLeader

        dev = SOLeader(SO101LeaderConfig(port=self.port, id=self.robot_id, use_degrees=True, dof_mode=7))
        if not dev.calibration:
            raise RuntimeError(
                f"리더 팔 캘리브 파일이 없습니다: {dev.calibration_fpath}\n"
                "먼저 캘리브를 하세요: bash ~/AIvision/deploy/calibrate_arms.sh leader"
            )
        dev.connect(calibrate=False)  # 리더는 힘을 끈 채로 둔다 (손으로 움직인다)
        self._dev = dev
        self._off = _enter_offsets_deg(dev.calibration)

    def read(self) -> dict[str, float]:
        a = self._dev.get_action()
        user = {n: float(a[f"{LEROBOT_FROM_URDF[n]}.pos"]) + self._off.get(n, 0.0) for n in ARM_JOINT_NAMES}
        user[GRIPPER_JOINT] = grip_100_to_user(float(a["gripper.pos"]))
        return user

    def disconnect(self) -> None:
        dev, self._dev = self._dev, None
        if dev is not None:
            try:
                dev.disconnect()
            except Exception:
                pass


class FakeLeader:
    """--sim: 리더 대신 초기자세 둘레를 천천히 움직인다."""

    def __init__(self) -> None:
        self._t0 = time.monotonic()

    def connect(self) -> None:
        self._t0 = time.monotonic()

    def read(self) -> dict[str, float]:
        t = time.monotonic() - self._t0
        q = dict(INIT_POSE_JOINTS_DEG)
        q["S1"] += 30.0 * math.sin(0.5 * t)
        q["S2"] += 10.0 * math.sin(0.35 * t)
        q["S5"] += 15.0 * math.sin(0.6 * t)
        q[GRIPPER_JOINT] = -20.0 + 15.0 * math.sin(0.8 * t)
        return q

    def disconnect(self) -> None:
        pass


class Teleop:
    def __init__(self, ctrl: Controller, leader, *, viz=None, plot=None) -> None:
        self.ctrl = ctrl
        self.leader = leader
        self.viz = viz
        self.plot = plot  # 관절 그래프 (Rerun)
        self.cmds: queue.Queue[str] = queue.Queue()
        self._last: dict = {}
        self._target: dict[str, float] | None = None  # 마지막으로 받아들인 리더 값
        self._leader_fail = 0

    def emit(self, state: str, msg: str = "", **extra) -> None:
        out = {"state": state, "step": "", "msg": msg, **extra}
        if out != self._last:
            self._last = dict(out)
            print(json.dumps(out, ensure_ascii=False), flush=True)

    def _draw(self) -> None:
        if self.viz is not None:
            try:
                self.viz.display(self.ctrl.snapshot().q)
            except Exception:
                pass
        if self.plot is not None:
            try:
                self.plot.log(self._target, self.ctrl.measured_joints_deg())
            except Exception as exc:
                _log.warning("그래프 그리기 실패, 그래프 끔: %s", exc)
                self.plot = None

    def _next_cmd(self) -> str | None:
        try:
            cmd = self.cmds.get_nowait()
        except queue.Empty:
            return None
        _log.info("명령 %s", cmd)
        return cmd

    # ------------------------------------------------------------ 리더 읽기

    def _read_leader(self) -> dict[str, float] | None:
        """리더 값. 튀는 관절은 지난 값을 쓴다. 못 읽으면 None."""
        try:
            raw = self.leader.read()
        except Exception as exc:
            self._leader_fail += 1
            _log.warning("리더 읽기 실패 %d/%d: %s", self._leader_fail, LEADER_FAIL_LIMIT, exc)
            return None
        self._leader_fail = 0
        prev = self._target
        if prev is not None:
            for n in URDF_JOINT_NAMES:
                if abs(raw[n] - prev[n]) > LEADER_JUMP_DEG:
                    _log.warning("리더 %s 값이 튐 (%.0f → %.0f) — 무시", n, prev[n], raw[n])
                    raw[n] = prev[n]
        self._target = raw
        return raw

    # ------------------------------------------------------------ 따라가기

    def _step_toward(self, target: dict[str, float], max_step: float) -> float:
        """팔로워를 target 쪽으로 max_step 만큼만 옮긴다. 남은 가장 큰 차이 (도) 를 돌려준다."""
        cur = self.ctrl.snapshot().joints_deg
        cmd, left = {}, 0.0
        for n in URDF_JOINT_NAMES:
            d = float(target[n]) - float(cur[n])
            left = max(left, abs(d))
            cmd[n] = float(cur[n]) + max(-max_step, min(max_step, d))
        self.ctrl.snap_to_joints(cmd)
        return left

    def follow(self) -> str:
        """켜기: 리더 자세로 맞춘 뒤 따라간다. stop · off · estop · fail 중 하나로 끝난다."""
        self._target = None
        self._leader_fail = 0
        aligned = False
        while True:
            t0 = time.monotonic()
            cmd = self._next_cmd()
            if cmd in ("stop", "off", "estop"):
                self.ctrl.stop_all()
                return cmd
            target = self._read_leader()
            if target is None:
                if self._leader_fail >= LEADER_FAIL_LIMIT:
                    self.ctrl.stop_all()
                    return "fail"
            else:
                left = self._step_toward(target, FOLLOW_MAX_STEP_DEG if aligned else ALIGN_DEG_S * TICK_S)
                if not aligned and left <= ALIGNED_DEG:
                    aligned = True
                    _log.info("리더 자세에 맞춤 — 따라가기 시작")
                st = self.ctrl.snapshot()
                if st.connected and not st.torque:
                    return "fail"
                fault = str(st.fault or "")
                if not aligned:
                    self.emit("aligning", step=f"남은 차이 {left:.0f}°")
                elif fault.startswith("충돌"):
                    self.emit("running", step="부딪히는 자세 — 이 자세로는 따라가지 않습니다")
                else:
                    self.emit("running", step="리더를 따라가는 중")
            self._draw()
            time.sleep(max(0.0, TICK_S - (time.monotonic() - t0)))

    def wait_command(self, allowed: tuple[str, ...]) -> str:
        """멈춘 채 allowed 중 하나를 기다린다."""
        while True:
            cmd = self._next_cmd()
            if cmd in allowed:
                return cmd
            self._draw()
            time.sleep(0.1)

    def shutdown(self) -> None:
        """앉은 자세로 간 뒤 (연결은 main 이 끊는다)."""
        st = self.ctrl.snapshot()
        if not (st.connected and st.torque):
            return
        self.emit("turning_off", "앉은 자세로 이동")
        self.ctrl.start_joint_goto(SHUTDOWN_JOINTS_DEG)
        end = time.monotonic() + 30.0
        time.sleep(0.1)
        while self.ctrl.goto_active() and time.monotonic() < end:
            self._draw()
            time.sleep(TICK_S)
        time.sleep(0.3)

    # ------------------------------------------------------------ 전체 흐름

    def run(self) -> int:
        how = self.follow()
        while True:
            if how == "start":
                how = self.follow()
            elif how == "off":
                self.shutdown()
                return 0
            elif how == "stop":
                self.emit("stopped", "정지중")
                how = self.wait_command(("start", "off", "estop"))
            elif how == "fail":
                st = self.ctrl.snapshot()
                if st.connected and not st.torque:
                    # 팔로워 통신이 여러 번 끊겨 제어 코드가 힘을 껐다. 움직일 수 없으니 끄기만 받는다.
                    self.emit("estop", "팔 통신 오류로 팔 힘이 꺼졌습니다.\n팔을 받치고 「끄기」를 누르세요.")
                    self.wait_command(("off",))
                    return 1
                self.emit("error", "리더 팔을 읽지 못해 멈췄습니다.\n리더 USB · 전원을 확인한 뒤 「켜기」, 그만하려면 「끄기」.")
                how = self.wait_command(("start", "off", "estop"))
            else:  # estop
                _log.warning("E-stop — 런처 (그 자리에 멈춤)")
                self.emit("estop", ESTOP_MSG)
                self.wait_command(("off",))
                self.shutdown()
                return 0


# 화면 맨 아래 작업 표시줄 등을 빼는 높이 (3D 창).
SCREEN_BOTTOM_MARGIN = 80


def _screen_layout(geometry: str | None) -> tuple[str | None, str | None]:
    """화면을 좌우 반으로. 그래프는 왼쪽 반에서 높이 절반, 세로 가운데. 3D 는 오른쪽 반 전체."""
    left, right = split_geometry(geometry)
    if left is None:
        return None, None
    x, y, w, h = (int(float(v)) for v in left.split(","))
    plot = f"{x},{y + h // 4},{w},{h // 2}"
    rx, ry, rw, rh = (int(float(v)) for v in right.split(","))
    return plot, f"{rx},{ry},{rw},{rh - SCREEN_BOTTOM_MARGIN}"


def main(argv: list[str] | None = None, warm: Warm | None = None) -> int:
    """warm: 런처가 미리 만들어 둔 팔 모델 · 제어 루프 (standby.py). 없으면 여기서 만든다."""
    ap = argparse.ArgumentParser(description="원격 조종 (런처가 켠다)")
    ap.add_argument("--sim", action="store_true", help="실제 팔 없이 가상 팔로 (리더 대신 정해진 움직임)")
    ap.add_argument("--no-viz", action="store_true", help="3D 화면을 띄우지 않음")
    ap.add_argument("--viz-geometry", default=None, help="화면 x,y,w,h (왼쪽 가운데 그래프, 오른쪽 반 3D)")
    args = ap.parse_args(argv)

    runlog.setup("teleop")
    warm = warm or prepare(args.sim, args.no_viz, _log)
    ctrl = warm.ctrl
    leader = FakeLeader() if args.sim else Leader()
    tele = Teleop(ctrl, leader, viz=warm.viz)
    plot_geom, viz_geom = _screen_layout(args.viz_geometry)
    if not args.no_viz:
        open_viz_window(warm.viz, viz_geom)
        try:
            from joint_plot import JointPlot

            tele.plot = JointPlot(plot_geom, _log)
        except Exception as exc:
            _log.warning("관절 그래프를 띄우지 못함: %s", exc)
    tele.cmds = warm.cmds

    code = 1
    try:
        tele.emit("connecting", "팔 연결 중")
        if not args.sim:
            busy = port_users(FOLLOWER_PORT) or port_users(LEADER_PORT)
            if busy:
                tele.emit("error", "팔을 다른 프로그램(펜던트 · 반복 데모 등)이 쓰고 있습니다. 그 프로그램을 끈 뒤 다시 켜세요.", fatal=True)
                return 1
            try:
                leader.connect()
            except Exception as exc:
                tele.emit("error", f"리더 팔에 연결하지 못했습니다.\n{exc}", fatal=True)
                return 1
            try:
                ctrl.connect()
            except Exception as exc:
                tele.emit("error", f"팔로워 팔에 연결하지 못했습니다.\n{exc}", fatal=True)
                return 1
        else:
            leader.connect()
        code = tele.run()
    finally:
        try:
            ctrl.stop()  # 팔로워 연결도 끊는다 (힘 꺼짐 — 앉은 자세에서)
        except Exception:
            pass
        leader.disconnect()
        if tele.plot is not None:
            tele.plot.close()
        if tele.viz is not None:
            try:
                tele.viz.close()
            except Exception:
                pass
        close_windows()
        tele.emit("off", "꺼짐")
    return code


if __name__ == "__main__":
    sys.exit(main())
