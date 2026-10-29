#!/usr/bin/env python3
"""반복 데모: 카메라 없이 정해진 두 자리 사이로 물체를 옮기기를 되풀이한다. 런처가 켜고 끈다.

  python launcher/repeat_demo.py            # 실제 팔 (자동 Connect)
  python launcher/repeat_demo.py --sim      # 가상 팔로 시험

한 사이클
  초기자세 → 물체가 있는 자리에서 집기 → 다른 자리에 놓기 → 초기자세
  다음 사이클은 반대 방향. 물체는 비스듬히 (Rx, Ry, Rz) = (180, 45, 0), 놓을 때도 같은 자세로 놓는다.
  CYCLES 번 하면 끄기와 같이 앉은 자세로 간 뒤 끝난다.

설정 파일 launcher/demo_settings.json (없으면 기본값으로 만든다, git 에는 안 올림)
  cycles         반복 횟수 (--cycles 를 주면 그 값)
  left_xyz_mm    왼쪽 자리 [x, y, z] (베이스 mm, +Y 가 로봇 기준 왼쪽, z 는 물체 중심 높이)
  right_xyz_mm   오른쪽 자리
  approach_d_mm  물체 표면 → 대기 위치 거리
  approach_a_mm  대기 위치 → 집기 위치 거리 (접근 방향)
  사이클마다 다시 읽으니, 데모 중에 고쳐 저장하면 다음 사이클부터 적용된다.

런처와 주고받기
  명령 (표준 입력, 한 줄에 하나)
    start  정지중이면 다시 시작
    stop   이번 사이클을 마치고 초기자세에서 멈춤
    off    이번 사이클을 마치고 초기자세 → 앉은 자세 → 끝냄
    estop  즉시 그 자리에 멈춤 (힘은 유지, 처지지 않음). 그 뒤에는 off 만 받는다.
  상태 (표준 출력, 한 줄에 JSON 하나)
    {"state": ..., "cycle": 완료 횟수, "cycles": 목표 횟수, "from": "왼쪽", "to": "오른쪽", "step": 단계, "msg": 글}
    state: connecting · homing · running · stopping · stopped · turning_off · off · estop · error
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from cad.model import approach_mm, bounds_mm  # noqa: E402
from common import SPEED_DEFAULT_PCT, Warm, apply_speed, close_windows, open_viz_window, prepare  # noqa: E402
from motion import runlog  # noqa: E402
from motion.controller import ALREADY_AT_TARGET, Controller  # noqa: E402
from motion.grasp import PlacePose, T_to_xyzrpy, compute_auto_pre  # noqa: E402
from motion.hw_controller import port_users  # noqa: E402
from motion.pick_place import PickPlaceRunner, build_pick_place_steps  # noqa: E402
from motion.robot_kinematics import (  # noqa: E402
    ARM_JOINT_NAMES,
    INIT_POSE_JOINTS_DEG,
    SHUTDOWN_JOINTS_DEG,
    RobotKinematics,
)

# 두 자리 (프로젝트 베이스 mm). +Y 가 로봇 기준 왼쪽.
SPOTS = {"left": (150.0, 180.0, 34.0), "right": (150.0, -180.0, 34.0)}
SPOT_KO = {"left": "왼쪽", "right": "오른쪽"}
OBJECT_RPY = (180.0, 45.0, 0.0)  # 비스듬히, 베이스 기준
CYCLES = 10
# 이만큼 (도) 안이면 초기자세에 있다고 본다.
AT_INIT_DEG = 2.0
TICK_S = 0.05
# 초기자세에 도착한 뒤 다음 사이클 전까지 쉬는 시간 (초).
INIT_PAUSE_S = 1.0
STATE_FILE_NAME = "repeat_demo_state.json"
SETTINGS_PATH = Path(__file__).resolve().with_name("demo_settings.json")
# 범위 밖이거나 형식이 틀리면 그 값만 기본값을 쓴다.
SETTING_LIMITS = {"approach_d_mm": (0.0, 60.0), "approach_a_mm": (5.0, 80.0)}  # mm
CYCLES_LIMITS = (1, 1000)
SPOT_KEYS = {"left_xyz_mm": "left", "right_xyz_mm": "right"}
SPOT_LIMITS = ((80.0, 350.0), (-350.0, 350.0), (0.0, 150.0))  # x, y, z mm (베이스)
SETTING_KEYS = ("cycles", "left_xyz_mm", "right_xyz_mm", "approach_d_mm", "approach_a_mm")
_log = runlog.get("demo")
ESTOP_MSG = (
    "E-stop — 팔이 그 자리에 멈췄습니다.\n"
    "「끄기」를 누르면 앉은 자세로 간 뒤 끕니다."
)


def _number_ok(v, lo: float, hi: float) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= float(v) <= hi


def _setting_ok(key: str, v) -> bool:
    if key == "cycles":
        return isinstance(v, int) and not isinstance(v, bool) and CYCLES_LIMITS[0] <= v <= CYCLES_LIMITS[1]
    if key in SPOT_KEYS:
        return isinstance(v, list) and len(v) == 3 and all(_number_ok(c, *lim) for c, lim in zip(v, SPOT_LIMITS))
    return _number_ok(v, *SETTING_LIMITS[key])


def default_demo_settings() -> dict:
    d, a = approach_mm()
    return {
        "cycles": CYCLES,
        "left_xyz_mm": list(SPOTS["left"]),
        "right_xyz_mm": list(SPOTS["right"]),
        "approach_d_mm": d,
        "approach_a_mm": a,
    }


def _write_settings(path: Path, data: dict) -> None:
    """한 줄에 키 하나 (좌표도 한 줄)로 써서 손으로 고치기 쉽게 한다."""
    order = [k for k in default_demo_settings() if k in data] + [k for k in data if k not in SETTING_KEYS]
    body = ",\n".join(f"  {json.dumps(k)}: {json.dumps(data[k], ensure_ascii=False)}" for k in order)
    try:
        path.write_text("{\n" + body + "\n}\n", encoding="utf-8")
    except OSError as exc:
        _log.warning("설정 파일을 못 씀 %s: %s", path, exc)


def load_demo_settings(path: Path = SETTINGS_PATH) -> dict:
    """demo_settings.json 값. 없으면 기본값으로 만들고, 빠진 키는 기본값으로 채워 넣는다.
    잘못된 값은 경고만 남기고 그 키만 기본값을 쓴다."""
    defaults = default_demo_settings()
    if not path.exists():
        _write_settings(path, defaults)
        return defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON 객체 { } 가 아님")
    except (OSError, ValueError) as exc:
        _log.warning("설정 파일을 못 읽음 %s: %s → 기본값", path, exc)
        return defaults
    out = dict(defaults)
    for key in defaults:
        if key not in data:
            continue
        if _setting_ok(key, data[key]):
            out[key] = data[key]
        else:
            _log.warning("설정 %s=%r 이 범위 밖이거나 형식이 틀림 → 기본값 %r", key, data[key], defaults[key])
    missing = [k for k in defaults if k not in data]
    if missing:
        _write_settings(path, {**data, **{k: defaults[k] for k in missing}})
    return out


def _other(side: str) -> str:
    return "right" if side == "left" else "left"


class StateFile:
    """물체가 지금 어느 자리에 있는지 기억한다 (껐다 켜도 이어서). 없으면 왼쪽."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> str:
        try:
            side = json.loads(self.path.read_text(encoding="utf-8")).get("object_at")
        except (OSError, ValueError):
            side = None
        return side if side in SPOTS else "left"

    def save(self, side: str) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"object_at": side, "time": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
        except OSError as exc:
            _log.warning("물체 위치 기록 실패: %s", exc)


class RepeatDemo:
    def __init__(self, ctrl: Controller, kin: RobotKinematics, state: StateFile, *, cycles: int | None, viz=None) -> None:
        self.ctrl = ctrl
        self.kin = kin
        self.state = state
        # None 이면 설정 파일의 cycles 를 따른다 (사이클마다 다시 읽음).
        self._cycles_arg = cycles
        self.cycles = int(cycles) if cycles is not None else CYCLES
        self.viz = viz
        self.runner = PickPlaceRunner(ctrl)
        self.cmds: queue.Queue[str] = queue.Queue()
        self.side = state.load()
        self.done = 0
        self.stop_after_cycle = False
        self.off_after_cycle = False
        self._last: dict = {}
        mins, maxs = bounds_mm()
        self._mins, self._maxs = mins, maxs
        self._settings: dict = {}
        self.reload_settings()

    # ------------------------------------------------------------ 주고받기

    def emit(self, state: str, msg: str = "", **extra) -> None:
        out = {
            "state": state,
            "cycle": self.done,
            "cycles": self.cycles,
            "from": SPOT_KO[self.side],
            "to": SPOT_KO[_other(self.side)],
            "step": self.runner.step_name if self.runner.busy() else "",
            "msg": msg,
            **extra,
        }
        if out != self._last:
            self._last = dict(out)
            print(json.dumps(out, ensure_ascii=False), flush=True)

    def _take_commands(self) -> None:
        """받은 명령을 표시만 해 둔다. estop 은 바로 처리한다."""
        while True:
            try:
                cmd = self.cmds.get_nowait()
            except queue.Empty:
                return
            _log.info("명령 %s", cmd)
            if cmd == "estop":
                raise _EStop()
            if cmd == "stop":
                self.stop_after_cycle = True
            elif cmd == "off":
                self.off_after_cycle = True
            elif cmd == "start":
                self.stop_after_cycle = False
            elif cmd.startswith("speed "):
                self._set_speed(cmd.split()[1])

    def _set_speed(self, value: str) -> None:
        try:
            v = apply_speed(self.ctrl, float(value))
        except ValueError:
            return
        _log.info("속도 %.0f%%  관절 %.0f°/s · 직선 %.0f mm/s", v["pct"], v["joint_deg_s"], v["lin_mm_s"])

    def _idle(self, sec: float = TICK_S) -> None:
        """기다리는 동안 3D 화면을 그리고 명령을 받는다."""
        end = time.monotonic() + sec
        while True:
            if self.viz is not None:
                try:
                    self.viz.display(self.ctrl.snapshot().q)
                except Exception:
                    pass
            self._take_commands()
            if time.monotonic() >= end:
                return
            time.sleep(min(TICK_S, max(0.0, end - time.monotonic())))

    # ------------------------------------------------------------ 움직이기

    def _wait_goto(self, state: str, timeout_s: float = 60.0) -> None:
        end = time.monotonic() + timeout_s
        self._idle(0.1)
        while self.ctrl.goto_active() and time.monotonic() < end:
            self.emit(state)
            self._idle()
        fault = str(self.ctrl.snapshot().fault or "")
        if fault and fault != ALREADY_AT_TARGET:
            raise _Failed(fault)

    def _at_init(self) -> bool:
        joints = self.ctrl.snapshot().joints_deg
        return all(abs(float(joints[j]) - float(INIT_POSE_JOINTS_DEG[j])) <= AT_INIT_DEG for j in ARM_JOINT_NAMES)

    def go_init(self) -> None:
        if self._at_init():
            return
        self.emit("homing", "초기자세로 이동")
        self.ctrl.init_pose()
        self._wait_goto("homing")

    def reload_settings(self) -> None:
        settings = load_demo_settings()
        if settings != self._settings:
            _log.info(
                "설정  반복 %s회  왼쪽 %s  오른쪽 %s  d=%g mm  a=%g mm",
                settings["cycles"] if self._cycles_arg is None else f"{self._cycles_arg}(--cycles)",
                settings["left_xyz_mm"], settings["right_xyz_mm"],
                settings["approach_d_mm"], settings["approach_a_mm"],
            )
            self._settings = settings
        if self._cycles_arg is None:
            self.cycles = int(settings["cycles"])

    def _cycle_steps(self):
        settings = self._settings
        spots = {side: tuple(float(c) for c in settings[key]) for key, side in SPOT_KEYS.items()}
        src, dst = spots[self.side], spots[_other(self.side)]
        d, a = settings["approach_d_mm"], settings["approach_a_mm"]
        obj = PlacePose(xyz_mm=src, rpy_deg=OBJECT_RPY)
        auto = compute_auto_pre(obj, self._mins, self._maxs, d, a, kin=self.kin, q=self.ctrl.snapshot().q)
        p_xyz, p_rpy = T_to_xyzrpy(auto.T_base)
        g_xyz, g_rpy = T_to_xyzrpy(auto.T_grasp_base)
        # 물체를 집은 손 모양 그대로 옮긴다: 손끝도 물체와 같은 만큼만 움직이고, 손목은 그대로.
        drop_xy = np.asarray(g_xyz[:2]) + (np.asarray(dst[:2]) - np.asarray(src[:2]))
        return build_pick_place_steps(
            p_xyz_mm=p_xyz, p_rpy_deg=p_rpy, g_xyz_mm=g_xyz, g_rpy_deg=g_rpy,
            drop_xy_mm=drop_xy, drop_rpy_deg=g_rpy,
        )

    def run_cycle(self) -> None:
        _log.info("사이클 %d/%d 시작  %s → %s", self.done + 1, self.cycles, SPOT_KO[self.side], SPOT_KO[_other(self.side)])
        if not self.runner.start(self._cycle_steps()):
            raise _Failed(self.runner.status)
        while True:
            phase = self.runner.tick()
            if phase == "done":
                break
            if phase == "fault":
                raise _Failed(self.runner.status)
            self.emit("stopping" if (self.stop_after_cycle or self.off_after_cycle) else "running")
            self._idle()
        self.side = _other(self.side)
        self.state.save(self.side)
        self.done += 1
        _log.info("사이클 %d/%d 완료  물체는 %s", self.done, self.cycles, SPOT_KO[self.side])

    def shutdown(self) -> None:
        """앉은 자세로 간 뒤 연결을 끊는다 (펜던트 창을 닫을 때와 같다). 쥔 물체는 놓지 않는다."""
        st = self.ctrl.snapshot()
        if st.connected and st.torque:
            self.emit("turning_off", "앉은 자세로 이동")
            self.ctrl.start_joint_goto(self.ctrl.keep_gripper_if_holding(SHUTDOWN_JOINTS_DEG))
            try:
                self._wait_goto("turning_off", timeout_s=30.0)
            except (_Failed, _EStop):
                pass

    # ------------------------------------------------------------ 전체 흐름

    def run(self) -> int:
        try:
            self.go_init()
            while True:
                self.reload_settings()
                if self.done >= self.cycles:
                    self.emit("turning_off", f"{self.cycles}회 완료 — 끕니다", finished=True)
                    _log.info("%d회 완료, 끔", self.cycles)
                    self.shutdown()
                    return 0
                if self.off_after_cycle:
                    self.shutdown()
                    return 0
                if self.stop_after_cycle:
                    self.emit("stopped", "정지중")
                    while self.stop_after_cycle and not self.off_after_cycle:
                        self._idle(0.1)
                    if self.off_after_cycle:
                        self.shutdown()
                        return 0
                    self.go_init()
                    continue
                self.emit("running", step="초기자세 대기")
                self._idle(INIT_PAUSE_S)
                if self.stop_after_cycle or self.off_after_cycle:
                    continue  # 쉬는 동안 정지·끄기를 눌렀으면 새 사이클을 시작하지 않는다
                self.run_cycle()
        except _EStop:
            self._hold_stop()
            self.state.save("left")  # 들고 있던 중일 수도 있어 위치를 모른다. 왼쪽에 두고 다시 켜게 한다.
            self.emit("estop", ESTOP_MSG)
            return self._after_estop()
        except _Failed as exc:
            self.state.save("left")
            lx, ly = self._settings["left_xyz_mm"][:2]
            msg = (
                f"실패: {exc}\n물체를 왼쪽 자리 ({lx:g}, {ly:g}) 에 놓고 「켜기」를 누르세요."
                if not self.ctrl.holding()
                else f"실패: {exc}\n물체를 쥔 채 멈췄습니다. 손으로 받친 뒤 「끄기」를 누르세요."
            )
            _log.warning("데모 실패: %s", exc)
            self.side = "left"
            self.emit("error", msg)
            return self._after_error()

    def _after_error(self) -> int:
        """실패 뒤: 켜기(start)면 처음부터, 끄기(off)면 앉은 자세로 간 뒤 끝."""
        self.stop_after_cycle = True
        try:
            while self.stop_after_cycle and not self.off_after_cycle:
                self._idle(0.1)
        except _EStop:
            self._hold_stop()
            self.state.save("left")
            self.emit("estop", ESTOP_MSG)
            return self._after_estop()
        if self.off_after_cycle:
            self.shutdown()
            return 0
        return self.run()

    def _hold_stop(self) -> None:
        """E-stop: 하던 동작을 끊고 지금 자세에서 힘을 유지한다. 쥔 물체도 그대로 쥔다."""
        _log.warning("E-stop — 런처 (그 자리에 멈춤)")
        self.runner.abort("E-stop")  # 내려놓기 없이 끊는다 (ctrl.stop_all 포함)

    def _after_estop(self) -> int:
        """E-stop 뒤에는 끄기만 받는다: 앉은 자세로 간 뒤 끝낸다."""
        while True:
            try:
                cmd = self.cmds.get(timeout=0.1)
            except queue.Empty:
                if self.viz is not None:
                    try:
                        self.viz.display(self.ctrl.snapshot().q)
                    except Exception:
                        pass
                continue
            if cmd != "off":
                continue
            try:
                self.shutdown()
            except Exception as exc:
                _log.warning("E-stop 뒤 앉은 자세 이동 실패: %s", exc)
            return 0


class _EStop(Exception):
    pass


class _Failed(Exception):
    pass


def main(argv: list[str] | None = None, warm: Warm | None = None) -> int:
    """warm: 런처가 미리 만들어 둔 팔 모델 · 제어 루프 (standby.py). 없으면 여기서 만든다."""
    ap = argparse.ArgumentParser(description="반복 데모 (런처가 켠다)")
    ap.add_argument("--sim", action="store_true", help="실제 팔 없이 가상 팔로")
    ap.add_argument("--cycles", type=int, default=None, help="반복 횟수. 안 주면 demo_settings.json 의 cycles")
    ap.add_argument("--speed", type=float, default=SPEED_DEFAULT_PCT, help="속도 %% (10–100). 관절 · 직선 이동 같이")
    ap.add_argument("--no-viz", action="store_true", help="3D 화면을 띄우지 않음")
    ap.add_argument("--viz-geometry", default=None, help="3D 화면 창 위치 x,y,w,h")
    ap.add_argument("--state-file", type=Path, default=None)
    args = ap.parse_args(argv)

    log_path = runlog.setup("demo")
    state_path = args.state_file or ((log_path.parent if log_path else Path.home() / "aivision_logs") / STATE_FILE_NAME)
    warm = warm or prepare(args.sim, args.no_viz, _log)
    ctrl, viz = warm.ctrl, warm.viz
    open_viz_window(viz, args.viz_geometry)
    demo = RepeatDemo(ctrl, warm.kin, StateFile(state_path), cycles=args.cycles, viz=viz)
    demo.cmds = warm.cmds
    demo._set_speed(str(args.speed))

    code = 1
    try:
        if not args.sim:
            demo.emit("connecting", "팔 연결 중")
            busy = port_users()
            if busy:
                demo.emit("error", "팔을 다른 프로그램(펜던트 등)이 쓰고 있습니다. 그 프로그램을 끈 뒤 다시 켜세요.", fatal=True)
                return 1
            try:
                ctrl.connect()
            except Exception as exc:
                demo.emit("error", f"팔에 연결하지 못했습니다.\n{exc}", fatal=True)
                return 1
        code = demo.run()
    finally:
        try:
            ctrl.stop()  # 연결도 끊는다
        except Exception:
            pass
        if viz is not None:
            try:
                viz.close()
            except Exception:
                pass
        close_windows()
        demo.emit("off", "꺼짐")
    return code


if __name__ == "__main__":
    sys.exit(main())
