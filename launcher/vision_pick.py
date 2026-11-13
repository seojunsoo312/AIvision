#!/usr/bin/env python3
"""카메라로 집기: 왼쪽 구역의 브라켓을 로봇에서 가까운 것부터 집어 흰색 · 검은색 바구니에 나눠 담는다.

  python launcher/vision_pick.py            # 실제 팔 (자동 Connect)
  python launcher/vision_pick.py --sim      # 가상 팔로 시험 (카메라 창은 켜 있어야 한다)

「켜기」 한 번 = 왼쪽 구역이 빌 때까지
  초기자세 → 카메라 창에 등록 요청 (구역 안, 베이스에서 가장 가까운 물체)
  → 처음 요청 때 보인 물체를 가까운 순으로 번호 매겨 기록
  → 대기 · 집기 자세 계산 (펜던트와 같은 compute_auto_pre) → 집기 → 색에 맞는 바구니 위에서 그리퍼 열기
  → 초기자세 → 다음 요청 … 구역이 비면 정지중.
  집기 직전마다 다시 찍어 등록하므로, 앞 물체를 집다가 다른 물체가 밀려도 그 자리에서 집는다.
  카메라 창 (런처 「카메라 창」) 이 켜져 있어야 한다.

설정 파일 launcher/vision_pick_settings.json (없으면 기본값으로 만든다, git 에는 안 올림, 반복 데모와 따로)
  pick_region_mm · pick_reach_mm · white_bin_xyz_mm · black_bin_xyz_mm · white_gray_min · approach_d_mm · approach_a_mm
  (launcher/demo_settings.py 참고). 물체 하나 집기 전마다 다시 읽는다.

런처와 주고받기 (반복 데모와 같은 방식)
  명령: start · stop · off · estop · speed N
  상태 (한 줄에 JSON 하나):
    {"state": ..., "picked": 옮긴 수, "white": 흰 수, "black": 검 수, "left": 남은 수 (마지막으로 본 것),
     "step": 단계, "msg": 글}
    state: connecting · homing · running · stopping · stopped · turning_off · off · estop · error
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from cad.model import bounds_mm  # noqa: E402
from common import SPEED_DEFAULT_PCT, Warm, close_windows, open_viz_window, prepare  # noqa: E402
from demo_settings import load_vision_settings  # noqa: E402
from motion import runlog  # noqa: E402
from motion.grasp import PlacePose, T_to_xyzrpy, compute_auto_pre  # noqa: E402
from motion.hw_controller import port_users  # noqa: E402
from motion.pick_place import PickPlaceRunner, build_pick_drop_steps  # noqa: E402
from motion.register_link import (  # noqa: E402
    REG_MSG_BUSY,
    REG_MSG_FAIL,
    REG_MSG_NO_REPLY,
    RegisterRequest,
)
from repeat_demo import ESTOP_MSG, RepeatDemo, _EStop, _Failed  # noqa: E402

_log = runlog.get("vision_pick")
# 카메라 창 응답 → 학생에게 보이는 글. no_box 는 실패가 아니라 「구역이 빔」.
REG_ERRORS = {
    "busy": REG_MSG_BUSY,
    "fail": REG_MSG_FAIL + "\n물체가 카메라 창에서 잘 보이는지 확인하세요.",
    "timeout": REG_MSG_NO_REPLY + "\n런처의 「카메라 창」을 켜고, 창이 뜬 뒤 다시 「켜기」.",
}
COLOR_KO = {"white": "흰", "black": "검"}
MSG_EMPTY = "왼쪽 구역이 비었습니다.\n브라켓을 놓고 「켜기」를 누르세요."
MSG_STOPPED = "정지중. 「켜기」로 이어서 합니다."
MSG_FAR = ("가장 가까운 물체 ({x:.0f}, {y:.0f}) 가 팔이 닿는 거리 ({lo:.0f}~{hi:.0f} mm) 밖입니다 ({d:.0f} mm).\n"
           "로봇 쪽으로 옮겨 놓고 「켜기」를 누르세요.")


@dataclass(frozen=True)
class Target:
    """카메라가 고른 물체 하나."""

    pose: PlacePose
    color: str  # white · black
    gray: float | None
    left: int  # 이 물체를 포함해 구역에 보인 수


class VisionPick(RepeatDemo):
    """반복 데모의 명령 받기 · 이동 기다리기 · E-stop · 끄기를 그대로 쓰고, 한 번 동작만 다르다."""

    def __init__(self, ctrl, kin, *, viz=None) -> None:
        # 부모 __init__ 은 부르지 않는다 (두 자리 상태 파일이 없다). 부모 메서드가 쓰는 값만 둔다.
        self.ctrl = ctrl
        self.kin = kin
        self.viz = viz
        self.runner = PickPlaceRunner(ctrl)
        self.cmds = None
        self.counts = {"white": 0, "black": 0}
        self.left: int | None = None
        self._numbered = False  # 이번 「켜기」에서 번호를 매겼는지
        self.stop_after_cycle = False
        self.off_after_cycle = False
        self._last: dict = {}
        self._mins, self._maxs = bounds_mm()
        self._settings: dict = {}
        self.reload_settings()

    @property
    def picked(self) -> int:
        return sum(self.counts.values())

    # ------------------------------------------------------------ 주고받기

    def emit(self, state: str, msg: str = "", **extra) -> None:
        out = {
            "state": state,
            "picked": self.picked,
            **self.counts,
            "left": self.left,
            "step": self.runner.step_name if self.runner.busy() else "",
            "msg": msg,
            **extra,
        }
        if out != self._last:
            self._last = dict(out)
            print(json.dumps(out, ensure_ascii=False), flush=True)

    def reload_settings(self) -> None:
        settings = load_vision_settings()
        if settings != self._settings:
            _log.info(
                "설정  구역 %s  흰 바구니 %s  검 바구니 %s  흰색 밝기 ≥ %g  d=%g mm  a=%g mm",
                settings["pick_region_mm"], settings["white_bin_xyz_mm"], settings["black_bin_xyz_mm"],
                settings["white_gray_min"], settings["approach_d_mm"], settings["approach_a_mm"],
            )
            self._settings = settings

    # ------------------------------------------------------------ 한 번 동작

    def find_object(self) -> Target | None:
        """카메라 창에 구역 안 가장 가까운 물체의 등록을 요청한다. 구역이 비면 None.
        기다리는 동안에도 E-stop · 정지 · 끄기를 받는다."""
        region = self._settings["pick_region_mm"]
        try:
            req = RegisterRequest.send(region_xy_mm=region, nearest=True)
        except OSError as exc:
            raise _Failed(REG_ERRORS["timeout"]) from exc
        while not req.done:
            self.emit("running", step="카메라로 물체 찾는 중")
            self._idle(0.1)
            req.poll()
        self.left = len(req.targets_xy_mm)
        if req.state == "no_box":
            return None
        if req.state != "ok":
            raise _Failed(REG_ERRORS.get(req.state, REG_MSG_FAIL))
        if not self._numbered:
            self._numbered = True
            listed = "  ".join(f"{k}:({x:.0f},{y:.0f})" for k, (x, y) in enumerate(req.targets_xy_mm, 1))
            _log.info("구역의 물체 %d개 (가까운 순)  %s", len(req.targets_xy_mm), listed)
        gray = req.gray
        color = "white" if gray is not None and gray >= float(self._settings["white_gray_min"]) else "black"
        _log.info("물체 %s  밝기 %s  xyz=%s  rpy=%s", COLOR_KO[color], "?" if gray is None else f"{gray:.0f}",
                  np.round(req.xyz_mm, 1).tolist(), np.round(req.rpy_deg, 1).tolist())
        return Target(PlacePose(xyz_mm=req.xyz_mm, rpy_deg=req.rpy_deg), color, gray, self.left)

    def pick_steps(self, target: Target):
        settings = self._settings
        auto = compute_auto_pre(
            target.pose, self._mins, self._maxs, settings["approach_d_mm"], settings["approach_a_mm"],
            kin=self.kin, q=self.ctrl.snapshot().q,
        )
        p_xyz, p_rpy = T_to_xyzrpy(auto.T_base)
        g_xyz, g_rpy = T_to_xyzrpy(auto.T_grasp_base)
        drop = settings[f"{target.color}_bin_xyz_mm"]
        _log.info("대기 %s %s  집기 %s %s  → %s 바구니 %s", np.round(p_xyz, 1).tolist(), np.round(p_rpy, 1).tolist(),
                  np.round(g_xyz, 1).tolist(), np.round(g_rpy, 1).tolist(), COLOR_KO[target.color], drop)
        return build_pick_drop_steps(
            p_xyz_mm=p_xyz, p_rpy_deg=p_rpy, g_xyz_mm=g_xyz, g_rpy_deg=g_rpy,
            drop_xyz_mm=np.asarray(drop, dtype=float),
        )

    def out_of_reach(self, target: Target) -> str:
        """팔이 닿는 거리 밖이면 알릴 글, 안이면 ""."""
        x, y = (float(v) for v in target.pose.xyz_mm[:2])
        lo, hi = (float(v) for v in self._settings["pick_reach_mm"])
        d = float(np.hypot(x, y))
        return "" if lo <= d <= hi else MSG_FAR.format(x=x, y=y, lo=lo, hi=hi, d=d)

    def pick_once(self) -> str:
        """물체 하나를 바구니에 담는다. "moved" · "empty" (구역이 빔) · "cancelled" (찾는 동안 정지 · 끄기)
        · 그 밖의 글 (가장 가까운 물체가 닿는 거리 밖, 팔은 안 움직임)."""
        self.reload_settings()
        target = self.find_object()
        if self.stop_after_cycle or self.off_after_cycle:
            return "cancelled"  # 찾는 동안 누른 정지 · 끄기: 움직이기 전에 멈춘다
        if target is None:
            _log.info("구역이 빔")
            return "empty"
        far = self.out_of_reach(target)
        if far:
            _log.info(far.replace("\n", " "))
            return far
        if not self.runner.start(self.pick_steps(target)):
            raise _Failed(self.runner.status)
        while True:
            phase = self.runner.tick()
            if phase == "done":
                break
            if phase == "fault":
                raise _Failed(self.runner.status)
            self.emit("stopping" if (self.stop_after_cycle or self.off_after_cycle) else "running")
            self._idle()
        self.counts[target.color] += 1
        self.left = max(0, target.left - 1)
        _log.info("옮김 %d개째 (흰 %d · 검 %d)", self.picked, self.counts["white"], self.counts["black"])
        return "moved"

    # ------------------------------------------------------------ 전체 흐름

    def _wait_start(self, msg: str, *, state: str = "stopped") -> bool:
        """정지중으로 알리고 「켜기」를 기다린다. 「끄기」면 False."""
        self.stop_after_cycle = True
        self.emit(state, msg)
        while self.stop_after_cycle and not self.off_after_cycle:
            self._idle(0.1)
        self._numbered = False  # 다시 켜면 번호를 새로 매긴다
        return not self.off_after_cycle

    def run(self) -> int:
        try:
            self.go_init()
            while True:
                if self.off_after_cycle:
                    self.shutdown()
                    return 0
                result = self.pick_once()  # 끝나면 초기자세 (집기 단계 마지막)
                if self.off_after_cycle:
                    self.go_init()
                    self.shutdown()
                    return 0
                if result == "moved" and not self.stop_after_cycle:
                    continue
                self.go_init()
                msg = {"empty": MSG_EMPTY, "cancelled": MSG_STOPPED}.get(result, result)
                if not self._wait_start(msg):
                    self.shutdown()
                    return 0
        except _EStop:
            self._hold_stop()
            self.emit("estop", ESTOP_MSG)
            return self._after_estop()
        except _Failed as exc:
            _log.warning("카메라로 집기 실패: %s", exc)
            msg = (
                f"실패: {exc}\n「켜기」로 다시, 「끄기」로 끝냅니다."
                if not self.ctrl.holding()
                else f"실패: {exc}\n물체를 쥔 채 멈췄습니다. 손으로 받친 뒤 「끄기」를 누르세요."
            )
            try:
                if not self._wait_start(msg, state="error"):
                    self.shutdown()
                    return 0
            except _EStop:
                self._hold_stop()
                self.emit("estop", ESTOP_MSG)
                return self._after_estop()
            return self.run()


def main(argv: list[str] | None = None, warm: Warm | None = None) -> int:
    """warm: 런처가 미리 만들어 둔 팔 모델 · 제어 루프 (standby.py). 없으면 여기서 만든다."""
    ap = argparse.ArgumentParser(description="카메라로 집기 (런처가 켠다)")
    ap.add_argument("--sim", action="store_true", help="실제 팔 없이 가상 팔로")
    ap.add_argument("--speed", type=float, default=SPEED_DEFAULT_PCT, help="속도 %% (10–100)")
    ap.add_argument("--no-viz", action="store_true", help="3D 화면을 띄우지 않음")
    ap.add_argument("--viz-geometry", default=None, help="3D 화면 창 위치 x,y,w,h")
    args = ap.parse_args(argv)

    runlog.setup("vision_pick")
    warm = warm or prepare(args.sim, args.no_viz, _log)
    ctrl, viz = warm.ctrl, warm.viz
    open_viz_window(viz, args.viz_geometry)
    job = VisionPick(ctrl, warm.kin, viz=viz)
    job.cmds = warm.cmds
    job._set_speed(str(args.speed))

    code = 1
    try:
        if not args.sim:
            job.emit("connecting", "팔 연결 중")
            if port_users():
                job.emit("error", "팔을 다른 프로그램(펜던트 등)이 쓰고 있습니다. 그 프로그램을 끈 뒤 다시 켜세요.", fatal=True)
                return 1
            try:
                ctrl.connect()
            except Exception as exc:
                job.emit("error", f"팔에 연결하지 못했습니다.\n{exc}", fatal=True)
                return 1
        code = job.run()
    finally:
        try:
            ctrl.stop()
        except Exception:
            pass
        if viz is not None:
            try:
                viz.close()
            except Exception:
                pass
        close_windows()
        job.emit("off", "꺼짐")
    return code


if __name__ == "__main__":
    sys.exit(main())
