"""런처가 켜는 프로그램(반복 데모 · 원격 조종)이 같이 쓰는 것: 명령 받기, 3D 화면 창."""

from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

# 3D 창 전용 Chrome 프로필. 평소 쓰는 Chrome 과 따로 떠야 창 자리 · 크기를 지정할 수 있고, 끌 때 창도 같이 닫힌다.
CHROME_PROFILE = Path.home() / ".cache" / "aivision" / "chrome_3d"
_windows: list[subprocess.Popen] = []


def read_commands(q: queue.Queue) -> None:
    """표준 입력의 명령을 한 줄씩 q 에 넣는다. 런처가 사라지면 (입력이 닫히면) off."""
    for line in sys.stdin:
        cmd = line.strip().lower()
        if cmd:
            q.put(cmd)
    q.put("off")


def open_meshcat(url: str, geometry: str | None) -> None:
    """Meshcat 을 Chrome 앱 창으로 연다. geometry = "x,y,w,h". close_windows() 로 닫는다."""
    browser = next((b for b in map(shutil.which, ("google-chrome", "chromium-browser", "chromium")) if b), None)
    if browser is None:
        import webbrowser

        webbrowser.open(url)
        return
    CHROME_PROFILE.mkdir(parents=True, exist_ok=True)
    cmd = [browser, f"--app={url}", f"--user-data-dir={CHROME_PROFILE}", "--no-first-run", "--no-default-browser-check"]
    if geometry:
        x, y, w, h = (int(float(v)) for v in geometry.split(","))
        cmd += [f"--window-position={x},{y}", f"--window-size={w},{h}"]
    _windows.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))


def close_windows() -> None:
    """open_meshcat 으로 연 창을 닫는다."""
    import os
    import signal

    while _windows:
        proc = _windows.pop()
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass


def start_viz(kin, q, geometry: str | None, log, *, open_window: bool = True):
    """3D 화면 (팔만). 못 띄우면 None. open_window=False 면 창은 나중에 open_meshcat 으로 연다."""
    try:
        from motion.visualizer import Visualizer

        viz = Visualizer(kin, open_browser=False)
        # 데모 화면은 단순하게: TCP 지나간 경로 없이, 팔은 회색 한 가지.
        viz.set_tcp_trail_enabled(False)
        viz.set_robot_colorful(False)
        viz.display(q)
        if viz.url and open_window:
            open_meshcat(viz.url, geometry)
        return viz
    except Exception as exc:
        log.warning("3D 화면을 띄우지 못함: %s", exc)
        return None


def open_viz_window(viz, geometry: str | None) -> None:
    if viz is not None and viz.url:
        open_meshcat(viz.url, geometry)


def split_geometry(geometry: str | None) -> tuple[str | None, str | None]:
    """"x,y,w,h" 를 왼쪽 · 오른쪽 반으로 나눈다."""
    if not geometry:
        return None, None
    x, y, w, h = (int(float(v)) for v in geometry.split(","))
    half = w // 2
    return f"{x},{y},{half},{h}", f"{x + half},{y},{w - half},{h}"


# 런처 속도 슬라이더 (%). 관절 이동과 직선 (xyzrpy) 이동을 같은 비율로 바꾼다.
SPEED_DEFAULT_PCT = 50


def speed_values(pct: float) -> dict[str, float]:
    """속도 % → 관절 °/s · 직선 mm/s · 자세 °/s. 펜던트 슬라이더와 같은 범위 (10% 최저, 100% 최고)."""
    from motion.controller import (
        JOG_ROT_MAX_DEG_S,
        JOG_ROT_MIN_DEG_S,
        JOG_VEL_MAX_MPS,
        JOG_VEL_MIN_MPS,
        SPEED_PCT_MAX,
        SPEED_PCT_MIN,
        speed_pct_to_joint_vel,
    )

    p = min(max(float(pct), SPEED_PCT_MIN), SPEED_PCT_MAX)
    t = (p - SPEED_PCT_MIN) / (SPEED_PCT_MAX - SPEED_PCT_MIN)
    return {
        "pct": p,
        "joint_deg_s": speed_pct_to_joint_vel(p),
        "lin_mm_s": (JOG_VEL_MIN_MPS + t * (JOG_VEL_MAX_MPS - JOG_VEL_MIN_MPS)) * 1000.0,
        "rot_deg_s": JOG_ROT_MIN_DEG_S + t * (JOG_ROT_MAX_DEG_S - JOG_ROT_MIN_DEG_S),
    }


def apply_speed(ctrl, pct: float) -> dict[str, float]:
    """다음 이동부터 이 속도로 (이미 시작한 이동은 그대로)."""
    v = speed_values(pct)
    ctrl.set_speed_pct(v["pct"])  # 관절 이동 · 그리퍼
    ctrl.set_tcp_jog_mm_s(v["lin_mm_s"])  # 직선 이동 (위치)
    ctrl.set_rpy_jog_deg_s(v["rot_deg_s"])  # 직선 이동 (자세)
    return v


def preload_lerobot() -> None:
    """lerobot 팔 코드를 미리 불러 둔다 (처음 부를 때 몇 초 걸린다)."""
    import lerobot.robots.so_follower  # noqa: F401
    import lerobot.teleoperators.so_leader  # noqa: F401


@dataclass
class Warm:
    """켜는 데 오래 걸리는 것 (팔 모델 10초 · lerobot 4초). 런처가 standby.py 로 미리 만들어 둔다."""

    kin: object
    ctrl: object
    viz: object
    cmds: queue.Queue


def prepare(sim: bool, no_viz: bool, log) -> Warm:
    """팔 모델 · 제어 루프 · 3D 화면 (창은 아직 안 엶) · 명령 받기. 팔은 아직 연결하지 않는다."""
    from motion.controller import Controller
    from motion.hw_controller import Hardware
    from motion.robot_kinematics import DEFAULT_URDF, RobotKinematics

    kin = RobotKinematics(DEFAULT_URDF)
    ctrl = Controller(kin, hardware=None if sim else Hardware())
    ctrl.start(pose_server=False)
    cmds: queue.Queue[str] = queue.Queue()
    threading.Thread(target=read_commands, args=(cmds,), daemon=True).start()
    viz = None if no_viz else start_viz(kin, ctrl.snapshot().q, None, log, open_window=False)
    if not sim:
        preload_lerobot()
    return Warm(kin=kin, ctrl=ctrl, viz=viz, cmds=cmds)
