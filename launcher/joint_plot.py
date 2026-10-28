"""원격 조종 관절 그래프: Rerun 창을 정해진 자리에 띄우고, 관절마다 리더(파랑) · 팔로워(주황) 두 줄을 그린다.

칸 배치는 deploy/teleoperate_7dof.py 와 같다 (2행 4열, 관절 하나씩). 값은 펜던트와 같은 관절 각도 (L자 = 0°).
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time

from motion.robot_kinematics import LEROBOT_FROM_URDF, URDF_JOINT_NAMES

LEADER_COLOR = [80, 160, 255]
FOLLOWER_COLOR = [255, 140, 40]
MEMORY_LIMIT = "300MB"
# 1초에 그리는 횟수 (제어는 30 번).
PLOT_HZ = 15.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_port(port: int, timeout_s: float) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _place_window(title_part: str, x: int, y: int, w: int, h: int, log, timeout_s: float = 20.0) -> None:
    """제목에 title_part 가 들어간 창이 뜨면 그 자리 · 크기로 옮긴다 (X11)."""
    try:
        from Xlib import X, display
    except Exception as exc:
        log.warning("창 자리를 못 옮김 (python-xlib 없음): %s", exc)
        return
    d = display.Display()
    root = d.screen().root
    net_list = d.intern_atom("_NET_CLIENT_LIST")
    net_name = d.intern_atom("_NET_WM_NAME")
    utf8 = d.intern_atom("UTF8_STRING")
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        prop = root.get_full_property(net_list, X.AnyPropertyType)
        for wid in (prop.value if prop else []):
            win = d.create_resource_object("window", wid)
            try:
                name = win.get_full_property(net_name, utf8)
                title = name.value.decode(errors="ignore") if name else (win.get_wm_name() or "")
            except Exception:
                continue
            if title_part in str(title):
                time.sleep(0.5)  # 창이 다 뜬 뒤에 옮겨야 창 관리자가 받아들인다
                win.configure(x=x, y=y, width=w, height=h)
                d.sync()
                return
        time.sleep(0.3)
    log.warning("Rerun 창을 찾지 못해 자리를 못 옮김")


class JointPlot:
    def __init__(self, geometry: str | None, log) -> None:
        import rerun as rr
        import rerun.blueprint as rrb

        self._rr = rr
        self._log = log
        self._last = 0.0
        self._t0 = time.monotonic()
        port = _free_port()
        cmd = [sys.executable, "-m", "rerun", "--port", str(port), "--memory-limit", MEMORY_LIMIT]
        if geometry:
            x, y, w, h = (int(float(v)) for v in geometry.split(","))
            cmd += ["--window-size", f"{w}x{h}"]
            threading.Thread(target=_place_window, args=("Rerun", x, y, w, h, log), daemon=True).start()
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        if not _wait_port(port, 15.0):
            raise RuntimeError("Rerun 창이 뜨지 않았습니다")
        rr.init("aivision_teleop")
        rr.connect_grpc(f"rerun+http://127.0.0.1:{port}/proxy")
        views = []
        for n in URDF_JOINT_NAMES:
            name = LEROBOT_FROM_URDF[n]
            views.append(
                rrb.TimeSeriesView(
                    name=name,
                    origin=f"/{name}",
                    overrides={
                        f"/{name}/leader": rr.SeriesLines(colors=LEADER_COLOR, names="leader"),
                        f"/{name}/follower": rr.SeriesLines(colors=FOLLOWER_COLOR, names="follower"),
                    },
                )
            )
        rr.send_blueprint(rrb.Blueprint(rrb.Grid(*views, grid_columns=4), collapse_panels=True))

    def log(self, leader: dict[str, float] | None, follower: dict[str, float]) -> None:
        now = time.monotonic()
        if now - self._last < 1.0 / PLOT_HZ:
            return
        self._last = now
        rr = self._rr
        rr.set_time("time", duration=now - self._t0)
        for n in URDF_JOINT_NAMES:
            name = LEROBOT_FROM_URDF[n]
            if leader is not None:
                rr.log(f"/{name}/leader", rr.Scalars(float(leader[n])))
            rr.log(f"/{name}/follower", rr.Scalars(float(follower[n])))

    def close(self) -> None:
        try:
            self._rr.disconnect()
        except Exception:
            pass
        import os
        import signal

        try:
            os.killpg(self._proc.pid, signal.SIGTERM)
        except Exception:
            pass
