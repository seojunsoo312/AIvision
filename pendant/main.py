#!/usr/bin/env python
"""관절 + 직교 조그, 메시 충돌, 절대 go-to. 실기는 Connect 로 연결한다."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
for path in (PROJECT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ui_style import apply_ui_scale, apply_ui_theme, enable_xft_tk

enable_xft_tk()

import customtkinter as ctk
import numpy as np

apply_ui_theme()

from gui_app import PendantGui
from motion import (
    DEFAULT_PORT,
    DEFAULT_ROBOT_ID,
    DEFAULT_URDF,
    Controller,
    Hardware,
    RobotKinematics,
    runlog,
)
from motion.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO101_6DOF pendant — joint / Cartesian jog and go-to")
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--no-open", action="store_true", help="Do not auto-open the Meshcat browser")
    p.add_argument(
        "--tcp-offset-mm",
        default=None,
        help="TCP in L6_1 (gripper body) as x,y,z mm. Default: fixed/moving jaw-tip midpoint. Use 0,0,0 for wrist_roll origin.",
    )
    p.add_argument("--port", default=DEFAULT_PORT, help="Feetech serial port")
    p.add_argument("--robot-id", default=DEFAULT_ROBOT_ID, help="LeRobot robot id (calib file stem)")
    p.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help=(
            "Directory with {robot-id}.json. Default: lerobot-calibrate cache "
            "(~/.cache/huggingface/lerobot/calibration/robots/so_follower)"
        ),
    )
    p.add_argument("--dof-mode", type=int, default=7, choices=(6, 7))
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help="펜던트 창·글자 배율 (기본 0.9, 작을수록 작아짐)",
    )
    p.add_argument(
        "--grasp",
        action="store_true",
        help="사물 위치 패널을 조작 가능하게 연다. 없으면 같은 패널이 잠긴다",
    )
    return p.parse_args()


def parse_tcp_offset(raw: str | None) -> np.ndarray | None:
    if raw is None:
        return None
    parts = [float(x) for x in raw.split(",")]
    if len(parts) != 3:
        raise SystemExit("--tcp-offset-mm needs x,y,z")
    return np.array(parts, dtype=float) / 1000.0


# Meshcat 창을 펜던트 오른쪽에 열 때 쓸 브라우저 (앞에 있는 것부터).
_APP_BROWSERS = ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium")
# 펜던트 오른쪽 빈자리가 이보다 좁으면 그냥 기본 브라우저로 연다.
_MIN_BESIDE_W = 320


def _open_meshcat_beside(root, url: str) -> None:
    """Meshcat 을 Chrome 앱 창 (주소창 없음) 으로 펜던트 오른쪽 빈자리에 맞춰 연다."""
    root.update_idletasks()
    x = root.winfo_rootx() + root.winfo_width() + 4
    y = max(0, root.winfo_y())
    w = root.winfo_screenwidth() - x
    h = root.winfo_height()
    browser = next((b for b in map(shutil.which, _APP_BROWSERS) if b), None)
    if browser is None or w < _MIN_BESIDE_W:
        webbrowser.open(url)
        return
    before = _app_windows()
    subprocess.Popen(
        [browser, f"--app={url}", f"--window-position={x},{y}", f"--window-size={w},{h}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Chrome 이 이미 켜져 있으면 --window-position 을 무시한다. 새 창이 뜨면 직접 옮긴다.
    threading.Thread(target=_move_new_app_window, args=(before, x, y, w, h), daemon=True).start()


# Chrome --app 창의 WM_CLASS 이름 (주소 127.0.0.1/static → "127.0.0.1__static").
_APP_WM_CLASS = "127.0.0.1__static"


def _app_windows() -> set[int]:
    """지금 떠 있는 Meshcat 앱 창 id. python-xlib 이 없으면 빈 집합."""
    try:
        from Xlib import display as xdisplay
    except ImportError:
        return set()
    d = xdisplay.Display()
    try:
        return {win.id for win in _iter_windows(d.screen().root) if _is_app_window(win)}
    finally:
        d.close()


def _iter_windows(win, depth: int = 0):
    """창 트리를 두 단계까지 (창 관리자 틀 → 실제 창)."""
    try:
        children = win.query_tree().children
    except Exception:
        return
    for child in children:
        yield child
        if depth < 1:
            yield from _iter_windows(child, depth + 1)


def _is_app_window(win) -> bool:
    try:
        cls = win.get_wm_class()
    except Exception:
        return False
    return bool(cls) and _APP_WM_CLASS in cls


def _move_new_app_window(before: set[int], x: int, y: int, w: int, h: int) -> None:
    """새로 뜬 Meshcat 앱 창을 (x, y, w, h) 로 옮긴다. 8초 안에 안 뜨면 그만둔다."""
    try:
        from Xlib import display as xdisplay
    except ImportError:
        return
    d = xdisplay.Display()
    try:
        for _ in range(32):
            time.sleep(0.25)
            new = [win for win in _iter_windows(d.screen().root) if _is_app_window(win) and win.id not in before]
            if new:
                for win in new:
                    win.configure(x=x, y=y, width=w, height=h)
                d.sync()
                return
    finally:
        d.close()


def _report_tk_error(exc_type, exc, tb) -> None:
    """버튼 콜백 예외: 원래처럼 터미널에 찍고, 로그 파일에도 남긴다."""
    import traceback

    traceback.print_exception(exc_type, exc, tb)
    runlog.get("gui").error("버튼 처리 중 예외", exc_info=(exc_type, exc, tb))


def main() -> None:
    args = parse_args()
    log_path = runlog.setup("pendant")
    kin = RobotKinematics(args.urdf, tcp_offset=parse_tcp_offset(args.tcp_offset_mm))
    q = kin.q_home()
    open_browser = not args.no_open
    # 브라우저는 펜던트 창이 뜬 뒤 그 오른쪽에 연다 (_open_meshcat_beside).
    viz = Visualizer(kin, open_browser=False)
    viz.display(q)
    if viz.url:
        # 창에는 주소를 띄우지 않는다. 브라우저를 닫았으면 이 주소로 다시 연다.
        print(f"Meshcat  {viz.url}")
    if log_path is not None:
        print(f"로그     {log_path}")

    hw = Hardware(
        port=args.port,
        robot_id=args.robot_id,
        calibration_dir=args.calibration_dir,
        dof_mode=args.dof_mode,
    )
    ctrl = Controller(kin, hardware=hw)
    ctrl.start()

    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    apply_ui_scale(root, args.scale)
    root.report_callback_exception = _report_tk_error
    PendantGui(root, ctrl, viz, meshcat_url=viz.url or "", grasp=args.grasp)
    if open_browser and viz.url:
        root.after(400, lambda: _open_meshcat_beside(root, viz.url))
    root.mainloop()


if __name__ == "__main__":
    main()
