#!/usr/bin/env python3
"""설치가 제대로 됐는지 항목별로 점검한다. AIvision 환경의 python 으로 실행한다.

  ~/miniforge3/envs/AIvision/bin/python ~/AIvision/deploy/check_board.py
  (conda activate AIvision 뒤면 python deploy/check_board.py)

팔·카메라가 안 꽂혀 있으면 그 항목은 '경고'로 둔다. '실패'가 있으면 종료 코드 1.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

PASS, WARN, FAIL = "통과", "경고", "실패"
_COLOR = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}
results: list[tuple[str, str, str]] = []


def check(name: str):
    def deco(fn):
        try:
            status, detail = fn()
        except Exception as exc:  # 점검 하나가 터져도 나머지는 계속한다
            status, detail = FAIL, f"{type(exc).__name__}: {exc}"
        results.append((status, name, str(detail)))
        print(f"  {_COLOR[status]}{status}\033[0m  {name}  —  {detail}", flush=True)
        return fn
    return deco


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


print("== 보드")


@check("JetPack 6 (R36)")
def _():
    s = Path("/etc/nv_tegra_release").read_text().splitlines()[0]
    return (PASS if "R36" in s else FAIL), s[:40]


@check("전원 모드 MAXN")
def _():
    m = re.search(r"NV Power Mode: (\S+)", sh("nvpmodel -q 2>/dev/null"))
    mode = m.group(1) if m else "?"
    return (PASS if mode.startswith("MAXN") else WARN), mode


@check("스왑 8GB 이상")
def _():
    total = sum(int(x) for x in sh("swapon --show=SIZE --bytes --noheadings").split()) / 2**30
    return (PASS if total >= 8 else WARN), f"{total:.1f} GB"


@check("한글 글꼴 (Noto Sans CJK)")
def _():
    n = int(sh('fc-list | grep -c "Noto Sans CJK"') or 0)
    return (PASS if n else FAIL), f"{n}개"


@check("시리얼·카메라 그룹 (dialout, video)")
def _():
    groups = set(sh("id -nG").split())
    miss = {"dialout", "video"} - groups
    return (PASS, "있음") if not miss else (WARN, f"없음 {sorted(miss)} — 다시 로그인 필요")


print("== 카메라")


@check("Orbbec SDK (libOrbbecSDK.so)")
def _():
    libs = glob.glob(str(PROJECT.parent / "Orbbec*" / "libOrbbecSDK.so")) + glob.glob(
        str(PROJECT.parent / "Orbbec*" / "lib" / "libOrbbecSDK.so")
    )
    return (PASS, libs[0]) if libs else (FAIL, f"{PROJECT.parent}/Orbbec* 에 없음")


@check("카메라 USB 권한 규칙")
def _():
    p = Path("/etc/udev/rules.d/99-obsensor-libusb.rules")
    return (PASS, str(p)) if p.exists() else (FAIL, "없음")


@check("카메라 연결 (lsusb 2bc5)")
def _():
    n = len([l for l in sh("lsusb").splitlines() if "2bc5" in l])
    return (PASS, f"{n}개 장치") if n >= 2 else (WARN, f"{n}개 — 카메라를 꽂으세요")


print("== Python 환경")


@check("Python 3.10")
def _():
    v = sys.version.split()[0]
    return (PASS if sys.version_info[:2] == (3, 10) else FAIL), f"{v}  ({sys.executable})"


@check("torch 2.8.0 + GPU")
def _():
    import torch

    ok = torch.__version__.startswith("2.8.0") and torch.cuda.is_available()
    return (PASS if ok else FAIL), f"{torch.__version__}  cuda={torch.cuda.is_available()}"


@check("numpy 1.26 과 torch 변환")
def _():
    import numpy as np
    import torch

    torch.from_numpy(np.ones(2)).cuda().cpu().numpy()
    return (PASS if np.__version__.startswith("1.26") else FAIL), np.__version__


@check("lerobot 7축 fork")
def _():
    import importlib.metadata as md
    import inspect
    import json

    import lerobot.robots.so_follower as so

    commit = json.loads(md.distribution("lerobot").read_text("direct_url.json"))["vcs_info"]["commit_id"]
    want = re.search(r"lerobot\.git@([0-9a-f]+)", (PROJECT / "docs/requirements.txt").read_text()).group(1)
    dof = "dof_mode" in inspect.signature(so.SO101FollowerConfig).parameters
    ok = dof and commit == want
    return (PASS if ok else FAIL), f"커밋 {commit[:8]} (문서 {want[:8]})  dof_mode={dof}"


@check("opencv 창 지원 (headless 아님)")
def _():
    import importlib.metadata as md

    import cv2

    try:
        md.version("opencv-python-headless")
        return FAIL, "opencv-python-headless 가 깔려 있음"
    except md.PackageNotFoundError:
        pass
    if not os.environ.get("DISPLAY"):
        return WARN, f"{cv2.__version__}  (화면 없음: 창 시험 생략)"
    cv2.namedWindow("check")
    cv2.destroyAllWindows()
    return PASS, cv2.__version__


@check("ultralytics, open3d, pinocchio, coal, meshcat, customtkinter")
def _():
    import coal  # noqa: F401
    import customtkinter  # noqa: F401
    import meshcat  # noqa: F401
    import open3d  # noqa: F401
    import pinocchio  # noqa: F401
    import ultralytics

    return PASS, f"ultralytics {ultralytics.__version__}"


@check("노트북 커널 AIvision")
def _():
    out = sh(f"{sys.executable} -m jupyter kernelspec list")
    return (PASS, "있음") if re.search(r"(?im)^\s*aivision\s", out) else (FAIL, "없음")


@check("펜던트 한글 글꼴 (Tk Xft)")
def _():
    if not os.environ.get("DISPLAY"):
        return WARN, "화면 없음: 생략"
    # 앞 점검의 customtkinter 가 tkinter 를 이미 불러왔으므로 새 프로세스에서 본다.
    out = subprocess.run(
        [sys.executable, "-c", "import ui_style; print(ui_style.enable_xft_tk())"],
        cwd=PROJECT / "pendant", capture_output=True, text=True,
    ).stdout.strip()
    return (PASS if out == "True" else FAIL), f"enable_xft_tk={out or '오류'}"


@check("가상 팔 (Arm virtual)")
def _():
    from motion import Arm

    a = Arm(open_browser=False)
    try:
        a.initial()
        a.moveL([10, 0, 0], "rel")
        xyz = a.where()["xyz_mm"]
    finally:
        a.close()
    return PASS, f"TCP {[round(v) for v in xyz]} mm"


print("== 보드마다 따로 하는 것")


@check("YOLO 가중치 yolo/weights/best.pt")
def _():
    p = PROJECT / "yolo/weights/best.pt"
    return (PASS, f"{p.stat().st_size // 1024} KB") if p.exists() else (WARN, "없음 — USB 로 복사")


@check("팔 시리얼 이름 (/dev/so101_*)")
def _():
    have = [n for n in ("so101_follower", "so101_leader") if os.path.exists(f"/dev/{n}")]
    if len(have) == 2:
        return PASS, "follower, leader"
    rules = Path("/etc/udev/rules.d/99-serial.rules")
    hint = "규칙은 있음 — 팔을 꽂으세요" if rules.exists() else "bash deploy/make_serial_rules.sh"
    return WARN, f"보임 {have or '없음'} — {hint}"


@check("팔 캘리브 (follower, leader)")
def _():
    base = Path.home() / ".cache/huggingface/lerobot/calibration"
    files = {
        "follower": base / "robots/so_follower/follower.json",
        "leader": base / "teleoperators/so_leader/leader.json",
    }
    have = [k for k, p in files.items() if p.exists()]
    if len(have) == 2:
        return PASS, "둘 다 있음"
    return WARN, f"있음 {have or '없음'} — bash deploy/calibrate_arms.sh"


@check("카메라 + YOLO (GPU)")
def _():
    weights = PROJECT / "yolo/weights/best.pt"
    if not weights.exists() or len([l for l in sh("lsusb").splitlines() if "2bc5" in l]) < 2:
        return WARN, "카메라 또는 best.pt 없음: 생략"
    if sh("pgrep -f 'roi_cloud.py|yolo/train/capture.py|OrbbecViewer'"):
        return WARN, "카메라를 쓰는 다른 프로그램이 켜져 있어 생략"
    sys.path.insert(0, str(PROJECT / "pendant"))
    from ultralytics import YOLO

    from vision.rgbd import OrbbecV1

    cam = OrbbecV1()
    try:
        frames = [cam.grab() for _ in range(15)]
    finally:
        cam.close()
    ok = [f for f in frames if f[0] is not None and f[1] is not None]
    if not ok:
        return FAIL, "프레임을 못 받음"
    r = YOLO(str(weights)).predict(ok[-1][0], device=0, verbose=False)
    dev = str(r[0].boxes.data.device)
    return (PASS if dev.startswith("cuda") else FAIL), f"컬러 {ok[-1][0].shape} 뎁스 {ok[-1][1].shape} YOLO {dev}"


n_fail = sum(1 for s, *_ in results if s == FAIL)
n_warn = sum(1 for s, *_ in results if s == WARN)
print(f"\n결과: 통과 {len(results) - n_fail - n_warn} · 경고 {n_warn} · 실패 {n_fail}")
if n_fail:
    print("실패 항목을 고친 뒤 bash deploy/setup_jetson.sh 를 다시 실행하세요 (끝난 단계는 건너뜁니다).")
sys.exit(1 if n_fail else 0)
