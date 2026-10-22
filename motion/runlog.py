"""실행 기록. 펜던트 · 학생 코드(Arm) · 카메라 창이 같은 날짜 파일에 남긴다.

  ~/aivision_logs/2026-10-22.log
  2026-10-22 14:03:11 [pendant] INFO  Connect 성공 ...

학생 보드에는 AI 가 없다. 문제가 생기면 이 파일을 받아 원인을 본다.
화면에 찍는 print 는 그대로 두고, 고장 · 명령 · 결과만 여기에 쓴다.

  from motion import runlog
  runlog.setup("pendant")                 # 프로그램 시작에서 한 번
  log = runlog.get("controller")          # 모듈마다
  log.warning("E-stop ...")

setup() 을 부르지 않은 프로그램(시험 스크립트 등)에서는 아무 데도 쓰지 않는다.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# 시험할 때는 AIVISION_LOG_DIR 로 다른 폴더를 쓴다.
LOG_DIR = Path(os.environ.get("AIVISION_LOG_DIR") or Path.home() / "aivision_logs")
# 이보다 오래된 날짜 파일은 setup() 때 지운다.
KEEP_DAYS = 30
_ROOT = "aivision"
_PROJECT = Path(__file__).resolve().parents[1]

_root = logging.getLogger(_ROOT)
_root.addHandler(logging.NullHandler())
_root.propagate = False
_setup_lock = threading.Lock()
_log_path: Path | None = None


def get(name: str) -> logging.Logger:
    """모듈별 기록기. setup() 전에는 아무 데도 쓰지 않는다."""
    return logging.getLogger(f"{_ROOT}.{name}")


def path() -> Path | None:
    """지금 쓰는 로그 파일. setup() 전이면 None."""
    return _log_path


def setup(program: str) -> Path | None:
    """오늘 날짜 파일에 [program] 표시로 기록을 시작한다. 여러 번 불러도 한 번만 한다.

    폴더를 못 만들면 (권한 등) 기록 없이 계속한다 — 로그 때문에 프로그램이 멈추면 안 된다.
    """
    global _log_path
    with _setup_lock:
        if _log_path is not None:
            return _log_path
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            _prune_old()
            target = LOG_DIR / f"{time.strftime('%Y-%m-%d')}.log"
            handler = logging.FileHandler(target, encoding="utf-8")
        except OSError as exc:
            print(f"로그 파일을 열지 못했습니다 ({exc}). 기록 없이 계속합니다.")
            return None
        handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s [{program}] %(levelname)-7s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        _root.addHandler(handler)
        _root.setLevel(logging.INFO)
        _log_path = target
    get("start").info(
        "시작  보드=%s  코드=%s  python=%s  명령=%s",
        socket.gethostname(),
        _git_revision(),
        sys.version.split()[0],
        " ".join(sys.argv),
    )
    _install_excepthooks()
    return _log_path


def _git_revision() -> str:
    """지금 코드의 커밋 (짧게). 고친 파일이 있으면 +수정 을 붙인다."""
    try:
        rev = subprocess.run(
            ["git", "-C", str(_PROJECT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(_PROJECT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "?"
    return f"{rev}+수정" if dirty else (rev or "?")


def _prune_old() -> None:
    cutoff = time.time() - KEEP_DAYS * 86400
    for old in LOG_DIR.glob("*.log"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass


def _install_excepthooks() -> None:
    """처리 안 된 예외(메인 · 스레드)도 파일에 남긴다. 원래 동작(화면 출력)은 그대로."""
    log = get("crash")
    prev_sys = sys.excepthook
    prev_thread = threading.excepthook

    def on_sys(exc_type, exc, tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            log.error("처리 안 된 예외", exc_info=(exc_type, exc, tb))
        prev_sys(exc_type, exc, tb)

    def on_thread(args):
        if args.exc_type is not SystemExit:
            name = getattr(args.thread, "name", "?")
            log.error("스레드 %s 예외", name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        prev_thread(args)

    sys.excepthook = on_sys
    threading.excepthook = on_thread
