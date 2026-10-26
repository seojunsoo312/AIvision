"""창(OpenCV · Tk)을 띄우기 전에 부르는 환경 정리. 카메라·학습 창이 쓸데없는 경고를 내지 않게 한다.

  quiet_gtk()   GTK 소리 모듈(canberra) 경고를 끄고, conda opencv 의 Qt 에 시스템 글꼴을 알려 준다.
"""

from __future__ import annotations

import os
from pathlib import Path


def _system_font_dir() -> Path | None:
    for fonts in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts/truetype"),
    ):
        if fonts.is_dir() and any(fonts.glob("*.ttf")):
            return fonts
    return None


def _link_cv2_qt_fonts(fonts: Path) -> None:
    """opencv-python 의 Qt 는 fontconfig 가 없고 cv2/qt/fonts 만 본다."""
    import sys

    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    qt_dirs = [
        Path(sys.executable).resolve().parent.parent / "lib" / ver / "site-packages" / "cv2" / "qt",
    ]
    for entry in sys.path:
        qt_dirs.append(Path(entry) / "cv2" / "qt")
    target = fonts.resolve()
    for qt in qt_dirs:
        if not qt.is_dir():
            continue
        dest = qt / "fonts"
        try:
            if dest.is_symlink() and dest.resolve() == target:
                return
            if dest.exists():
                return
            dest.symlink_to(target, target_is_directory=True)
        except OSError:
            continue
        return


def quiet_gtk() -> None:
    for key in ("GTK_MODULES", "GTK3_MODULES"):
        raw = os.environ.get(key)
        if not raw:
            continue
        os.environ[key] = ":".join(
            p for p in raw.split(":") if p and "canberra" not in p.lower()
        )
    # conda opencv-python 의 Qt 는 글꼴을 안 넣고, cv2/qt/fonts 가 없으면 경고만 낸다.
    fonts = _system_font_dir()
    if fonts is None:
        return
    os.environ["QT_QPA_FONTDIR"] = str(fonts)
    _link_cv2_qt_fonts(fonts)
