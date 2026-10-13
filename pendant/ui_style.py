"""펜던트 UI 글꼴과 버튼 기본값.

conda 의 Tk 는 Xft 없이 빌드되어 Noto 를 못 보고, 한글이 비트맵 글꼴
``fixed`` 로 나온다. 시스템 Tk 는 Xft 를 쓴다. tkinter 보다 먼저 그 라이브러리를
읽고, Tk 가 맞는 시스템 스크립트 폴더를 보게 한다.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# 노트북은 x86_64-linux-gnu, Jetson 은 aarch64-linux-gnu.
_SYS_TK = next(iter(sorted(Path("/usr/lib").glob("*-linux-gnu/libtk8.6.so"))), Path("/usr/lib/libtk8.6.so"))
_SYS_TK_LIB = Path("/usr/share/tcltk/tk8.6")
_SYS_TCL_LIB = Path("/usr/share/tcltk/tcl8.6")

UI_FAMILY = "Noto Sans CJK KR"
MONO_FAMILY = "Noto Sans Mono CJK KR"

_enabled = False


def enable_xft_tk() -> bool:
    """tkinter 를 import 하기 전에 시스템 Xft Tk 를 읽는다. 여러 번 불러도 된다."""
    global _enabled
    if _enabled:
        return True
    if not sys.platform.startswith("linux"):
        return False
    if not _SYS_TK.is_file() or not (_SYS_TK_LIB / "tk.tcl").is_file():
        return False
    if "tkinter" in sys.modules or "_tkinter" in sys.modules:
        return False
    os.environ["TK_LIBRARY"] = str(_SYS_TK_LIB)
    if (_SYS_TCL_LIB / "init.tcl").is_file():
        os.environ["TCL_LIBRARY"] = str(_SYS_TCL_LIB)
    ctypes.CDLL(str(_SYS_TK), mode=ctypes.RTLD_GLOBAL)
    _enabled = True
    return True


def apply_ui_theme() -> None:
    """CustomTkinter 기본 글꼴을 Noto Sans CJK KR 로, 버튼은 조금 더 둥글게."""
    import customtkinter as ctk
    from customtkinter.windows.widgets.theme import ThemeManager

    if not ThemeManager.theme:
        ctk.set_default_color_theme("blue")
    font = ThemeManager.theme.get("CTkFont")
    if isinstance(font, dict) and "family" in font:
        font["family"] = UI_FAMILY
        font["size"] = 14
    button = ThemeManager.theme.get("CTkButton")
    if isinstance(button, dict):
        button["corner_radius"] = 10


def apply_root_fonts(root) -> None:
    """이름 있는 Tk 글꼴 (입력 칸, 대화 상자) 도 같은 글꼴로 맞춘다."""
    import tkinter as tk
    import tkinter.font as tkfont

    for name, family, size in (
        ("TkDefaultFont", UI_FAMILY, 11),
        ("TkTextFont", UI_FAMILY, 11),
        ("TkMenuFont", UI_FAMILY, 11),
        ("TkHeadingFont", UI_FAMILY, 12),
        ("TkCaptionFont", UI_FAMILY, 12),
        ("TkSmallCaptionFont", UI_FAMILY, 10),
        ("TkIconFont", UI_FAMILY, 10),
        ("TkTooltipFont", UI_FAMILY, 10),
        ("TkFixedFont", MONO_FAMILY, 11),
    ):
        try:
            tkfont.nametofont(name).configure(family=family, size=size)
        except tk.TclError:
            pass
    root.option_add("*Font", f"{{{UI_FAMILY}}} 11")


def ui_font(size: int | None = None, weight: str = "normal"):
    import customtkinter as ctk

    kwargs: dict = {"family": UI_FAMILY, "weight": weight}
    if size is not None:
        kwargs["size"] = size
    return ctk.CTkFont(**kwargs)


def mono_font(size: int = 14):
    import customtkinter as ctk

    return ctk.CTkFont(family=MONO_FAMILY, size=size)
