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


BUTTON_CORNER_RADIUS = 6


def apply_ui_theme() -> None:
    """CustomTkinter 기본 글꼴을 Noto Sans CJK KR 로, 버튼 모서리는 작게."""
    import customtkinter as ctk
    from customtkinter.windows.widgets.theme import ThemeManager
    from customtkinter.windows.widgets.core_rendering import DrawEngine

    # 둥근 모서리를 다각형으로 그린다. 리눅스 기본 (글꼴 모양) 은 0.9 같은 소수
    # 배율에서 아래 모서리가 버튼 몸통과 1px 어긋나 점처럼 튀어나온다.
    # 다각형은 계단이 지므로 반지름을 작게 (BUTTON_CORNER_RADIUS) 둔다.
    DrawEngine.preferred_drawing_method = "polygon_shapes"

    if not ThemeManager.theme:
        ctk.set_default_color_theme("blue")
    font = ThemeManager.theme.get("CTkFont")
    if isinstance(font, dict) and "family" in font:
        font["family"] = UI_FAMILY
        font["size"] = 14
    button = ThemeManager.theme.get("CTkButton")
    if isinstance(button, dict):
        # 0.9 같은 소수 배율에서 큰 반지름은 아래 모서리가 1px 어긋나 찌그러져 보인다.
        button["corner_radius"] = BUTTON_CORNER_RADIUS
    # 슬라이더 진행 막대를 트랙과 같은 색으로. 소수 배율에서 진행 막대가
    # 양 끝 회색 조각으로만 그려져 손잡이 뒤에 겹쳐 보인다.
    _default_button_font()
    slider = ThemeManager.theme.get("CTkSlider")
    if isinstance(slider, dict) and "fg_color" in slider:
        slider["progress_color"] = slider["fg_color"]


# 기본 버튼 글자 크기. 14 는 0.9 배율에서 글자 칸(24px)이 버튼(26px)을 거의 채워
# 글자 배경이 버튼 아래로 1~2px 삐져나온다. 글꼴을 직접 준 버튼은 그대로.
BUTTON_FONT_SIZE = 13


def _default_button_font() -> None:
    """font 를 안 준 CTkButton 에 BUTTON_FONT_SIZE 글꼴을 넣는다. 여러 번 불러도 된다."""
    import customtkinter as ctk

    if getattr(ctk.CTkButton.__init__, "_ui_style_patched", False):
        return
    orig = ctk.CTkButton.__init__

    def init(self, *args, **kwargs):
        if kwargs.get("font") is None:
            kwargs["font"] = ctk.CTkFont(family=UI_FAMILY, size=BUTTON_FONT_SIZE)
        orig(self, *args, **kwargs)

    init._ui_style_patched = True
    ctk.CTkButton.__init__ = init


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


# 펜던트 원래 크기 (배율 1.0, gui_app.py 의 geometry).
PENDANT_BASE_W = 980
PENDANT_BASE_H = 1126
# 기본 배율. --scale 로 바꾼다.
DEFAULT_UI_SCALE = 0.9


def apply_ui_scale(root, scale: float | None = None) -> float:
    """창 크기와 글자를 함께 줄이거나 키운다. 위젯을 만들기 전에 부른다."""
    import customtkinter as ctk

    scale = float(DEFAULT_UI_SCALE if scale is None else scale)
    ctk.set_widget_scaling(scale)
    ctk.set_window_scaling(scale)
    return scale
