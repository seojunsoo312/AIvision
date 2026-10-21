"""관절 + 직교 조그와 절대 go-to."""

from __future__ import annotations

import sys
from pathlib import Path

# pendant/ 와 프로젝트 폴더를 import 경로에 넣는다 (python pendant/main.py 로 실행해도 되게).
_PENDANT = Path(__file__).resolve().parent
_PROJECT = _PENDANT.parent
for path in (_PROJECT, _PENDANT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ui_style import (
    PENDANT_BASE_H,
    PENDANT_BASE_W,
    apply_root_fonts,
    apply_ui_theme,
    enable_xft_tk,
    mono_font,
    ui_font,
)

# 한글 글꼴: tkinter 를 import 하기 전에 시스템 Xft Tk 를 먼저 읽어야 한다.
enable_xft_tk()

import time
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import numpy as np

apply_ui_theme()

from motion.controller import (
    ALREADY_AT_TARGET,
    DEFAULT_SPEED_PCT,
    JOG_ROT_DEG_S,
    JOG_ROT_MAX_DEG_S,
    JOG_ROT_MIN_DEG_S,
    JOG_VEL_MAX_MPS,
    JOG_VEL_MIN_MPS,
    JOG_VEL_MPS,
    ROT_FRAME_BASE,
    ROT_FRAME_TCP,
    SPEED_PCT_MAX,
    SPEED_PCT_MIN,
    SPEED_PRESETS,
    Controller,
    speed_pct_to_joint_vel,
)
from motion.hw_controller import grip_100_to_user, grip_user_to_100
from motion.robot_kinematics import GRIPPER_JOINT, LEROBOT_FROM_URDF, SHUTDOWN_JOINTS_DEG, URDF_JOINT_NAMES
from motion.visualizer import Visualizer

# 화면 갱신 간격 (ms). 약 30 fps.
DISPLAY_MS = 33


def _set_text(widget, text: str) -> None:
    """글자가 바뀔 때만 configure. CTk 위젯은 configure 마다 다시 그린다."""
    if widget.cget("text") != text:
        widget.configure(text=text)


def _joint_ui_name(name: str) -> str:
    """펜던트 표시 이름: S1–S6 → J1–J6. 그리퍼는 S7 그대로."""
    if name.startswith("S") and name[1:].isdigit() and 1 <= int(name[1:]) <= 6:
        return f"J{name[1:]}"
    return name


# Rx/Ry/Rz 글자색. 지금은 모두 흰색 (기준 좌표에 따라 색을 바꿀 자리).
_RPY_COLOR_WHITE = "#FFFFFF"
_RPY_COLOR = {
    "Rx": _RPY_COLOR_WHITE,
    "Ry": _RPY_COLOR_WHITE,
    "Rz": _RPY_COLOR_WHITE,
    "roll": _RPY_COLOR_WHITE,
    "pitch": _RPY_COLOR_WHITE,
    "yaw": _RPY_COLOR_WHITE,
}


class PendantGui:
    """펜던트 창. 왼쪽 관절, 가운데 TCP, 오른쪽 두 칸은 사물 위치 패널 (teach_grasp).

    화면은 DISPLAY_MS 마다 컨트롤러 상태를 읽어 다시 그린다. 실제 제어는 Controller 가 한다.
    """

    def __init__(
        self,
        root: ctk.CTk,
        controller: Controller,
        visualizer: Visualizer,
        *,
        meshcat_url: str = "",
        grasp: bool = False,
    ) -> None:
        self.root = root
        self._ctrl = controller
        self._viz = visualizer
        self._closing = False
        self._closed = False
        self._grasp_enabled = bool(grasp)
        self._teach = None
        self.root.title("로봇 컨트롤러")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        apply_root_fonts(self.root)

        # --- 맨 위: 왼쪽 = 안내 한 줄과 현재 TCP·오차·모드, 오른쪽 = 큰 버튼 (_build_top_buttons) ---
        header = ctk.CTkFrame(root, fg_color="transparent")
        header.pack(side="top", fill="x", padx=12, pady=(8, 6))
        info = ctk.CTkFrame(header, fg_color="transparent")
        info.pack(side="left", anchor="n")
        self._top_btns = ctk.CTkFrame(header, fg_color="transparent")
        self._top_btns.pack(side="right", anchor="n")

        # 상태 글자 4줄. 줄 높이를 줄여 오른쪽 큰 버튼 두 줄과 비슷한 높이로 맞춘다.
        line = {"height": 22, "anchor": "w"}
        top_row = ctk.CTkFrame(info, fg_color="transparent")
        top_row.pack(side="top", anchor="w")
        self.mode_label = ctk.CTkLabel(top_row, text="mode: virtual", font=mono_font(), **line)
        self.mode_label.pack(side="left")
        ctk.CTkLabel(
            top_row, text=f"  ·  Meshcat: {meshcat_url or 'printed URL'}", **line
        ).pack(side="left")
        self.fault_label = ctk.CTkLabel(top_row, text="", text_color="#f87171", **line)
        self.fault_label.pack(side="left", padx=(12, 0))

        self.tcp_label = ctk.CTkLabel(info, text="TCP: —", font=mono_font(14), **line)
        self.rpy_label = ctk.CTkLabel(info, text="RxRyRz: —", font=mono_font(14), **line)
        self.err_label = ctk.CTkLabel(info, text="err:  (virtual)", font=mono_font(14), **line)
        for lab in (self.tcp_label, self.rpy_label, self.err_label):
            lab.pack(side="top", anchor="w")

        # --- 본문 2×2: 위 = 관절 | TCP, 아래 = 사물 위치 | 사물 위치 동작 버튼 ---
        # 작은 모니터에서도 오른쪽에 Meshcat 브라우저 자리가 남도록 세로로 쌓는다.
        body = ctk.CTkFrame(root, fg_color="transparent")
        body.grid_columnconfigure(0, weight=1, uniform="col")
        body.grid_columnconfigure(1, weight=1, uniform="col")

        self.root.minsize(PENDANT_BASE_W - 100, PENDANT_BASE_H - 200)
        self.root.geometry(f"{PENDANT_BASE_W}x{PENDANT_BASE_H}+0+0")  # 왼쪽 위. 오른쪽은 Meshcat 자리
        left = ctk.CTkFrame(body)
        right = ctk.CTkFrame(body)
        self._grasp_col = ctk.CTkFrame(body)
        self._grasp_act_col = ctk.CTkFrame(body)
        left.grid(row=0, column=0, sticky="new", padx=(0, 6))
        right.grid(row=0, column=1, sticky="new", padx=(6, 0))
        self._grasp_col.grid(row=1, column=0, sticky="new", padx=(0, 6), pady=(12, 0))
        self._grasp_act_col.grid(row=1, column=1, sticky="new", padx=(6, 0), pady=(12, 0))

        # --- 왼쪽 칸: 관절마다 [이름, 현재값, −, +, 목표 입력칸, 단위] ---
        ctk.CTkLabel(left, text="Joints", anchor="w").pack(
            anchor="w", padx=8, pady=(8, 4)
        )
        self.joint_labels: dict[str, ctk.CTkLabel] = {}
        self.joint_entries: dict[str, ctk.CTkEntry] = {}
        for name in URDF_JOINT_NAMES:
            row = ctk.CTkFrame(left)
            row.pack(fill="x", pady=4, padx=6)
            alias = LEROBOT_FROM_URDF[name]
            unit = "0–100" if name == GRIPPER_JOINT else "deg"
            ctk.CTkLabel(
                row,
                text=f"{_joint_ui_name(name)} ({alias})",
                width=156,
                anchor="w",
                font=mono_font(),
            ).pack(side="left", padx=(4, 2))
            lab = ctk.CTkLabel(
                row,
                text="—",
                width=64,
                anchor="e",
                font=mono_font(),
            )
            lab.pack(side="left", padx=(0, 4))
            self.joint_labels[name] = lab
            self._hold_button(
                row,
                "−",
                lambda j=name: self._ctrl.set_joint_jog(j, -1),
                lambda j=name: self._ctrl.set_joint_jog(j, 0),
            )
            self._hold_button(
                row,
                "+",
                lambda j=name: self._ctrl.set_joint_jog(j, 1),
                lambda j=name: self._ctrl.set_joint_jog(j, 0),
            )
            ent = ctk.CTkEntry(row, width=72)
            ent.insert(0, "50" if name == GRIPPER_JOINT else "0")
            ent.pack(side="left", padx=(8, 4))
            self.joint_entries[name] = ent
            ctk.CTkLabel(row, text=unit, width=40, anchor="w").pack(side="left")
        # 그리퍼 (S7 아래): 누르고 있는 동안만 열리거나 닫힌다.
        grip = ctk.CTkFrame(left)
        grip.pack(fill="x", pady=4, padx=6)
        ctk.CTkLabel(grip, text="Gripper", width=70, anchor="w").pack(side="left", padx=(4, 8))
        self._hold_button(
            grip,
            "Close",
            lambda: self._ctrl.set_joint_jog(GRIPPER_JOINT, -1),
            lambda: self._ctrl.set_joint_jog(GRIPPER_JOINT, 0),
            width=8,
        )
        self._hold_button(
            grip,
            "Open",
            lambda: self._ctrl.set_joint_jog(GRIPPER_JOINT, 1),
            lambda: self._ctrl.set_joint_jog(GRIPPER_JOINT, 0),
            width=8,
        )

        # TCP의 현재값/이동 버튼과 같은 높이가 되도록 아래에서 늘린다.
        self._joint_btn_spacer = ctk.CTkFrame(left, fg_color="transparent", height=0)
        self._joint_btn_spacer.pack(fill="x")
        self._joint_btn_spacer.pack_propagate(False)
        goto_btn = ctk.CTkFrame(left, fg_color="transparent")
        self._joint_goto_row = goto_btn
        goto_btn.pack(fill="x", padx=6, pady=(8, 4))
        ctk.CTkButton(goto_btn, text="현재값 넣기", width=120, command=self._fill_joints).pack(
            side="left", padx=(0, 6)
        )
        ctk.CTkButton(goto_btn, text="관절 이동", width=110, command=self._move_joints).pack(
            side="left"
        )

        # 관절 속도: 프리셋 버튼 + 슬라이더 (%). 조그와 관절 go-to 에 같이 쓰인다.
        ctk.CTkLabel(left, text="관절 속도", anchor="w").pack(
            anchor="w", padx=8, pady=(12, 4)
        )
        speed = ctk.CTkFrame(left)
        speed.pack(fill="x", padx=6, pady=(0, 4))
        preset_row = ctk.CTkFrame(speed, fg_color="transparent")
        preset_row.pack(fill="x", padx=4, pady=(6, 2))
        for name, pct in SPEED_PRESETS:
            ctk.CTkButton(
                preset_row,
                text=name,
                width=70,
                command=lambda p=pct: self._set_speed_pct(p),
            ).pack(side="left", padx=(0, 6))
        self._speed_label = ctk.CTkLabel(
            speed,
            text=self._speed_label_text(DEFAULT_SPEED_PCT),
            anchor="w",
            font=mono_font(),
        )
        self._speed_label.pack(anchor="w", padx=4, pady=(4, 0))
        self._speed_slider = ctk.CTkSlider(
            speed,
            from_=SPEED_PCT_MIN,
            to=SPEED_PCT_MAX,
            number_of_steps=int(SPEED_PCT_MAX - SPEED_PCT_MIN),
            command=self._on_speed_slider,
        )
        self._speed_slider.set(DEFAULT_SPEED_PCT)
        self._speed_slider.pack(fill="x", padx=4, pady=(4, 8))
        self._ctrl.set_speed_pct(DEFAULT_SPEED_PCT)

        # --- 가운데 칸: TCP 축마다 [이름, 현재값, −, +, 목표 입력칸, 단위] ---
        ctk.CTkLabel(right, text="TCP", anchor="w").pack(
            anchor="w", padx=8, pady=(8, 4)
        )
        self.ee_entries: dict[str, ctk.CTkEntry] = {}
        self._xyz_frame_seg = self._jog_frame_bar(right, "XYZ jog frame", self._on_xyz_frame)
        self._tcp_live_labels: dict[str, ctk.CTkLabel] = {}
        self._rpy_jog_labels: dict[str, ctk.CTkLabel] = {}
        self._cart_row(right, "X", "x", entry_key="x", unit="mm")
        self._cart_row(right, "Y", "y", entry_key="y", unit="mm")
        self._cart_row(right, "Z", "z", entry_key="z", unit="mm")

        self._rot_frame_seg = self._jog_frame_bar(
            right, "Rx/Ry/Rz jog frame", self._on_rot_frame, top=10
        )

        self._cart_row(right, "Rx", "wx", entry_key="roll", unit="deg", color=_RPY_COLOR["Rx"], store_label=True)
        self._cart_row(right, "Ry", "wy", entry_key="pitch", unit="deg", color=_RPY_COLOR["Ry"], store_label=True)
        self._cart_row(right, "Rz", "wz", entry_key="yaw", unit="deg", color=_RPY_COLOR["Rz"], store_label=True)
        ee_btn = ctk.CTkFrame(right, fg_color="transparent")
        self._ee_goto_row = ee_btn
        ee_btn.pack(fill="x", padx=6, pady=(8, 4))
        ctk.CTkButton(ee_btn, text="현재값 넣기", width=120, command=self._fill_ee).pack(
            side="left", padx=(0, 6)
        )
        ctk.CTkButton(ee_btn, text="TCP 이동", width=110, command=self._move_ee).pack(side="left")

        # TCP 조그 속도: 프리셋 버튼 + 위치 (mm/s) · 자세 (deg/s) 슬라이더.
        vel = ctk.CTkFrame(right)
        vel.pack(fill="x", padx=6, pady=(8, 8))
        jog_preset = ctk.CTkFrame(vel, fg_color="transparent")
        jog_preset.pack(fill="x", padx=4, pady=(6, 2))
        for name, mm_s, deg_s in (
            ("느림", JOG_VEL_MIN_MPS * 1000.0, JOG_ROT_MIN_DEG_S),
            ("보통", JOG_VEL_MPS * 1000.0, JOG_ROT_DEG_S),
            ("빠름", JOG_VEL_MAX_MPS * 1000.0, JOG_ROT_MAX_DEG_S),
        ):
            ctk.CTkButton(
                jog_preset,
                text=name,
                width=70,
                command=lambda m=mm_s, d=deg_s: self._set_jog_preset(m, d),
            ).pack(side="left", padx=(0, 6))
        self._tcp_vel_label = ctk.CTkLabel(
            vel,
            text=f"XYZ jog  {JOG_VEL_MPS * 1000:.0f} mm/s",
            anchor="w",
            height=22,
        )
        self._tcp_vel_label.pack(anchor="w", padx=4, pady=(4, 0))
        self._tcp_vel_slider = ctk.CTkSlider(
            vel,
            from_=JOG_VEL_MIN_MPS * 1000.0,
            to=JOG_VEL_MAX_MPS * 1000.0,
            number_of_steps=int(JOG_VEL_MAX_MPS * 1000.0 - JOG_VEL_MIN_MPS * 1000.0),
            command=self._on_tcp_jog_speed,
        )
        self._tcp_vel_slider.set(JOG_VEL_MPS * 1000.0)
        self._tcp_vel_slider.pack(fill="x", padx=4, pady=(4, 4))
        self._ctrl.set_tcp_jog_mm_s(JOG_VEL_MPS * 1000.0)

        self._rpy_vel_label = ctk.CTkLabel(
            vel,
            text=f"Rx/Ry/Rz jog  {JOG_ROT_DEG_S:.0f} deg/s",
            anchor="w",
            height=22,
        )
        self._rpy_vel_label.pack(anchor="w", padx=4, pady=(4, 0))
        self._rpy_vel_slider = ctk.CTkSlider(
            vel,
            from_=JOG_ROT_MIN_DEG_S,
            to=JOG_ROT_MAX_DEG_S,
            number_of_steps=int(JOG_ROT_MAX_DEG_S - JOG_ROT_MIN_DEG_S),
            command=self._on_rpy_jog_speed,
        )
        self._rpy_vel_slider.set(JOG_ROT_DEG_S)
        self._rpy_vel_slider.pack(fill="x", padx=4, pady=(4, 6))
        self._ctrl.set_rpy_jog_deg_s(JOG_ROT_DEG_S)

        body.pack(side="top", fill="both", expand=True, padx=12, pady=(4, 10))
        self._build_top_buttons(self._top_btns)

        # 조그 버튼 밖에서 마우스를 떼도 조그가 멈추게 한다.
        self.root.bind_all("<ButtonRelease-1>", self._on_global_release, add="+")
        # 비상 단축키: 창 어디에 초점이 있어도 동작한다 (입력 칸에서도).
        self.root.bind_all("<space>", self._on_key_stop)
        self.root.bind_all("<Escape>", self._on_key_estop)
        # 입력 칸은 자기 처리 (공백 입력) 가 먼저라서, 그 단계에서 가로채 공백을 막는다.
        self.root.bind_class("Entry", "<space>", self._on_key_stop)
        self._mount_grasp_panel()
        self._fill_joints()
        self._fill_ee()
        self._schedule_display()
        self.root.after(80, self._align_joint_goto_row)
        self.root.bind("<Map>", lambda _e: self.root.after(50, self._align_joint_goto_row), add="+")

    def _build_top_buttons(self, parent: ctk.CTkFrame) -> None:
        """위 오른쪽 큰 버튼: [Stop][E-stop][Connect] / [HOME][초기자세][TCP 경로]."""
        big = {"width": 150, "height": 44, "font": ui_font(16, "bold")}
        ctk.CTkButton(parent, text="Stop (Space)", command=self._ctrl.stop_all, **big).grid(
            row=0, column=0, padx=4, pady=4
        )
        ctk.CTkButton(
            parent,
            text="E-stop (Esc)",
            fg_color="#b91c1c",
            hover_color="#991b1b",
            command=self._ctrl.estop,
            **big,
        ).grid(row=0, column=1, padx=4, pady=4)
        self.connect_btn = ctk.CTkButton(parent, text="Connect", command=self._toggle_connect, **big)
        self.connect_btn.grid(row=0, column=2, padx=4, pady=4)
        ctk.CTkButton(parent, text="HOME", command=self._ctrl.home, **big).grid(
            row=1, column=0, padx=4, pady=4
        )
        ctk.CTkButton(parent, text="초기자세", command=self._goto_init_pose, **big).grid(
            row=1, column=1, padx=4, pady=4
        )
        # 오른쪽 아래 칸: Meshcat 표시 체크 두 개를 위아래로.
        view = ctk.CTkFrame(parent, fg_color="transparent")
        view.grid(row=1, column=2, padx=(14, 4), pady=0, sticky="w")
        check = {"font": ui_font(14, "bold"), "checkbox_width": 20, "checkbox_height": 20, "height": 24}
        self._trail_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            view, text="TCP 경로", variable=self._trail_var, command=self._on_trail_toggle, **check
        ).pack(anchor="w")
        self._color_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            view, text="링크 색 구분", variable=self._color_var, command=self._on_color_toggle, **check
        ).pack(anchor="w")

    def _mount_grasp_panel(self) -> None:
        """아래 두 칸에 사물 위치 패널 (teach_grasp) 을 붙인다. 실패하면 안내 글자만."""
        from teach_grasp import attach_teach_panel

        self._teach = attach_teach_panel(
            self._grasp_col,
            self._viz,
            meshcat_url="",
            controller=self._ctrl,
            enabled=self._grasp_enabled,
            actions_parent=self._grasp_act_col,
        )
        if self._teach is None:
            ctk.CTkLabel(
                self._grasp_col,
                text="사물 위치 패널을 열지 못했습니다",
                text_color="#6b7280",
                anchor="w",
            ).pack(anchor="w", padx=12, pady=12)

    def _align_joint_goto_row(self) -> None:
        """관절 현재값/이동 버튼을 TCP 현재값/이동 버튼과 같은 높이에 둔다."""
        self.root.update_idletasks()
        dy = self._ee_goto_row.winfo_rooty() - self._joint_goto_row.winfo_rooty()
        if dy > 1:
            self._joint_btn_spacer.configure(height=int(dy))

    def _jog_frame_bar(self, parent: ctk.CTkFrame, text: str, command, *, top: int = 0) -> ctk.CTkSegmentedButton:
        """base/TCP 전환을 Go-to 입력칸과 같은 x에 둔다."""
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.pack(fill="x", padx=6, pady=(top, 2))
        # _cart_row에서 입력칸이 시작되는 x와 맞춘다. (축 이름 + 현재값 + −/+)
        ctk.CTkLabel(bar, text=text, width=200, anchor="w").pack(side="left", padx=(4, 2))
        seg = ctk.CTkSegmentedButton(bar, values=["base", "TCP"], command=command, width=148)
        seg.set("base")
        seg.pack(side="left", padx=(8, 4))
        return seg

    def _cart_row(
        self,
        parent: ctk.CTkFrame,
        title: str,
        axis: str,
        *,
        entry_key: str,
        unit: str,
        color: str | None = None,
        store_label: bool = False,
    ) -> None:
        """TCP 한 축 줄: [이름, 현재값, −, +, 목표 입력칸, 단위]. axis 는 컨트롤러 조그 축 이름."""
        row = ctk.CTkFrame(parent)
        row.pack(fill="x", pady=4, padx=6)
        label_kw: dict = {"text": title, "width": 36, "anchor": "w"}
        if color is not None:
            label_kw["text_color"] = color
        lab = ctk.CTkLabel(row, **label_kw)
        lab.pack(side="left", padx=(4, 2))
        if store_label:
            self._rpy_jog_labels[title] = lab
        live = ctk.CTkLabel(
            row,
            text="—",
            width=72,
            anchor="e",
            font=mono_font(),
        )
        live.pack(side="left", padx=(0, 4))
        self._tcp_live_labels[entry_key] = live
        self._hold_button(
            row,
            "−",
            lambda a=axis: self._ctrl.set_cart_jog(a, -1),
            lambda a=axis: self._ctrl.set_cart_jog(a, 0),
        )
        self._hold_button(
            row,
            "+",
            lambda a=axis: self._ctrl.set_cart_jog(a, 1),
            lambda a=axis: self._ctrl.set_cart_jog(a, 0),
        )
        ent = ctk.CTkEntry(row, width=72)
        ent.insert(0, "0")
        ent.pack(side="left", padx=(8, 4))
        self.ee_entries[entry_key] = ent
        unit_kw: dict = {"text": unit, "width": 36, "anchor": "w"}
        if color is not None:
            unit_kw["text_color"] = color
        ctk.CTkLabel(row, **unit_kw).pack(side="left")

    def _hold_button(self, parent: ctk.CTkFrame, text: str, on_press, on_release, *, width: int = 3) -> None:
        """누르고 있는 동안 동작하는 조그 버튼. 누르면 on_press, 떼면 on_release."""
        # width <= 4 는 예전 Tk 글자 폭 기준 +/− 버튼. 더 크면 글자 버튼.
        px = 40 if width <= 4 else 88
        btn = ctk.CTkButton(
            parent,
            text=text,
            width=px,
            height=32,
            corner_radius=6,
            font=ui_font(16 if width <= 4 else 13),
        )
        # _on_global_release 가 조그 버튼인지 알아보는 표시.
        btn._is_jog = True  # type: ignore[attr-defined]
        btn.pack(side="left", padx=2)
        btn.bind("<ButtonPress-1>", lambda _e: on_press())
        btn.bind("<ButtonRelease-1>", lambda _e: on_release())

    def _on_key_stop(self, _event=None) -> str:
        """Space: 모든 동작 멈춤 (토크는 유지). 입력 칸에 공백이 들어가지 않게 막는다."""
        self._ctrl.stop_all()
        return "break"

    def _on_key_estop(self, _event=None) -> str:
        """Esc: 비상 정지 (토크 끔 — 팔이 힘을 잃는다)."""
        self._ctrl.estop()
        return "break"

    def _on_global_release(self, event: tk.Event) -> None:
        """창 어디서든 마우스를 떼면 호출. 조그 버튼 위가 아니면 조그를 멈춘다."""
        w = event.widget
        while w is not None:
            if getattr(w, "_is_jog", None):
                return
            w = getattr(w, "master", None)
        # 조그 버튼 밖에서 마우스를 떼면 조그는 멈추지만 go-to 는 취소하면 안 된다.
        self._ctrl.clear_jog()

    def _joint_display(self, name: str, user_deg: float) -> float:
        """화면에 보일 값. 그리퍼는 0–100, 나머지는 도 그대로."""
        if name == GRIPPER_JOINT:
            return grip_user_to_100(user_deg)
        return float(user_deg)

    def _fill_joints(self) -> None:
        """「현재값 넣기」: 관절 입력칸에 지금 관절값을 채운다."""
        st = self._ctrl.snapshot()
        for name, deg in st.joints_deg.items():
            self.joint_entries[name].delete(0, "end")
            self.joint_entries[name].insert(0, f"{self._joint_display(name, deg):.2f}")

    def _fill_ee(self) -> None:
        """「현재값 넣기」: TCP 입력칸에 지금 위치·자세를 채운다."""
        st = self._ctrl.snapshot()
        xyz = st.pose.xyz_mm
        rpy = st.pose.rpy_deg
        vals = {
            "x": xyz[0],
            "y": xyz[1],
            "z": xyz[2],
            "roll": rpy[0],
            "pitch": rpy[1],
            "yaw": rpy[2],
        }
        for key, val in vals.items():
            self.ee_entries[key].delete(0, "end")
            self.ee_entries[key].insert(0, f"{val:.2f}")

    def _move_joints(self) -> None:
        """「관절 이동」: 입력칸의 관절값으로 이동을 시작한다."""
        try:
            target: dict[str, float] = {}
            for name in URDF_JOINT_NAMES:
                raw = float(self.joint_entries[name].get())
                target[name] = grip_100_to_user(raw) if name == GRIPPER_JOINT else raw
        except ValueError:
            messagebox.showerror("관절 이동", "관절 값은 숫자로 넣으세요.")
            return
        self._ctrl.start_joint_goto(target)

    def _goto_init_pose(self) -> None:
        """「초기자세」: 입력칸을 초기자세 값으로 바꾸고 그 자세로 이동한다."""
        from motion.robot_kinematics import INIT_POSE_JOINTS_DEG

        for name in URDF_JOINT_NAMES:
            shown = (
                grip_user_to_100(INIT_POSE_JOINTS_DEG[name])
                if name == GRIPPER_JOINT
                else INIT_POSE_JOINTS_DEG[name]
            )
            self.joint_entries[name].delete(0, "end")
            self.joint_entries[name].insert(0, f"{shown:.2f}")
        self._ctrl.init_pose()

    def _move_ee(self) -> None:
        """「TCP 이동」: 입력칸의 위치·자세로 이동을 시작한다."""
        try:
            xyz = np.array([float(self.ee_entries[k].get()) for k in ("x", "y", "z")], dtype=float)
            rpy = np.array(
                [float(self.ee_entries[k].get()) for k in ("roll", "pitch", "yaw")],
                dtype=float,
            )
        except ValueError:
            messagebox.showerror("TCP 이동", "TCP 값은 숫자로 넣으세요.")
            return
        self._ctrl.start_ee_goto(xyz, rpy)

    def _on_tcp_jog_speed(self, value: float) -> None:
        """XYZ 조그 속도 슬라이더 (mm/s)."""
        lo = int(JOG_VEL_MIN_MPS * 1000.0)
        hi = int(JOG_VEL_MAX_MPS * 1000.0)
        mm = int(round(float(value)))
        mm = max(lo, min(hi, mm))
        self._ctrl.set_tcp_jog_mm_s(mm)
        self._tcp_vel_label.configure(text=f"XYZ jog  {mm} mm/s")

    def _on_rpy_jog_speed(self, value: float) -> None:
        """Rx/Ry/Rz 조그 속도 슬라이더 (deg/s)."""
        deg = int(round(float(value)))
        deg = max(int(JOG_ROT_MIN_DEG_S), min(int(JOG_ROT_MAX_DEG_S), deg))
        self._ctrl.set_rpy_jog_deg_s(deg)
        self._rpy_vel_label.configure(text=f"Rx/Ry/Rz jog  {deg} deg/s")

    def _set_jog_preset(self, mm_s: float, deg_s: float) -> None:
        """느림/보통/빠름 버튼: 두 슬라이더를 함께 옮긴다."""
        self._tcp_vel_slider.set(mm_s)
        self._rpy_vel_slider.set(deg_s)
        self._on_tcp_jog_speed(mm_s)
        self._on_rpy_jog_speed(deg_s)

    def _speed_label_text(self, pct: float) -> str:
        joint = speed_pct_to_joint_vel(pct)
        return f"관절 {joint:.0f}°/s"

    def _set_speed_pct(self, pct: float) -> None:
        value = max(SPEED_PCT_MIN, min(SPEED_PCT_MAX, float(pct)))
        self._speed_slider.set(value)
        self._apply_speed_pct(value)

    def _on_speed_slider(self, value: float) -> None:
        self._apply_speed_pct(value)

    def _apply_speed_pct(self, value: float) -> None:
        """관절 속도 % 를 컨트롤러에 넣고 표시를 deg/s 로 바꾼다."""
        pct = int(round(float(value)))
        pct = max(int(SPEED_PCT_MIN), min(int(SPEED_PCT_MAX), pct))
        self._ctrl.set_speed_pct(pct)
        self._speed_label.configure(text=self._speed_label_text(pct))

    def _on_color_toggle(self) -> None:
        """「링크 색 구분」 체크: 켜면 링크마다 다른 색, 끄면 로봇 전체 회색."""
        self._viz.set_robot_colorful(bool(self._color_var.get()))

    def _on_trail_toggle(self) -> None:
        """「TCP 경로」 체크: Meshcat 에 TCP 지나간 선을 그릴지."""
        self._viz.set_tcp_trail_enabled(bool(self._trail_var.get()))

    def _on_xyz_frame(self, value: str) -> None:
        """XYZ 조그 기준 전환: base (로봇 베이스 축) / TCP (그리퍼 축)."""
        frame = ROT_FRAME_TCP if str(value).upper() == "TCP" else ROT_FRAME_BASE
        self._ctrl.set_xyz_frame(frame)

    def _on_rot_frame(self, value: str) -> None:
        """Rx/Ry/Rz 조그 기준 전환: base / TCP."""
        frame = ROT_FRAME_TCP if str(value).upper() == "TCP" else ROT_FRAME_BASE
        self._ctrl.set_rot_frame(frame)
        self._apply_rpy_label_colors(frame)

    def _rpy_colors(self, frame: str) -> dict[str, str]:
        return _RPY_COLOR

    def _apply_rpy_label_colors(self, frame: str) -> None:
        colors = self._rpy_colors(frame)
        for title, lab in self._rpy_jog_labels.items():
            lab.configure(text_color=colors[title])

    def _toggle_connect(self) -> None:
        """Connect / Disconnect 버튼. 연결하면 입력칸을 실기 값으로 채운다."""
        st = self._ctrl.snapshot()
        if st.connected:
            self._ctrl.disconnect()
            self.connect_btn.configure(text="Connect")
            return
        self.connect_btn.configure(state="disabled")
        try:
            self._ctrl.connect()
            self.connect_btn.configure(text="Disconnect")
            self._fill_joints()
            self._fill_ee()
        except Exception as exc:
            messagebox.showerror("Connect", str(exc))
            self.connect_btn.configure(text="Connect")
        finally:
            self.connect_btn.configure(state="normal")

    def _schedule_display(self) -> None:
        """컨트롤러 상태를 읽어 Meshcat 과 모든 표시를 갱신하고, DISPLAY_MS 뒤에 다시 부른다."""
        if self._closed:
            return
        st = self._ctrl.snapshot()
        self._viz.display(st.q)
        xyz = st.pose.xyz_mm
        rpy = st.pose.rpy_deg
        _set_text(self.tcp_label, f"TCP:  x={xyz[0]:8.2f}  y={xyz[1]:8.2f}  z={xyz[2]:8.2f}  mm")
        _set_text(self.rpy_label, f"rot:  Rx={rpy[0]:8.2f}  Ry={rpy[1]:8.2f}  Rz={rpy[2]:8.2f}  deg")
        for key, val in (
            ("x", xyz[0]),
            ("y", xyz[1]),
            ("z", xyz[2]),
            ("roll", rpy[0]),
            ("pitch", rpy[1]),
            ("yaw", rpy[2]),
        ):
            _set_text(self._tcp_live_labels[key], f"{val:7.2f}")
        # 오차는 실기일 때만 있다: 명령 TCP − 실측 TCP.
        if st.ee_err_mm is None or st.err_xyz_mm is None:
            _set_text(self.err_label, "err:  (virtual)")
        else:
            d = st.err_xyz_mm
            _set_text(
                self.err_label,
                f"err:  |Δ|={st.ee_err_mm:7.2f}  "
                f"Δx={d[0]:+7.2f}  Δy={d[1]:+7.2f}  Δz={d[2]:+7.2f}  mm",
            )
        torque = "torque on" if st.torque else "torque off"
        if st.connected:
            _set_text(self.mode_label, f"mode: real · {torque}")
            _set_text(self.connect_btn, "Disconnect")
        else:
            _set_text(self.mode_label, "mode: virtual")
            if self.connect_btn.cget("state") != "disabled":
                _set_text(self.connect_btn, "Connect")
        # 기준 좌표 버튼이 컨트롤러 상태와 다르면 상태에 맞춘다.
        xyz_val = "TCP" if st.xyz_frame == ROT_FRAME_TCP else "base"
        if self._xyz_frame_seg.get() != xyz_val:
            self._xyz_frame_seg.set(xyz_val)
        seg_val = "TCP" if st.rot_frame == ROT_FRAME_TCP else "base"
        if self._rot_frame_seg.get() != seg_val:
            self._rot_frame_seg.set(seg_val)
            self._apply_rpy_label_colors(st.rot_frame)
        # "이미 목표 관절에 있음" 은 오류가 아니라서 빨간 글자로 보이지 않는다.
        _set_text(self.fault_label, "" if st.fault == ALREADY_AT_TARGET else st.fault)
        for name, deg in st.joints_deg.items():
            shown = self._joint_display(name, deg)
            unit = "" if name == GRIPPER_JOINT else "°"
            _set_text(self.joint_labels[name], f"{shown:7.2f}{unit}")
        if not self._closed:
            self.root.after(DISPLAY_MS, self._schedule_display)

    def on_close(self) -> None:
        """창 닫기. 실기에 토크가 켜져 있으면 먼저 쉬는 자세로 옮긴 뒤 닫는다."""
        if self._closing:
            return
        self._closing = True
        if self._teach is not None:
            self._teach.abort_user_pick_place()
        st = self._ctrl.snapshot()
        if st.connected and st.torque:
            duration_s = self._ctrl.start_joint_goto(dict(SHUTDOWN_JOINTS_DEG))
            self.fault_label.configure(text="종료: rest 자세로 이동 후 연결 해제")
            self._poll_shutdown(time.monotonic() + float(duration_s) + 1.5)
            return
        self._finish_close()

    def _poll_shutdown(self, deadline: float) -> None:
        """쉬는 자세 이동이 끝나거나 시간이 지나면 닫는다."""
        if self._closed:
            return
        if not self._ctrl.goto_active() or time.monotonic() >= deadline:
            self._finish_close()
            return
        self.root.after(50, lambda: self._poll_shutdown(deadline))

    def _finish_close(self) -> None:
        """제어 루프와 Meshcat 을 끄고 창을 없앤다."""
        if self._closed:
            return
        self._closed = True
        try:
            self._ctrl.stop()
        except Exception:
            pass
        try:
            self._viz.close()
        except Exception:
            pass
        if self._teach is not None:
            self._teach._cancel_register_poll()
        self.root.destroy()
