#!/usr/bin/env python3
"""사물 위치 패널 화면: 물체 자세 칸, 버튼, 상태 글자. 펜던트 오른쪽 두 칸이 이것이다.

  python pendant/main.py              # 펜던트만 (사물 위치 칸은 잠김)
  python pendant/main.py --grasp      # 같은 창에서 사물 위치 칸 조작
  python pendant/teach_grasp.py       # 물체 창만 (가상 팔로 P/G 이동)

이 파일은 화면만 다룬다.
  pendant/teach_logic.py   계산 · 실행 (Rx/Ry/Rz 표기, 대기·집기 자세, 픽앤플레이스 단계, 랜덤 배치, 유저생성)
  pendant/teach_scene.py   3D 화면(Meshcat)에 그리기
  motion/grasp.py          접근 자세 · 랜덤 배치 · RPY 변환 계산 (랜덤 세 자세의 Rx/Ry/Rz 설명도 여기)
  cad/model.py             yaml · STL 읽기
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PENDANT = Path(__file__).resolve().parent
PROJECT = PENDANT.parent
for path in (PROJECT, PENDANT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ui_style import apply_root_fonts, apply_ui_theme, enable_xft_tk, ui_font  # noqa: E402

enable_xft_tk()
apply_ui_theme()

from cad.model import (  # noqa: E402
    CAD_YAML,
    approach_mm,
    drop_xy_mm,
    load_mesh_m,
    mesh_path as cad_mesh_path,
)
from motion import runlog  # noqa: E402
from motion.grasp import (  # noqa: E402
    RANDOM_PLACE_MAX_TRIES,
    RANDOM_PLACE_MODES,
    AutoPreResult,
    GraspSpec,
    PlacePose,
    T_to_xyzrpy,
    load_specs,
)
from motion.pick_place import PickPlaceRunner  # noqa: E402
from motion.register_link import (  # noqa: E402
    REG_MSG_FAIL,
    REG_MSG_NO_BOX,
    REG_MSG_NO_REPLY,
    REG_MSG_RUN,
    REG_POLL_MS,
    RegisterRequest,
)
from teach_logic import (  # noqa: E402
    MODE_KO,
    PlaceRpy,
    UserPickPlace,
    auto_label,
    auto_pre,
    pick_place_steps,
    place_text,
    r1,
    random_place,
)
from teach_scene import TeachScene, tcp_matches_pre  # noqa: E402
from vision.gui_env import quiet_gtk  # noqa: E402

# 입력 칸 이름 (입력 오류 문구에 쓴다).
_ENTRY_LABELS = {
    "place_x": "사물 x", "place_y": "사물 y", "place_z": "사물 z",
    "place_roll": "사물 Rx", "place_pitch": "사물 Ry", "place_yaw": "사물 Rz",
    "approach_d": "d mm (대기)", "approach_a": "a mm (집기)",
    "drop_x": "놓기 x", "drop_y": "놓기 y",
}
DISPLAY_MS = 33
RPY_SLIDER_MIN = -180.0
RPY_SLIDER_MAX = 180.0
# 사물 XYZ 슬라이더 (프로젝트 베이스 mm). XY 는 랜덤 배치 고리
# (r < RANDOM_R_MAX_MM, motion/grasp.py) 보다 넓게 해서 최대 도달 근처도 시험할 수 있다.
PLACE_XY_SLIDER_MIN = -300.0
PLACE_XY_SLIDER_MAX = 300.0
PLACE_Z_SLIDER_MIN = 0.0
PLACE_Z_SLIDER_MAX = 40.0
RPY_LIVE_APPLY_MS = 40
# 상태 글자 줄바꿈 너비 (배율 1.0 px). 오른쪽 아래 칸 너비에 맞춘다.
STATUS_WRAP_PX = 400
NO_CONTROLLER = "컨트롤러 없음. python pendant/main.py --grasp"
_log = runlog.get("teach")


class TeachGraspGui:
    def __init__(
        self,
        root,
        *,
        visualizer,
        mesh_path: Path,
        vertices_m: np.ndarray,
        faces: np.ndarray,
        place: PlacePose,
        grasp: GraspSpec,
        meshcat_url: str,
        kinematics=None,
        controller=None,
        own_robot: bool = True,
        drive_viz: bool = False,
        embedded: bool = False,
        enabled: bool = True,
        actions_parent=None,
    ) -> None:
        import customtkinter as ctk

        self.root = root
        self._embedded = bool(embedded)
        self._panel_enabled = bool(enabled) if self._embedded else True
        self._panel_labels: list = []
        self._panel_buttons: list = []
        self._mode_cbs: list = []
        if not self._embedded:
            apply_root_fonts(root)
        self._scene = TeachScene(visualizer)
        self._kin = kinematics
        self._ctrl = controller
        self._own_robot = bool(own_robot) and controller is None
        self._drive_viz = bool(drive_viz) and controller is not None
        self._yaml_path = CAD_YAML
        self._grasp_spec = grasp
        self._grasp_hide_var = None
        self._region_hide_var = None
        self._auto_pre: AutoPreResult | None = None
        self._pre_tcp_matched: bool | None = None
        self._status: ctk.CTkLabel | None = None
        self.entries: dict[str, ctk.CTkEntry] = {}
        self.sliders: dict[str, ctk.CTkSlider] = {}
        self._slider_sync = False  # 코드가 칸·슬라이더를 채우는 중이면 True (이벤트 무시)
        self._live_apply_after: str | None = None
        self._pp_after: str | None = None
        self._reg_after: str | None = None
        self._reg_req: RegisterRequest | None = None
        self._pp_runner: PickPlaceRunner | None = PickPlaceRunner(controller) if controller is not None else None
        self._user_pp = UserPickPlace(controller)
        verts = np.asarray(vertices_m, dtype=float).reshape(-1, 3)
        self._cad_verts_m = verts
        self._cad_mins_mm = verts.min(axis=0) * 1000.0
        self._cad_maxs_mm = verts.max(axis=0) * 1000.0
        self._rng = np.random.default_rng()
        self._warn_label: ctk.CTkLabel | None = None
        self._rand_mode: str | None = None
        self._rand_mode_vars: dict[str, object] = {}
        approach_d0, approach_a0 = approach_mm(CAD_YAML)
        self._approach_d_mm = float(approach_d0)
        self._approach_a_mm = float(approach_a0)
        self._drop_xy = drop_xy_mm(CAD_YAML)
        # 물체 회전: 저장은 베이스 기준, 화면 숫자는 물체 축(OBJ)/베이스 표기. 버튼으로 표기만 바꾼다.
        self._rpy = PlaceRpy.from_place(place.rpy_deg)
        shown = self._rpy.shown()

        if not self._embedded:
            root.title("물체 집기 티칭")
            root.minsize(980, 720)
            root.geometry("1080x860")
            root.protocol("WM_DELETE_WINDOW", self.on_close)
            split = ctk.CTkFrame(root, fg_color="transparent")
            split.pack(fill="both", expand=True)
            host = ctk.CTkFrame(split)
            actions_host = ctk.CTkFrame(split)
            host.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=8)
            actions_host.pack(side="left", fill="y", padx=(4, 8), pady=8)
        else:
            host = root
            actions_host = actions_parent if actions_parent is not None else root
        self._host = host

        self._build_place_box(host, place, shown)
        self._build_actions(actions_host)
        self._build_random_row(actions_host)
        self._build_entries_row(actions_host, approach_d0, approach_a0)

        # 상태 글자는 두 줄 자리를 미리 잡고 칸 너비에서 줄바꿈한다.
        # 한 줄로 두면 긴 글자가 칸을 오른쪽으로 늘려 버튼까지 같이 길어진다.
        self._status = ctk.CTkLabel(
            actions_host, text="", anchor="nw", justify="left", height=40, wraplength=STATUS_WRAP_PX
        )
        self._status.pack(fill="x", padx=14, pady=(0, 4))
        btn_load = ctk.CTkButton(
            actions_host,
            text="물체 위치 불러오기",
            command=self.load_registered_place,
            fg_color="#86EFAC",
            hover_color="#4ADE80",
            text_color="#14532D",
        )
        btn_load.pack(fill="x", padx=14, pady=(0, 12))
        self._panel_buttons.append(btn_load)

        if self._embedded and not self._panel_enabled:
            self._lock_place_panel()
            return

        if not self._embedded:
            root.bind("<Return>", lambda _e: self.apply())
        self._scene.draw_cad(vertices_m, faces, region=not self._region_hidden())
        self.apply()
        # TCP 가 움직이는 동안 대기 구 색을 바꾼다. drive_viz 가 아니면 display(q) 는 펜던트가 한다.
        if self._ctrl is not None:
            self._schedule_display()

    # ================================================================ 화면 만들기

    def _build_place_box(self, host, place: PlacePose, shown: dict[str, float]) -> None:
        """왼쪽 칸: 사물 x/y/z, Rx/Ry/Rz (표기 버튼), 숨김 체크."""
        import customtkinter as ctk

        # TCP 칸과 같은 시작 높이·같은 줄 간격. 박스는 TCP 행과 같은 CTkFrame.
        title = ctk.CTkLabel(host, text="사물의 위치", anchor="w")
        title.pack(anchor="w", padx=8, pady=(8, 4))
        self._panel_labels.append(title)
        for key, label, val in (
            ("place_x", "x (mm)", place.xyz_mm[0]),
            ("place_y", "y (mm)", place.xyz_mm[1]),
            ("place_z", "z (mm)", place.xyz_mm[2]),
        ):
            self._slider_row(host, key, float(val), label, on_slide=self._on_xyz_slider, on_entry=self._on_xyz_entry)

        rpy_head = ctk.CTkFrame(host, fg_color="transparent")
        rpy_head.pack(fill="x", padx=6, pady=(10, 2))
        rpy_title = ctk.CTkLabel(rpy_head, text="Rx/Ry/Rz", width=200, anchor="w")
        rpy_title.pack(side="left", padx=(4, 2))
        self._panel_labels.append(rpy_title)
        self._place_rpy_seg = ctk.CTkSegmentedButton(
            rpy_head, values=["base", "OBJ"], command=self._on_place_rpy_frame, width=148
        )
        self._place_rpy_seg.set("OBJ")
        self._place_rpy_seg.pack(side="left", padx=(8, 4))
        for key, label, axis in (("place_roll", "Rx", "roll"), ("place_pitch", "Ry", "pitch"), ("place_yaw", "Rz", "yaw")):
            self._slider_row(host, key, float(shown[axis]), label, on_slide=self._on_rpy_slider, on_entry=self._on_rpy_entry)

        hide_row = ctk.CTkFrame(host, fg_color="transparent")
        hide_row.pack(fill="x", padx=8, pady=(2, 2))
        self._grasp_hide_var = ctk.BooleanVar(value=False)
        self._grasp_hide_cb = ctk.CTkCheckBox(
            hide_row, text="비활성화 (축 숨김)", variable=self._grasp_hide_var, command=self._on_grasp_hide_toggle, width=160
        )
        self._grasp_hide_cb.pack(side="left")
        self._region_hide_var = ctk.BooleanVar(value=False)
        self._region_hide_cb = ctk.CTkCheckBox(
            hide_row, text="비활성화 (영역 숨김)", variable=self._region_hide_var, command=self._on_region_hide_toggle, width=170
        )
        self._region_hide_cb.pack(side="left", padx=(12, 0))

    def _build_actions(self, actions_host) -> None:
        """오른쪽 칸 위: 적용 · 물체 초기화 · 대기/집기 위치로 · 픽앤플레이스 두 개 · 초기자세."""
        import customtkinter as ctk

        actions = ctk.CTkFrame(actions_host, fg_color="transparent")
        actions.pack(fill="x", padx=14, pady=(4, 8))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)
        btn_apply = ctk.CTkButton(actions, text="적용", command=self.apply)
        btn_apply.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 8))
        btn_reset = ctk.CTkButton(actions, text="물체 초기화", command=self.reset_place)
        btn_reset.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 8))
        self._panel_buttons.extend((btn_apply, btn_reset))
        self._btn_pre = ctk.CTkButton(actions, text="대기 위치로", command=self.goto_pre)
        self._btn_grasp = ctk.CTkButton(actions, text="집기 위치로", command=self.goto_grasp)
        self._btn_pre.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(0, 8))
        self._btn_grasp.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(0, 8))
        self._btn_seq = ctk.CTkButton(
            actions, text="픽앤플레이스", command=self.start_pick_place, height=46, corner_radius=6, font=ui_font(18)
        )
        self._btn_seq.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._btn_user_seq = ctk.CTkButton(
            actions,
            text="픽앤플레이스(유저생성)",
            command=self.start_user_pick_place,
            height=46,
            corner_radius=6,
            font=ui_font(18),
        )
        self._btn_user_seq.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._btn_init = ctk.CTkButton(actions, text="초기자세로 이동", command=self.snap_init_pose)
        self._btn_init.grid(row=4, column=0, columnspan=2, sticky="ew")
        robot_buttons = (self._btn_pre, self._btn_grasp, self._btn_seq, self._btn_user_seq, self._btn_init)
        self._panel_buttons.extend(robot_buttons)
        if self._ctrl is None:
            for btn in robot_buttons:
                btn.configure(state="disabled")

    def _build_random_row(self, actions_host) -> None:
        """「물체 랜덤 생성」 버튼, 충돌 경고, 자세 종류 체크 네 개."""
        import customtkinter as ctk

        rand_row = ctk.CTkFrame(actions_host, fg_color="transparent")
        rand_row.pack(fill="x", padx=14, pady=(4, 4))
        btn_rand = ctk.CTkButton(rand_row, text="물체 랜덤 생성", width=140, command=self.randomize_place)
        btn_rand.pack(side="left")
        self._panel_buttons.append(btn_rand)
        self._warn_label = ctk.CTkLabel(rand_row, text="", anchor="w", text_color="#f87171")
        self._warn_label.pack(side="left", padx=(12, 0), fill="x", expand=True)

        mode_row = ctk.CTkFrame(actions_host, fg_color="transparent")
        mode_row.pack(fill="x", padx=14, pady=(4, 8))
        for key, label in (("stand", "세우기"), ("lie", "눕히기"), ("slant", "비스듬히"), ("free", "자유 포즈")):
            var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                mode_row,
                text=label,
                variable=var,
                command=lambda k=key: self._on_rand_mode_toggle(k),
                checkbox_width=18,
                checkbox_height=18,
            )
            cb.pack(side="left", padx=(0, 8))
            self._rand_mode_vars[key] = var
            self._mode_cbs.append(cb)

    def _build_entries_row(self, actions_host, approach_d0: float, approach_a0: float) -> None:
        """두 줄 × 두 칸: 왼쪽 = 접근 거리 d/a, 오른쪽 = 내려놓을 위치 x/y (베이스 mm)."""
        import customtkinter as ctk

        foot = ctk.CTkFrame(actions_host, fg_color="transparent")
        foot.pack(fill="x", padx=14, pady=(0, 4))
        drop_x0, drop_y0 = self._drop_xy
        row_d = ctk.CTkFrame(foot, fg_color="transparent")
        row_d.pack(fill="x", pady=3)
        self._pair(row_d, "approach_d", f"{approach_d0:.1f}", "d mm (대기)")
        self._pair(row_d, "drop_x", f"{drop_x0:.1f}", "놓기 x mm")
        row_a = ctk.CTkFrame(foot, fg_color="transparent")
        row_a.pack(fill="x", pady=3)
        self._pair(row_a, "approach_a", f"{approach_a0:.1f}", "a mm (집기)")
        self._pair(row_a, "drop_y", f"{drop_y0:.1f}", "놓기 y mm")
        for key in ("approach_d", "approach_a"):
            self.entries[key].bind("<FocusOut>", lambda _e: self._schedule_live_apply())
            self.entries[key].bind("<Return>", lambda _e: self._schedule_live_apply())
        for key in ("drop_x", "drop_y"):
            self.entries[key].bind("<FocusOut>", lambda _e: self._draw_drop_marker())
            self.entries[key].bind("<Return>", lambda _e: self._draw_drop_marker())

    def _pair(self, row, key: str, value: str, label: str) -> None:
        """한 줄 안의 [이름, 짧은 입력 칸] 한 쌍. 쌍마다 줄의 절반을 차지한다."""
        import customtkinter as ctk

        cell = ctk.CTkFrame(row, fg_color="transparent")
        cell.pack(side="left", fill="x", expand=True)
        lab = ctk.CTkLabel(cell, text=label, width=92, anchor="w")
        lab.pack(side="left", padx=(0, 6))
        self._panel_labels.append(lab)
        ent = ctk.CTkEntry(cell, width=76)
        ent.insert(0, value)
        ent.pack(side="left")
        self.entries[key] = ent

    def _slider_limits(self, key: str) -> tuple[float, float]:
        if key in ("place_x", "place_y"):
            return PLACE_XY_SLIDER_MIN, PLACE_XY_SLIDER_MAX
        if key == "place_z":
            return PLACE_Z_SLIDER_MIN, PLACE_Z_SLIDER_MAX
        return RPY_SLIDER_MIN, RPY_SLIDER_MAX

    def _slider_row(self, parent, key: str, value: float, label: str, *, on_slide, on_entry) -> None:
        """[이름, 입력 칸, 슬라이더] 한 줄. TCP `_cart_row` 와 같은 프레임, 높이는 조그 버튼(32)."""
        import customtkinter as ctk

        lo, hi = self._slider_limits(key)
        row = ctk.CTkFrame(parent)
        row.pack(fill="x", pady=4, padx=6)
        lab = ctk.CTkLabel(row, text=label, width=110, anchor="w")
        lab.pack(side="left")
        self._panel_labels.append(lab)
        ent = ctk.CTkEntry(row, width=72, height=32)
        ent.insert(0, f"{value:.1f}")
        ent.pack(side="left", padx=(6, 4))
        slider = ctk.CTkSlider(
            row,
            from_=lo,
            to=hi,
            number_of_steps=max(int(round(hi - lo)), 1),
            command=lambda v, k=key: on_slide(k, v),
        )
        self._slider_sync = True
        try:
            slider.set(float(np.clip(value, lo, hi)))
        finally:
            self._slider_sync = False
        slider.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.entries[key] = ent
        self.sliders[key] = slider
        ent.bind("<FocusOut>", lambda _e, k=key: on_entry(k))
        ent.bind("<Return>", lambda _e, k=key: on_entry(k))

    def _lock_place_panel(self) -> None:
        """펜던트를 --grasp 없이 켰을 때: 보이기만 하고 모두 잠근다."""
        gray = "#6b7280"
        for lab in self._panel_labels:
            try:
                lab.configure(text_color=gray)
            except Exception:
                pass
        for ent in self.entries.values():
            ent.configure(text_color=gray, state="disabled")
        for slider in self.sliders.values():
            slider.configure(state="disabled")
        self._place_rpy_seg.configure(state="disabled")
        for cb in (self._grasp_hide_cb, self._region_hide_cb, *self._mode_cbs):
            cb.configure(state="disabled", text_color_disabled=gray)
        for btn in self._panel_buttons:
            btn.configure(state="disabled")

    # ================================================================ 칸 · 슬라이더

    def _locked(self) -> bool:
        return self._embedded and not self._panel_enabled

    def _write_entry(self, key: str, val: float, *, slider: bool = True) -> None:
        """칸에 숫자를 쓰고, 슬라이더도 맞춘다 (슬라이더 이벤트는 무시)."""
        self._slider_sync = True
        try:
            self.entries[key].delete(0, "end")
            self.entries[key].insert(0, f"{val:.1f}")
            if slider and key in self.sliders:
                lo, hi = self._slider_limits(key)
                self.sliders[key].set(float(np.clip(val, lo, hi)))
        finally:
            self._slider_sync = False

    def _set_entry(self, key: str, val: float) -> None:
        self._write_entry(key, val)

    def _on_xyz_slider(self, key: str, value: float) -> None:
        if self._slider_sync:
            return
        self._write_entry(key, float(value), slider=False)
        self._schedule_live_apply()

    def _on_xyz_entry(self, key: str) -> None:
        if self._slider_sync or key not in self.sliders:
            return
        try:
            val = float(self.entries[key].get().strip())
        except ValueError:
            return
        lo, hi = self._slider_limits(key)
        self._write_entry(key, float(np.clip(val, lo, hi)))
        self._schedule_live_apply()

    def _show_place_rpy(self, shown: dict[str, float], *, keep_axis: str | None = None) -> None:
        """Rx/Ry/Rz 칸을 채운다. 끌고 있는 축의 슬라이더는 건드리지 않는다."""
        for axis, val in shown.items():
            self._write_entry(f"place_{axis}", val, slider=axis != keep_axis)

    def _on_place_rpy_frame(self, value: str) -> None:
        if self._locked():
            return
        if self._rpy.set_frame("base" if str(value) == "base" else "body"):
            self._show_place_rpy(self._rpy.refresh())

    def _on_rpy_slider(self, key: str, value: float) -> None:
        if self._slider_sync:
            return
        axis = key.removeprefix("place_")
        self._show_place_rpy(self._rpy.nudge_axis(axis, float(value)), keep_axis=axis)
        self._schedule_live_apply()

    def _on_rpy_entry(self, key: str) -> None:
        if self._slider_sync or key not in self.sliders:
            return
        try:
            val = float(self.entries[key].get().strip())
        except ValueError:
            return
        val = float(np.clip(val, RPY_SLIDER_MIN, RPY_SLIDER_MAX))
        self._show_place_rpy(self._rpy.set_axis(key.removeprefix("place_"), val))
        self._schedule_live_apply()

    def _schedule_live_apply(self) -> None:
        """끌거나 칠 때마다 다시 그리면 느리므로, 잠깐 뒤에 한 번만 적용한다."""
        if self._live_apply_after is not None:
            try:
                self.root.after_cancel(self._live_apply_after)
            except Exception:
                pass
        self._live_apply_after = self.root.after(RPY_LIVE_APPLY_MS, self._live_apply)

    def _live_apply(self) -> None:
        self._live_apply_after = None
        try:
            self.apply()
        except Exception:
            pass

    def _read_float(self, key: str) -> float:
        text = self.entries[key].get().strip()
        try:
            return float(text)
        except ValueError:
            name = _ENTRY_LABELS.get(key, key)
            raise ValueError(f"{name} 칸에 숫자를 넣으세요 (지금: '{text}')") from None

    def _read_place(self) -> PlacePose:
        return PlacePose(
            xyz_mm=(self._read_float("place_x"), self._read_float("place_y"), self._read_float("place_z")),
            rpy_deg=tuple(float(x) for x in self._rpy.base_rpy),
        )

    def _read_da(self) -> tuple[float, float]:
        """d (대기 거리) · a (집기 깊이). 칸이 비었거나 숫자가 아니면 yaml 값."""
        try:
            d_mm = self._read_float("approach_d")
        except (KeyError, ValueError):
            d_mm = self._approach_d_mm
        try:
            a_mm = self._read_float("approach_a")
        except (KeyError, ValueError):
            a_mm = self._approach_a_mm
        return float(d_mm), float(a_mm)

    def _fill_place(self, place: PlacePose) -> None:
        """새 물체 자세를 칸에 넣는다 (랜덤 · 불러오기 · 초기화)."""
        self._rpy.set_place(place.rpy_deg)
        for key, val in zip(("place_x", "place_y", "place_z"), place.xyz_mm):
            self._write_entry(key, float(val))
        self._show_place_rpy(self._rpy.shown())

    def _set_status(self, text: str, *, error: bool = False, info: bool = False) -> None:
        if self._status is None:
            return
        color = "#f87171" if error else "#e5e7eb" if info else "#86efac"
        self._status.configure(text=text, text_color=color)

    def _post_status(self, text: str, error: bool = False) -> None:
        """다른 스레드에서 부를 때: 화면 스레드로 넘겨 상태 글자를 바꾼다."""
        try:
            self.root.after(0, lambda: self._set_status(text, error=error))
        except Exception:
            pass

    # ================================================================ 3D 화면

    def _grasp_section_hidden(self) -> bool:
        var = getattr(self, "_grasp_hide_var", None)
        return bool(var.get()) if var is not None else False

    def _region_hidden(self) -> bool:
        var = getattr(self, "_region_hide_var", None)
        return bool(var.get()) if var is not None else False

    def _on_grasp_hide_toggle(self) -> None:
        self.apply()

    def _on_region_hide_toggle(self) -> None:
        if self._region_hidden():
            self._scene.hide_region()
        else:
            self._scene.show_region()

    def _compute_auto(self, place: PlacePose) -> AutoPreResult:
        d_mm, a_mm = self._read_da()
        auto = auto_pre(place, self._cad_mins_mm, self._cad_maxs_mm, d_mm, a_mm, kin=self._kin, ctrl=self._ctrl)
        self._auto_pre = auto
        return auto

    def apply(self) -> None:
        """칸 숫자로 물체·대기·집기·놓기 표시를 다시 그린다."""
        if self._locked():
            return
        try:
            place = self._read_place()
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        d_mm, _a_mm = self._read_da()
        auto = self._compute_auto(place)
        matched = self._ctrl is not None and tcp_matches_pre(auto, self._ctrl.snapshot().pose.xyz_m)
        self._pre_tcp_matched = matched
        self._scene.draw_place(place, auto, d_mm=d_mm, matched=matched, grasp_hidden=self._grasp_section_hidden())
        self._draw_drop_marker()
        if self._own_robot:
            self._scene.show_home_robot(self._kin, self._grasp_spec.gripper)

    def _draw_drop_marker(self) -> None:
        """놓기 x/y 칸이 숫자면 바닥에 빨간 원, 아니면 지운다."""
        try:
            dx, dy = self._read_float("drop_x"), self._read_float("drop_y")
        except (KeyError, ValueError):
            self._scene.clear_drop()
            return
        self._scene.draw_drop(dx, dy)

    def _schedule_display(self) -> None:
        """33 ms 마다: (물체 창만 띄웠으면) 팔을 그리고, TCP 가 대기 구에 닿으면 구를 초록으로."""
        if self._ctrl is None:
            return
        try:
            if not int(self.root.winfo_exists()):
                return
        except Exception:
            return
        st = self._ctrl.snapshot()
        if self._drive_viz:
            self._scene.display(st.q)
        if self._auto_pre is not None:
            matched = tcp_matches_pre(self._auto_pre, st.pose.xyz_m)
            if matched is not self._pre_tcp_matched:
                self._pre_tcp_matched = matched
                self._scene.draw_approach_markers(self._auto_pre, matched=matched)
        self.root.after(DISPLAY_MS, self._schedule_display)

    # ================================================================ 버튼: 물체

    def _on_rand_mode_toggle(self, key: str) -> None:
        """한 번에 한 모드. 나머지도 누를 수 있다 (잠그지 않음). 다시 누르면 해제."""
        if bool(self._rand_mode_vars[key].get()):
            self._rand_mode = key
            for k, other in self._rand_mode_vars.items():
                if k != key:
                    other.set(False)
        else:
            self._rand_mode = None

    def randomize_place(self) -> None:
        """선택한 모드로 랜덤 배치 (선택 없음 = 자유). 바닥 아래로 안 들어간다."""
        self._warn_label.configure(text="")
        mode = self._rand_mode if self._rand_mode in RANDOM_PLACE_MODES else "free"
        res = random_place(self._cad_verts_m, mode, self._rng, kin=self._kin, ctrl=self._ctrl)
        if res.place is None:
            self._set_status(f"랜덤 실패({mode}): {RANDOM_PLACE_MAX_TRIES}회 내 바닥/영역 만족 샘플 없음", error=True)
            return
        self._fill_place(res.place)
        self.apply()
        if res.error:
            self._set_status(f"랜덤({mode}) {place_text(res.place)}  (충돌검사 실패: {res.error})", error=True)
            return
        mode_ko = MODE_KO.get(mode, mode)
        if res.collided:
            self._warn_label.configure(text="경고: 물체가 현재 로봇과 충돌합니다. 다시 랜덤 생성하세요.")
            self._set_status(f"랜덤({mode_ko}/충돌) {place_text(res.place)}", error=True)
        else:
            self._set_status(f"랜덤({mode_ko}) {place_text(res.place)}")

    def reset_place(self) -> None:
        place, _ = load_specs(self._yaml_path)
        self._fill_place(place)
        self.apply()
        self._set_status(f"물체 초기화: {self._yaml_path.name} place")

    def load_registered_place(self) -> None:
        """카메라 창에 c 와 같은 전체 등록을 요청하고, 끝난 자세만 칸에 넣는다."""
        if self._locked():
            return
        self._cancel_register_poll()
        try:
            self._reg_req = RegisterRequest.send()
        except OSError:
            self._reg_req = None
            self._set_status(REG_MSG_NO_REPLY, error=True)
            _log.warning("물체 위치 불러오기: 요청 파일 쓰기 실패")
            return
        _log.info("물체 위치 불러오기: 카메라 창에 등록 요청")
        self._set_status(REG_MSG_RUN, info=True)
        self._reg_after = self.root.after(REG_POLL_MS, self._poll_register)

    def _poll_register(self) -> None:
        self._reg_after = None
        try:
            if not int(self.root.winfo_exists()):
                return
        except Exception:
            return
        req = self._reg_req
        if req is None:
            return
        state = req.poll()
        if req.done:
            if state == "ok":
                _log.info("등록 결과 ok  xyz=%s  rpy=%s", r1(req.xyz_mm), r1(req.rpy_deg))
            else:
                _log.warning("등록 결과 %s", state)
        if state == "ok":
            place = PlacePose(xyz_mm=req.xyz_mm, rpy_deg=req.rpy_deg)
            self._fill_place(place)
            self.apply()
            self._set_status(f"불러옴  xyz=({place.xyz_mm[0]:.1f}, {place.xyz_mm[1]:.1f}, {place.xyz_mm[2]:.1f})")
            return
        messages = {
            "run": (REG_MSG_RUN, False),
            "busy": (REG_MSG_RUN, False),
            "no_box": (REG_MSG_NO_BOX, True),
            "fail": (REG_MSG_FAIL, True),
            "timeout": (REG_MSG_NO_REPLY, True),
        }
        if state in messages:
            text, error = messages[state]
            self._set_status(text, error=error, info=not error)
        if not req.done:
            self._reg_after = self.root.after(REG_POLL_MS, self._poll_register)

    def _cancel_register_poll(self) -> None:
        if self._reg_after is None:
            return
        try:
            self.root.after_cancel(self._reg_after)
        except Exception:
            pass
        self._reg_after = None

    # ================================================================ 버튼: 팔

    def _goto_auto(self, target: str) -> None:
        """「대기 위치로」(pre) · 「집기 위치로」(grasp)."""
        try:
            place = self._read_place()
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        auto = self._compute_auto(place)
        if self._ctrl is None:
            self._set_status(NO_CONTROLLER, error=True)
            return
        self.apply()
        xyz_mm, rpy_deg = T_to_xyzrpy(auto.T_base if target == "pre" else auto.T_grasp_base)
        duration_s = self._ctrl.start_ee_goto(xyz_mm, rpy_deg)
        self._set_status(
            f"{auto_label(auto, target)} 이동  xyz=[{xyz_mm[0]:.1f}, {xyz_mm[1]:.1f}, {xyz_mm[2]:.1f}]  "
            f"rpy=[{rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}]  "
            f"({duration_s:.1f}s)"
        )

    def goto_pre(self) -> None:
        self._goto_auto("pre")

    def goto_grasp(self) -> None:
        self._goto_auto("grasp")

    def start_pick_place(self) -> None:
        if self._ctrl is None or self._pp_runner is None:
            self._set_status(NO_CONTROLLER, error=True)
            return
        if self._user_pick_running():
            self._set_status("픽앤플레이스(유저생성) 실행 중", error=True)
            return
        if self._pp_runner.busy():
            self._set_status("픽앤플레이스 이미 실행 중", error=True)
            return
        try:
            place = self._read_place()
            drop_xy = (self._read_float("drop_x"), self._read_float("drop_y"))
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        auto = self._compute_auto(place)
        _log.info("픽앤플레이스 입력  물체 xyz=%s rpy=%s  놓기 xy=%s", r1(place.xyz_mm), r1(place.rpy_deg), r1(drop_xy))
        if not self._pp_runner.start(pick_place_steps(auto, drop_xy)):
            self._set_status(self._pp_runner.status, error=True)
            return
        self._set_status(f"픽앤플레이스  {self._pp_runner.step_name}")
        self._tick_pick_place()

    def _tick_pick_place(self) -> None:
        self._pp_after = None
        runner = self._pp_runner
        if runner is None:
            return
        phase = runner.tick()
        if phase == "running":
            self._set_status(f"픽앤플레이스  {runner.step_name}")
            self._pp_after = self.root.after(50, self._tick_pick_place)
        elif phase == "done":
            self._set_status("픽앤플레이스 완료")
        elif phase == "fault":
            self._set_status(f"픽앤플레이스 실패: {runner.status}", error=True)

    def _user_pick_running(self) -> bool:
        return self._user_pp.running()

    def start_user_pick_place(self) -> None:
        if self._ctrl is None:
            self._set_status(NO_CONTROLLER, error=True)
            return
        if self._user_pick_running():
            self._set_status("픽앤플레이스(유저생성) 이미 실행 중", error=True)
            return
        if self._pp_runner is not None and self._pp_runner.busy():
            self._set_status("픽앤플레이스 실행 중", error=True)
            return
        self._user_pp.start(self._post_status)
        self._set_status("픽앤플레이스(유저생성) 실행")

    def abort_user_pick_place(self) -> None:
        self._user_pp.abort()

    def snap_init_pose(self) -> None:
        if self._ctrl is None:
            self._set_status(NO_CONTROLLER, error=True)
            return
        how = self._ctrl.snap_init_pose()
        if how == "snap":
            self._set_status("초기자세로 즉시 이동 (가상)")
        elif how == "goto":
            self._set_status("초기자세로 이동 중 (실기 · 관절 go-to)")
        else:
            self._set_status("초기자세 이동 실패 (E-stop/충돌 등)", error=True)

    def on_close(self) -> None:
        self.abort_user_pick_place()
        if self._pp_runner is not None and self._pp_runner.busy():
            self._pp_runner.abort("창 닫힘")
        if self._pp_after is not None:
            try:
                self.root.after_cancel(self._pp_after)
            except Exception:
                pass
            self._pp_after = None
        self._cancel_register_poll()
        self.root.destroy()


def attach_teach_panel(
    parent,
    visualizer,
    *,
    meshcat_url: str,
    controller=None,
    enabled: bool = False,
    actions_parent=None,
) -> TeachGraspGui | None:
    """로봇 펜던트 열에 사물 위치 블록을 붙인다. enabled 가 아니면 표시만 하고 잠근다."""
    try:
        mesh_path = cad_mesh_path()
        vertices_m, faces = load_mesh_m(mesh_path)
    except Exception as exc:
        print(f"사물 위치 패널: CAD 로드 실패 ({exc})")
        if enabled:
            return None
        mesh_path = Path("cad")
        vertices_m = np.zeros((1, 3))
        faces = np.zeros((0, 3), dtype=np.uint32)
    try:
        place, grasp = load_specs(CAD_YAML)
    except Exception as exc:
        print(f"사물 위치 패널: yaml 로드 실패 ({exc})")
        return None
    return TeachGraspGui(
        parent,
        visualizer=visualizer,
        mesh_path=mesh_path,
        vertices_m=vertices_m,
        faces=faces,
        place=place,
        grasp=grasp,
        meshcat_url=meshcat_url,
        kinematics=getattr(controller, "_kin", None) if controller is not None else None,
        controller=controller,
        own_robot=False,
        drive_viz=False,
        embedded=True,
        enabled=enabled,
        actions_parent=actions_parent,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="물체 배치·grasp 값을 Meshcat에서 확인·이동")
    p.add_argument("--no-open", action="store_true", help="Meshcat 브라우저를 자동으로 열지 않음")
    return p.parse_args()


def main() -> None:
    quiet_gtk()
    args = parse_args()
    try:
        import customtkinter as ctk  # noqa: F401
        import open3d as o3d  # noqa: F401
    except ImportError as exc:
        raise SystemExit("customtkinter / open3d 가 필요합니다. conda activate AIvision") from exc

    from motion import DEFAULT_URDF, Controller, RobotKinematics
    from motion.visualizer import Visualizer

    mesh_path = cad_mesh_path()
    vertices_m, faces = load_mesh_m(mesh_path)
    place, grasp = load_specs(CAD_YAML)

    kin = RobotKinematics(DEFAULT_URDF)
    viz = Visualizer(kin, open_browser=not args.no_open)
    url = viz.url or "(meshcat server running)"
    print(f"CAD      {mesh_path}")
    print(f"Meshcat  {url}")

    ctrl = Controller(kin)
    ctrl.start()

    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    gui = TeachGraspGui(
        root,
        visualizer=viz,
        kinematics=kin,
        controller=ctrl,
        own_robot=False,
        drive_viz=True,
        mesh_path=mesh_path,
        vertices_m=vertices_m,
        faces=faces,
        place=place,
        grasp=grasp,
        meshcat_url=url,
    )

    def _on_close() -> None:
        gui._cancel_register_poll()
        ctrl.stop()
        try:
            viz.close()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
