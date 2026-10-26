"""사물 위치 패널이 3D 화면(Meshcat)에 그리는 것. Tk 화면도 계산도 없다.

  물체 CAD · 물체 축, 대기(P) 축과 주황 구, 집기(G) 축과 초록 구, P→G 선, 대각선 N 축,
  랜덤 배치 영역(초록 부채꼴), 놓기 위치(바닥의 빨간 원), 물체 창만 띄웠을 때의 홈 자세 팔.

좌표는 프로젝트 베이스(+X 앞, +Y 왼쪽, +Z 위)로 받아서 Meshcat(URDF) 좌표로 바꿔 그린다.
"""

from __future__ import annotations

import numpy as np

from motion.base_frame import T_urdf_from_user
from motion.grasp import N_CAD, AutoPreResult, PlacePose, place_region_overlay_m, pose_to_T

APPROACH_MARKER_RADIUS_M = 0.004  # 4 mm
# TCP 와 주황 대기 구 비교: 가까우면 초록으로 (go-to 디버그용).
PRE_TCP_MATCH_POS_MM = 1.5
APPROACH_PRE_COLOR = 0xFB923C
APPROACH_PRE_MATCH_COLOR = 0x22C55E
APPROACH_OTHER_COLOR = 0xC2410C
# Meshcat 의 놓기 위치 표시: 바닥에 붙은 빨간 원판 (반지름 · 두께 m).
DROP_MARKER_COLOR = 0xEF4444
DROP_MARKER_RADIUS_M = 0.003
DROP_MARKER_THICK_M = 0.0005


def T_meshcat(T_user: np.ndarray) -> np.ndarray:
    """프로젝트 베이스 자세 → Pinocchio/Meshcat (URDF 그대로) 월드."""
    return T_urdf_from_user(T_user)


def tcp_matches_pre(auto: AutoPreResult | None, tcp_xyz_m) -> bool:
    """TCP 가 주황 대기 구 위에 있으면 True (프로젝트 베이스 위치)."""
    if auto is None:
        return False
    err_m = float(np.linalg.norm(np.asarray(tcp_xyz_m, dtype=float) - auto.T_base[:3, 3]))
    return err_m * 1000.0 < PRE_TCP_MATCH_POS_MM


class TeachScene:
    """Meshcat 오버레이 이름과 색을 한곳에서 관리한다."""

    def __init__(self, viz) -> None:
        self._viz = viz
        self._cad_loaded = False

    # ------------------------------------------------------------ 물체 · 영역

    def draw_cad(self, vertices_m: np.ndarray, faces: np.ndarray, *, region: bool) -> None:
        if region:
            self.show_region()
        self._viz.set_overlay_mesh("cad", vertices_m, faces, color=0xC4C4C8, opacity=0.92)
        self._cad_loaded = True

    def show_region(self) -> None:
        region_v, region_f, region_loop = place_region_overlay_m()
        self._viz.set_overlay_mesh("place_region", region_v, region_f, color=0x86EFAC, opacity=0.28)
        self._viz.set_overlay_lines("place_region_edge", region_loop, color=0x4ADE80, closed=True)

    def hide_region(self) -> None:
        for name in ("place_region", "place_region_edge"):
            self._viz.clear_overlay(name)

    def hide_grasp(self) -> None:
        for name in ("grasp", "grasp_marker", "approach"):
            self._viz.clear_overlay(name)

    # ------------------------------------------------------------ 물체 자세 · 대기 · 집기

    def draw_place(
        self, place: PlacePose, auto: AutoPreResult, *, d_mm: float, matched: bool, grasp_hidden: bool
    ) -> None:
        """물체와 대기(P)·집기(G) 표시를 다시 그린다 (「적용」 한 번)."""
        T_cad = T_meshcat(pose_to_T(np.array(place.xyz_mm), np.array(place.rpy_deg)))
        T_p = T_meshcat(auto.T_base)
        T_g = T_meshcat(auto.T_grasp_base)
        if self._cad_loaded:
            self._viz.set_overlay_transform("cad", T_cad)
        self._viz.set_overlay_axes("cad_axes", T_cad, scale=0.02, labels=False)
        self._viz.clear_overlay("base_at_obj")
        self._viz.set_overlay_axes("pregrasp", T_p, scale=0.02)
        # 가상 N 축 (x=z, y=0) 은 그 접근을 골랐을 때만 그린다.
        if auto.axis_name == "N":
            n_world = T_cad[:3, :3] @ N_CAD
            o = T_cad[:3, 3]
            half = max(float(d_mm), 20.0) / 1000.0
            self._viz.set_overlay_segment("n_axis", o - n_world * half, o + n_world * half, color=0xC026D3)
        else:
            self._viz.clear_overlay("n_axis")
        self.draw_approach_markers(auto, matched=matched)
        if grasp_hidden:
            self.hide_grasp()
        else:
            self._viz.set_overlay_axes("grasp", T_g, scale=0.0225)
            self._viz.set_overlay_segment("approach", T_p[:3, 3], T_g[:3, 3])
            self._viz.set_overlay_spheres(
                "grasp_marker", T_g[:3, 3].reshape(1, 3), radius_m=APPROACH_MARKER_RADIUS_M, color=0x22C55E
            )

    def draw_approach_markers(self, auto: AutoPreResult | None, *, matched: bool) -> None:
        """대기 위치 구. TCP 가 그 위에 있으면 초록, 아니면 주황. 쓰지 않은 눕힘 대기는 진한 주황."""
        if auto is None:
            return
        centers = [T_meshcat(auto.T_base)[:3, 3]]
        colors = [APPROACH_PRE_MATCH_COLOR if matched else APPROACH_PRE_COLOR]
        if auto.other_pre_base is not None:
            T_other = np.eye(4)
            T_other[:3, 3] = np.asarray(auto.other_pre_base, dtype=float)
            centers.append(T_meshcat(T_other)[:3, 3])
            colors.append(APPROACH_OTHER_COLOR)
        self._viz.clear_overlay("approach_markers")
        self._viz.set_overlay_spheres(
            "approach_markers", np.stack(centers, axis=0), radius_m=APPROACH_MARKER_RADIUS_M, colors=colors
        )

    # ------------------------------------------------------------ 놓기 위치

    def draw_drop(self, dx_mm: float, dy_mm: float) -> None:
        """놓기 x/y 위치를 바닥에 빨간 원판으로 보인다."""
        T = np.eye(4)
        T[:3, 3] = [dx_mm / 1000.0, dy_mm / 1000.0, 0.0]
        # 바닥(z=0) 바로 위에 두께의 절반만 올려 바닥 격자와 겹쳐 깜박이지 않게 한다.
        center = T_meshcat(T)[:3, 3] + np.array([0.0, 0.0, DROP_MARKER_THICK_M / 2.0])
        self._viz.clear_overlay("drop_marker")
        self._viz.set_overlay_disc(
            "drop_marker", center, radius_m=DROP_MARKER_RADIUS_M, thickness_m=DROP_MARKER_THICK_M, color=DROP_MARKER_COLOR
        )

    def clear_drop(self) -> None:
        self._viz.clear_overlay("drop_marker")

    # ------------------------------------------------------------ 팔

    def show_home_robot(self, kin, gripper_100: float) -> None:
        """물체 창만 띄웠을 때: 팔을 홈 자세로 그린다."""
        if kin is None:
            return
        from motion.hw_controller import grip_100_to_user
        from motion.robot_kinematics import GRIPPER_JOINT, HOME_JOINTS_DEG, URDF_JOINT_NAMES

        joints = {name: float(HOME_JOINTS_DEG[name]) for name in URDF_JOINT_NAMES}
        joints[GRIPPER_JOINT] = grip_100_to_user(gripper_100)
        self._viz.display(kin.q_from_deg(joints))

    def display(self, q) -> None:
        self._viz.display(q)
