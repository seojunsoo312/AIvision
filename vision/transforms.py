"""로봇과 무관한 4x4 강체 변환 (mm)."""

from __future__ import annotations

import numpy as np


def rt_to_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def apply_T(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    if len(xyz) == 0:
        return xyz
    ones = np.ones((len(xyz), 1), dtype=np.float64)
    homo = np.hstack([xyz.astype(np.float64), ones])
    out = (T @ homo.T).T
    return out[:, :3].astype(np.float32)


def transform_plane(plane: np.ndarray, T: np.ndarray) -> np.ndarray:
    """src 좌표의 평면 ax+by+cz+d=0 → dst 좌표. p_dst = T @ p_src (mm)."""
    n = np.asarray(plane[:3], dtype=np.float64).reshape(3)
    d = float(plane[3])
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    n_dst = T[:3, :3] @ n
    d_dst = d - float(n_dst @ T[:3, 3])
    return np.array([n_dst[0], n_dst[1], n_dst[2], d_dst], dtype=np.float64)


def plane_tilt_from_z_deg(plane: np.ndarray) -> float:
    """평면과 XY (z=상수) 사이 각도. 0 = 수평."""
    n = np.asarray(plane[:3], dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(n))
    if length < 1e-12:
        return 90.0
    return float(np.degrees(np.arccos(np.clip(abs(n[2]) / length, 0.0, 1.0))))


def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """R @ a_hat = b_hat 인 회전 R."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    c = float(np.clip(a @ b, -1.0, 1.0))
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / float(np.linalg.norm(axis))
        return axis_angle_R(axis, np.pi)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    k = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + k + k @ k * ((1.0 - c) / (s * s))


def axis_angle_R(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def align_T_base_cam_to_desk_z(T_base_cam: np.ndarray, plane_cam: np.ndarray) -> np.ndarray:
    """책상 법선이 ±Z 와 나란하도록 카메라 원점을 중심으로 T 를 돌린다.

    이동 성분 (베이스에서 본 카메라 원점) 은 그대로다. 로봇이 책상에 수평으로
    놓였을 때 쓴다: 노란 RANSAC 평면이 로봇 XY 와 나란해야 한다.
    """
    T = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4).copy()
    plane_b = transform_plane(plane_cam, T)
    n = np.asarray(plane_b[:3], dtype=np.float64).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if float(n @ z) < 0.0:
        z = -z
    R_align = rotation_aligning(n, z)
    T[:3, :3] = R_align @ T[:3, :3]
    return T


def align_T_base_cam_to_base_desk_plane(
    T_base_cam: np.ndarray, plane_base: np.ndarray
) -> np.ndarray:
    """align_T_base_cam_to_desk_z 와 같지만 평면이 이미 베이스 좌표다."""
    T = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4)
    plane_cam = transform_plane(plane_base, invert_T(T))
    return align_T_base_cam_to_desk_z(T, plane_cam)
