"""프로젝트 베이스 좌표와 Pinocchio/URDF 월드 좌표의 관계.

사용자가 보는 베이스 좌표 (펜던트, 놓기 자세 가르치기, 손눈 캘리브, register --base):
  +X = 로봇 앞, +Z = 위, +Y = 왼쪽 (오른손 좌표계).

예전 Rz(180°) 사용자 좌표를 같은 원점에서 Rz(+90°) 돌린 것이다.
같은 말로 p_urdf = Rz(-90°) @ p_user.

  p_urdf = R @ p_user
  R_urdf = R @ R_user
  T_urdf = T_urdf_from_user @ T_user

FK·IK·충돌 계산은 안에서 계속 URDF 좌표로 한다.
"""

from __future__ import annotations

import numpy as np

from motion.robot_kinematics import TcpPose, rotmat_to_rpy_deg, rpy_deg_to_rotmat

USER_BASE_FRAME = "project_xfwd"
LEGACY_BASE_FRAME = "project_rz180"

# p_urdf = R_URDF_FROM_USER @ p_user  (Rz(-90°))
R_URDF_FROM_USER = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
R_USER_FROM_URDF = R_URDF_FROM_USER.T

# 예전 사용자 좌표(+X 오른쪽, +Y 앞) → 지금 사용자 좌표(+X 앞, +Y 왼쪽).
R_USER_FROM_LEGACY = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def T_urdf_from_user_matrix() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_URDF_FROM_USER
    return T


def xyz_urdf_from_user(xyz: np.ndarray) -> np.ndarray:
    return R_URDF_FROM_USER @ np.asarray(xyz, dtype=float).reshape(3)


def xyz_user_from_urdf(xyz: np.ndarray) -> np.ndarray:
    return R_USER_FROM_URDF @ np.asarray(xyz, dtype=float).reshape(3)


def rot_urdf_from_user(R_user: np.ndarray) -> np.ndarray:
    return R_URDF_FROM_USER @ np.asarray(R_user, dtype=float).reshape(3, 3)


def rot_user_from_urdf(R_urdf: np.ndarray) -> np.ndarray:
    return R_USER_FROM_URDF @ np.asarray(R_urdf, dtype=float).reshape(3, 3)


def T_urdf_from_user(T_user: np.ndarray) -> np.ndarray:
    return T_urdf_from_user_matrix() @ np.asarray(T_user, dtype=float).reshape(4, 4)


def T_user_from_stored(T: np.ndarray, base_frame: str | None) -> np.ndarray:
    """`base_frame` 기준으로 저장된 4×4 를 지금 사용자 베이스 좌표로 옮긴다."""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    frame = (base_frame or LEGACY_BASE_FRAME).strip()
    if frame == USER_BASE_FRAME:
        return T
    if frame == LEGACY_BASE_FRAME:
        out = np.eye(4)
        out[:3, :3] = R_USER_FROM_LEGACY
        return out @ T
    raise ValueError(f"unknown base_frame: {base_frame!r}")


def tcp_pose_user_from_urdf(pose: TcpPose) -> TcpPose:
    R_u = rot_user_from_urdf(pose.rotation)
    xyz_u = xyz_user_from_urdf(pose.xyz_m)
    return TcpPose(xyz_m=xyz_u, rpy_deg=rotmat_to_rpy_deg(R_u), rotation=R_u)


def ee_target_urdf_from_user(xyz_mm: np.ndarray, rpy_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """사용자 Go-to 값 (mm, deg) → URDF 미터 위치 + 회전 행렬."""
    xyz_u = np.asarray(xyz_mm, dtype=float).reshape(3) / 1000.0
    rpy = np.asarray(rpy_deg, dtype=float).reshape(3)
    R_u = rpy_deg_to_rotmat(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    return xyz_urdf_from_user(xyz_u), rot_urdf_from_user(R_u)
