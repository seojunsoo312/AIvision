"""브라켓 CAD 의 회전 대칭을 수치로 구한다.

T 와 T @ S 가 같은 모양을 같은 자리에 놓으면 (S 가 대칭) 두 자세는 같은 답으로 친다.
축 방향 회전 24개 중, CAD 점군을 중심 기준으로 돌렸을 때 자기 자신과 거리 중앙값이 SYM_TOL_MM 이하인 것만 남긴다.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

SYM_TOL_MM = 0.5


def _axis_rotations() -> list[np.ndarray]:
    """좌표축끼리만 바꾸는 회전 24개."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            R = np.zeros((3, 3))
            for i, j in enumerate(perm):
                R[i, j] = signs[i]
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


@lru_cache(maxsize=1)
def _symmetries_cached() -> tuple[tuple[float, ...], ...]:
    import open3d as o3d

    from cad.model import load_mesh_m

    verts_m, faces = load_mesh_m()
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(verts_m, float) * 1000.0),
        o3d.utility.Vector3iVector(np.asarray(faces, np.int32)),
    )
    # 표면을 촘촘히 뽑아 1 mm 격자로 줄인다 (register.load_cad_xyz 와 같은 방식, yolo 를 부르지 않으려고 따로 둔다).
    o3d.utility.random.seed(0)
    pcd = mesh.sample_points_uniformly(number_of_points=80000).voxel_down_sample(1.0)
    P = np.asarray(pcd.points, dtype=np.float64)
    c = P.mean(axis=0)
    ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    found = []
    for R in _axis_rotations():
        Q = (P - c) @ R.T + c
        d = np.asarray(ref.compute_point_cloud_distance(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q))))
        if float(np.median(d)) <= SYM_TOL_MM:
            S = np.eye(4)
            S[:3, :3] = R
            S[:3, 3] = c - R @ c
            found.append(tuple(S.ravel()))
    return tuple(found)


def cad_symmetries() -> list[np.ndarray]:
    """CAD 프레임 → CAD 프레임 4×4 대칭 목록. 항상 단위 행렬을 포함한다."""
    return [np.asarray(s, float).reshape(4, 4) for s in _symmetries_cached()]
