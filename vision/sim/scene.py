"""학생 책상 장면을 MuJoCo 로 만들고 컬러 · 뎁스 · 분할 · 정답 자세를 저장한다.

좌표는 프로젝트 베이스(+X 앞, +Y 왼쪽, +Z 위, 책상 z=0). MuJoCo 안은 미터, 저장은 mm.
브라켓은 CAD 프레임(mesh_rpy 적용 후) 정점으로 만들어, body 자세가 곧 정합 결과 T(CAD → 베이스)와 같은 뜻이다.

    python -m vision.sim.scene --count 1 --objects 1 --seed 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# 화면 없는 렌더. 노트북에서 창으로 보려면 MUJOCO_GL=glfw 로 바꾼다.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco

from cad.model import load_mesh_m
from vision.sim.data import HEIGHT, OUT_DIR, WIDTH, Scene, save_scene

# 실제 Orbbec(fx ≈ 454)과 같은 배율이 되는 세로 화각. 브라켓이 약 45 픽셀로 보인다.
FOVY_DEG = 55.7

# 2026-10-29 손눈 결과 (학생 책상). 180° 돌린 영상 기준, OpenCV 카메라 축.
T_BASE_CAM = np.array(
    [
        [-0.07449, -0.99647, -0.03875, 149.427],
        [-0.99719, 0.07413, 0.01057, -21.326],
        [-0.00766, 0.03943, -0.99919, 452.327],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# 브라켓 놓는 범위 (베이스 mm). 로봇 베이스 둘레는 뺀다.
PLACE_X_MM = (100.0, 260.0)
PLACE_Y_MM = (-180.0, 180.0)
ROBOT_KEEPOUT_MM = 90.0
# 브라켓끼리 이만큼은 떨어져 있어야 한다 (가장 가까운 점 사이 xy 거리).
MIN_GAP_MM = 30.0

DESK_RGBA = (0.74, 0.74, 0.76, 1.0)
# 학습 데이터의 브라켓 색 비율: 흰색 · 은색 · 검정. 수업은 흰색을 쓰지만 다른 색도 견디게.
BRACKET_COLOR_P = (0.6, 0.2, 0.2)
# 학생 로봇: 흰색 3D 출력 부품 + 검은 서보.
ROBOT_WHITE_RGBA = (0.93, 0.93, 0.92, 1.0)
ROBOT_SERVO_RGBA = (0.12, 0.12, 0.13, 1.0)
BRACKET_RGBA = (0.94, 0.94, 0.93, 1.0)

# 정지 판정: 모든 속도가 이 값보다 작은 상태가 SETTLE_HOLD_S 동안 이어지면 멈춘 것으로 본다.
SETTLE_VEL = 2e-3
SETTLE_HOLD_S = 0.2
SETTLE_MAX_S = 4.0

# 분할에서 이보다 작은 조각은 경계 색 섞임으로 보고 버린다.
SEG_MIN_PART_PX = 20


def _orthonormal(R: np.ndarray) -> np.ndarray:
    """손눈 결과의 회전을 가장 가까운 회전 행렬로 다듬는다 (소수점 반올림 오차 제거)."""
    U, _s, Vt = np.linalg.svd(np.asarray(R, dtype=float))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def default_T_base_cam() -> np.ndarray:
    T = T_BASE_CAM.copy()
    T[:3, :3] = _orthonormal(T[:3, :3])
    return T


def camera_K(fovy_deg: float = FOVY_DEG, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """MuJoCo 렌더 카메라의 K. 정사각 픽셀, 주점은 화면 정중앙 (픽셀 중심이 정수인 OpenCV 기준)."""
    f = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def rpy_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """XYZ 외부 회전 RPY (도): R = Rz @ Ry @ Rx. motion.robot_kinematics 와 같은 약속."""
    r, p, y = np.deg2rad([roll, pitch, yaw])
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# 자세 클래스마다 "CAD 프레임에서 본 위쪽 방향". register.class_base_rpys 의 yaw=0 대표 자세에서 구한다.
_CLASS_REP_RPY = {0: [(0, 0, 0)], 1: [(90, 0, 0), (-90, 0, 0)], 2: [(180, 45, 180)]}
CLASS_TOL_DEG = 15.0


def pose_class(R_base_obj: np.ndarray) -> tuple[int, float]:
    """놓인 자세 → (자세 클래스, 대표 자세와의 각도 차). 어느 것과도 CLASS_TOL_DEG 넘게 다르면 −1.

    대칭으로 같은 자세(예: 세운 L 을 반대쪽 판으로 세움)도 같은 클래스로 본다.
    """
    from vision.sim.symmetry import cad_symmetries

    best = (-1, 180.0)
    for S in cad_symmetries():
        up_cad = (np.asarray(R_base_obj, dtype=float) @ S[:3, :3]).T @ np.array([0.0, 0.0, 1.0])
        for cid, rpys in _CLASS_REP_RPY.items():
            for rpy in rpys:
                ref = rpy_R(*rpy).T @ np.array([0.0, 0.0, 1.0])
                ang = float(np.degrees(np.arccos(np.clip(up_cad @ ref, -1.0, 1.0))))
                if ang < best[1]:
                    best = (cid, ang)
    if best[1] > CLASS_TOL_DEG:
        return -1, best[1]
    return best


def _fmt(vals) -> str:
    return " ".join(f"{float(v):.6g}" for v in np.asarray(vals).ravel())


def build_xml(
    n_objects: int,
    *,
    T_base_cam: np.ndarray,
    fovy_deg: float = FOVY_DEG,
    width: int = WIDTH,
    height: int = HEIGHT,
    n_distractors: int = 0,
    robot_parts: list[dict] | None = None,
) -> str:
    """책상 + 조명 + 카메라 + 브라켓 n 개 (각자 freejoint) + 방해물 n_distractors 개 MJCF 문자열.

    방해물은 물리 없이 자리만 정하는 mocap 물체다 (상자 · 원통 · 캡슐 · 타원체 · 구를 돌아가며).
    크기 · 색 · 자리는 장면마다 SceneMaker.place_distractors 가 바꾼다.
    robot_parts 를 주면 로봇 팔 메시를 부품마다 mocap 물체로 넣는다 (자세는 SceneMaker.set_robot).
    """
    verts_m, faces = load_mesh_m()
    R = T_base_cam[:3, :3]
    cam_pos = T_base_cam[:3, 3] / 1000.0
    # MuJoCo 카메라: x 오른쪽, y 위, −z 앞. OpenCV: x 오른쪽, y 아래, +z 앞.
    xyaxes = np.concatenate([R[:, 0], -R[:, 1]])
    bodies = []
    for i in range(n_objects):
        bodies.append(
            f"""
    <body name="bracket{i}" pos="{0.15 + 0.06 * i} 0 0.1">
      <freejoint name="free{i}"/>
      <geom name="bracket{i}" type="mesh" mesh="bracket" material="bracket"
            friction="0.6 0.005 0.0001" condim="4" solref="0.004 1"/>
    </body>"""
        )
    for j in range(n_distractors):
        kind = DISTRACTOR_KINDS[j % len(DISTRACTOR_KINDS)]
        bodies.append(
            f"""
    <body name="distract{j}" mocap="true" pos="2 {2 + 0.1 * j} 0.02">
      <geom name="distract{j}" type="{kind}" size="0.01 0.01 0.01" rgba="0.5 0.5 0.5 1"
            contype="0" conaffinity="0"/>
    </body>"""
        )
    robot_assets = []
    for part in robot_parts or []:
        bodies.append(
            f"""
    <body name="{part['name']}" mocap="true" pos="2 -2 -0.5">
      <geom name="{part['name']}" type="mesh" mesh="{part['mesh']}" material="{part['material']}"
            contype="0" conaffinity="0"/>
    </body>"""
        )
    for mesh, (path, scale) in {p_["mesh"]: (p_["file"], p_["scale"]) for p_ in robot_parts or []}.items():
        robot_assets.append(f'<mesh name="{mesh}" file="{path}" scale="{_fmt(scale)}"/>')
    return f"""
<mujoco model="sim_desk">
  <option timestep="0.002"/>
  <statistic extent="1" center="0.15 0 0.2"/>
  <visual>
    <global offwidth="{width}" offheight="{height}"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.15 0.15 0.15" specular="0 0 0"/>
    <map znear="0.01" zfar="5"/>
  </visual>
  <asset>
    <material name="desk" rgba="{_fmt(DESK_RGBA)}" specular="0.1" shininess="0.2"/>
    <material name="bracket" rgba="{_fmt(BRACKET_RGBA)}" specular="0.3" shininess="0.4"/>
    <mesh name="bracket" vertex="{_fmt(verts_m)}" face="{_fmt(faces)}"/>
    <material name="robot_white" rgba="{_fmt(ROBOT_WHITE_RGBA)}" specular="0.2" shininess="0.3"/>
    <material name="robot_servo" rgba="{_fmt(ROBOT_SERVO_RGBA)}" specular="0.3" shininess="0.5"/>
    {''.join(robot_assets)}
  </asset>
  <worldbody>
    <!-- 방향광(directional)은 그림자에 줄무늬가 생겨서 스포트라이트를 쓴다. -->
    <light name="key" directional="false" pos="0.45 0.3 0.9" dir="-0.3 -0.3 -1" cutoff="70" exponent="1"
           diffuse="0.65 0.65 0.65" specular="0.1 0.1 0.1" castshadow="true"/>
    <geom name="desk" type="plane" size="1.2 1.2 0.01" material="desk"/>
    <camera name="cam" pos="{_fmt(cam_pos)}" xyaxes="{_fmt(xyaxes)}" fovy="{fovy_deg}"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>
"""


# 방해물 모양 (MuJoCo 기본 도형). 과자 봉지 · 테이프 · 케이블 · 로봇 · 다른 부품을 대충 흉내 낸다.
DISTRACTOR_KINDS = ("box", "cylinder", "capsule", "ellipsoid", "box", "capsule", "sphere", "cylinder")
# 방해물을 놓는 범위 (베이스 mm). 카메라 화면 전체와 그 바깥 조금.
DISTRACT_X_MM = (-40.0, 340.0)
DISTRACT_Y_MM = (-230.0, 190.0)
# 방해물과 브라켓 사이 최소 간격 (xy, mm). 브라켓을 가리면 정답 라벨이 틀어진다.
DISTRACT_GAP_MM = 15.0


# 카메라 흔들기: 위치 ± mm, 회전 ± 도 (손눈 결과가 교실마다 이 정도 다르다고 본다).
CAM_JITTER_MM = 8.0
CAM_JITTER_DEG = 2.0


def jitter_camera(T: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """카메라 자세를 무작위로 조금 흔든 T_base_cam."""
    from vision.transforms import axis_angle_R

    out = np.asarray(T, float).copy()
    axis = rng.normal(size=3)
    angle = np.deg2rad(rng.uniform(0.0, CAM_JITTER_DEG))
    out[:3, :3] = axis_angle_R(axis / np.linalg.norm(axis), angle) @ out[:3, :3]
    out[:3, 3] += rng.uniform(-CAM_JITTER_MM, CAM_JITTER_MM, 3)
    return out


def _random_quat(rng: np.random.Generator) -> np.ndarray:
    """고르게 퍼진 무작위 회전 (w, x, y, z)."""
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def _sample_xy(rng: np.random.Generator) -> np.ndarray:
    while True:
        x = rng.uniform(*PLACE_X_MM)
        y = rng.uniform(*PLACE_Y_MM)
        if np.hypot(x, y) >= ROBOT_KEEPOUT_MM + 30.0:
            return np.array([x, y])


class SceneMaker:
    """같은 모델로 장면을 여러 개 만든다. 장면마다 브라켓을 떨어뜨려 물리로 눕힌 뒤 렌더한다."""

    def __init__(
        self,
        n_objects: int = 1,
        *,
        T_base_cam: np.ndarray | None = None,
        fovy_deg: float = FOVY_DEG,
        n_distractors: int = 0,
        robot: bool = False,
    ) -> None:
        self.n = int(n_objects)
        self.n_distract = int(n_distractors)
        self.kin = None
        parts: list[dict] = []
        if robot:
            from motion.robot_kinematics import DEFAULT_URDF, RobotKinematics

            self.kin = RobotKinematics(DEFAULT_URDF)
            for g in self.kin.visual_model.geometryObjects:
                stem = Path(g.meshPath).stem
                parts.append({
                    "name": f"robot_{g.name}",
                    "mesh": f"rm_{stem}_{'_'.join(f'{v:g}' for v in g.meshScale)}",
                    "file": g.meshPath,
                    "scale": np.asarray(g.meshScale, float),
                    "material": "robot_servo" if stem.startswith("sts3215") else "robot_white",
                })
        self.T_base_cam = default_T_base_cam() if T_base_cam is None else np.asarray(T_base_cam, float)
        self.K = camera_K(fovy_deg)
        self.model = mujoco.MjModel.from_xml_string(
            build_xml(self.n, T_base_cam=self.T_base_cam, fovy_deg=fovy_deg, n_distractors=self.n_distract,
                      robot_parts=parts)
        )
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
        self.geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"bracket{i}") for i in range(self.n)
        ]
        self.body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"bracket{i}") for i in range(self.n)
        ]
        self.qadr = [
            int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"free{i}")])
            for i in range(self.n)
        ]
        self.distract_geoms = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"distract{j}") for j in range(self.n_distract)
        ]
        self.distract_mocap = [
            int(self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"distract{j}")])
            for j in range(self.n_distract)
        ]
        # 이번 장면에 보이는 방해물 수. make() 전에 정한다.
        self.distract_active = 0
        self.robot_mocap = [
            int(self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, p_["name"])])
            for p_ in parts
        ]
        self._mat_robot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "robot_white")
        # 다음 make() 에서 쓸 로봇 관절 (펜던트 각도, 도). None 이면 로봇을 화면 밖에 둔다.
        self.robot_joints_deg: dict[str, float] | None = None
        if self.kin is not None:
            from motion.robot_kinematics import INIT_POSE_JOINTS_DEG

            self.robot_joints_deg = dict(INIT_POSE_JOINTS_DEG)
        verts_m, _faces = load_mesh_m()
        self.verts_mm = np.asarray(verts_m, float) * 1000.0
        # 떨어뜨릴 브라켓 수. 나머지는 카메라 밖 책상에 치워 둔다 (모델을 다시 만들지 않으려고).
        self.active = self.n
        self._mat_desk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "desk")
        self._mat_bracket = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "bracket")
        self._look0 = (
            self.model.mat_rgba.copy(),
            self.model.light_pos.copy(),
            self.model.light_dir.copy(),
            self.model.light_diffuse.copy(),
            self.model.vis.headlight.ambient.copy(),
            self.model.mat_specular.copy(),
            self.model.mat_shininess.copy(),
        )

    # ---- 카메라 · 겉모습 (학습 데이터용) ----

    def set_camera(self, T_base_cam: np.ndarray) -> None:
        """카메라 자세를 바꾼다 (손눈 결과가 조금 달라지는 것을 흉내). K 는 그대로."""
        T = np.asarray(T_base_cam, float).copy()
        T[:3, :3] = _orthonormal(T[:3, :3])
        R = T[:3, :3]
        # MuJoCo 카메라 축: x 오른쪽, y 위, z 뒤 (OpenCV 의 y, z 를 뒤집은 것).
        R_mj = np.column_stack([R[:, 0], -R[:, 1], -R[:, 2]])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_mj.ravel())
        self.model.cam_pos[self.cam_id] = T[:3, 3] / 1000.0
        self.model.cam_quat[self.cam_id] = quat
        self.T_base_cam = T

    def randomize_look(self, rng: np.random.Generator) -> None:
        """책상 · 브라켓 색, 조명 방향 · 세기를 무작위로. 실제 교실 조명 차이를 견디게 하려고."""
        g = rng.uniform(0.5, 0.88)
        tint = rng.uniform(-0.03, 0.03, 3)
        self.model.mat_rgba[self._mat_desk, :3] = np.clip(g + tint, 0, 1)
        self.set_bracket_color(str(rng.choice(["white", "silver", "black"], p=BRACKET_COLOR_P)), rng)
        if self._mat_robot >= 0:
            wr = rng.uniform(0.82, 0.98)
            self.model.mat_rgba[self._mat_robot, :3] = np.clip(wr + rng.uniform(-0.02, 0.02, 3), 0, 1)
        # 스포트라이트를 작업 구역 위 무작위 자리에 두고 작업 구역 가운데를 비춘다.
        pos = np.array([0.18 + rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0.8, 1.4)])
        d = np.array([0.18, 0.0, 0.0]) - pos
        self.model.light_pos[0] = pos
        self.model.light_dir[0] = d / np.linalg.norm(d)
        self.model.light_diffuse[0] = rng.uniform(0.35, 0.85)
        self.model.vis.headlight.ambient[:] = rng.uniform(0.2, 0.5)

    def set_bracket_color(self, color: str, rng: np.random.Generator | None = None) -> None:
        """브라켓 색: white (흰 출력물) · silver (은색 금속, 반짝임) · black (검정)."""
        rng = np.random.default_rng(0) if rng is None else rng
        m, i = self.model, self._mat_bracket
        if color == "white":
            g, spec, shin = rng.uniform(0.8, 1.0), 0.3, 0.4
        elif color == "silver":
            g, spec, shin = rng.uniform(0.55, 0.75), rng.uniform(0.6, 0.9), rng.uniform(0.6, 0.9)
        elif color == "black":
            g, spec, shin = rng.uniform(0.04, 0.18), rng.uniform(0.2, 0.5), rng.uniform(0.3, 0.6)
        else:
            raise ValueError(f"모르는 브라켓 색: {color}")
        m.mat_rgba[i, :3] = np.clip(g + rng.uniform(-0.02, 0.02, 3), 0, 1)
        m.mat_specular[i] = spec
        m.mat_shininess[i] = shin

    def reset_look(self) -> None:
        mat, lpos, ldir, ldiff, amb, spec, shin = self._look0
        self.model.mat_rgba[:] = mat
        self.model.mat_specular[:] = spec
        self.model.mat_shininess[:] = shin
        self.model.light_pos[:] = lpos
        self.model.light_dir[:] = ldir
        self.model.light_diffuse[:] = ldiff
        self.model.vis.headlight.ambient[:] = amb

    def set_look(self, name: str, rng: np.random.Generator) -> None:
        """이름 붙은 겉모습. default · random · lowcontrast (밝은 책상, 흰 브라켓과 비슷) · dim (어두움) · harsh (낮은 강한 조명, 짙은 그림자)."""
        self.reset_look()
        if name == "default":
            return
        if name in ("silver", "black"):
            self.set_bracket_color(name, rng)
            return
        if name == "random":
            self.randomize_look(rng)
            return
        m = self.model
        if name == "lowcontrast":
            m.mat_rgba[self._mat_desk, :3] = 0.86
            m.mat_rgba[self._mat_bracket, :3] = 0.92
        elif name == "dim":
            m.light_diffuse[0] = 0.25
            m.vis.headlight.ambient[:] = 0.15
        elif name == "harsh":
            pos = np.array([0.18 + 0.7, 0.5, 0.45])
            m.light_pos[0] = pos
            m.light_dir[0] = (np.array([0.18, 0.0, 0.0]) - pos) / np.linalg.norm(np.array([0.18, 0.0, 0.0]) - pos)
            m.light_diffuse[0] = 0.95
            m.vis.headlight.ambient[:] = 0.12
        else:
            raise ValueError(f"모르는 겉모습: {name}")

    # ---- 자세 ----

    def T_base_obj(self, i: int) -> np.ndarray:
        """i 번째 브라켓의 CAD 프레임 → 베이스 (mm)."""
        b = self.body_ids[i]
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[b].reshape(3, 3)
        T[:3, 3] = self.data.xpos[b] * 1000.0
        return T

    def _world_verts(self, i: int) -> np.ndarray:
        T = self.T_base_obj(i)
        return self.verts_mm @ T[:3, :3].T + T[:3, 3]

    def start_drop(self, rng: np.random.Generator) -> None:
        """브라켓을 범위 안 공중에 무작위 방향으로 놓는다. 아직 떨어뜨리지는 않는다."""
        mujoco.mj_resetData(self.model, self.data)
        starts: list[np.ndarray] = []
        for i in range(self.active, self.n):
            # 쓰지 않는 브라켓은 카메라 밖 책상 위에 둔다.
            a = self.qadr[i]
            self.data.qpos[a : a + 3] = [0.9, -0.9 + 0.1 * i, 0.03]
            self.data.qpos[a + 3 : a + 7] = [1.0, 0.0, 0.0, 0.0]
        for i in range(self.active):
            for _ in range(200):
                xy = _sample_xy(rng)
                if all(np.linalg.norm(xy - s) > 70.0 for s in starts):
                    break
            starts.append(xy)
            a = self.qadr[i]
            self.data.qpos[a : a + 3] = [xy[0] / 1000.0, xy[1] / 1000.0, 0.06 + 0.01 * i]
            self.data.qpos[a + 3 : a + 7] = _random_quat(rng)
        mujoco.mj_forward(self.model, self.data)
        self._still_s = 0.0
        self._t_s = 0.0

    def step(self) -> bool:
        """물리 한 스텝. 다 멈췄거나 시간이 다 되면 True."""
        mujoco.mj_step(self.model, self.data)
        dt = float(self.model.opt.timestep)
        self._t_s += dt
        if float(np.max(np.abs(self.data.qvel))) < SETTLE_VEL:
            self._still_s += dt
        else:
            self._still_s = 0.0
        return self._still_s >= SETTLE_HOLD_S or self._t_s >= SETTLE_MAX_S

    def freeze(self) -> None:
        """멈춘 자세로 고정하고 렌더용 위치를 다시 계산한다."""
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _drop(self, rng: np.random.Generator) -> None:
        self.start_drop(rng)
        while not self.step():
            pass
        self.freeze()

    def placement_ok(self) -> bool:
        """범위 안, 로봇 둘레 밖, 책상에 붙어 있고, 서로 MIN_GAP_MM 이상 떨어져 있으면 True."""
        pts = [self._world_verts(i) for i in range(self.active)]
        for i, p in enumerate(pts):
            c = p.mean(axis=0)
            if not (PLACE_X_MM[0] - 20 <= c[0] <= PLACE_X_MM[1] + 20 and PLACE_Y_MM[0] - 20 <= c[1] <= PLACE_Y_MM[1] + 20):
                return False
            if np.hypot(c[0], c[1]) < ROBOT_KEEPOUT_MM:
                return False
            # 책상 밑으로 파고들거나 공중에 뜨면 버린다.
            if p[:, 2].min() < -1.0 or p[:, 2].min() > 1.0:
                return False
        for i in range(self.active):
            for j in range(i + 1, self.active):
                d = np.linalg.norm(pts[i][::4, None, :2] - pts[j][None, ::4, :2], axis=2).min()
                if d < MIN_GAP_MM:
                    return False
        return True

    # ---- 렌더 ----

    @property
    def renderer(self) -> mujoco.Renderer:
        # 처음 렌더할 때 만든다. 뷰어만 쓸 때는 GL 문맥을 하나 덜 만든다.
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH)
        return self._renderer

    def render(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self.renderer
        r.update_scene(self.data, camera=self.cam_id)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.data, camera=self.cam_id)
        depth_m = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(self.data, camera=self.cam_id)
        seg_raw = r.render().copy()
        r.disable_segmentation_rendering()
        seg = np.zeros(seg_raw.shape[:2], np.uint8)
        is_geom = seg_raw[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        for i, gid in enumerate(self.geom_ids):
            mask = is_geom & (seg_raw[..., 0] == gid)
            # 안티에일리어싱이 경계 색을 섞어 엉뚱한 id 로 읽히는 점이 1~8 픽셀씩 흩어져 생긴다. 작은 조각은 버린다.
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            for c in range(1, n):
                if stats[c, cv2.CC_STAT_AREA] >= SEG_MIN_PART_PX:
                    seg[lab == c] = i + 1
        depth_mm = (depth_m * 1000.0).astype(np.float32)
        depth_mm[~np.isfinite(depth_mm)] = 0.0
        return rgb, depth_mm, seg

    def set_robot(self, joints_deg: dict[str, float] | None) -> None:
        """로봇 팔 부품들을 관절 각도대로 놓는다 (펜던트 각도, 도). None 이면 화면 밖으로."""
        if self.kin is None:
            return
        import pinocchio as pin

        from motion.base_frame import rot_user_from_urdf, xyz_user_from_urdf

        d = self.data
        if joints_deg is None:
            for mid in self.robot_mocap:
                d.mocap_pos[mid] = [2.0, -2.0, -0.5]
            return
        k = self.kin
        q = k.q_from_deg(joints_deg)
        pin.forwardKinematics(k.model, k.data, q)
        gdata = k.visual_model.createData() if not hasattr(self, "_vis_data") else self._vis_data
        self._vis_data = gdata
        pin.updateGeometryPlacements(k.model, k.data, k.visual_model, gdata)
        for mid, oMg in zip(self.robot_mocap, gdata.oMg):
            R = rot_user_from_urdf(np.asarray(oMg.rotation))
            p = xyz_user_from_urdf(np.asarray(oMg.translation))
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R).ravel())
            d.mocap_pos[mid] = p
            d.mocap_quat[mid] = quat

    def random_robot_joints(self, rng: np.random.Generator) -> dict[str, float]:
        """학습용 로봇 자세: 대개 시작 자세 근처, 가끔 크게 다른 자세."""
        from motion.robot_kinematics import INIT_POSE_JOINTS_DEG

        out = dict(INIT_POSE_JOINTS_DEG)
        spread = 15.0 if rng.random() < 0.6 else 60.0
        for name in ("S1", "S2", "S3", "S4", "S5", "S6"):
            out[name] = float(out[name] + rng.uniform(-spread, spread))
        out["S7"] = float(rng.uniform(-50.0, 50.0))
        return out

    def place_distractors(self, rng: np.random.Generator) -> None:
        """방해물을 브라켓과 겹치지 않게 책상 위에 놓는다. 쓰지 않는 것은 화면 밖에 둔다.

        크기는 케이블처럼 가는 것부터 로봇 팔 · 과자 봉지만 한 것까지, 색은 흰색 · 은색 · 보라 등 무작위.
        흰색 · 회색 물건을 일부러 많이 넣는다 (흰 브라켓과 헷갈리는 것을 배워야 해서).
        """
        m, d = self.model, self.data
        foot = [self._world_verts(i)[::8, :2] for i in range(self.active)]
        for j, (gid, mid) in enumerate(zip(self.distract_geoms, self.distract_mocap)):
            if j >= self.distract_active:
                d.mocap_pos[mid] = [2.0, 2.0 + 0.1 * j, -0.5]
                continue
            kind = int(m.geom_type[gid])
            for _ in range(50):
                if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE) and rng.random() < 0.6:
                    # 케이블: 가늘고 길게.
                    r, half = rng.uniform(0.0015, 0.004), rng.uniform(0.03, 0.12)
                    size = np.array([r, half, 0.0])
                else:
                    a = rng.uniform(0.006, 0.05, 3)
                    if rng.random() < 0.2:  # 로봇 · 상자처럼 큰 것
                        a = rng.uniform(0.03, 0.08, 3)
                    size = a
                xy = np.array([rng.uniform(*DISTRACT_X_MM), rng.uniform(*DISTRACT_Y_MM)]) / 1000.0
                yaw = rng.uniform(0, 2 * np.pi)
                lying = kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)) and rng.random() < 0.6
                if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    z = size[0]
                elif kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
                    z = size[0] if lying else size[1] + (size[0] if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0)
                else:
                    z = size[2]
                reach = float(np.max(size[:2]) + (size[1] if lying else 0.0)) * 1000.0
                if all(np.min(np.linalg.norm(f - xy * 1000.0, axis=1)) > reach + DISTRACT_GAP_MM for f in foot):
                    break
            else:
                d.mocap_pos[mid] = [2.0, 2.0 + 0.1 * j, -0.5]
                continue
            m.geom_size[gid] = size
            # 흰 · 회색 40%, 진한 색 40%, 은색처럼 반짝이는 것 20%.
            u = rng.random()
            if u < 0.4:
                g = rng.uniform(0.6, 1.0)
                rgb = np.clip(g + rng.uniform(-0.03, 0.03, 3), 0, 1)
            elif u < 0.8:
                rgb = rng.uniform(0.0, 1.0, 3)
            else:
                rgb = np.clip(rng.uniform(0.7, 0.95) + rng.uniform(-0.05, 0.05, 3), 0, 1)
            m.geom_rgba[gid] = [*rgb, 1.0]
            q_yaw = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            if lying:
                q_lie = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])  # x 축으로 90°: 세로 축을 눕힌다
                q = np.zeros(4)
                mujoco.mju_mulQuat(q, q_yaw, q_lie)
            else:
                q = q_yaw
            d.mocap_pos[mid] = [xy[0], xy[1], z]
            d.mocap_quat[mid] = q
        mujoco.mj_forward(m, d)

    def make(self, rng: np.random.Generator, *, max_tries: int = 50) -> Scene:
        for _ in range(max_tries):
            self._drop(rng)
            if self.placement_ok():
                break
        else:
            raise RuntimeError("조건에 맞게 놓지 못함 (범위 · 간격)")
        if self.n_distract:
            self.place_distractors(rng)
        if self.kin is not None:
            self.set_robot(self.robot_joints_deg)
            mujoco.mj_forward(self.model, self.data)
        rgb, depth_mm, seg = self.render()
        objects = []
        for i in range(self.active):
            T = self.T_base_obj(i)
            cid, ang = pose_class(T[:3, :3])
            objects.append(
                {
                    "id": i + 1,
                    "T_base_obj": T.tolist(),
                    "pose_class": cid,
                    "class_angle_deg": round(ang, 2),
                    "height_mm": round(float(self._world_verts(i)[:, 2].max()), 2),
                    "visible_px": int(np.count_nonzero(seg == i + 1)),
                }
            )
        return Scene(rgb, depth_mm, seg, self.K.copy(), self.T_base_cam.copy(), objects)


def main() -> None:
    ap = argparse.ArgumentParser(description="MuJoCo 책상 장면 만들기")
    ap.add_argument("--count", type=int, default=1, help="장면 수")
    ap.add_argument("--objects", type=int, default=1, help="장면마다 브라켓 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "scenes")
    ap.add_argument(
        "--look", default="default", choices=("default", "random", "lowcontrast", "dim", "harsh", "silver", "black"),
        help="겉모습. 놓인 자세는 시드로만 정해져서, 같은 시드면 겉모습만 다른 같은 장면이 된다",
    )
    ap.add_argument("--cam-jitter", action="store_true", help="장면마다 카메라 자세를 손눈 오차만큼 흔든다")
    ap.add_argument("--distractors", type=int, default=0, help="장면마다 방해물 0~N 개")
    ap.add_argument("--robot", action="store_true", help="로봇 팔을 시작 자세(INIT_POSE)로 넣는다")
    args = ap.parse_args()
    maker = SceneMaker(args.objects, n_distractors=args.distractors, robot=args.robot)
    rng = np.random.default_rng(args.seed)
    # 겉모습 · 카메라 난수는 따로 둔다. 물리(놓인 자세)는 rng 만 쓰므로 겉모습을 바꿔도 같은 자세가 나온다.
    look_rng = np.random.default_rng(args.seed + 10_000)
    T0 = maker.T_base_cam.copy()
    for k in range(args.count):
        maker.set_look(args.look, look_rng)
        if args.cam_jitter:
            maker.set_camera(jitter_camera(T0, look_rng))
        if args.distractors:
            maker.distract_active = int(look_rng.integers(0, args.distractors + 1))
        scene = maker.make(rng)
        folder = args.out / f"scene_{k:04d}"
        save_scene(scene, folder)
        desc = ", ".join(f"클래스 {o['pose_class']} 높이 {o['height_mm']:.0f}mm {o['visible_px']}px" for o in scene.objects)
        print(f"{folder.name}: {desc}")


if __name__ == "__main__":
    main()
