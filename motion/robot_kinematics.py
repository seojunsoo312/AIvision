"""Pinocchio 래퍼: URDF 읽기, FK, 야코비안, DLS 서보."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import coal
import numpy as np
import pinocchio as pin

# 팔 설계도와 부품 STL. URDF 의 package://meshes/ 는 이 폴더의 meshes/ 이다.
ROBOT_DIR = Path(__file__).resolve().parent / "robot"
DEFAULT_URDF = ROBOT_DIR / "SO101_6DOF.urdf"

# URDF S1..S7 = Feetech ID 1..7 (Meshcat 에서 조그로 확인).
URDF_JOINT_NAMES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
ARM_JOINT_NAMES = ("S1", "S2", "S3", "S4", "S5", "S6")
GRIPPER_JOINT = "S7"
# 공식 SO-101 그리퍼 관절: 아래 한계에서 끝이 맞닿고, 위 한계에서 완전히 열린다.
# 펜던트 0 = 닫힘, 100 = 완전히 열림. 사용자 0 = 펜던트 50 (HOME, 반쯤 열림).
GRIPPER_CLOSED_CAD_DEG = -10.0
GRIPPER_OPEN_CAD_DEG = 100.0
GRIPPER_USER_MID = 50.0  # 사용자 각도 0 = 버스/펜던트 50 (HOME, 반쯤 열림)
EE_FRAME = "L6_1"  # 그리퍼 몸체 (wrist_roll 의 자식). 움직이는 집게가 아니다
TCP_FRAME = "tcp"
TCP_PARENT_JOINT = "S6"  # 몸체 고정. S7 을 열어도 TCP 가 움직이면 안 된다
# 그리퍼 링크 (S6). 손가락 깊이는 공식 SO-101 그리퍼 프레임 (~98 mm).
# 공식 x=-7.9 mm 는 고정 집게 안쪽 면 위다. SO-100 TCP 는 그 면에서 8 mm 떨어진
# 벌어짐 가운데였으므로 여기서는 x 를 0 으로 둔다 (패드가 ±7.9 mm 로 대칭).
# Rx(+90°) 로 예전 공구 축을 유지한다: +X 위, +Y 는 팔뚝을 따라 뒤, 손가락은 -Y 쪽.
TCP_OFFSET_IN_L6 = np.array([0.0, -0.000218121, -0.0981274])
TCP_RPY_IN_L6_DEG = (90.0, 0.0, 0.0)

# 나중에 하드웨어와 맞출 LeRobot 이름. 조그로 확인함, 체인 순서 = 서보 ID.
LEROBOT_FROM_URDF = {
    "S1": "shoulder_pan",
    "S2": "shoulder_lift",
    "S3": "elbow_flex",
    "S4": "elbow_roll",
    "S5": "wrist_flex",
    "S6": "wrist_roll",
    "S7": "gripper",
}
# 펜던트 0° 일 때의 CAD 각도. 새 URDF 의 q=0 은 이미 위팔이 위, 팔뚝이 +X 다.
# S1 +90° 면 그 팔뚝이 사용자 앞 (URDF −Y) 으로 돈다. SO-100 홈과 같은
# L 자세라서 대기 자세가 매트 위에 있다.
JOINT_ZERO_OFFSET_DEG = {
    "S1": 90.0,
    "S2": 0.0,
    "S3": 0.0,
    "S4": 0.0,
    "S5": 0.0,
    "S6": 0.0,
    "S7": 0.0,
}

# 펜던트 + 와 URDF + 의 부호. 새 관절 축에 맞춰서 + 가 SO-100 모델 때와
# 같은 방향으로 각 링크를 움직이게 했다.
JOINT_SIGN = {
    "S1": 1.0,
    "S2": 1.0,
    "S3": 1.0,
    "S4": 1.0,
    "S5": 1.0,
    "S6": 1.0,
    "S7": 1.0,  # GUI + 는 연다 (CAD 가 GRIPPER_OPEN_CAD_DEG 쪽으로)
}

# 펜던트(사용자) 소프트 한계. 키가 없으면 제한 없음 (S7 은 그리퍼 CAD 범위를 쓴다).
JOINT_LIMIT_USER_DEG: dict[str, tuple[float, float]] = {
    "S6": (-165.0, 165.0),  # wrist_roll
}
S6_LIMIT_MARGIN_DEG = 1.5

# 사용자 기준 HOME (도). 0 = URDF 의 JOINT_ZERO_OFFSET_DEG.
# L 자세에서 캘리브 Enter 를 누르면 그 자세가 사용자 0 이 된다. S5 를 따로 더 옮기지 않는다.
# 그리퍼 HOME 사용자 0 = 펜던트 50 (반쯤 열림). CAD 로는 GRIPPER_OPEN_CAD_DEG 까지의 중간.
HOME_JOINTS_DEG = {name: 0.0 for name in URDF_JOINT_NAMES}

# Pendant "초기자세" — 작업 시작용. S7 user −50 = GUI 0–100 의 0 (완전 닫힘).
INIT_POSE_JOINTS_DEG = {
    "S1": 0.0,
    "S2": -40.0,
    "S3": 40.0,
    "S4": 0.0,
    "S5": -90.0,
    "S6": 0.0,
    "S7": -GRIPPER_USER_MID,
}

# 펜던트 [x] 쉬는 자세. GUI 0–100 그리퍼 21.53 → 사용자 각도 (HOME=50).
SHUTDOWN_JOINTS_DEG = {
    "S1": 0.0,
    "S2": -98.0,
    "S3": 90.0,
    "S4": 0.0,
    "S5": -60.0,
    "S6": 0.0,
    "S7": 21.53 - 50.0,
}

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


def gripper_user_to_cad(user_deg: float) -> float:
    """사용자 0 = 펜던트 50. 펜던트 0–100 을 CAD 닫힘–열림에 맞춘다."""
    grip_100 = float(np.clip(user_deg + GRIPPER_USER_MID, 0.0, 100.0))
    span = GRIPPER_OPEN_CAD_DEG - GRIPPER_CLOSED_CAD_DEG
    return GRIPPER_CLOSED_CAD_DEG + (grip_100 / 100.0) * span


def gripper_cad_to_user(cad_deg: float) -> float:
    span = GRIPPER_OPEN_CAD_DEG - GRIPPER_CLOSED_CAD_DEG
    grip_100 = (float(cad_deg) - GRIPPER_CLOSED_CAD_DEG) / span * 100.0
    return float(np.clip(grip_100, 0.0, 100.0)) - GRIPPER_USER_MID


# 이웃한 Fusion STL 은 공유 모터에서 겹친다. 힌지 둘레 이 반지름 안의 삼각형을 빼고
# 남은 링크 몸체끼리 충돌을 검사한다.
ADJACENT_JOINT_CLEARANCE_M = 0.04

IK_ITERS = 12
MAX_CART_STEP_M = 0.006
MAX_HOLD_CORR_M = 0.0015
HOLD_DEADBAND_M = 0.002  # 2 mm. 쓰지 않는 축을 너무 꽉 잡으면 S1/S4 가 왔다 갔다 한다.
TILT_DEADBAND_RAD = 0.004
MAX_DQ_ITER_RAD = 1.5 * DEG2RAD
MAX_DQ_FRAME_RAD = 5.0 * DEG2RAD
DLS_LAMBDA = 3.0  # mm 환산. 클수록 영공간 떨림 (X 조그 때 S1/S4) 이 줄어든다.
TILT_MM_PER_RAD = 80.0
TILT_WEIGHT = 1.0
TILT_WEIGHT_EPS = 0.05
XYZ_TOL_M = 0.0003


@dataclass(frozen=True)
class TcpPose:
    xyz_m: np.ndarray  # (3,)
    rpy_deg: np.ndarray  # XYZ 외부 회전, 도
    rotation: np.ndarray  # 3x3

    @property
    def xyz_mm(self) -> np.ndarray:
        return self.xyz_m * 1000.0


def rpy_deg_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """XYZ 외부 회전 RPY (도): R = Rz @ Ry @ Rx."""
    rpy = np.deg2rad([float(roll), float(pitch), float(yaw)])
    return np.asarray(pin.rpy.rpyToMatrix(rpy), dtype=float)


def rotmat_to_rpy_deg(rot: np.ndarray) -> np.ndarray:
    """XYZ 외부 회전 RPY (도). rpy_deg_to_rotmat 의 역변환."""
    rpy = pin.rpy.matrixToRpy(np.asarray(rot, dtype=float))
    return np.degrees(np.asarray(rpy, dtype=float)).reshape(3)


def _dls_dq(jac: np.ndarray, err: np.ndarray, *, damping: float, max_dq: float) -> np.ndarray:
    j = np.asarray(jac, dtype=float)
    e = np.asarray(err, dtype=float)
    if j.size == 0 or e.size == 0:
        return np.zeros(j.shape[1] if j.ndim == 2 else 0)
    n = j.shape[1]
    lam = float(max(damping, 0.0))
    dq = np.linalg.solve(j.T @ j + (lam * lam) * np.eye(n), j.T @ e)
    peak = float(np.max(np.abs(dq))) if dq.size else 0.0
    if peak > max_dq > 0:
        dq = dq * (max_dq / peak)
    return dq


def _clip_position_error(err_p: np.ndarray, cmd_mask: np.ndarray | None) -> np.ndarray:
    err_p = np.asarray(err_p, dtype=float).copy()
    if cmd_mask is None:
        nerr = float(np.linalg.norm(err_p))
        if nerr > MAX_CART_STEP_M:
            err_p = err_p * (MAX_CART_STEP_M / nerr)
        return err_p
    cmd = np.asarray(cmd_mask, dtype=bool).reshape(3)
    hold = ~cmd
    if np.any(cmd):
        e_cmd = np.where(cmd, err_p, 0.0)
        n_cmd = float(np.linalg.norm(e_cmd))
        if n_cmd > MAX_CART_STEP_M:
            err_p[cmd] = e_cmd[cmd] * (MAX_CART_STEP_M / n_cmd)
    if np.any(hold):
        err_p[hold] = np.where(np.abs(err_p[hold]) < HOLD_DEADBAND_M, 0.0, err_p[hold])
        e_hold = np.where(hold, err_p, 0.0)
        n_hold = float(np.linalg.norm(e_hold))
        if n_hold > MAX_HOLD_CORR_M:
            err_p[hold] = e_hold[hold] * (MAX_HOLD_CORR_M / n_hold)
    return err_p


class RobotKinematics:
    """SO101_6DOF.urdf 를 읽는다. FK + DLS 직교 서보. 하드웨어 매핑은 다른 곳에 있다."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        package_dirs: list[str] | None = None,
        tcp_offset: np.ndarray | None = None,
    ) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        if package_dirs is None:
            # package://meshes/*.stl → URDF 옆 폴더, 없으면 그 상위 폴더에서 찾는다.
            package_dirs = [str(self.urdf_path.parent), str(self.urdf_path.parent.parent)]

        self.tcp_offset = (
            np.asarray(TCP_OFFSET_IN_L6, dtype=float).reshape(3)
            if tcp_offset is None
            else np.asarray(tcp_offset, dtype=float).reshape(3)
        )

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        missing = [n for n in URDF_JOINT_NAMES if not self.model.existJointName(n)]
        if missing:
            raise ValueError(f"URDF missing joints {missing}; have {self.model.names}")
        if not self.model.existFrame(EE_FRAME):
            raise ValueError(f"URDF missing EE frame {EE_FRAME!r}")
        self._add_tcp_frame()
        self.data = self.model.createData()
        self.visual_model = pin.buildGeomFromUrdf(
            self.model,
            str(self.urdf_path),
            pin.GeometryType.VISUAL,
            package_dirs=package_dirs,
        )
        self.collision_model = pin.buildGeomFromUrdf(
            self.model,
            str(self.urdf_path),
            pin.GeometryType.COLLISION,
            package_dirs=package_dirs,
        )
        self._init_collision_pairs()
        self.ee_frame_id = self.model.getFrameId(TCP_FRAME)

        # URDF 관절마다 q / v 인덱스 (names[0] 의 universe 는 건너뜀).
        self._q_index: dict[str, int] = {}
        self._v_index: dict[str, int] = {}
        for name in URDF_JOINT_NAMES:
            jid = self.model.getJointId(name)
            self._q_index[name] = int(self.model.idx_qs[jid])
            self._v_index[name] = int(self.model.idx_vs[jid])
        self._arm_v_cols = [self._v_index[n] for n in ARM_JOINT_NAMES]
        self._arm_q_cols = [self._q_index[n] for n in ARM_JOINT_NAMES]

    def _add_tcp_frame(self) -> None:
        jid = self.model.getJointId(TCP_PARENT_JOINT)
        rot = rpy_deg_to_rotmat(*TCP_RPY_IN_L6_DEG)
        placement = pin.SE3(rot, self.tcp_offset)
        self.model.addFrame(pin.Frame(TCP_FRAME, jid, placement, pin.FrameType.OP_FRAME))

    def _init_collision_pairs(self) -> None:
        """한 칸 이상 떨어진 링크는 전체 메시끼리, 이웃 링크는 힌지 여유를 둔 몸체로 접힘 충돌을 본다."""
        cm = self.collision_model
        cm.addAllCollisionPairs()
        keep: list[tuple[int, int]] = []
        for pair in cm.collisionPairs:
            ja = int(cm.geometryObjects[pair.first].parentJoint)
            jb = int(cm.geometryObjects[pair.second].parentJoint)
            if abs(ja - jb) <= 1:
                continue
            keep.append((int(pair.first), int(pair.second)))
        cm.removeAllCollisionPairs()
        for i, j in keep:
            cm.addCollisionPair(pin.CollisionPair(i, j))
        self.collision_data = pin.GeometryData(cm)
        self._adj_body_pairs = self._build_adjacent_body_pairs()

    def _build_adjacent_body_pairs(self) -> list[tuple[int, int, object, object]]:
        """공유 모터 CAD 를 잘라 낸 링크 몸체 BVH. BASE–L1 은 건너뜀 (회전축)."""
        cm = self.collision_model
        pin.forwardKinematics(self.model, self.data, pin.neutral(self.model))
        gd = pin.GeometryData(cm)
        pin.updateGeometryPlacements(self.model, self.data, cm, gd)
        n = len(cm.geometryObjects)
        cache: dict[tuple[int, int], object] = {}
        pairs: list[tuple[int, int, object, object]] = []
        for i in range(n):
            for j in range(i + 1, n):
                ja = int(cm.geometryObjects[i].parentJoint)
                jb = int(cm.geometryObjects[j].parentJoint)
                # BASE–L1 회전은 CAD 끼리 겹치는 것이고, L6–L7 집게는 원래 닫히는 것이다.
                if abs(ja - jb) != 1 or min(ja, jb) == 0:
                    continue
                if max(ja, jb) == int(self.model.getJointId(GRIPPER_JOINT)):
                    continue
                jid = max(ja, jb)
                bodies = []
                for gi in (i, j):
                    key = (gi, jid)
                    if key not in cache:
                        cache[key] = self._body_bvh_far_from_joint(
                            cm.geometryObjects[gi], gd.oMg[gi], jid
                        )
                    bodies.append(cache[key])
                if bodies[0] is None or bodies[1] is None:
                    continue
                pairs.append((i, j, bodies[0], bodies[1]))
        return pairs

    def _body_bvh_far_from_joint(self, go: pin.GeometryObject, oMg: pin.SE3, jid: int):
        geom = go.geometry
        verts = np.asarray(geom.vertices(), dtype=float)
        joint_local = np.asarray(
            oMg.inverse().act(np.asarray(self.data.oMi[jid].translation, dtype=float))
        )
        far = np.linalg.norm(verts - joint_local, axis=1) >= ADJACENT_JOINT_CLEARANCE_M
        kept: list[tuple[int, int, int]] = []
        for k in range(int(geom.num_tris)):
            tri = geom.tri_indices(k)
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            if far[a] and far[b] and far[c]:
                kept.append((a, b, c))
        if len(kept) < 8:
            return None
        used = sorted({idx for tri in kept for idx in tri})
        remap = {old: new for new, old in enumerate(used)}
        pts = coal.StdVec_Vec3s()
        for idx in used:
            pts.append(np.asarray(verts[idx], dtype=float))
        tris = coal.StdVec_Triangle()
        for a, b, c in kept:
            tris.append(coal.Triangle(remap[a], remap[b], remap[c]))
        bvh = coal.BVHModelOBBRSS()
        bvh.beginModel(len(tris), len(pts))
        bvh.addSubModel(pts, tris)
        bvh.endModel()
        bvh.computeLocalAABB()
        return bvh

    def in_collision(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=float)
        if pin.computeCollisions(
            self.model, self.data, self.collision_model, self.collision_data, q, True
        ):
            return True
        return self._adjacent_bodies_in_collision(q)

    def _adjacent_bodies_in_collision(self, q: np.ndarray) -> bool:
        pin.forwardKinematics(self.model, self.data, q)
        gd = pin.GeometryData(self.collision_model)
        pin.updateGeometryPlacements(self.model, self.data, self.collision_model, gd)
        req = coal.CollisionRequest()
        for i, j, bi, bj in self._adj_body_pairs:
            ti, tj = gd.oMg[i], gd.oMg[j]
            oi = coal.CollisionObject(
                bi, coal.Transform3s(np.asarray(ti.rotation), np.asarray(ti.translation))
            )
            oj = coal.CollisionObject(
                bj, coal.Transform3s(np.asarray(tj.rotation), np.asarray(tj.translation))
            )
            res = coal.CollisionResult()
            coal.collide(oi, oj, req, res)
            if res.isCollision():
                return True
        return False

    def q_index(self, joint_name: str) -> int:
        return self._q_index[joint_name]

    def joints_deg(self, q: np.ndarray) -> dict[str, float]:
        """펜던트 각도 = URDF 각도 − L 자세 영점 오프셋."""
        out: dict[str, float] = {}
        for name in URDF_JOINT_NAMES:
            cad = float(q[self._q_index[name]] * RAD2DEG)
            if name == GRIPPER_JOINT:
                out[name] = gripper_cad_to_user(cad)
            else:
                out[name] = JOINT_SIGN[name] * (cad - JOINT_ZERO_OFFSET_DEG[name])
        return out

    def s6_at_limit(self, q: np.ndarray) -> bool:
        s6 = float(self.joints_deg(q)["S6"])
        lo, hi = JOINT_LIMIT_USER_DEG["S6"]
        return bool(s6 <= lo + S6_LIMIT_MARGIN_DEG or s6 >= hi - S6_LIMIT_MARGIN_DEG)

    def clamp_q(self, q: np.ndarray) -> np.ndarray:
        """그리퍼 CAD 범위와 JOINT_LIMIT_USER_DEG 에 있는 관절을 자른다."""
        q = np.asarray(q, dtype=float).copy()
        i = self._q_index[GRIPPER_JOINT]
        lo = min(GRIPPER_CLOSED_CAD_DEG, GRIPPER_OPEN_CAD_DEG) * DEG2RAD
        hi = max(GRIPPER_CLOSED_CAD_DEG, GRIPPER_OPEN_CAD_DEG) * DEG2RAD
        q[i] = float(np.clip(q[i], lo, hi))
        for name, (user_lo, user_hi) in JOINT_LIMIT_USER_DEG.items():
            j = self._q_index[name]
            cad = float(q[j] * RAD2DEG)
            user = JOINT_SIGN[name] * (cad - JOINT_ZERO_OFFSET_DEG[name])
            user = float(np.clip(user, user_lo, user_hi))
            cad = JOINT_ZERO_OFFSET_DEG[name] + JOINT_SIGN[name] * user
            q[j] = cad * DEG2RAD
        return q

    def q_home(self) -> np.ndarray:
        return self.q_from_deg(HOME_JOINTS_DEG)

    def q_from_deg(self, joints_deg: dict[str, float]) -> np.ndarray:
        """펜던트(사용자) 각도로 Pinocchio q 를 만든다."""
        q = pin.neutral(self.model).copy()
        for name in URDF_JOINT_NAMES:
            user = float(joints_deg.get(name, HOME_JOINTS_DEG[name]))
            if name == GRIPPER_JOINT:
                cad = gripper_user_to_cad(user)
            else:
                cad = JOINT_ZERO_OFFSET_DEG[name] + JOINT_SIGN[name] * user
            q[self._q_index[name]] = cad * DEG2RAD
        return q

    def forward_tcp(self, q: np.ndarray) -> TcpPose:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.ee_frame_id]
        rot = np.asarray(oMf.rotation, dtype=float).copy()
        xyz = np.asarray(oMf.translation, dtype=float).copy()
        return TcpPose(xyz_m=xyz, rpy_deg=rotmat_to_rpy_deg(rot), rotation=rot)

    def tcp_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """TCP 자세와 팔 관절 6개의 월드 야코비안. Jv: m/rad, Jw: 1/rad."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.ee_frame_id]
        jac = np.asarray(
            pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
            ),
            dtype=float,
        )[:, self._arm_v_cols]
        return (
            np.asarray(oMf.translation, dtype=float).copy(),
            np.asarray(oMf.rotation, dtype=float).copy(),
            jac[0:3].copy(),
            jac[3:6].copy(),
        )

    def servo_toward(
        self,
        q: np.ndarray,
        target_xyz: np.ndarray,
        *,
        R_ref: np.ndarray | None = None,
        cmd_mask: np.ndarray | None = None,
        ori_weight: float = TILT_WEIGHT,
    ) -> np.ndarray:
        """DLS IK 한 제어 프레임. 새 전체 `q` 를 돌려준다 (그리퍼는 그대로)."""
        q_out = self.clamp_q(np.asarray(q, dtype=float).copy())
        q_arm = q_out[self._arm_q_cols].copy()
        q_start = q_arm.copy()
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        cmd = None if cmd_mask is None else np.asarray(cmd_mask, dtype=bool).reshape(3)
        use_ori = R_ref is not None and ori_weight >= TILT_WEIGHT_EPS
        for _ in range(IK_ITERS):
            pos, rot, jv, jw = self.tcp_jacobian(self._q_with_arm(q_out, q_arm))
            err_p = _clip_position_error(target_xyz - pos, cmd)
            nerr = float(np.linalg.norm(target_xyz - pos))
            jac_p = jv * 1000.0
            err_p_mm = err_p * 1000.0
            j_extra: np.ndarray | None = None
            e_extra: np.ndarray | None = None
            if use_ori:
                assert R_ref is not None
                err_r = np.clip(np.asarray(pin.log3(R_ref @ rot.T)).reshape(3), -0.04, 0.04)
                err_r = np.where(np.abs(err_r) < TILT_DEADBAND_RAD, 0.0, err_r)
                if nerr < XYZ_TOL_M and float(np.linalg.norm(err_r)) < 0.002:
                    break
                tw = float(ori_weight)
                # S6 한계: 닿지 않는 손목 roll 을 S1–S5 로 메우지 않는다 (TCP 가 점에서 끌려 나감).
                if self.s6_at_limit(self._q_with_arm(q_out, q_arm)):
                    tw *= 0.08
                if tw >= TILT_WEIGHT_EPS:
                    j_extra = jw * TILT_MM_PER_RAD * tw
                    e_extra = err_r * TILT_MM_PER_RAD * tw
                elif nerr < XYZ_TOL_M:
                    break
            elif nerr < XYZ_TOL_M:
                break
            if cmd is None:
                if j_extra is None:
                    dq = _dls_dq(jac_p, err_p_mm, damping=DLS_LAMBDA, max_dq=MAX_DQ_ITER_RAD)
                else:
                    dq = _dls_dq(
                        np.vstack([jac_p, j_extra]),
                        np.concatenate([err_p_mm, e_extra]),
                        damping=DLS_LAMBDA,
                        max_dq=MAX_DQ_ITER_RAD,
                    )
            else:
                # XYZ 는 DLS 한 번으로. 유지를 먼저 푸는 계층 방식이면 Z 에 Jxy 의 작은 나머지만
                # 남아서 (책상에 접힌 자세: S2/S3 가 XY 유지에 바쁨) +Z 가 안 움직이는 것처럼 보인다.
                if j_extra is None:
                    dq = _dls_dq(jac_p, err_p_mm, damping=DLS_LAMBDA, max_dq=MAX_DQ_ITER_RAD)
                else:
                    dq = _dls_dq(
                        np.vstack([jac_p, j_extra]),
                        np.concatenate([err_p_mm, e_extra]),
                        damping=DLS_LAMBDA,
                        max_dq=MAX_DQ_ITER_RAD,
                    )
            q_arm = q_arm + dq
            q_out = self.clamp_q(self._q_with_arm(q_out, q_arm))
            q_arm = q_out[self._arm_q_cols].copy()
            if float(np.max(np.abs(q_arm - q_start))) >= MAX_DQ_FRAME_RAD:
                break
        return q_out

    def _q_with_arm(self, q_full: np.ndarray, q_arm: np.ndarray) -> np.ndarray:
        out = np.asarray(q_full, dtype=float).copy()
        out[self._arm_q_cols] = q_arm
        return out
