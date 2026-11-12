"""카메라에 물체 등록을 요청하고, 그 6D로 교육용 픽앤플레이스 좌표를 만든다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cad.model import approach_mm, bounds_mm
from motion.arm import Arm, ArmError
from motion.grasp import PlacePose, T_to_xyzrpy, compute_auto_pre
from motion.pick_place import rpy_tcp_rx_parallel_base_x
from motion.robot_kinematics import DEFAULT_URDF, RobotKinematics
from motion.register_link import (
    REG_MSG_BUSY,
    REG_MSG_FAIL,
    REG_MSG_NO_BOX,
    REG_MSG_NO_REPLY,
    RegisterRequest,
)

# 등록이 끝난 state → 학생에게 보이는 오류.
_REG_ERRORS = {
    "busy": REG_MSG_BUSY,
    "no_box": REG_MSG_NO_BOX,
    "fail": REG_MSG_FAIL,
    "timeout": REG_MSG_NO_REPLY,
}


@dataclass(frozen=True)
class PickTargets:
    """대기 P, 집기 G, 놓기 자세. 위치는 mm, 자세는 deg."""

    p_xyz: list[float]
    p_rpy: list[float]
    g_xyz: list[float]
    g_rpy: list[float]
    place_rpy: list[float]


@dataclass(frozen=True)
class SeenObject:
    """카메라가 찾은 브라켓 하나 (load_all_pick_targets). 위치는 mm, 자세는 deg."""

    xyz: list[float]  # 물체 위치 (로봇 베이스)
    rpy: list[float]  # 물체 방향
    pose: str  # 놓인 자세: 세우기 / 눕히기 / 비스듬히
    color: str  # 흰 / 검
    verdict: str  # 양품 / 불량 / 검사 안 함 (검은 브라켓)
    pick: PickTargets | None  # 대기 · 집기 · 놓기 자세. 계산하지 못하면 None (닿는지는 움직일 때 확인)

    @property
    def distance(self) -> float:
        """로봇 베이스 중심에서 물체까지 바닥 거리 (mm)."""
        return round(float(np.hypot(self.xyz[0], self.xyz[1])), 1)


def load_pick_targets(arm: Arm | None = None) -> PickTargets:
    """카메라 창에 물체 등록을 요청하고, 그 6D로 대기·집기·놓기 자세를 만든다.

    펜던트 「물체 위치 불러오기」와 같은 요청이다. 결과가 올 때까지 이 줄에서 기다린다.
    arm을 주면 그 관절로 손목 자세를 고르고, 없으면 홈 관절을 쓴다.
    """
    place = _request_place()
    return _targets(place, *_seed(arm))


def load_all_pick_targets(arm: Arm | None = None) -> list[SeenObject]:
    """카메라 창에 보이는 브라켓을 모두 요청한다. 브라켓마다 위치 · 자세 · 색 · 판정과 집기 좌표.

    로봇에서 가까운 순서로 돌려준다. 순서를 바꾸려면 sorted(objs, key=...) 로 직접 정한다.
    카메라 창의 자세 방식 (CAD 정합 / 키포인트) 으로 구하고, 판정은 카메라 창 i 검사와 같다.
    집기 자세를 계산하지 못한 브라켓은 pick 이 None 이다. 팔이 닿는지는 moveL 할 때 확인된다.
    """
    try:
        req = RegisterRequest.send(all_objects=True)
    except OSError as exc:
        raise ArmError(REG_MSG_NO_REPLY) from exc
    state = req.wait()
    if state != "ok":
        raise ArmError(_REG_ERRORS[state])
    kin, q = _seed(arm)
    out = []
    for o in req.objects or []:
        place = PlacePose(xyz_mm=o["xyz_mm"], rpy_deg=o["rpy_deg"])
        try:
            pick = _targets(place, kin, q)
        except ArmError:
            raise
        except Exception:
            pick = None  # 이 자세로는 집기 자세를 못 만듦
        out.append(SeenObject(xyz=_vec(o["xyz_mm"]), rpy=_vec(o["rpy_deg"]), pose=o["pose"],
                              color=o["color"], verdict=o["verdict"], pick=pick))
    return out


def _targets(place: PlacePose, kin, q) -> PickTargets:
    """물체 6D → 대기 · 집기 · 놓기 자세."""
    try:
        mins_mm, maxs_mm = bounds_mm()
    except Exception as exc:
        raise ArmError(f"CAD를 읽지 못했습니다 ({exc})") from exc
    d_mm, a_mm = approach_mm()
    auto = compute_auto_pre(
        place,
        mins_mm,
        maxs_mm,
        d_mm,
        a_mm,
        kin=kin,
        q=q,
    )
    p_xyz, p_rpy = T_to_xyzrpy(auto.T_base)
    g_xyz, g_rpy = T_to_xyzrpy(auto.T_grasp_base)
    place_rpy = rpy_tcp_rx_parallel_base_x(np.asarray(g_rpy, dtype=float))
    return PickTargets(
        p_xyz=_vec(p_xyz),
        p_rpy=_vec(p_rpy),
        g_xyz=_vec(g_xyz),
        g_rpy=_vec(g_rpy),
        place_rpy=_vec(place_rpy),
    )


def _request_place() -> PlacePose:
    """등록 요청을 보내고, 이번 요청의 응답 6D를 돌려준다."""
    try:
        req = RegisterRequest.send()
    except OSError as exc:
        raise ArmError(REG_MSG_NO_REPLY) from exc
    state = req.wait()
    if state != "ok":
        raise ArmError(_REG_ERRORS[state])
    return PlacePose(xyz_mm=req.xyz_mm, rpy_deg=req.rpy_deg)


def _seed(arm: Arm | None):
    if arm is not None:
        ctrl = arm._ctrl
        if ctrl is None:
            raise ArmError("이미 종료된 Arm 입니다")
        return ctrl._kin, ctrl.snapshot().q
    kin = RobotKinematics(DEFAULT_URDF)
    return kin, kin.q_home()


def _vec(value) -> list[float]:
    arr = np.asarray(value, dtype=float).reshape(3)
    out: list[float] = []
    for v in arr:
        r = round(float(v), 2)
        out.append(0.0 if r == 0.0 else r)
    return out
