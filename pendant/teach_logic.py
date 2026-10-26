"""사물 위치 패널의 계산과 실행. 화면(Tk)도 3D 화면(Meshcat)도 없다.

  - PlaceRpy            물체 Rx/Ry/Rz 를 「물체 축」/「베이스」 두 표기로 보여 주고 고친다.
  - auto_pre            물체 자세 → 대기(P)·집기(G) 자세 (motion/grasp.py compute_auto_pre).
  - pick_place_steps    대기·집기 자세와 놓기 x/y → 픽앤플레이스 단계 (motion/pick_place.py).
  - random_place        랜덤 배치 + 팔과 겹치는지 검사.
  - UserPickPlace       「픽앤플레이스(유저생성)」: user_pickandplace.py 의 run(arm) 을 따로 돌린다.

화면 쪽은 pendant/teach_grasp.py, 3D 화면에 그리는 쪽은 pendant/teach_scene.py.
"""

from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from motion import runlog
from motion.grasp import (
    AutoPreResult,
    PlacePose,
    T_to_xyzrpy,
    canonicalize_extrinsic_rpy,
    compute_auto_pre,
    object_collides_robot,
    place_rpy_base_delta,
    place_rpy_body_delta,
    rpy_body_xyz_to_extrinsic,
    rpy_extrinsic_to_body_xyz,
    sample_random_place,
)
from motion.pick_place import PickPlaceStep, build_pick_place_steps

PROJECT = Path(__file__).resolve().parent.parent
AXES = ("roll", "pitch", "yaw")
# 랜덤 배치 모드 → 화면 글자.
MODE_KO = {"stand": "세우기", "lie": "눕히기", "slant": "비스듬히", "free": "자유"}
_log = runlog.get("teach")


def r1(values) -> list[float]:
    """로그용: 소수 1자리 숫자 목록."""
    return [round(float(v), 1) for v in values]


# ---------------------------------------------------------------- 물체 Rx/Ry/Rz


@dataclass
class PlaceRpy:
    """물체 회전. 저장은 늘 베이스 기준(extrinsic) 값이고, 화면 숫자만 표기를 바꾼다.

    frame: "body" = 물체 축 기준 (화면 OBJ), "base" = 베이스 기준.
    slider: 지금 슬라이더가 가리키는 축별 값. 슬라이더를 끌 때 이전 값과의 차이만큼 돌린다.
    """

    base_rpy: tuple[float, float, float]
    frame: str = "body"
    slider: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_place(cls, rpy_deg) -> "PlaceRpy":
        st = cls(base_rpy=canonicalize_extrinsic_rpy(rpy_deg))
        st.slider = st.shown()
        return st

    def set_place(self, rpy_deg) -> None:
        """새 물체 자세 (랜덤 · 불러오기 · 초기화)."""
        self.base_rpy = canonicalize_extrinsic_rpy(rpy_deg)
        self.slider = self.shown()

    def shown(self) -> dict[str, float]:
        """지금 표기로 푼 Rx/Ry/Rz."""
        if self.frame == "base":
            rpy = self.base_rpy
        else:
            rpy = rpy_extrinsic_to_body_xyz(self.base_rpy)
        return {axis: float(v) for axis, v in zip(AXES, rpy)}

    def refresh(self, keep_axis: str | None = None) -> dict[str, float]:
        """화면에 쓸 값. 끌고 있는 축(keep_axis)을 뺀 슬라이더 값도 이 값으로 맞춘다."""
        shown = self.shown()
        for axis, val in shown.items():
            if axis != keep_axis:
                self.slider[axis] = val
        return shown

    def set_frame(self, frame: str) -> bool:
        """표기를 바꾼다. 바뀌었으면 True (그때 refresh() 로 화면 값을 다시 받는다)."""
        frame = "base" if frame == "base" else "body"
        if frame == self.frame:
            return False
        self.frame = frame
        return True

    def set_axis(self, axis: str, value: float) -> dict[str, float]:
        """칸에 숫자를 넣었을 때: 그 축을 그 값으로 (다른 축은 지금 슬라이더 값). 화면 값을 돌려준다."""
        cmd = dict(self.slider)
        cmd[axis] = float(value)
        rpy = (float(cmd["roll"]), float(cmd["pitch"]), float(cmd["yaw"]))
        if self.frame == "base":
            self.base_rpy = canonicalize_extrinsic_rpy(rpy)
        else:
            self.base_rpy = rpy_body_xyz_to_extrinsic(*rpy)
        self.slider = cmd
        return self.refresh()

    def nudge_axis(self, axis: str, value: float) -> dict[str, float]:
        """슬라이더를 끌 때: 이전 슬라이더 값과의 차이만큼 그 축으로 돌린다. 화면 값을 돌려준다."""
        delta = float(value) - float(self.slider[axis])
        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0
        if self.frame == "base":
            self.base_rpy = place_rpy_base_delta(self.base_rpy, axis=axis, delta_deg=delta)
        else:
            self.base_rpy = place_rpy_body_delta(self.base_rpy, axis=axis, delta_deg=delta)
        self.slider[axis] = float(value)
        return self.refresh(keep_axis=axis)


# ---------------------------------------------------------------- 대기 · 집기 · 픽앤플레이스


def robot_seed(kin, ctrl):
    """IK 시작 자세. 컨트롤러가 있으면 지금 관절, 없으면 홈."""
    if kin is None and ctrl is not None:
        kin = getattr(ctrl, "_kin", None)
    q = None
    if ctrl is not None:
        q = ctrl.snapshot().q
    elif kin is not None:
        q = kin.q_home()
    return kin, q


def auto_pre(place: PlacePose, mins_mm, maxs_mm, d_mm: float, a_mm: float, *, kin, ctrl) -> AutoPreResult:
    kin, q = robot_seed(kin, ctrl)
    return compute_auto_pre(place, mins_mm, maxs_mm, d_mm, a_mm, kin=kin, q=q)


def auto_label(auto: AutoPreResult, target: str) -> str:
    """상태 글자: 「대기 위치 (Z pitch+)」, 「집기 위치 (Z a=30 pitch+)」."""
    flip = "pitch−" if auto.pitch_flipped else "pitch+"
    form = f" {auto.form}" if auto.form else ""
    if target == "pre":
        return f"대기 위치 ({auto.axis_name}{form} {flip})"
    return f"집기 위치 ({auto.axis_name}{form} a={auto.a_mm:.0f} {flip})"


def pick_place_steps(auto: AutoPreResult, drop_xy) -> list[PickPlaceStep]:
    p_xyz, p_rpy = T_to_xyzrpy(auto.T_base)
    g_xyz, g_rpy = T_to_xyzrpy(auto.T_grasp_base)
    return build_pick_place_steps(
        p_xyz_mm=p_xyz,
        p_rpy_deg=p_rpy,
        g_xyz_mm=g_xyz,
        g_rpy_deg=g_rpy,
        drop_xy_mm=np.array(drop_xy, dtype=float),
    )


# ---------------------------------------------------------------- 랜덤 배치


@dataclass(frozen=True)
class RandomResult:
    place: PlacePose | None
    collided: bool = False
    error: str = ""  # 충돌 검사가 실패했을 때


def random_place(verts_m, mode: str, rng, *, kin, ctrl) -> RandomResult:
    """랜덤 배치 하나. 못 찾으면 place=None."""
    place = sample_random_place(verts_m, mode=mode, rng=rng)
    if place is None:
        return RandomResult(None)
    kin, q = robot_seed(kin, ctrl)
    if kin is None or q is None:
        return RandomResult(place)
    try:
        return RandomResult(place, collided=object_collides_robot(kin, q, verts_m, place))
    except Exception as exc:
        return RandomResult(place, error=str(exc))


def place_text(place: PlacePose) -> str:
    return f"xyz={list(np.round(place.xyz_mm, 1))}  rpy={list(np.round(place.rpy_deg, 1))}"


# ---------------------------------------------------------------- 픽앤플레이스(유저생성)


def load_user_pick_run(path: Path = PROJECT / "user_pickandplace.py"):
    """프로젝트 루트의 user_pickandplace.py 를 매번 다시 읽는다 (고친 내용이 바로 반영)."""
    if not path.is_file():
        raise FileNotFoundError(f"파일 없음: {path}")
    spec = importlib.util.spec_from_file_location("user_pickandplace_live", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("user_pickandplace.py 를 읽지 못했습니다")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise RuntimeError("user_pickandplace.py 에 run(arm) 함수가 없습니다")
    return run


class UserPickPlace:
    """run(arm) 을 따로 스레드에서 돌린다. 끝나면 on_done(글자, error) 를 부른다 (그 스레드에서)."""

    def __init__(self, ctrl) -> None:
        self._ctrl = ctrl
        self._thread: threading.Thread | None = None

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, on_done: Callable[[str, bool], None]) -> None:
        self._thread = threading.Thread(target=self._work, args=(on_done,), name="user-pick-place", daemon=True)
        self._thread.start()

    def abort(self) -> None:
        if self.running() and self._ctrl is not None:
            self._ctrl.stop_all()

    def _work(self, on_done) -> None:
        try:
            run = load_user_pick_run()
            from motion.arm import Arm

            _log.info("픽앤플레이스(유저생성) 시작")
            run(Arm.attach(self._ctrl))
        except Exception as exc:
            _log.warning("픽앤플레이스(유저생성) 실패: %s", exc, exc_info=not isinstance(exc, RuntimeError))
            on_done(f"픽앤플레이스(유저생성) 실패: {exc}", True)
        else:
            _log.info("픽앤플레이스(유저생성) 완료")
            on_done("픽앤플레이스(유저생성) 완료", False)
