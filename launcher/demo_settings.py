"""반복 데모 설정 파일 launcher/demo_settings.json 읽기 · 만들기.

런처가 켜질 때 파일이 없으면 기본값으로 만들고 (학생이 바로 고칠 수 있게), 반복 데모는
사이클마다 다시 읽는다. 무거운 모듈(팔 모델 · lerobot)은 부르지 않는다.

  cycles         반복 횟수
  left_xyz_mm    왼쪽 자리 [x, y, z] (베이스 mm, +Y 가 로봇 기준 왼쪽, z 는 물체 중심 높이)
  right_xyz_mm   오른쪽 자리
  approach_d_mm  물체 표면 → 대기 위치 거리
  approach_a_mm  대기 위치 → 집기 위치 거리 (접근 방향)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cad.model import approach_mm

SETTINGS_PATH = Path(__file__).resolve().with_name("demo_settings.json")
# 두 자리 기본값 (프로젝트 베이스 mm). +Y 가 로봇 기준 왼쪽.
SPOTS = {"left": (150.0, 180.0, 34.0), "right": (150.0, -180.0, 34.0)}
CYCLES = 10
# 범위 밖이거나 형식이 틀리면 그 값만 기본값을 쓴다.
SETTING_LIMITS = {"approach_d_mm": (0.0, 60.0), "approach_a_mm": (5.0, 80.0)}  # mm
CYCLES_LIMITS = (1, 1000)
SPOT_KEYS = {"left_xyz_mm": "left", "right_xyz_mm": "right"}
SPOT_LIMITS = ((80.0, 350.0), (-350.0, 350.0), (0.0, 150.0))  # x, y, z mm (베이스)
SETTING_KEYS = ("cycles", "left_xyz_mm", "right_xyz_mm", "approach_d_mm", "approach_a_mm")
_log = logging.getLogger("aivision.demo")


def _number_ok(v, lo: float, hi: float) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= float(v) <= hi


def _setting_ok(key: str, v) -> bool:
    if key == "cycles":
        return isinstance(v, int) and not isinstance(v, bool) and CYCLES_LIMITS[0] <= v <= CYCLES_LIMITS[1]
    if key in SPOT_KEYS:
        return isinstance(v, list) and len(v) == 3 and all(_number_ok(c, *lim) for c, lim in zip(v, SPOT_LIMITS))
    return _number_ok(v, *SETTING_LIMITS[key])


def default_demo_settings() -> dict:
    d, a = approach_mm()
    return {
        "cycles": CYCLES,
        "left_xyz_mm": list(SPOTS["left"]),
        "right_xyz_mm": list(SPOTS["right"]),
        "approach_d_mm": d,
        "approach_a_mm": a,
    }


def _write_settings(path: Path, data: dict) -> None:
    """한 줄에 키 하나 (좌표도 한 줄)로 써서 손으로 고치기 쉽게 한다."""
    order = [k for k in default_demo_settings() if k in data] + [k for k in data if k not in SETTING_KEYS]
    body = ",\n".join(f"  {json.dumps(k)}: {json.dumps(data[k], ensure_ascii=False)}" for k in order)
    try:
        path.write_text("{\n" + body + "\n}\n", encoding="utf-8")
    except OSError as exc:
        _log.warning("설정 파일을 못 씀 %s: %s", path, exc)


def load_demo_settings(path: Path = SETTINGS_PATH) -> dict:
    """demo_settings.json 값. 없으면 기본값으로 만들고, 빠진 키는 기본값으로 채워 넣는다.
    잘못된 값은 경고만 남기고 그 키만 기본값을 쓴다."""
    defaults = default_demo_settings()
    if not path.exists():
        _write_settings(path, defaults)
        return defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON 객체 { } 가 아님")
    except (OSError, ValueError) as exc:
        _log.warning("설정 파일을 못 읽음 %s: %s → 기본값", path, exc)
        return defaults
    out = dict(defaults)
    for key in defaults:
        if key not in data:
            continue
        if _setting_ok(key, data[key]):
            out[key] = data[key]
        else:
            _log.warning("설정 %s=%r 이 범위 밖이거나 형식이 틀림 → 기본값 %r", key, data[key], defaults[key])
    missing = [k for k in defaults if k not in data]
    if missing:
        _write_settings(path, {**data, **{k: defaults[k] for k in missing}})
    return out


def ensure_settings_file(path: Path = SETTINGS_PATH) -> None:
    """파일이 없거나 키가 빠졌으면 기본값으로 만든다 · 채운다 (런처 시작 때)."""
    load_demo_settings(path)
