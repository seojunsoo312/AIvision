"""데모 설정 파일 읽기 · 만들기. 둘 다 git 에 안 올리는 보드별 파일이다.

  launcher/demo_settings.json         반복 데모
  launcher/vision_pick_settings.json  카메라로 집기 (반복 데모와 따로)

런처가 켜질 때 파일이 없으면 기본값으로 만들고 (학생이 바로 고칠 수 있게), 데모는
한 번 움직이기 전마다 다시 읽는다. 무거운 모듈(팔 모델 · lerobot)은 부르지 않는다.

demo_settings.json
  cycles         반복 횟수
  left_xyz_mm    왼쪽 자리 [x, y, z] (베이스 mm, +Y 가 로봇 기준 왼쪽, z 는 물체 중심 높이)
  right_xyz_mm   오른쪽 자리
  approach_d_mm  물체 표면 → 대기 위치 거리
  approach_a_mm  대기 위치 → 집기 위치 거리 (접근 방향)

vision_pick_settings.json
  pick_region_mm   집을 구역 [[x 최소, x 최대], [y 최소, y 최대]] (베이스 mm). 가운데가 이 안인 물체만 집는다.
  pick_reach_mm    베이스에서 물체까지 [최소, 최대] 거리 (mm). 밖이면 집지 않고 옮겨 달라고 알린다.
  white_bin_xyz_mm 흰 브라켓 바구니: 그리퍼를 열 자리 [x, y, z] (z 는 바구니 가장자리보다 3~5 cm 위)
  black_bin_xyz_mm 검은 브라켓 바구니
  white_gray_min   물체 밝기 (0–255) 가 이 이상이면 흰색
  approach_d_mm    물체 표면 → 대기 위치 거리
  approach_a_mm    대기 위치 → 집기 위치 거리
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cad.model import approach_mm

SETTINGS_PATH = Path(__file__).resolve().with_name("demo_settings.json")
VISION_SETTINGS_PATH = Path(__file__).resolve().with_name("vision_pick_settings.json")
# 두 자리 기본값 (프로젝트 베이스 mm). +Y 가 로봇 기준 왼쪽.
SPOTS = {"left": (150.0, 180.0, 34.0), "right": (150.0, -180.0, 34.0)}
CYCLES = 10
# 범위 밖이거나 형식이 틀리면 그 값만 기본값을 쓴다.
SETTING_LIMITS = {"approach_d_mm": (0.0, 60.0), "approach_a_mm": (5.0, 80.0)}  # mm
CYCLES_LIMITS = (1, 1000)
SPOT_KEYS = {"left_xyz_mm": "left", "right_xyz_mm": "right"}
SPOT_LIMITS = ((80.0, 350.0), (-350.0, 350.0), (0.0, 150.0))  # x, y, z mm (베이스)
# 카메라로 집기 기본값. 브라켓은 왼쪽 (+Y) 구역, 바구니는 오른쪽 (-Y): 위(먼 쪽) 흰색, 아래(가까운 쪽) 검은색.
VISION_REGION = ((60.0, 300.0), (60.0, 260.0))
# 팔이 제 몸에 걸리지 않고 옆으로 쥔 물체를 가져갈 수 있는 곳 (IK · 충돌 검사로 ±2 cm, z 70~80 mm 확인).
VISION_BINS = {"white_bin_xyz_mm": (160.0, -110.0, 80.0), "black_bin_xyz_mm": (95.0, -170.0, 80.0)}
VISION_WHITE_GRAY = 120.0
# 세 자세 · 여러 방향 모두 IK · 충돌 검사를 통과한 거리 (11 cm 안은 몸통, 27~30 cm 는 팔이 제 몸에 걸림).
VISION_REACH = (110.0, 270.0)
REGION_LIMITS = ((-100.0, 400.0), (-400.0, 400.0))  # x, y mm
VISION_LIMITS = {"white_gray_min": (1.0, 254.0), **SETTING_LIMITS}
_log = logging.getLogger("aivision.demo")


def _number_ok(v, lo: float, hi: float) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= float(v) <= hi


def _setting_ok(key: str, v) -> bool:
    if key == "cycles":
        return isinstance(v, int) and not isinstance(v, bool) and CYCLES_LIMITS[0] <= v <= CYCLES_LIMITS[1]
    if key in SPOT_KEYS:
        return isinstance(v, list) and len(v) == 3 and all(_number_ok(c, *lim) for c, lim in zip(v, SPOT_LIMITS))
    return _number_ok(v, *SETTING_LIMITS[key])


def _vision_setting_ok(key: str, v) -> bool:
    if key in VISION_BINS:
        return isinstance(v, list) and len(v) == 3 and all(_number_ok(c, *lim) for c, lim in zip(v, SPOT_LIMITS))
    if key == "pick_reach_mm":
        return isinstance(v, list) and len(v) == 2 and all(_number_ok(c, 0.0, 400.0) for c in v) and v[0] < v[1]
    if key == "pick_region_mm":
        return (
            isinstance(v, list) and len(v) == 2
            and all(isinstance(p, list) and len(p) == 2 and all(_number_ok(c, *lim) for c in p) and p[0] < p[1]
                    for p, lim in zip(v, REGION_LIMITS))
        )
    return _number_ok(v, *VISION_LIMITS[key])


def default_vision_settings() -> dict:
    d, a = approach_mm()
    return {
        "pick_region_mm": [list(p) for p in VISION_REGION],
        "pick_reach_mm": list(VISION_REACH),
        **{k: list(v) for k, v in VISION_BINS.items()},
        "white_gray_min": VISION_WHITE_GRAY,
        "approach_d_mm": d,
        "approach_a_mm": a,
    }


def default_demo_settings() -> dict:
    d, a = approach_mm()
    return {
        "cycles": CYCLES,
        "left_xyz_mm": list(SPOTS["left"]),
        "right_xyz_mm": list(SPOTS["right"]),
        "approach_d_mm": d,
        "approach_a_mm": a,
    }


def _write_settings(path: Path, data: dict, defaults: dict | None = None) -> None:
    """한 줄에 키 하나 (좌표도 한 줄)로 써서 손으로 고치기 쉽게 한다. 키 순서는 defaults 를 따른다."""
    keys = default_demo_settings() if defaults is None else defaults
    order = [k for k in keys if k in data] + [k for k in data if k not in keys]
    body = ",\n".join(f"  {json.dumps(k)}: {json.dumps(data[k], ensure_ascii=False)}" for k in order)
    try:
        path.write_text("{\n" + body + "\n}\n", encoding="utf-8")
    except OSError as exc:
        _log.warning("설정 파일을 못 씀 %s: %s", path, exc)


def load_demo_settings(path: Path = SETTINGS_PATH) -> dict:
    """demo_settings.json 값. 없으면 기본값으로 만들고, 빠진 키는 기본값으로 채워 넣는다.
    잘못된 값은 경고만 남기고 그 키만 기본값을 쓴다."""
    return _load(path, default_demo_settings(), _setting_ok)


def load_vision_settings(path: Path = VISION_SETTINGS_PATH) -> dict:
    """vision_pick_settings.json 값. 읽는 규칙은 load_demo_settings 와 같다."""
    return _load(path, default_vision_settings(), _vision_setting_ok)


def _load(path: Path, defaults: dict, ok) -> dict:
    if not path.exists():
        _write_settings(path, defaults, defaults)
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
        if ok(key, data[key]):
            out[key] = data[key]
        else:
            _log.warning("설정 %s=%r 이 범위 밖이거나 형식이 틀림 → 기본값 %r", key, data[key], defaults[key])
    missing = [k for k in defaults if k not in data]
    if missing:
        _write_settings(path, {**data, **{k: defaults[k] for k in missing}}, defaults)
    return out


def ensure_settings_file(path: Path = SETTINGS_PATH, vision_path: Path = VISION_SETTINGS_PATH) -> None:
    """파일이 없거나 키가 빠졌으면 기본값으로 만든다 · 채운다 (런처 시작 때). 두 파일 모두."""
    load_demo_settings(path)
    load_vision_settings(vision_path)
