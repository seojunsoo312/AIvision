"""카메라 창(roi_cloud)에 물체 등록을 요청하고 결과를 받는다. JSON 파일 두 개로 주고받는다.

요청 쪽(펜던트 「물체 위치 불러오기」, load_pick_targets)이 요청 파일에 {"id"} 를 쓰면,
roi_cloud 가 같은 id 로 상태 파일에 state 를 쓴다.
요청에 더 넣을 수 있는 것 (런처 「카메라로 집기」가 쓴다, 모두 베이스 mm):
  "exclude_xy_mm": [[x, y, r], ...]       이 원 안의 물체는 고르지 않는다
  "region_xy_mm": [[x0, x1], [y0, y1]]    가운데가 이 사각형 안인 물체만 고른다
  "nearest": true                         가장 높은 것 대신 베이스 원점에서 가장 가까운 것을 고른다
그러면 상태에 "gray" (고른 물체 점들의 밝기 중앙값 0–255) 와
"targets_xy_mm" (고를 수 있던 물체 가운데 [x, y], 고르는 순서대로) 가 같이 온다.

  run     요청을 받아 등록 중
  ok      끝. xyz_mm, rpy_deg 가 같이 온다 (로봇 베이스, mm·deg)
          "all" 요청 ({"id", "all": true}, load_all_pick_targets) 이면 objects 목록이 온다:
          브라켓마다 {xyz_mm, rpy_deg, pose, color, verdict}. 로봇에서 가까운 순.
  busy    이미 다른 등록 중이라 받지 않음
  no_box  바운딩 박스가 없음
  fail    등록 실패

요청 쪽은 REG_ACK_S 안에 run 이 없거나, run 뒤 REG_JOB_S 안에 끝나지 않으면 timeout 으로 본다.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from motion import runlog

# 요청 · 상태 파일은 카메라 창(yolo/pose/roi_cloud.py)의 기록 폴더에 둔다.
_ROI_DIR = Path(__file__).resolve().parent.parent / "yolo" / "runs" / "roi"

REGISTER_REQUEST_JSON = _ROI_DIR / "register_request.json"
REGISTER_STATUS_JSON = _ROI_DIR / "register_status.json"
REG_ACK_S = 2.0  # 요청 → run 까지
REG_JOB_S = 30.0  # run → 끝까지
REG_POLL_MS = 200
REG_MSG_RUN = "현재 등록 중입니다."
REG_MSG_BUSY = "이미 등록 중입니다."
REG_MSG_NO_BOX = "바운딩 박스가 없습니다. 카메라 창을 확인해 주세요."
REG_MSG_FAIL = "등록에 실패했습니다. 카메라 창을 확인해 주세요."
REG_MSG_NO_REPLY = "카메라 창이 응답하지 않습니다. 실행중인지 확인해주세요"
# poll() 이 이 중 하나를 돌려주면 끝. 나머지는 "wait"(아직 응답 없음), "run".
DONE_STATES = ("ok", "busy", "no_box", "fail", "timeout")
_log = runlog.get("register")


def _write_json(path: Path, payload: dict) -> None:
    """tmp 에 쓰고 바꿔 끼운다. 읽는 쪽이 반쯤 쓴 파일을 보지 않게."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


# --- roi_cloud 쪽 ---


def read_request() -> str | None:
    """요청 id. 없거나 깨졌으면 None."""
    rid, _all = read_request_full()
    return rid


def read_request_full() -> tuple[str | None, bool]:
    """(요청 id, 모든 브라켓을 원하는지). 없거나 깨졌으면 (None, False)."""
    data = _read_json(REGISTER_REQUEST_JSON)
    rid = data.get("id") if data is not None else None
    if not isinstance(rid, str) or not rid:
        return None, False
    return rid, bool(data.get("all", False))


@dataclass(frozen=True)
class RequestOptions:
    """요청에 붙은 고르는 조건. 아무것도 없으면 예전처럼 카메라 창이 고른 (가장 높은) 물체."""

    exclude: tuple[tuple[float, float, float], ...] = ()
    region: tuple[tuple[float, float], tuple[float, float]] | None = None
    nearest: bool = False

    @property
    def active(self) -> bool:
        return bool(self.exclude) or self.region is not None or self.nearest


def read_request_options() -> RequestOptions:
    """요청의 고르는 조건. 없거나 깨진 항목은 뺀다."""
    data = _read_json(REGISTER_REQUEST_JSON) or {}
    exclude = []
    items = data.get("exclude_xy_mm")
    for item in items if isinstance(items, list) else []:
        try:
            x, y, r = (float(v) for v in item)
        except (TypeError, ValueError):
            continue
        if r > 0:
            exclude.append((x, y, r))
    region = None
    try:
        (x0, x1), (y0, y1) = ((float(a), float(b)) for a, b in data["region_xy_mm"])
        if x0 < x1 and y0 < y1:
            region = ((x0, x1), (y0, y1))
    except (KeyError, TypeError, ValueError):
        pass
    return RequestOptions(exclude=tuple(exclude), region=region, nearest=data.get("nearest") is True)


def write_status(
    req_id: str, state: str, *, xyz=None, rpy=None, objects: list[dict] | None = None, extra: dict | None = None
) -> None:
    """state: run, busy, no_box, fail, ok. ok 이면 xyz(mm), rpy(deg) 를, all 요청이면 objects 를 같이 준다.
    extra: 같이 실을 값 (gray, targets_xy_mm 등)."""
    payload: dict = {"id": str(req_id), "state": state, **(extra or {})}
    if xyz is not None and rpy is not None:
        payload["xyz_mm"] = [float(v) for v in xyz]
        payload["rpy_deg"] = [float(v) for v in rpy]
    if objects is not None:
        payload["objects"] = objects
    _write_json(REGISTER_STATUS_JSON, payload)
    if state == "ok" and objects is not None:
        _log.info("요청 %s → ok  브라켓 %d개", req_id[:8], len(objects))
    elif state == "ok":
        _log.info(
            "요청 %s → ok  xyz=%s  rpy=%s",
            req_id[:8],
            [round(v, 1) for v in payload["xyz_mm"]],
            [round(v, 1) for v in payload["rpy_deg"]],
        )
    elif state != "run":
        _log.warning("요청 %s → %s", req_id[:8], state)


# --- 요청 쪽 ---


class RegisterRequest:
    """등록 요청 하나. send() 로 보내고 poll() 이나 wait() 로 결과를 본다.

    GUI 는 타이머에서 poll() 을, 스크립트는 wait() 을 부른다.
    ok 이면 xyz_mm, rpy_deg 에 결과가 있다.
    """

    def __init__(self, all_objects: bool = False) -> None:
        self.id = uuid.uuid4().hex
        self.state = "wait"
        self.all_objects = bool(all_objects)
        self.xyz_mm: tuple[float, float, float] | None = None
        self.rpy_deg: tuple[float, float, float] | None = None
        # all 요청의 결과: 브라켓마다 {"xyz_mm", "rpy_deg", "pose", "color", "verdict"}
        self.objects: list[dict] | None = None
        # 요청에 고르는 조건을 붙였을 때만 온다.
        self.gray: float | None = None
        self.targets_xy_mm: list[tuple[float, float]] = []
        self._accepted = False
        self._ack_deadline = 0.0
        self._job_deadline = 0.0

    @classmethod
    def send(
        cls, all_objects: bool = False, *, exclude_xy_mm=None, region_xy_mm=None, nearest: bool = False
    ) -> RegisterRequest:
        """요청 파일을 쓴다. all_objects 면 보이는 브라켓 모두를 요청한다. 쓰지 못하면 OSError.

        exclude_xy_mm: [(x, y, r), ...] 이 원 안의 물체는 고르지 않는다.
        region_xy_mm: ((x0, x1), (y0, y1)) 이 사각형 안의 물체만. nearest: 베이스에서 가장 가까운 것부터.
        """
        req = cls(all_objects)
        now = time.monotonic()
        req._ack_deadline = now + REG_ACK_S
        req._job_deadline = now + REG_JOB_S
        payload: dict = {"id": req.id}
        if all_objects:
            payload["all"] = True
        if exclude_xy_mm:
            payload["exclude_xy_mm"] = [[float(v) for v in item] for item in exclude_xy_mm]
        if region_xy_mm is not None:
            payload["region_xy_mm"] = [[float(v) for v in pair] for pair in region_xy_mm]
        if nearest:
            payload["nearest"] = True
        _write_json(REGISTER_REQUEST_JSON, payload)
        return req

    @property
    def done(self) -> bool:
        return self.state in DONE_STATES

    def poll(self) -> str:
        """상태 파일을 한 번 읽고 state 를 돌려준다. 끝난 뒤에는 그 값을 그대로."""
        if self.done:
            return self.state
        data = _read_json(REGISTER_STATUS_JSON)
        if data is not None and data.get("id") == self.id:
            state = str(data.get("state") or "")
            if state == "run":
                self._accepted = True
                self.state = "run"
            elif state in ("busy", "no_box", "fail"):
                self.state = state
                self._take_extra(data)
                return self.state
            elif state == "ok":
                self.state = "ok" if self._take_pose(data) else "fail"
                return self.state
        now = time.monotonic()
        deadline = self._job_deadline if self._accepted else self._ack_deadline
        if now >= deadline:
            self.state = "timeout"
        return self.state

    def wait(self) -> str:
        """끝날 때까지 이 줄에서 기다린다."""
        while not self.done:
            self.poll()
            if not self.done:
                time.sleep(REG_POLL_MS / 1000.0)
        return self.state

    def _take_pose(self, data: dict) -> bool:
        if self.all_objects:
            return self._take_objects(data)
        xyz = data.get("xyz_mm")
        rpy = data.get("rpy_deg")
        if not isinstance(xyz, list) or not isinstance(rpy, list) or len(xyz) < 3 or len(rpy) < 3:
            return False
        try:
            self.xyz_mm = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            self.rpy_deg = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
        except (TypeError, ValueError):
            return False
        self._take_extra(data)
        return True

    def _take_extra(self, data: dict) -> None:
        try:
            self.gray = None if data.get("gray") is None else float(data["gray"])
        except (TypeError, ValueError):
            self.gray = None
        out = []
        for item in data.get("targets_xy_mm") or []:
            try:
                out.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError, IndexError):
                pass
        self.targets_xy_mm = out

    def _take_objects(self, data: dict) -> bool:
        objs = data.get("objects")
        if not isinstance(objs, list):
            return False
        out = []
        try:
            for o in objs:
                out.append({
                    "xyz_mm": tuple(float(v) for v in o["xyz_mm"][:3]),
                    "rpy_deg": tuple(float(v) for v in o["rpy_deg"][:3]),
                    "pose": str(o.get("pose", "")),
                    "color": str(o.get("color", "")),
                    "verdict": str(o.get("verdict", "")),
                })
        except (KeyError, TypeError, ValueError, IndexError):
            return False
        self.objects = out
        return True
