#!/usr/bin/env python3
"""런처가 미리 켜 두는 프로그램. 켜는 데 오래 걸리는 것 (팔 모델 · lerobot, 15초 정도) 을 먼저 해 둔다.

팔은 연결하지 않는다. 런처의 「켜기」가 보내는 명령에 따라 그 프로그램이 된다.
  start demo x,y,w,h [인자…]     → 반복 데모 (repeat_demo.main, 뒤 인자는 그대로 넘김)
  start teleop x,y,w,h [인자…]   → 원격 조종 (teleop.main)
  off                            → 그냥 끝
메모리를 많이 써서 (1GB 넘게) 프로그램마다 따로 두지 않고 하나만 둔다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from common import prepare  # noqa: E402
from motion import runlog  # noqa: E402

_log = runlog.get("standby")


def main() -> int:
    ap = argparse.ArgumentParser(description="런처가 미리 켜 두는 프로그램")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    warm = prepare(args.sim, args.no_viz, _log)
    print(json.dumps({"state": "ready"}), flush=True)
    while True:
        parts = warm.cmds.get().split()
        if parts[0] == "off":
            warm.ctrl.stop()
            if warm.viz is not None:
                warm.viz.close()
            return 0
        if parts[0] == "start" and len(parts) >= 2 and parts[1] in ("demo", "teleop"):
            break
    if parts[1] == "demo":
        import repeat_demo as job
    else:
        import teleop as job
    argv = (["--sim"] if args.sim else []) + (["--no-viz"] if args.no_viz else [])
    if len(parts) > 2:
        argv += ["--viz-geometry", parts[2]]
    argv += parts[3:]
    return job.main(argv, warm=warm)


if __name__ == "__main__":
    sys.exit(main())
