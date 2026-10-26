"""lerobot-teleoperate 와 같지만, --display_data=true 일 때 Rerun 그래프를 관절마다 따로 그린다.

lerobot 은 관절 7개의 값을 그래프 하나에 모두 겹쳐 그린다. 여기서는 2행 4열 칸에 관절 하나씩
(마지막 칸은 비어 있음), 칸마다 리더(명령)와 팔로워(실제) 두 줄만 그린다.

인자는 lerobot-teleoperate 와 같다.

  python deploy/teleoperate_7dof.py \\
    --robot.type=so101_follower --robot.port=/dev/so101_follower --robot.id=follower --robot.dof_mode=7 \\
    --teleop.type=so101_leader --teleop.port=/dev/so101_leader --teleop.id=leader --teleop.dof_mode=7 \\
    --display_data=true
"""

from __future__ import annotations

import sys

import rerun as rr
import rerun.blueprint as rrb
from lerobot.scripts import lerobot_teleoperate as tele

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "elbow_roll", "wrist_flex", "wrist_roll", "gripper")
LEADER_COLOR = [80, 160, 255]  # 리더가 보낸 명령 (action)
FOLLOWER_COLOR = [255, 140, 40]  # 팔로워가 실제로 간 위치 (observation)


def _blueprint() -> rrb.Blueprint:
    views = []
    for j in JOINTS:
        act, obs = f"/action.{j}.pos", f"/observation.{j}.pos"
        views.append(
            rrb.TimeSeriesView(
                name=j,
                origin="/",
                contents=[act, obs],
                overrides={
                    act: rr.SeriesLines(colors=LEADER_COLOR, names="leader"),
                    obs: rr.SeriesLines(colors=FOLLOWER_COLOR, names="follower"),
                },
            )
        )
    return rrb.Blueprint(rrb.Grid(*views, grid_columns=4), collapse_panels=True)


_original_init_rerun = tele.init_rerun


def _init_rerun_with_grid(*args, **kwargs) -> None:
    _original_init_rerun(*args, **kwargs)
    rr.send_blueprint(_blueprint())


tele.init_rerun = _init_rerun_with_grid

if __name__ == "__main__":
    sys.exit(tele.main())
