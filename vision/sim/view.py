"""시뮬 장면을 MuJoCo 창으로 본다. 브라켓이 떨어져 멈추는 모습을 실제 속도로 보여 주고, 멈추면 다음 장면으로 간다.

    python -m vision.sim.view                # 브라켓 1개, 장면 계속
    python -m vision.sim.view --objects 3 --seed 5

창 두 개가 뜬다.
  "camera 640x480"  실제 카메라와 같은 화소 · 화각으로 렌더한 화면 (정합에 쓰는 그 영상). 멈추면 오른쪽에 뎁스.
  MuJoCo 창         3D 장면. 창 비율에 맞춰 그리므로 실제 카메라 시점이어도 위아래 · 좌우가 잘려 보일 수 있다.

MuJoCo 창 조작
  [ / ]     카메라 바꾸기 (처음엔 실제 카메라 시점 "cam")
  Esc       자유 시점 (마우스 왼쪽 회전, 오른쪽 이동, 휠 확대)
  Space     일시 정지
  창 닫기   끝
"""

from __future__ import annotations

import argparse
import os
import time

# 창을 띄우려면 GLFW 를 써야 한다. scene 을 부르기 전에 정한다.
os.environ["MUJOCO_GL"] = "glfw"

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from vision.depth_cloud import colorize_depth
from vision.sim.scene import SceneMaker, pose_class

# 멈춘 뒤 다음 장면까지 기다리는 시간.
HOLD_S = 2.5
# 떨어지는 동안 카메라 화면을 다시 그리는 간격.
CAM_PERIOD_S = 1.0 / 20.0
CAM_WIN = "camera 640x480"


def _show_rgb(maker: SceneMaker) -> None:
    """실제 카메라와 같은 640×480 화면을 그대로 보여 준다."""
    r = maker.renderer
    r.update_scene(maker.data, camera=maker.cam_id)
    cv2.imshow(CAM_WIN, cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def _show_settled(maker: SceneMaker) -> None:
    """멈춘 장면: 컬러 | 뎁스 (정합 코드가 받는 그 두 장)."""
    rgb, depth_mm, _seg = maker.render()
    both = np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), colorize_depth(depth_mm)])
    cv2.imshow(CAM_WIN, both)
    cv2.waitKey(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="시뮬 장면 보기")
    ap.add_argument("--objects", type=int, default=1, help="장면마다 브라켓 수")
    ap.add_argument("--seed", type=int, default=0, help="scene.py 와 같은 시드면 같은 장면 순서")
    ap.add_argument("--free", action="store_true", help="실제 카메라 시점 대신 자유 시점으로 시작")
    args = ap.parse_args()

    maker = SceneMaker(args.objects)
    rng = np.random.default_rng(args.seed)
    model, data = maker.model, maker.data
    dt = float(model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as v:
        if not args.free:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            v.cam.fixedcamid = maker.cam_id
        k = 0
        while v.is_running():
            maker.start_drop(rng)
            v.sync()
            t0 = time.perf_counter()
            sim_t = 0.0
            done = False
            next_cam = 0.0
            # 실제 시간에 맞춰 물리를 돌린다.
            while v.is_running() and not done:
                while sim_t < time.perf_counter() - t0 and not done:
                    done = maker.step()
                    sim_t += dt
                v.sync()
                if time.perf_counter() >= next_cam:
                    _show_rgb(maker)
                    next_cam = time.perf_counter() + CAM_PERIOD_S
                time.sleep(0.005)
            maker.freeze()
            v.sync()
            _show_settled(maker)
            ok = "OK" if maker.placement_ok() else "버림 (범위 · 간격)"
            info = []
            for i in range(maker.n):
                T = maker.T_base_obj(i)
                cid, ang = pose_class(T[:3, :3])
                info.append(f"#{i + 1} 클래스 {cid} ({ang:.0f}°) xy=({T[0, 3]:.0f}, {T[1, 3]:.0f})")
            print(f"장면 {k}: {ok}  " + "  ".join(info), flush=True)
            k += 1
            t_end = time.perf_counter() + HOLD_S
            while v.is_running() and time.perf_counter() < t_end:
                v.sync()
                cv2.waitKey(20)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
