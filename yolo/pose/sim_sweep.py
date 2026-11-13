"""여러 시험 조건을 한 번에 돌려 표로 남긴다. 실험용.

    python -m yolo.pose.sim_sweep kpt --weights vision/sim/out/pose_runs/v2/weights/best.pt
    python -m yolo.pose.sim_sweep rgb --weights ... --tag v2

결과: vision/sim/out/results/sweep_<묶음>_<tag>.md (표) 와 .json (물체마다 결과).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from vision.sim.data import OUT_DIR
from vision.sim.noise import NoiseConfig
from vision.sim.rgb_noise import PRESETS
from yolo.pose.sim_eval import Settings, run_eval

SCENES = {
    "default": OUT_DIR / "scenes",
    "lowcontrast": OUT_DIR / "scenes_lowcontrast",
    "dim": OUT_DIR / "scenes_dim",
    "harsh": OUT_DIR / "scenes_harsh",
    "random": OUT_DIR / "scenes_random",
    "x3": OUT_DIR / "scenes3",
    "x3_random": OUT_DIR / "scenes3_random",
    "distract": OUT_DIR / "scenes_distract",
    "robot": OUT_DIR / "scenes_robot",
    "silver": OUT_DIR / "scenes_silver",
    "black": OUT_DIR / "scenes_black",
    "robot_distract": OUT_DIR / "scenes_robot_distract",
}


def _cases(group: str) -> list[tuple[str, str, dict]]:
    """(이름, 장면 묶음, Settings 에 바꿀 값). 뎁스 노이즈는 모두 켠다."""
    if group == "kpt":
        out = [("기준 (흔들기 없음)", "default", {})]
        out += [(f"키포인트 ±{px:g}px", "default", {"kpt_noise_px": px}) for px in (0.5, 1, 2, 3, 5)]
        out += [(f"엉뚱한 점 {int(p * 100)}%", "default", {"kpt_outlier_p": p}) for p in (0.1, 0.2, 0.3)]
        out += [("±2px + 엉뚱한 점 10%", "default", {"kpt_noise_px": 2.0, "kpt_outlier_p": 0.1})]
        out += [
            ("손눈 2mm · 0.5°", "default", {"he_mm": 2.0, "he_deg": 0.5}),
            ("손눈 5mm · 1°", "default", {"he_mm": 5.0, "he_deg": 1.0}),
        ]
        return out
    if group == "rgb":
        out = [(f"컬러 {name}", "default", {"rgb": PRESETS[name]}) for name in PRESETS]
        out += [("왜곡 + PnP 왜곡 보정", "default", {"rgb": PRESETS["distort"], "pnp_dist": True})]
        out += [("real_hard + PnP 왜곡 보정", "default", {"rgb": PRESETS["real_hard"], "pnp_dist": True})]
        out += [(f"장면 {name}", name, {}) for name in ("lowcontrast", "dim", "harsh", "random")]
        out += [("은색 브라켓", "silver", {}), ("검은 브라켓", "black", {})]
        out += [("장면 random + real_hard", "random", {"rgb": PRESETS["real_hard"]})]
        out += [("브라켓 3개", "x3", {}), ("브라켓 3개 random + real_mild", "x3_random", {"rgb": PRESETS["real_mild"]})]
        out += [("방해물 0~8개", "distract", {}), ("방해물 + real_mild", "distract", {"rgb": PRESETS["real_mild"]})]
        out += [("로봇 (시작 자세)", "robot", {}), ("로봇 + 방해물 + real_mild", "robot_distract", {"rgb": PRESETS["real_mild"]})]
        return out
    if group == "icp":
        # 기존 방식도 같은 조건에서 (컬러 열화는 영향 없음, 뎁스 · 장면 · 손눈만).
        return [
            ("기준", "default", {}),
            ("손눈 2mm · 0.5°", "default", {"he_mm": 2.0, "he_deg": 0.5}),
            ("손눈 5mm · 1°", "default", {"he_mm": 5.0, "he_deg": 1.0}),
            ("렌즈 왜곡", "default", {"rgb": PRESETS["distort"]}),
            ("브라켓 3개", "x3", {}),
        ]
    raise ValueError(group)


def main() -> None:
    ap = argparse.ArgumentParser(description="시뮬 시험 조건 묶음 돌리기")
    ap.add_argument("group", choices=("kpt", "rgb", "icp"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--method", default=None, help="기본: icp 묶음은 icp, 나머지는 pnp_depth")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    method = args.method or ("icp" if args.group == "icp" else "pnp_depth")
    base = Settings(noise=NoiseConfig(), cls_mode="gt", method=method, weights=args.weights)
    table = [
        "| 조건 | 성공 | 세우기 | 눕히기 | 비스듬히 | 자신 있게 틀림 | 미검출 | 엉뚱한 검출 | 위치 mm | 회전 ° | 시간 s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    dump = {}
    stem = f"sweep_{args.group}_{method}" + (f"_{args.tag}" if args.tag else "")
    res_dir = OUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    for name, scenes, change in _cases(args.group):
        folders = sorted(p for p in Path(SCENES[scenes]).iterdir() if (p / "gt.json").is_file())[: args.n]
        rows, sm = run_eval(folders, replace(base, **change), workers=args.workers)
        bc = sm["by_class"]
        line = (
            f"| {name} | {sm['success']}/{sm['n']} | {bc.get('0', '-')} | {bc.get('1', '-')} | {bc.get('2', '-')} | "
            f"{sm['confident_wrong']} | {sm['no_detect']} | {sm.get('false_det', '-')} | {sm['pos_mm_median_ok']} | "
            f"{sm['rot_deg_median_ok']} | "
            f"{sm['time_median_s']} |"
        )
        table.append(line)
        print(line, flush=True)
        dump[name] = {"scenes": scenes, "change": {k: str(v) for k, v in change.items()}, "summary": sm, "rows": rows}
        (res_dir / f"{stem}.md").write_text("\n".join(table) + "\n", encoding="utf-8")
        (res_dir / f"{stem}.json").write_text(json.dumps(dump, indent=1, default=str), encoding="utf-8")
    print(f"저장: {res_dir / (stem + '.md')}")


if __name__ == "__main__":
    main()
