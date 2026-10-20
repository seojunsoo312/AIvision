#!/usr/bin/env bash
# 7축 팔 캘리브. 기존 캘리브 파일을 백업한 뒤 lerobot-calibrate 를 실행한다.
# conda activate 를 하지 않아도 AIvision 환경의 lerobot-calibrate 를 찾는다.
#
#   bash ~/AIvision/deploy/calibrate_arms.sh            # 팔로워 → 리더
#   bash ~/AIvision/deploy/calibrate_arms.sh follower   # 팔로워만
#   bash ~/AIvision/deploy/calibrate_arms.sh leader     # 리더만
#
# 각 단계에서 lerobot-calibrate 가 안내하는 대로 팔을 가운데 두고 Enter, 관절을 끝까지
# 천천히 돌린 뒤 Enter. 펜던트(pendant/main.py)는 꺼 둔다 (같은 포트를 열 수 없다).
set -euo pipefail

which="${1:-all}"
case "$which" in all|follower|leader) ;; *) echo "인자는 all, follower, leader 중 하나"; exit 1;; esac

CAL=""
for d in "$HOME/miniforge3" "$HOME/miniconda3"; do
  [ -x "$d/envs/AIvision/bin/lerobot-calibrate" ] && CAL="$d/envs/AIvision/bin/lerobot-calibrate" && break
done
[ -n "$CAL" ] || { echo "AIvision 환경의 lerobot-calibrate 가 없습니다. 먼저 bash deploy/setup_jetson.sh"; exit 1; }
if pgrep -f "pendant/main.py" >/dev/null; then
  echo "펜던트가 켜져 있습니다. 펜던트 창을 닫고 다시 실행하세요."; exit 1
fi

CAL_DIR="$HOME/.cache/huggingface/lerobot/calibration"
backup() {  # $1 = 캘리브 파일
  [ -f "$1" ] || return 0
  local b="$1.bak.$(date +%Y%m%d_%H%M%S)"
  cp "$1" "$b"
  echo "   기존 파일 백업: $b"
}

run_one() {  # $1 = follower|leader
  local role="$1" port file args
  if [ "$role" = follower ]; then
    port=/dev/so101_follower
    file="$CAL_DIR/robots/so_follower/follower.json"
    args=(--robot.type=so101_follower --robot.port="$port" --robot.id=follower)
  else
    port=/dev/so101_leader
    file="$CAL_DIR/teleoperators/so_leader/leader.json"
    args=(--teleop.type=so101_leader --teleop.port="$port" --teleop.id=leader)
  fi
  echo
  echo "== $role 캘리브 ($port)"
  [ -e "$port" ] || { echo "$port 가 없습니다. 팔을 꽂거나 bash deploy/make_serial_rules.sh 를 먼저 하세요."; return 1; }
  backup "$file"
  "$CAL" "${args[@]}"
  [ -f "$file" ] && echo "   저장: $file" || { echo "   캘리브 파일이 생기지 않았습니다."; return 1; }
}

[ "$which" = leader ] || run_one follower
[ "$which" = follower ] || run_one leader
echo
echo "끝. 펜던트로 확인: python pendant/main.py → Connect → 팔이 튀지 않고 지금 자세에서 조그가 되는지."
