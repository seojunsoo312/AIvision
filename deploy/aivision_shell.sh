# 「AIvision 터미널」 각 칸이 처음 읽는 파일 (bash --init-file).
# 평소 설정(~/.bashrc)을 읽고, AIvision 환경을 켜고, 이 칸에서 칠 명령을 한 줄 보여 준다
# (네 번째 칸, AIvision 폴더는 빈 칸이라 안내 없음).
# 폴더는 terminator 레이아웃이 칸마다 정해 준다 (deploy/install_terminal_icon.sh).

[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"

# ~/.bashrc 에 conda 설정이 없어도 켜지게 한다.
if ! command -v conda >/dev/null 2>&1; then
  for _d in "$HOME/miniforge3" "$HOME/miniconda3"; do
    [ -f "$_d/etc/profile.d/conda.sh" ] && . "$_d/etc/profile.d/conda.sh" && break
  done
  unset _d
fi
conda activate AIvision 2>/dev/null || printf '\033[31m! AIvision 환경을 켜지 못했습니다\033[0m\n'

case "$(basename "$PWD")" in
  pendant) AIVISION_HINT="python main.py   (픽앤플레이스: --grasp)" ;;
  vision)  AIVISION_HINT="python handeye/vision_test.py · get_intrinsic.py · capture.py · compute.py" ;;
  yolo)    AIVISION_HINT="python train/capture.py · label.py · prepare.py · train.py · detect.py
   카메라 창: python pose/roi_cloud.py" ;;
  *)       AIVISION_HINT="" ;;
esac

_aivision_hint() { [ -z "$AIVISION_HINT" ] || printf '\033[1;36m▶ %s\033[0m\n' "$AIVISION_HINT"; }

# clear 나 Ctrl+L 로 화면을 지워도 안내 줄이 다시 나오게 한다.
clear() { command clear "$@"; _aivision_hint; }
bind -x '"\C-l": clear' 2>/dev/null

_aivision_hint
