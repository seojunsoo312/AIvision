#!/usr/bin/env bash
# Jetson(Orin Nano, JetPack 6) 한 대에 이 프로젝트를 까는 스크립트.
# docs/jetson_setup.md 2~5번을 순서대로 하고, 끝나면 deploy/check_board.py 로 점검한다.
# 이미 끝난 단계는 건너뛰므로 중간에 멈추면 같은 명령을 다시 실행하면 된다.
#
#   git clone https://github.com/seojunsoo312/AIvision.git ~/AIvision
#   bash ~/AIvision/deploy/setup_jetson.sh
#
# 인터넷이 필요하다. 처음에 sudo 비밀번호를 한 번 묻는다.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="$(dirname "$PROJECT_DIR")"   # Orbbec SDK 는 프로젝트 옆에 둔다 (docs/vision.md)
LOG="$HOME/aivision_setup_$(date +%Y%m%d_%H%M%S).log"
ENV_NAME="AIvision"

TORCH_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"
TORCH_PINS=$'torch==2.8.0\ntorchvision==0.23.0\n'
NUMPY_PIN="numpy==1.26.4"            # Jetson torch 2.8.0 은 NumPy 1.x 로 빌드됨
OPENCV_PIN="opencv-python<4.12"      # 4.12 부터 numpy 2 를 요구
ORBBEC_URL="https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.27/OrbbecViewer_v1.10.27_202509260950_arm64_release.zip"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
# lerobot fork 는 docs/requirements.txt 에 적힌 커밋 한 곳만 정본으로 쓴다.
LEROBOT_SPEC="$(grep -oE 'lerobot\[feetech\] @ git\+[^"]+' "$PROJECT_DIR/docs/requirements.txt" | head -1)"
LEROBOT_COMMIT="${LEROBOT_SPEC##*@}"

exec > >(tee -a "$LOG") 2>&1
say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
skip() { printf '   \033[90m- %s (이미 됨)\033[0m\n' "$*"; }
warn() { printf '   \033[33m! %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n로그: %s\n' "$*" "$LOG"; exit 1; }

# ---------------------------------------------------------------- 0. 확인
say "0. 보드 확인"
[ "$(uname -m)" = "aarch64" ] || die "Jetson(aarch64)이 아닙니다: $(uname -m). 노트북은 docs/laptop_setup.md"
grep -q "R36" /etc/nv_tegra_release 2>/dev/null || die "JetPack 6(R36)이 아닙니다: $(head -1 /etc/nv_tegra_release 2>/dev/null)"
[ "$(id -u)" -ne 0 ] || die "sudo 로 실행하지 마세요. 일반 사용자로 실행하면 필요할 때 비밀번호를 묻습니다."
[ -n "$LEROBOT_SPEC" ] || die "docs/requirements.txt 에서 lerobot 설치 줄을 찾지 못했습니다."
ok "$(head -1 /etc/nv_tegra_release | cut -c1-40)"
ok "프로젝트 $PROJECT_DIR"
ok "로그 $LOG"
sudo -v || die "sudo 권한이 필요합니다."
# 설치가 길어도 sudo 를 다시 묻지 않게 유지한다.
( while true; do sudo -n true; sleep 50; kill -0 "$$" 2>/dev/null || exit; done ) 2>/dev/null &

# ---------------------------------------------------------------- 1. apt
say "1. 시스템 패키지"
PKGS=(git curl unzip build-essential fonts-noto-cjk tk8.6 libtk8.6)
missing=()
for p in "${PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
if [ ${#missing[@]} -gt 0 ]; then
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
  ok "설치: ${missing[*]}"
else
  skip "${PKGS[*]}"
fi

# ---------------------------------------------------------------- 2. 권한
say "2. 팔(시리얼)·카메라 권한"
need_relogin=0
for g in dialout video; do
  if id -nG "$USER" | tr ' ' '\n' | grep -qx "$g"; then skip "그룹 $g"
  else sudo usermod -aG "$g" "$USER"; ok "그룹 $g 추가"; need_relogin=1; fi
done

# ---------------------------------------------------------------- 3. 전원 모드
say "3. 전원 모드 (MAXN)"
# 모드 번호는 보드마다 다르다. Orin Nano Super 는 2 = MAXN_SUPER (0 은 15W).
maxn_id="$(grep -oE 'POWER_MODEL ID=[0-9]+ NAME=MAXN[A-Z_]*' /etc/nvpmodel.conf | head -1 | sed -E 's/.*ID=([0-9]+).*/\1/')"
cur_mode="$(sudo nvpmodel -q 2>/dev/null | grep -oE 'NV Power Mode: .*' | sed 's/NV Power Mode: //')"
if [ -z "$maxn_id" ]; then
  warn "MAXN 모드를 찾지 못했습니다. 지금 모드: $cur_mode"
elif [[ "$cur_mode" == MAXN* ]]; then
  skip "$cur_mode"
else
  # 재부팅을 물으면 '아니오'로 답한다. 설정은 저장되고 다음 부팅부터 확실히 적용된다.
  echo no | sudo nvpmodel -m "$maxn_id" >/dev/null 2>&1 || true
  ok "모드 $maxn_id 로 설정 ($(sudo nvpmodel -q 2>/dev/null | grep -oE 'NV Power Mode: .*'))"
fi

# ---------------------------------------------------------------- 4. 스왑
say "4. 스왑 8GB (YOLO 학습용)"
if [ -f /swapfile ]; then
  skip "/swapfile"
else
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  ok "/swapfile 8GB 생성"
fi
swapon --show=NAME --noheadings | grep -qx /swapfile || sudo swapon /swapfile
grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
ok "부팅 때 자동으로 켜짐"

# ---------------------------------------------------------------- 5. Orbbec SDK
say "5. Orbbec 카메라 SDK (ARM64)"
sdk_lib="$(find "$HOME_DIR"/OrbbecViewer_* "$HOME_DIR"/OrbbecSDK_* -maxdepth 2 -name 'libOrbbecSDK.so' 2>/dev/null | head -1 || true)"
if [ -n "$sdk_lib" ]; then
  skip "$sdk_lib"
else
  tmp="$(mktemp -d)"
  curl -fL --retry 3 -o "$tmp/orbbec.zip" "$ORBBEC_URL" || die "Orbbec SDK 다운로드 실패: $ORBBEC_URL"
  unzip -q "$tmp/orbbec.zip" -d "$tmp/x"
  # 압축이 폴더를 한 겹 더 만들어도 libOrbbecSDK.so 가 있는 폴더를 홈 바로 아래로 옮긴다.
  lib="$(find "$tmp/x" -name 'libOrbbecSDK.so' | head -1)"
  [ -n "$lib" ] || die "압축 안에 libOrbbecSDK.so 가 없습니다."
  top="$(dirname "$lib")"; [ "$(basename "$top")" = "lib" ] && top="$(dirname "$top")"
  name="$(basename "$ORBBEC_URL" .zip)"
  mv "$top" "$HOME_DIR/$name"
  rm -rf "$tmp"
  sdk_lib="$(find "$HOME_DIR/$name" -maxdepth 2 -name 'libOrbbecSDK.so' | head -1)"
  ok "$HOME_DIR/$name"
fi
if [ -f /etc/udev/rules.d/99-obsensor-libusb.rules ]; then
  skip "카메라 USB 권한 규칙"
else
  sdk_dir="$(dirname "$sdk_lib")"; [ "$(basename "$sdk_dir")" = "lib" ] && sdk_dir="$(dirname "$sdk_dir")"
  rules_sh="$(find "$sdk_dir" -name 'install_udev_rules.sh' | head -1)"
  [ -n "$rules_sh" ] || die "SDK 안에 install_udev_rules.sh 가 없습니다. 노트북의 /etc/udev/rules.d/99-obsensor-libusb.rules 를 복사하세요."
  (cd "$(dirname "$rules_sh")" && sudo bash "$(basename "$rules_sh")")
  ok "카메라 USB 권한 규칙 설치"
fi

# ---------------------------------------------------------------- 6. conda
say "6. conda (Python 만 conda 로 만든다)"
CONDA_BASE=""
for d in "$HOME/miniforge3" "$HOME/miniconda3"; do [ -x "$d/bin/conda" ] && CONDA_BASE="$d" && break; done
if [ -n "$CONDA_BASE" ]; then
  skip "$CONDA_BASE"
else
  tmp="$(mktemp -d)"
  curl -fL --retry 3 -o "$tmp/mf.sh" "$MINIFORGE_URL" || die "Miniforge 다운로드 실패"
  bash "$tmp/mf.sh" -b -p "$HOME/miniforge3"
  rm -rf "$tmp"
  CONDA_BASE="$HOME/miniforge3"
  "$CONDA_BASE/bin/conda" init bash >/dev/null
  ok "$CONDA_BASE 설치 (새 터미널부터 conda 명령 사용 가능)"
fi
ENV_DIR="$CONDA_BASE/envs/$ENV_NAME"
PY="$ENV_DIR/bin/python"
if [ -x "$PY" ]; then
  skip "환경 $ENV_NAME"
else
  # numpy 등을 conda 로 받으면 Jetson NVPL 과 엮여 numpy 가 깨진다. Python 만.
  "$CONDA_BASE/bin/conda" create -y -n "$ENV_NAME" python=3.10
  ok "환경 $ENV_NAME (Python 3.10)"
fi
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version' || die "$ENV_NAME 환경이 Python 3.10 이 아닙니다."

# ---------------------------------------------------------------- 7. pip 패키지
say "7. Python 패키지 (순서가 중요: torch → lerobot → opencv → 나머지 → numpy)"
CONS="$(mktemp)"; printf '%s' "$TORCH_PINS" > "$CONS"
pip_i() { "$PY" -m pip install --disable-pip-version-check "$@"; }

if "$PY" -c 'import torch, sys; sys.exit(0 if torch.__version__.startswith("2.8.0") and torch.cuda.is_available() else 1)' 2>/dev/null; then
  skip "torch 2.8.0 (GPU)"
else
  pip_i torch==2.8.0 torchvision==0.23.0 --index-url "$TORCH_INDEX"
  ok "torch 2.8.0 (Jetson GPU)"
fi

have_commit="$("$PY" - <<'PYEOF' 2>/dev/null || true
import json, importlib.metadata as m
print(json.loads(m.distribution("lerobot").read_text("direct_url.json"))["vcs_info"]["commit_id"])
PYEOF
)"
if [ "$have_commit" = "$LEROBOT_COMMIT" ]; then
  skip "lerobot 7축 fork ${LEROBOT_COMMIT:0:8}"
else
  pip_i -c "$CONS" "$LEROBOT_SPEC"
  ok "lerobot 7축 fork ${LEROBOT_COMMIT:0:8}"
fi

# lerobot 이 창 없는 opencv-python-headless 를 깐다. 카메라 창이 필요하니 바꾼다.
if "$PY" -m pip show opencv-python-headless >/dev/null 2>&1 || ! "$PY" -m pip show opencv-python >/dev/null 2>&1; then
  "$PY" -m pip uninstall -y opencv-python-headless opencv-python >/dev/null 2>&1 || true
  pip_i -c "$CONS" "$OPENCV_PIN"
  ok "opencv-python (창 있는 판)"
else
  skip "opencv-python (창 있는 판)"
fi

pip_i -c "$CONS" ultralytics open3d pin meshcat customtkinter Pillow jupyter ipykernel
ok "ultralytics, open3d, pinocchio, meshcat, customtkinter, jupyter"

# 마지막에 numpy 를 1.26 으로. lerobot·rerun 등이 numpy>=2 경고를 내지만 이 프로젝트에서 쓰는 기능은 1.26 에서 확인했다.
pip_i "$NUMPY_PIN" 2>&1 | grep -vE "requires numpy|dependency conflicts|resolver does not" || true
ok "$NUMPY_PIN"
rm -f "$CONS"

"$PY" -m ipykernel install --user --name "$ENV_NAME" --display-name "$ENV_NAME" >/dev/null
ok "노트북 커널 $ENV_NAME"

# ---------------------------------------------------------------- 8. 브라우저
say "8. Chrome 그래픽 가속 (Meshcat 3D 화면)"
LOCAL_STATE="$HOME/.config/google-chrome/Local State"
if [ ! -f "$LOCAL_STATE" ]; then
  skip "Chrome 설정 없음 (기본값이 가속 켜짐)"
elif pgrep -x chrome >/dev/null; then
  warn "Chrome 이 켜져 있어 확인을 건너뜁니다. 흰 화면이면 chrome://settings/system 에서 「가능한 경우 그래픽 가속 사용」을 켜세요."
else
  "$PY" - "$LOCAL_STATE" <<'PYEOF'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
mode = d.setdefault("hardware_acceleration_mode", {})
if mode.get("enabled") is False:
    mode["enabled"] = True
    json.dump(d, open(p, "w", encoding="utf-8"))
    print("   \033[32m✓\033[0m 꺼져 있던 그래픽 가속을 켰습니다")
else:
    print("   \033[90m- 그래픽 가속 켜짐 (이미 됨)\033[0m")
PYEOF
fi

# ---------------------------------------------------------------- 바탕화면 아이콘
say "바탕화면 아이콘 (AIvision 업데이트 · AIvision 터미널)"
bash "$PROJECT_DIR/deploy/install_update_icon.sh" || warn "아이콘을 만들지 못했습니다. 나중에 bash deploy/install_update_icon.sh"
bash "$PROJECT_DIR/deploy/install_terminal_icon.sh" || warn "터미널 아이콘을 만들지 못했습니다. 나중에 bash deploy/install_terminal_icon.sh"

# ---------------------------------------------------------------- 정리
say "캐시 정리 (복제할 SSD 용량을 줄인다)"
"$PY" -m pip cache purge >/dev/null 2>&1 || true
"$CONDA_BASE/bin/conda" clean -a -y >/dev/null 2>&1 || true
ok "pip, conda 캐시 ($(df -h / | awk 'NR==2 {print $3}') 사용 중)"

# ---------------------------------------------------------------- 9. 수동으로 할 일
say "9. 스크립트가 하지 않는 일"
[ -f "$PROJECT_DIR/yolo/weights/best.pt" ] && skip "yolo/weights/best.pt" \
  || warn "yolo/weights/best.pt 가 없습니다. 학습한 가중치를 USB 로 복사하세요 (git 에 없음)."
[ -e /dev/so101_follower ] && [ -e /dev/so101_leader ] && skip "팔 시리얼 이름" \
  || warn "팔 시리얼 이름 규칙: bash deploy/make_serial_rules.sh (팔을 하나씩 꽂으며 진행)"
[ -f "$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json" ] && skip "팔로워 캘리브" \
  || warn "팔 캘리브: bash deploy/calibrate_arms.sh"
[ "$need_relogin" -eq 1 ] && warn "그룹 권한이 바뀌었습니다. 로그아웃 후 다시 로그인(또는 재부팅)해야 팔·카메라가 열립니다."

# ---------------------------------------------------------------- 10. 점검
say "10. 점검"
"$PY" "$PROJECT_DIR/deploy/check_board.py" || true
printf '\n로그: %s\n' "$LOG"
