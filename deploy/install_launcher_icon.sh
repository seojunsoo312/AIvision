#!/usr/bin/env bash
# 바탕화면에 「AIvision 런처」 아이콘을 만든다. 더블클릭하면 launcher/launcher.py 가 버튼 창으로 뜬다
# (명령어 대신 버튼으로 반복 데모 등을 켜고 끈다). 여러 번 실행해도 같은 아이콘 하나.
#
#   bash ~/AIvision/deploy/install_launcher_icon.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
[ -d "$DESKTOP" ] || DESKTOP="$HOME/Desktop"

PY=""
for d in "$HOME/miniforge3" "$HOME/miniconda3"; do
  [ -x "$d/envs/AIvision/bin/python" ] && PY="$d/envs/AIvision/bin/python" && break
done
[ -n "$PY" ] || { echo "   ! AIvision 환경을 찾지 못했습니다. 먼저 bash deploy/setup_jetson.sh"; exit 1; }

entry="[Desktop Entry]
Type=Application
Name=AIvision 런처
Comment=버튼으로 반복 데모 등을 켜고 끕니다
Exec=\"$PY\" \"$PROJECT_DIR/launcher/launcher.py\"
Path=$PROJECT_DIR
Icon=applications-science
Terminal=false
Categories=Utility;
"
mkdir -p "$DESKTOP" "$HOME/.local/share/applications"
for target in "$HOME/.local/share/applications/aivision-launcher.desktop" "$DESKTOP/aivision-launcher.desktop"; do
  printf '%s' "$entry" > "$target"
  chmod +x "$target"
done
gio set "$DESKTOP/aivision-launcher.desktop" metadata::trusted true 2>/dev/null || true
echo "   ✓ 바탕화면 아이콘: $DESKTOP/aivision-launcher.desktop"
