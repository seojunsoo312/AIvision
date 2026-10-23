#!/usr/bin/env bash
# 바탕화면에 「AIvision 업데이트」 아이콘을 만든다. 더블클릭하면 터미널에서 update_board.sh 가 돈다.
# setup_jetson.sh 가 부른다. 따로 실행해도 된다 (여러 번 실행해도 같은 아이콘 하나).
#
#   bash ~/AIvision/deploy/install_update_icon.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
[ -d "$DESKTOP" ] || DESKTOP="$HOME/Desktop"
mkdir -p "$DESKTOP" "$HOME/.local/share/applications"

entry="[Desktop Entry]
Type=Application
Name=AIvision 업데이트
Comment=AIvision 을 GitHub 최신 버전으로 맞춥니다 (고친 파일은 바탕화면 AIvision_백업 에 보관)
Exec=bash \"$PROJECT_DIR/deploy/update_board.sh\"
Icon=system-software-update
Terminal=true
Categories=Utility;
"
for target in "$HOME/.local/share/applications/aivision-update.desktop" "$DESKTOP/aivision-update.desktop"; do
  printf '%s' "$entry" > "$target"
  chmod +x "$target"
done
# Ubuntu 바탕화면은 처음 보는 실행 아이콘을 막는다. 믿을 수 있는 아이콘으로 표시해 둔다.
gio set "$DESKTOP/aivision-update.desktop" metadata::trusted true 2>/dev/null || true
echo "   ✓ 바탕화면 아이콘: $DESKTOP/aivision-update.desktop"
