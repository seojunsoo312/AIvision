#!/usr/bin/env bash
# 바탕화면에 「AIvision 터미널」 아이콘을 만든다. 더블클릭하면 terminator 가 4칸으로 열리고,
# 칸마다 AIvision 환경이 켜진 채 폴더가 잡혀 있다.
#
#   ┌─────────────┬─────────────┐
#   │ pendant     │ vision      │
#   ├─────────────┼─────────────┤
#   │ yolo        │ AIvision    │
#   └─────────────┴─────────────┘
#
# setup_jetson.sh 가 부른다. 따로 실행해도 된다 (여러 번 실행해도 같은 아이콘 하나).
# terminator 설정(~/.config/terminator/config)에는 aivision 레이아웃만 넣고 나머지는 그대로 둔다.
#
#   bash ~/AIvision/deploy/install_terminal_icon.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
[ -d "$DESKTOP" ] || DESKTOP="$HOME/Desktop"

if ! command -v terminator >/dev/null 2>&1; then
  echo "   ! terminator 가 없습니다: sudo apt install -y terminator 후 다시 실행하세요"
  exit 1
fi

# terminator 는 시스템 python3 의 configobj 로 설정을 읽는다. 같은 것으로 고쳐야 모양이 안 깨진다.
/usr/bin/python3 - "$HOME/.config/terminator/config" "$PROJECT_DIR" <<'PYEOF'
import sys
from pathlib import Path
from configobj import ConfigObj

path, project = Path(sys.argv[1]), sys.argv[2]
path.parent.mkdir(parents=True, exist_ok=True)
cfg = ConfigObj(str(path), encoding="utf-8")
shell = f'bash --init-file "{project}/deploy/aivision_shell.sh"'

def term(parent, order, folder):
    return {"type": "Terminal", "parent": parent, "order": order, "profile": "default",
            "command": shell, "directory": f"{project}/{folder}".rstrip("/")}

cfg.setdefault("layouts", {})
cfg["layouts"]["aivision"] = {
    "window0": {"type": "Window", "parent": "", "order": 0},
    "rows":    {"type": "VPaned", "parent": "window0", "order": 0, "ratio": 0.5},
    "top":     {"type": "HPaned", "parent": "rows", "order": 0, "ratio": 0.5},
    "bottom":  {"type": "HPaned", "parent": "rows", "order": 1, "ratio": 0.5},
    "pendant": term("top", 0, "pendant"),
    "vision":  term("top", 1, "vision"),
    "yolo":    term("bottom", 0, "yolo"),
    "root":    term("bottom", 1, ""),
}
cfg.write()
PYEOF

# -u: 이미 켜진 terminator 에 창을 맡기지 않고 새로 띄운다. 맡기면 그 terminator 가 켜질 때
#     읽어 둔 옛 설정으로 열어서 4칸 레이아웃이 무시된다.
entry="[Desktop Entry]
Type=Application
Name=AIvision 터미널
Comment=터미널 4칸 (pendant · vision · yolo · AIvision), AIvision 환경이 켜진 채로 열립니다
Exec=terminator -u --maximise -l aivision
Icon=utilities-terminal
Terminal=false
Categories=Utility;
"
mkdir -p "$DESKTOP" "$HOME/.local/share/applications"
for target in "$HOME/.local/share/applications/aivision-terminal.desktop" "$DESKTOP/aivision-terminal.desktop"; do
  printf '%s' "$entry" > "$target"
  chmod +x "$target"
done
gio set "$DESKTOP/aivision-terminal.desktop" metadata::trusted true 2>/dev/null || true
echo "   ✓ 바탕화면 아이콘: $DESKTOP/aivision-terminal.desktop"
