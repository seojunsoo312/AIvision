#!/usr/bin/env bash
# 이 보드를 GitHub 최신 버전으로 맞춘다. git 을 몰라도 되게, 바탕화면 「AIvision 업데이트」
# 아이콘이 이 스크립트를 실행한다 (아이콘 설치: bash deploy/install_update_icon.sh).
#
#   bash ~/AIvision/deploy/update_board.sh
#
# 하는 일
#   1. 펜던트 · 카메라 창 · 노트북(Jupyter)이 켜져 있으면 멈춘다 (켜진 채 바꾸면 꼬인다).
#   2. 학생이 고친 파일을 바탕화면 AIvision_백업/날짜_시각/ 에 복사해 둔다.
#   3. 저장소를 GitHub 과 똑같이 맞춘다 (git reset --hard). 새로 만든 파일, 캘리브,
#      가중치, 로그처럼 git 이 모르는 파일은 건드리지 않는다.
#   3-1. 바탕화면 「실행 코드.md」 · 「노트북 실습.md」를 docs/ 의 것으로 맞춘다 (고친 것은 백업 폴더에 보관).
#   4. 설치 목록(docs/requirements.txt, setup_jetson.sh)이 바뀌었으면 setup_jetson.sh 를 다시 돌린다.
#   5. check_board.py 로 점검하고, 버전과 결과를 한 줄로 보여 준다.
#
# 인터넷이 필요하다. 시험할 때는 AIVISION_BACKUP_DIR 로 백업 위치를 바꿀 수 있다.
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="main"
# 저장소 주소. 예전 이름(SO-ARM_xyzrpy_control)으로 받은 보드는 이 주소로 바꿔 둔다.
REPO_URL="https://github.com/seojunsoo312/AIvision.git"
OLD_REPO_NAME="SO-ARM_xyzrpy_control"
STAMP="$(date +%Y-%m-%d_%H%M)"
DESKTOP="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
[ -d "$DESKTOP" ] || DESKTOP="$HOME/Desktop"
BACKUP_ROOT="${AIVISION_BACKUP_DIR:-$DESKTOP/AIvision_백업}"
BACKUP_DIR="$BACKUP_ROOT/$STAMP"
RUNLOG_DIR="${AIVISION_LOG_DIR:-$HOME/aivision_logs}"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m! %s\033[0m\n' "$*"; }
finish() {  # $1 = 0 성공 / 그 외 실패. 아이콘으로 열면 창이 바로 닫히지 않게 기다린다.
  if [ -t 0 ] && [ -z "${AIVISION_NO_WAIT:-}" ]; then
    printf '\nEnter 를 누르면 창이 닫힙니다.'
    read -r _ || true
  fi
  exit "$1"
}
die()  { printf '\n\033[1;31m✗ %s\033[0m\n' "$*"; runlog "WARNING" "업데이트 실패: $*"; finish 1; }
runlog() {  # 실행 기록 (motion/runlog.py 와 같은 파일 · 같은 모양)
  mkdir -p "$RUNLOG_DIR" 2>/dev/null || return 0
  printf '%s [update] %-7s aivision.update: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$2" \
    >> "$RUNLOG_DIR/$(date +%Y-%m-%d).log" 2>/dev/null || true
}

cd "$PROJECT_DIR" || die "프로젝트 폴더가 없습니다: $PROJECT_DIR"
git rev-parse --git-dir >/dev/null 2>&1 || die "$PROJECT_DIR 는 git 저장소가 아닙니다."

# 펜던트가 켜져 있으면 0. 「AIvision 터미널」 pendant 칸에서는 `python main.py` 로 켜서
# 명령줄에 pendant/ 가 없으므로, 그 프로세스의 작업 폴더도 본다.
pendant_running() {
  local pid
  pgrep -f "pendant/main.py" >/dev/null && return 0
  for pid in $(pgrep -f "main.py"); do
    [ "$(readlink "/proc/$pid/cwd" 2>/dev/null)" = "$PROJECT_DIR/pendant" ] && return 0
  done
  return 1
}

# ---------------------------------------------------------------- 1. 켜진 프로그램
say "1. 켜진 프로그램 확인"
busy=""
pendant_running && busy="$busy 펜던트"
pgrep -f "roi_cloud.py" >/dev/null && busy="$busy 카메라창"
pgrep -f "jupyter" >/dev/null && busy="$busy 노트북(Jupyter)"
[ -z "$busy" ] || die "먼저 닫아 주세요:$busy  (켜진 채 바꾸면 저장할 때 옛 내용으로 덮어쓸 수 있습니다)"
ok "닫혀 있음"

# ---------------------------------------------------------------- 2. 최신 버전 받기
say "2. GitHub 에서 최신 버전 확인"
old_rev="$(git rev-parse HEAD)"
if git remote get-url origin 2>/dev/null | grep -q "$OLD_REPO_NAME"; then
  git remote set-url origin "$REPO_URL" && ok "저장소 주소를 새 이름으로 바꿨습니다: $REPO_URL"
fi
git fetch --quiet origin "$BRANCH" || die "GitHub 에 연결하지 못했습니다. 인터넷 연결을 확인하세요."
new_rev="$(git rev-parse "origin/$BRANCH")"
ok "지금 $(git rev-parse --short "$old_rev")  →  최신 $(git rev-parse --short "$new_rev")"

# ---------------------------------------------------------------- 3. 학생이 고친 파일 백업
say "3. 고친 파일 보관"
backed=0
backup_one() {  # $1 = 저장소 안 경로
  [ -f "$1" ] || return 0
  mkdir -p "$BACKUP_DIR/$(dirname "$1")" && cp -p "$1" "$BACKUP_DIR/$1" && backed=$((backed + 1))
  printf '     %s\n' "$1"
}
# 파일 이름은 -z (NUL 구분) 로 받는다. 그냥 받으면 git 이 한글 이름을 "\353\241..." 로 바꿔
# 내보내서 「로봇 제어 실습_학생.ipynb」를 못 찾고 백업 없이 지나간다.
# (a) git 에 있는 파일 중 고친 것
while IFS= read -r -d '' f; do
  [ -n "$f" ] && backup_one "$f"
done < <(git diff --name-only -z HEAD 2>/dev/null)
# (b) 새 버전이 같은 이름으로 추가하는 파일과 겹치는, git 이 모르던 파일 (덮어써지기 전에)
while IFS= read -r -d '' f; do
  [ -n "$f" ] && [ -f "$f" ] && ! git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 && backup_one "$f"
done < <(git diff --name-only -z --diff-filter=A "$old_rev" "$new_rev" 2>/dev/null)
if [ "$backed" -gt 0 ]; then
  ok "$backed 개를 보관했습니다: $BACKUP_DIR"
else
  ok "고친 파일 없음"
fi

# ---------------------------------------------------------------- 4. 맞추기
say "4. 최신 버전으로 맞추기"
if [ "$old_rev" = "$new_rev" ] && [ "$backed" -eq 0 ]; then
  ok "이미 최신입니다"
else
  git reset --quiet --hard "origin/$BRANCH" || die "맞추기에 실패했습니다 (git reset)."
  ok "$(git log -1 --format='%h  %ad  %s' --date=format:'%Y-%m-%d %H:%M' | cut -c1-90)"
fi

# 바탕화면 안내 파일을 저장소의 docs/ 에 있는 것으로 맞춘다. 학생이 바탕화면 파일을
# 고쳐 두었으면 덮어쓰기 전에 백업 폴더에 보관한다.
for DOC_NAME in "실행 코드.md" "노트북 실습.md"; do
  DOC_SRC="$PROJECT_DIR/docs/$DOC_NAME"
  DOC_DST="$DESKTOP/$DOC_NAME"
  [ -f "$DOC_SRC" ] || continue
  if [ -f "$DOC_DST" ] && ! cmp -s "$DOC_SRC" "$DOC_DST"; then
    mkdir -p "$BACKUP_DIR/바탕화면" && cp -p "$DOC_DST" "$BACKUP_DIR/바탕화면/" && backed=$((backed + 1))
    printf '     바탕화면/%s (예전 것 보관)\n' "$DOC_NAME"
  fi
  cp "$DOC_SRC" "$DOC_DST" && ok "바탕화면 「$DOC_NAME」 최신으로"
done

# ---------------------------------------------------------------- 5. 설치 다시 할지
say "5. 설치 목록 확인"
if [ "$old_rev" != "$new_rev" ] && ! git diff --quiet "$old_rev" "$new_rev" -- docs/requirements.txt deploy/setup_jetson.sh; then
  warn "설치 목록이 바뀌었습니다. 설치 스크립트를 다시 실행합니다 (sudo 비밀번호를 물을 수 있습니다)."
  AIVISION_NO_WAIT=1 bash "$PROJECT_DIR/deploy/setup_jetson.sh" || die "설치 스크립트가 실패했습니다. 화면 위쪽 오류와 ~/aivision_setup_*.log 를 확인하세요."
else
  ok "바뀐 것 없음 (다시 설치하지 않음)"
fi

# ---------------------------------------------------------------- 6. 점검
say "6. 점검"
PY=""
for d in "$HOME/miniforge3" "$HOME/miniconda3"; do
  [ -x "$d/envs/AIvision/bin/python" ] && PY="$d/envs/AIvision/bin/python" && break
done
summary="점검 건너뜀 (AIvision 환경 없음)"
if [ -n "$PY" ]; then
  check_out="$("$PY" "$PROJECT_DIR/deploy/check_board.py" 2>&1)"
  printf '%s\n' "$check_out" | grep -E "실패|경고" | sed 's/^/   /' || true
  summary="$(printf '%s\n' "$check_out" | grep -E '^결과:' | tail -1 | sed 's/^결과: //')"
  [ -n "$summary" ] || summary="점검 결과를 읽지 못함"
fi

# ---------------------------------------------------------------- 끝
ver="$(git rev-parse --short HEAD)"
runlog "INFO" "업데이트 $(git rev-parse --short "$old_rev") → $ver  보관 ${backed}개  점검 $summary"
printf '\n\033[1;32m✅ 업데이트 완료 — 버전 %s · %s\033[0m\n' "$ver" "$summary"
if [ "$backed" -gt 0 ]; then
  printf '\033[1m📁 고친 파일 %d개를 %s 에 보관했습니다.\033[0m\n' "$backed" "$BACKUP_DIR"
  printf '   실습 노트북은 처음 상태로 돌아갔습니다. 내가 고친 것은 위 폴더에서 열면 됩니다.\n'
fi
finish 0
