#!/usr/bin/env bash
# 팔 USB 보드 번호로 /dev/so101_follower, /dev/so101_leader 이름을 붙인다.
# 팔마다 번호가 다르므로 보드마다 새로 만든다. 다른 보드의 파일을 복사하지 않는다.
#
#   bash ~/AIvision/deploy/make_serial_rules.sh
#
# 안내에 따라 팔을 하나씩 꽂는다. 리더 없이 팔로워만 쓰면 리더 단계에서 s 를 누른다.
set -euo pipefail

RULES=/etc/udev/rules.d/99-serial.rules

acm_list() { ls /dev/ttyACM* 2>/dev/null || true; }
serial_of() { udevadm info -q property -n "$1" | sed -n 's/^ID_SERIAL_SHORT=//p'; }
wait_new_port() {  # $1 = 이미 있던 포트 목록. 새로 생긴 포트 하나를 돌려준다.
  for _ in $(seq 1 20); do
    for p in $(acm_list); do
      grep -qx "$p" <<<"$1" || { echo "$p"; return 0; }
    done
    sleep 0.5
  done
  return 1
}

echo "== 팔 시리얼 이름 규칙 만들기"
read -rp "1) 팔 USB 두 개를 모두 뽑고 Enter: " _
if [ -n "$(acm_list)" ]; then
  echo "아직 꽂힌 시리얼 장치가 있습니다: $(acm_list | tr '\n' ' ')"
  echo "팔 말고 다른 장치라면 그대로 진행해도 됩니다."
fi
before="$(acm_list)"

read -rp "2) 팔로워(모터 달린 팔)만 꽂고 Enter: " _
port="$(wait_new_port "$before")" || { echo "새 장치가 보이지 않습니다. USB 를 다시 꽂고 처음부터 하세요."; exit 1; }
follower="$(serial_of "$port")"
[ -n "$follower" ] || { echo "$port 의 보드 번호를 읽지 못했습니다."; exit 1; }
echo "   팔로워 $port  번호 $follower"
before="$(acm_list)"

leader=""
read -rp "3) 리더(손으로 잡는 팔)도 꽂고 Enter (리더가 없으면 s): " ans
if [ "$ans" != "s" ]; then
  port="$(wait_new_port "$before")" || { echo "새 장치가 보이지 않습니다. USB 를 다시 꽂고 처음부터 하세요."; exit 1; }
  leader="$(serial_of "$port")"
  [ -n "$leader" ] || { echo "$port 의 보드 번호를 읽지 못했습니다."; exit 1; }
  [ "$leader" != "$follower" ] || { echo "두 팔의 번호가 같습니다($leader). 다른 팔을 꽂았는지 확인하세요."; exit 1; }
  echo "   리더   $port  번호 $leader"
fi

tmp="$(mktemp)"
{
  echo "SUBSYSTEM==\"tty\", ATTRS{serial}==\"$follower\", SYMLINK+=\"so101_follower\""
  [ -z "$leader" ] || echo "SUBSYSTEM==\"tty\", ATTRS{serial}==\"$leader\", SYMLINK+=\"so101_leader\""
} > "$tmp"
echo
echo "== $RULES 에 씁니다:"
cat "$tmp"
if [ -f "$RULES" ]; then
  sudo cp "$RULES" "$RULES.bak.$(date +%Y%m%d_%H%M%S)"
  echo "   (기존 파일은 .bak 으로 백업)"
fi
sudo install -m 644 "$tmp" "$RULES"
rm -f "$tmp"
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
sleep 1
echo
ls -l /dev/so101_* 2>/dev/null || echo "아직 이름이 안 보이면 팔 USB 를 뽑았다 다시 꽂으세요."
