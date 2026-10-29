#!/usr/bin/env python3
"""AIvision 런처: 명령어 대신 버튼으로 프로그램을 켜고 끈다 (데모 · 수업 도구).

  python launcher/launcher.py          # 실제 팔
  python launcher/launcher.py --sim    # 가상 팔로 시험

버튼은 줄마다 [켜기] [끄기] [정지]. 지금은 「반복 데모」와 「원격 조종」이 동작하고 나머지는 준비 중이다.
둘 다 같은 팔을 쓰므로 한 번에 하나만 켠다.
각 프로그램 (launcher/repeat_demo.py · launcher/teleop.py) 을 따로 켜서 표준 입력으로 명령을 보내고,
표준 출력으로 상태를 받는다.
팔 모델 · lerobot 준비에 15초 가까이 걸려서, 런처가 뜰 때 launcher/standby.py 를 하나 미리 켜 둔다
(팔은 연결하지 않음). 「켜기」를 누르면 그것에 "start demo|teleop" 을 보내 그 프로그램이 되게 한다.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PENDANT = PROJECT / "pendant"  # 글꼴·테마 (ui_style) 를 같이 쓴다
for _p in (PROJECT, PENDANT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ui_style import apply_root_fonts, apply_ui_scale, apply_ui_theme, enable_xft_tk, ui_font  # noqa: E402

enable_xft_tk()
apply_ui_theme()

import customtkinter as ctk  # noqa: E402

POLL_MS = 100
WIN_W, WIN_H = 640, 740
SETTINGS = Path.home() / ".config" / "aivision" / "launcher.json"
SPEED_PRESETS = (25, 50, 75, 100)
LAMP = {"off": ("● 꺼짐", "#9ca3af"), "on": ("● 켜짐", "#4ade80"), "stopped": ("● 정지됨", "#fbbf24"),
        "busy": ("● 처리 중", "#60a5fa"), "estop": ("● E-stop", "#f87171"), "error": ("● 멈춤 (실패)", "#f87171")}
STATE_LAMP = {"off": "off", "running": "on", "stopped": "stopped", "estop": "estop", "error": "error"}
# E-stop 을 누를 수 있는 상태.
ESTOP_STATES = ("connecting", "homing", "aligning", "running", "stopping", "stopped", "error")


@dataclass(frozen=True)
class Job:
    """런처 줄 하나가 켜는 프로그램."""

    title: str
    script: str
    state_ko: dict
    enable: dict  # 상태 → 누를 수 있는 버튼


JOBS = {
    "demo": Job(
        title="반복 데모",
        script="repeat_demo.py",
        state_ko={"connecting": "팔 연결 중", "homing": "초기자세로 이동", "running": "반복 중",
                  "stopping": "이번 사이클 후 멈춤", "stopped": "정지중", "turning_off": "끄는 중", "off": "꺼짐",
                  "estop": "E-stop", "error": "멈춤 (실패)"},
        enable={"connecting": ("off", "stop"), "homing": ("off", "stop"), "running": ("off", "stop"),
                "stopping": ("off",), "stopped": ("start", "off"), "error": ("start", "off"), "estop": ("off",)},
    ),
    "teleop": Job(
        title="원격 조종",
        script="teleop.py",
        state_ko={"connecting": "팔 연결 중", "aligning": "리더 자세로 맞추는 중", "running": "따라가는 중",
                  "stopped": "정지중", "turning_off": "끄는 중", "off": "꺼짐", "estop": "E-stop",
                  "error": "멈춤 (실패)"},
        enable={"connecting": ("off",), "aligning": ("off", "stop"), "running": ("off", "stop"),
                "stopped": ("start", "off"), "error": ("start", "off"), "estop": ("off",)},
    ),
}


class JobProcess:
    """프로그램 하나를 켜고 명령을 보내고 상태 줄을 받는다."""

    def __init__(self, script: str, sim: bool, *, geometry: str | None = None, extra: list[str] | None = None) -> None:
        cmd = [sys.executable, str(HERE / script)]
        if geometry:
            cmd += ["--viz-geometry", geometry]
        if sim:
            cmd.append("--sim")
        cmd += extra or []
        self.events: queue.Queue[dict] = queue.Queue()
        self.stderr_tail: deque[str] = deque(maxlen=12)
        self.proc = subprocess.Popen(
            cmd, cwd=str(PROJECT), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()

    def _read_out(self) -> None:
        for line in self.proc.stdout:
            try:
                self.events.put(json.loads(line))
            except ValueError:
                pass
        self.events.put({"state": "exited", "code": self.proc.wait()})

    def _read_err(self) -> None:
        for line in self.proc.stderr:
            if " INFO " not in line:
                self.stderr_tail.append(line.rstrip())

    def send(self, cmd: str) -> None:
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            pass

    def alive(self) -> bool:
        return self.proc.poll() is None


class Popup:
    """가운데 뜨는 안내 창 하나. 글을 바꿀 수 있고 「확인」으로 닫는다."""

    def __init__(self, root) -> None:
        self.root = root
        self.win = None
        self.label = None

    def show(self, text: str, *, color: str = "#e5e7eb", title: str = "AIvision") -> None:
        if self.win is None or not self.win.winfo_exists():
            self.win = ctk.CTkToplevel(self.root)
            self.win.resizable(False, False)
            self.label = ctk.CTkLabel(self.win, text="", font=ui_font(17), wraplength=420, justify="center")
            self.label.pack(padx=28, pady=(26, 14))
            ctk.CTkButton(self.win, text="확인", width=120, command=self.close).pack(pady=(0, 20))
            self.win.transient(self.root)
            self.root.update_idletasks()
            x = self.root.winfo_rootx() + 60
            y = self.root.winfo_rooty() + 160
            self.win.geometry(f"+{x}+{y}")
        self.win.title(title)
        self.label.configure(text=text, text_color=color)
        self.win.lift()

    def close(self) -> None:
        if self.win is not None and self.win.winfo_exists():
            self.win.destroy()
        self.win = None


class Launcher:
    def __init__(self, root, *, sim: bool) -> None:
        self.root = root
        self.sim = sim
        # 같은 팔을 쓰므로 켜져 있는 프로그램은 늘 하나 (active).
        self.active: str | None = None
        self.proc: JobProcess | None = None
        # 미리 켜 둔 프로그램 (팔은 아직 연결하지 않음). 「켜기」를 누르면 active 로 넘어간다.
        self.standby: JobProcess | None = None
        self._standby_tries = 0
        self._closing = False
        self.state = "off"
        self.info_ev: dict = {}
        self.popup = Popup(root)
        self._popup_kind = ""  # stop · off · finish · error · estop
        root.title("AIvision 런처" + (" (가상)" if sim else ""))
        root.geometry(f"{WIN_W}x{WIN_H}+0+0")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        apply_root_fonts(root)

        top = ctk.CTkFrame(root, fg_color="transparent")
        top.pack(fill="x", padx=16, pady=(14, 6))
        ctk.CTkLabel(top, text="AIvision 런처", font=ui_font(22, "bold")).pack(side="left")
        self.estop_btn = ctk.CTkButton(
            top, text="E-stop  (즉시 멈춤)", fg_color="#b91c1c", hover_color="#991b1b", width=190, height=40,
            font=ui_font(16, "bold"), command=self.on_estop,
        )
        self.estop_btn.pack(side="right")

        self.rows: dict[str, dict] = {}
        self._section("데모")
        self._row("demo", "반복 데모", stop=True)
        self._row("teleop", "원격 조종", stop=True)
        self._row("vision_pick", "카메라로 집기", stop=True, dummy=True)
        self._section("도구")
        self._row("pendant", "펜던트", stop=True, dummy=True)
        self._row("camera", "카메라 창", stop=False, dummy=True)

        self.speed_pct = self._load_speed()
        self._speed_section()

        self.info = ctk.CTkLabel(root, text="", anchor="w", justify="left", font=ui_font(15), wraplength=WIN_W - 40)
        self.info.pack(fill="x", padx=20, pady=(14, 2))
        self.status = ctk.CTkLabel(root, text="", anchor="w", justify="left", text_color="#9ca3af", wraplength=WIN_W - 40)
        self.status.pack(fill="x", padx=20, pady=(0, 10))
        self._refresh()
        root.after(POLL_MS, self._poll)
        root.after(300, self._spawn_standby)

    # ------------------------------------------------------------ 속도

    def _speed_section(self) -> None:
        """펜던트처럼 프리셋 + 슬라이더. 관절 이동과 직선 이동을 같은 비율로 바꾼다 (반복 데모)."""
        self._section("속도")
        f = ctk.CTkFrame(self.root)
        f.pack(fill="x", padx=16, pady=3)
        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(8, 2))
        self.speed_btns = []
        for pct in SPEED_PRESETS:
            b = ctk.CTkButton(row, text=f"{pct}%", width=70, command=lambda p=pct: self._set_speed(p))
            b.pack(side="left", padx=4)
            self.speed_btns.append(b)
        self.speed_label = ctk.CTkLabel(row, text="", anchor="w")
        self.speed_label.pack(side="left", padx=(12, 0))
        self.speed_slider = ctk.CTkSlider(f, from_=10, to=100, number_of_steps=18, command=self._set_speed)
        self.speed_slider.pack(fill="x", padx=12, pady=(4, 10))
        self._set_speed(self.speed_pct, send=False)

    def _set_speed(self, pct: float, *, send: bool = True) -> None:
        pct = int(round(float(pct) / 5.0) * 5)
        changed = pct != self.speed_pct
        self.speed_pct = pct
        self.speed_slider.set(pct)
        try:
            from common import speed_values

            v = speed_values(pct)
            self.speed_label.configure(text=f"{pct}%   관절 {v['joint_deg_s']:.0f}°/s · 직선 {v['lin_mm_s']:.0f} mm/s")
        except Exception:
            self.speed_label.configure(text=f"{pct}%")
        if not (send and changed):
            return
        self._save_speed()
        # 켜져 있는 반복 데모에는 바로 보낸다 (다음 이동부터).
        if self.active == "demo" and self.proc is not None and self.proc.alive():
            self.proc.send(f"speed {pct}")

    def _load_speed(self) -> int:
        try:
            return int(json.loads(SETTINGS.read_text(encoding="utf-8")).get("speed_pct", 50))
        except (OSError, ValueError, TypeError):
            return 50

    def _save_speed(self) -> None:
        try:
            SETTINGS.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS.write_text(json.dumps({"speed_pct": self.speed_pct}), encoding="utf-8")
        except OSError:
            pass

    def _job_args(self, key: str) -> list[str]:
        return ["--speed", str(self.speed_pct)] if key == "demo" else []

    # ------------------------------------------------------------ 미리 켜 두기

    def _spawn_standby(self) -> None:
        # 팔을 쓰는 프로그램이 켜져 있는 동안은 두지 않는다 (메모리를 1GB 넘게 쓴다).
        if self._closing or self.standby is not None or self.active is not None:
            return
        if self._standby_tries >= 3:  # 계속 바로 꺼지면 그만둔다 (켜기를 누르면 그때 새로 켠다)
            return
        self._standby_tries += 1
        try:
            self.standby = JobProcess("standby.py", self.sim)
        except OSError:
            pass

    def _poll_standby(self) -> None:
        sp = self.standby
        if sp is None:
            return
        while True:
            try:
                ev = sp.events.get_nowait()
            except queue.Empty:
                return
            if ev.get("state") == "ready":
                self._standby_tries = 0
            elif ev.get("state") == "exited":
                self.standby = None
                self.root.after(3000, self._spawn_standby)
                return

    def _viz_geometry(self, key: str) -> str:
        """화면 창 자리. 원격 조종은 화면 전체 (그래프 · 3D 반씩, 런처는 그 뒤에 가려짐), 그 밖은 런처 오른쪽."""
        self.root.update_idletasks()
        if key == "teleop":  # 화면 전체를 넘긴다. 나누는 것은 teleop.py
            return f"0,0,{self.root.winfo_screenwidth()},{self.root.winfo_screenheight()}"
        x = self.root.winfo_rootx() + self.root.winfo_width() + 4
        return f"{x},0,{max(self.root.winfo_screenwidth() - x, 400)},{self.root.winfo_screenheight() - 80}"

    # ------------------------------------------------------------ 화면

    def _section(self, title: str) -> None:
        ctk.CTkLabel(self.root, text=title, anchor="w", font=ui_font(17, "bold")).pack(fill="x", padx=18, pady=(14, 4))

    def _row(self, key: str, label: str, *, stop: bool, dummy: bool = False) -> None:
        f = ctk.CTkFrame(self.root)
        f.pack(fill="x", padx=16, pady=3)
        ctk.CTkLabel(f, text=label, width=150, anchor="w", font=ui_font(16)).pack(side="left", padx=(12, 4), pady=8)
        lamp = ctk.CTkLabel(f, text="", width=120, anchor="w")
        lamp.pack(side="left")
        if dummy:
            handlers = (self._dummy,) * 3
        else:
            handlers = (lambda: self.on_start(key), lambda: self.on_off(key), lambda: self.on_stop(key))
        btns = {}
        for name, text, fn in zip(("start", "off", "stop"), ("켜기", "끄기", "정지"), handlers):
            if name == "stop" and not stop:
                continue
            b = ctk.CTkButton(f, text=text, width=78, command=fn)
            b.pack(side="left", padx=4)
            btns[name] = b
        self.rows[key] = {"lamp": lamp, "btns": btns, "dummy": dummy}

    def _dummy(self) -> None:
        self.status.configure(text="아직 준비 중인 기능입니다.")

    def _set_lamp(self, key: str, kind: str) -> None:
        text, color = LAMP[kind]
        self.rows[key]["lamp"].configure(text=text, text_color=color)

    def _refresh(self) -> None:
        """켜져 있는 프로그램 상태에 맞춰 불과 버튼을 바꾼다."""
        st = self.state
        for key, row in self.rows.items():
            if row["dummy"]:
                self._set_lamp(key, "off")
                continue
            if key == self.active:
                self._set_lamp(key, STATE_LAMP.get(st, "busy"))
                enable = JOBS[key].enable.get(st, ())
            else:
                self._set_lamp(key, "off")
                # 다른 프로그램이 팔을 쓰는 동안은 켤 수 없다.
                enable = ("start",) if self.active is None else ()
            for name, btn in row["btns"].items():
                btn.configure(state="normal" if name in enable else "disabled")
        self.estop_btn.configure(state="normal" if self.active is not None and st in ESTOP_STATES else "disabled")
        # 원격 조종은 리더를 따라가기만 해서 속도와 상관없다.
        speed_state = "disabled" if self.active == "teleop" else "normal"
        for b in self.speed_btns:
            b.configure(state=speed_state)
        self.speed_slider.configure(state=speed_state)
        self.info.configure(text=self._info_text())

    def _info_text(self) -> str:
        if self.active is None:
            return "꺼짐 — 켤 프로그램의 「켜기」를 누르세요."
        job, st, ev = JOBS[self.active], self.state, self.info_ev
        if self.active == "demo":
            state_ko = "이번 사이클 후 끔" if (st == "stopping" and self._popup_kind == "off") else job.state_ko.get(st, st)
            line = f"{job.title}: {state_ko}   ·   완료 {ev.get('cycle', 0)} / {ev.get('cycles', '?')} 회"
            if st in ("running", "stopping") and ev.get("from"):
                line += f"\n물체 {ev['from']} → {ev['to']}"
                if ev.get("step"):
                    line += f"   ·   단계: {ev['step']}"
            return line
        line = f"{job.title}: {job.state_ko.get(st, st)}"
        if st == "stopped":
            line += "\n팔이 그 자리에 멈춰 있습니다. 다시 따라가려면 「켜기」."
        elif ev.get("step"):
            line += f"\n{ev['step']}"
        return line

    def _popup(self, text: str, **kw) -> None:
        self.popup.show(text, title=JOBS[self.active].title if self.active else "AIvision", **kw)

    # ------------------------------------------------------------ 버튼

    def on_start(self, key: str) -> None:
        if self.proc is not None and self.proc.alive():
            if key == self.active and self.state in ("stopped", "error"):
                self.popup.close()
                self.proc.send("start")
            return
        geometry = self._viz_geometry(key)
        self.active, self.info_ev, self.state, self._popup_kind = key, {}, "connecting", ""
        self.status.configure(text="")
        sp, self.standby = self.standby, None
        if sp is not None and sp.alive():
            self.proc = sp
            sp.send(" ".join(["start", key, geometry, *self._job_args(key)]))
            self._refresh()
            return
        try:
            self.proc = JobProcess(JOBS[key].script, self.sim, geometry=geometry, extra=self._job_args(key))
        except OSError as exc:
            self._popup(f"{JOBS[key].title}을(를) 켜지 못했습니다.\n{exc}", color="#f87171")
            self.active, self.proc, self.state = None, None, "off"
        self._refresh()

    def on_stop(self, key: str) -> None:
        if key != self.active or self.proc is None or not self.proc.alive():
            return
        self.proc.send("stop")
        if key == "demo":
            self._popup_kind = "stop"
            self._popup("이번 사이클 수행 후 정지하겠습니다.")

    def on_off(self, key: str) -> None:
        if key != self.active or self.proc is None or not self.proc.alive():
            return
        self.proc.send("off")
        self._popup_kind = "off"
        if key == "demo" and self.state not in ("stopped", "error", "estop"):
            self._popup("이번 사이클 수행 후 끄겠습니다.")
        else:
            self._popup("앉은 자세로 간 뒤 끕니다.")

    def on_estop(self) -> None:
        if self.proc is not None and self.proc.alive():
            self.proc.send("estop")

    # ------------------------------------------------------------ 상태 받기

    def _poll(self) -> None:
        self._poll_standby()
        proc = self.proc
        if proc is not None:
            while True:
                try:
                    ev = proc.events.get_nowait()
                except queue.Empty:
                    break
                self._on_event(ev)
        self.root.after(POLL_MS, self._poll)

    def _on_event(self, ev: dict) -> None:
        st = ev.get("state")
        if st == "ready":  # 미리 켜 둔 것이 준비를 마쳤다는 알림. 켜기를 이미 눌렀으면 곧 연결한다.
            return
        if st == "exited":
            code = ev.get("code")
            # 실패 안내를 이미 띄웠으면 (연결 실패 등) 덮어쓰지 않는다.
            if self.state != "off" and code not in (0, None) and self._popup_kind != "error":
                tail = "\n".join(list(self.proc.stderr_tail)[-4:]) if self.proc else ""
                self._popup(f"{JOBS[self.active].title}이(가) 오류로 꺼졌습니다.\n{tail}", color="#f87171")
            self.active, self.proc, self.state = None, None, "off"
            self._refresh()
            self.root.after(1000, self._spawn_standby)  # 다음 「켜기」를 위해 다시 준비
            return
        self.info_ev = ev
        prev, self.state = self.state, st
        msg = ev.get("msg") or ""
        if st == "stopped" and self._popup_kind == "stop":
            self._popup("정지중\n다시 시작하려면 「켜기」, 끄려면 「끄기」를 누르세요.", color="#fbbf24")
        elif st == "turning_off" and ev.get("finished"):
            self._popup_kind = "finish"
            self._popup(f"{ev.get('cycles')}회 완료 — 앉은 자세로 간 뒤 끕니다.")
        elif st == "off" and prev != "off":
            if self._popup_kind in ("off", "finish"):
                self._popup("꺼졌습니다." if self._popup_kind == "off" else f"{ev.get('cycles')}회 완료 — 꺼졌습니다.")
            self._popup_kind = ""
        elif st == "estop" and prev != "estop":
            self._popup_kind = "estop"
            self._popup(msg, color="#f87171")
        elif st == "error" and prev != "error":
            self._popup_kind = "error"
            self._popup(msg or "실패", color="#f87171")
        self._refresh()

    def on_close(self) -> None:
        """런처를 닫으면 켜져 있는 프로그램도 끈다 (앉은 자세로 간 뒤)."""
        self._closing = True
        if self.standby is not None:
            self.standby.send("off")  # 팔에 손대지 않은 채 바로 끝난다
            self.standby = None
        if self.proc is not None and self.proc.alive():
            if self.state != "turning_off":
                self.on_off(self.active)
            self.status.configure(text=f"{JOBS[self.active].title}을(를) 끄는 중입니다. 꺼지면 런처도 닫힙니다.")
            self._close_when_done()
            return
        self.root.destroy()

    def _close_when_done(self) -> None:
        if self.proc is None or not self.proc.alive():
            self.root.destroy()
            return
        self.root.after(300, self._close_when_done)


def main() -> None:
    ap = argparse.ArgumentParser(description="AIvision 런처")
    ap.add_argument("--sim", action="store_true", help="실제 팔 없이 가상 팔로")
    ap.add_argument("--scale", type=float, default=None)
    args = ap.parse_args()
    try:
        # 반복 데모 설정 파일을 미리 만들어 둔다. 데모를 켜기 전에도 고칠 수 있게.
        from demo_settings import ensure_settings_file

        ensure_settings_file()
    except Exception as exc:  # 설정 파일 때문에 런처가 안 뜨면 안 된다
        print(f"반복 데모 설정 파일을 못 만듦: {exc}", file=sys.stderr)
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    apply_ui_scale(root, args.scale)
    Launcher(root, sim=args.sim)
    root.mainloop()


if __name__ == "__main__":
    main()
