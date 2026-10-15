# -*- coding: utf-8 -*-
"""
app.py -- Drift Fix desktop app.

Everything in one window:
  * live view of both sticks: raw reading (orange) vs corrected output
    (green), so you can SEE the drift and the fix
  * one-click guided calibration per stick, with on-screen prompts and
    target guides, followed by automatic training of the from-scratch MLP
  * a report comparing the model against the plain offset fix
  * per-stick mode: learned model / offset fix / off
  * Start/Stop the virtual Xbox 360 controller your game reads
  * remembers your controller, modes and auto-start choice

Usage:
    python app.py              # real controller
    python app.py --simulate   # demo with a simulated drifting pad
                                   # (drag inside a stick circle to move it)
"""
import argparse
import math
import os
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import reader  # noqa: E402  (sets SDL hints before pygame starts)
import pygame  # noqa: E402

import config  # noqa: E402
import settings as settings_file  # noqa: E402
from calibration import CalibrationSession, angle_to_xy, full_script, rest_script  # noqa: E402
from corrector import StickCorrector  # noqa: E402
from reader import STICKS, PadState  # noqa: E402
from training import (TrainingError, clear_data, data_summary, load_sessions,  # noqa: E402
                      save_result, save_session, train_stick)
from virtual_pad import open_virtual_pad  # noqa: E402

W, H = 1040, 680
HEADER_H = 72
PANEL_Y, PANEL_H = 84, 410
BOTTOM_Y = PANEL_Y + PANEL_H + 12

BG = (18, 20, 26)
PANEL = (28, 31, 40)
PANEL_HI = (40, 44, 58)
TEXT = (232, 234, 240)
MUTED = (140, 146, 162)
GRID = (55, 60, 76)
RAW = (255, 145, 60)
OUT = (70, 220, 140)
ACCENT = (90, 150, 255)
DANGER = (235, 85, 85)
WARN = (240, 200, 80)

MODE_NAMES = {"model": "Learned model", "offset": "Offset fix", "off": "Off (raw)"}
MODE_COLORS = {"model": OUT, "offset": ACCENT, "off": MUTED}
OWN_PAD_GRACE_S = 5.0  # devices appearing this soon after we create the virtual pad are ours


# ---------------------------------------------------------------------------
# small drawing helpers
# ---------------------------------------------------------------------------
class Fonts:
    def __init__(self):
        self._cache = {}
        self._text_cache = {}

    def get(self, size, bold=False):
        key = (size, bold)
        if key not in self._cache:
            self._cache[key] = pygame.font.SysFont("segoeui,arial", size, bold=bold)
        return self._cache[key]

    def render(self, text, size, color, bold=False):
        key = (text, size, color, bold)
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 600:
                self._text_cache.clear()
            surf = self.get(size, bold).render(text, True, color)
            self._text_cache[key] = surf
        return surf


class Button:
    def __init__(self, rect, label, on_click, enabled=lambda: True, style="normal"):
        self.rect = pygame.Rect(rect)
        self.label = label          # str or callable -> str
        self.on_click = on_click
        self.enabled = enabled
        self.style = style          # normal / primary / danger, or callable

    def text(self):
        return self.label() if callable(self.label) else self.label

    def draw(self, surf, fonts, mouse):
        enabled = self.enabled()
        style = self.style() if callable(self.style) else self.style
        base = {"primary": ACCENT, "danger": DANGER, "go": (46, 160, 100)}.get(style, PANEL_HI)
        if not enabled:
            color = (34, 37, 46)
        elif self.rect.collidepoint(mouse):
            color = tuple(min(255, c + 22) for c in base)
        else:
            color = base
        pygame.draw.rect(surf, color, self.rect, border_radius=8)
        label = fonts.render(self.text(), 15, TEXT if enabled else (90, 95, 108), bold=True)
        surf.blit(label, label.get_rect(center=self.rect.center))

    def click(self, pos):
        if self.rect.collidepoint(pos) and self.enabled():
            self.on_click()
            return True
        return False


# ---------------------------------------------------------------------------
# the app
# ---------------------------------------------------------------------------
class App:
    def __init__(self, simulate=False):
        reader.init()
        sys.setswitchinterval(0.001)  # keep the live loop smooth while training runs
        pygame.display.set_caption("Drift Fix - controller drift correction")
        self.screen = pygame.display.set_mode((W, H))
        self.fonts = Fonts()
        self.clock = pygame.time.Clock()
        self.simulate = simulate
        self.settings = settings_file.load()

        self.reader = None
        self.own_ids = set()          # instance ids of our own virtual pad(s)
        self.vpad = None
        self.vpad_created_at = 0.0
        self.known_ids = set()

        self.correctors = {s: StickCorrector(s) for s in STICKS}
        self.summaries = {}
        for s in STICKS:
            self._load_model(s, initial=True)

        self.state = PadState()
        self.out = {s: (0.0, 0.0) for s in STICKS}
        self.trails = {s: deque(maxlen=45) for s in STICKS}

        self.calib = None             # (stick, CalibrationSession)
        self.pilot = None
        self.training = None          # dict(stick, progress) while training
        self.train_result = None      # set by the worker thread
        self.dialog = None            # (title, lines)
        self.confirm_reset = {}       # stick -> deadline
        self.messages = deque(maxlen=6)
        self.dragging = None

        self._build_buttons()
        if simulate:
            from simulator import SimulatedReader
            self.reader = SimulatedReader()
            self.log("Simulation mode: drag inside a stick circle to move it.")
        else:
            self._open_preferred_device()
        if self.settings.get("autostart"):
            self.start_output()

    # ----- logging / settings ---------------------------------------------
    def log(self, text):
        self.messages.append((time.strftime("%H:%M:%S"), text))
        print(text)

    def persist(self):
        self.settings["modes"] = {s: c.mode for s, c in self.correctors.items()}
        if self.reader is not None and not self.simulate:
            self.settings["device_guid"] = self.reader.guid
        try:
            settings_file.save(self.settings)
        except OSError as e:
            self.log(f"Could not save settings: {e}")

    # ----- models -----------------------------------------------------------
    def _load_model(self, stick, initial=False):
        corr = self.correctors[stick]
        try:
            ok = corr.load()
        except (OSError, ValueError, KeyError) as e:
            ok = False
            self.log(f"Could not load {stick} model: {e}")
        if not ok:
            corr.unload()
        elif initial:
            saved = self.settings.get("modes", {}).get(stick)
            if saved:
                corr.set_mode(saved)
        self.summaries[stick] = data_summary(stick)

    # ----- devices ----------------------------------------------------------
    def _devices(self):
        return reader.list_devices(exclude_instance_ids=self.own_ids)

    def _open_device(self, info):
        if self.reader is not None:
            self.reader.close()
        try:
            self.reader = reader.ControllerReader(info.index)
        except (RuntimeError, pygame.error) as e:
            self.reader = None
            self.log(f"Could not open controller: {e}")
            return
        for c in self.correctors.values():
            c.reset_history()
        how = "auto-mapped" if self.reader.mapped else "raw mapping from config.py"
        self.log(f"Using controller: {self.reader.name} ({how})")
        self.persist()

    def _open_preferred_device(self):
        devices = self._devices()
        self.known_ids = {d.instance_id for d in reader.list_devices()}
        if not devices:
            self.log("No controller found -- plug one in.")
            return
        want = self.settings.get("device_guid")
        pick = next((d for d in devices if d.guid == want), devices[0])
        self._open_device(pick)

    def switch_device(self):
        devices = self._devices()
        if not devices:
            self.log("No controllers connected.")
            return
        cur = self.reader.instance_id if self.reader else None
        ids = [d.instance_id for d in devices]
        nxt = devices[(ids.index(cur) + 1) % len(devices)] if cur in ids else devices[0]
        self._open_device(nxt)

    def _on_device_added(self, device_index):
        try:
            iid = pygame.joystick.Joystick(device_index).get_instance_id()
        except pygame.error:
            return
        if iid in self.known_ids:
            return
        self.known_ids.add(iid)
        if self.vpad is not None and time.time() - self.vpad_created_at < OWN_PAD_GRACE_S:
            self.own_ids.add(iid)  # that's our virtual pad showing up -- never read it
            return
        if self.reader is None and not self.simulate:
            info = next((d for d in self._devices() if d.instance_id == iid), None)
            if info:
                self._open_device(info)

    def _on_device_removed(self, instance_id):
        self.known_ids.discard(instance_id)
        if self.reader is not None and not self.simulate and instance_id == self.reader.instance_id:
            self.reader.close()
            self.reader = None
            self.log("Controller disconnected.")
            if self.calib:
                self.cancel_calibration()

    # ----- virtual output ---------------------------------------------------
    def start_output(self):
        if self.vpad is not None:
            return
        self.known_ids |= {d.instance_id for d in reader.list_devices()}
        pad, err = open_virtual_pad()
        if err:
            self.log(err)
            self.dialog = ("Virtual controller unavailable", [
                err, "",
                "Install the ViGEmBus driver:",
                "https://github.com/ViGEm/ViGEmBus/releases",
                "then restart this app.",
                "", "You can still calibrate and preview corrections."])
            return
        self.vpad = pad
        self.vpad_created_at = time.time()
        self.log("Correction ON -- games should use the virtual Xbox 360 controller.")

    def stop_output(self):
        if self.vpad is None:
            return
        self.vpad.close()
        self.vpad = None
        self.log("Correction OFF -- virtual controller removed.")

    def toggle_output(self):
        self.stop_output() if self.vpad else self.start_output()

    def toggle_autostart(self):
        self.settings["autostart"] = not self.settings.get("autostart", False)
        self.persist()

    # ----- calibration ------------------------------------------------------
    def busy(self):
        return self.calib is not None or self.training is not None

    def start_calibration(self, stick, quick=False):
        if self.reader is None:
            self.log("Connect a controller first.")
            return
        steps = rest_script() if quick else full_script()
        self.calib = (stick, CalibrationSession(steps))
        self.correctors[stick].reset_history()
        if self.simulate:
            from simulator import Autopilot
            self.pilot = Autopilot(seed=int(time.time()))
        kind = "rest refresh" if quick else "calibration"
        self.log(f"{stick.capitalize()} stick {kind} started (Esc to cancel).")

    def cancel_calibration(self):
        if self.calib:
            self.log("Calibration cancelled -- nothing saved.")
        self.calib = None
        self.pilot = None

    def _finish_calibration(self):
        stick, session = self.calib
        self.calib = None
        self.pilot = None
        save_session(stick, session.rows)
        self.log(f"Saved {len(session.rows)} samples. Training {stick} stick model...")
        self.training = {"stick": stick, "progress": 0.0}
        threading.Thread(target=self._train_worker, args=(stick,), daemon=True).start()

    def _train_worker(self, stick):
        def progress(epoch, epochs, loss):
            self.training["progress"] = epoch / epochs
        try:
            result = train_stick(load_sessions(stick), stick, progress=progress)
            save_result(stick, result)
            self.train_result = (stick, result, None)
        except TrainingError as e:
            self.train_result = (stick, None, str(e))
        except Exception as e:  # never kill the UI thread's state
            self.train_result = (stick, None, f"{type(e).__name__}: {e}")

    def _poll_training(self):
        if self.train_result is None:
            return
        stick, result, err = self.train_result
        self.train_result = None
        self.training = None
        if err:
            self.log(f"Training failed: {err}")
            self.dialog = ("Training failed", [err])
            return
        corr = self.correctors[stick]
        corr.set_model(result.mlp, result.x_std, result.meta)
        corr.set_mode(result.meta["preferred_mode"])
        self.summaries[stick] = data_summary(stick)
        self.persist()
        self.log(f"{stick.capitalize()} stick model ready -> mode: "
                 f"{MODE_NAMES[corr.mode]}.")
        self.dialog = (f"{stick.capitalize()} stick calibrated", result.report)

    def cycle_mode(self, stick):
        self.correctors[stick].cycle_mode()
        self.persist()

    def reset_stick(self, stick):
        deadline = self.confirm_reset.get(stick, 0)
        if time.time() > deadline:
            self.confirm_reset[stick] = time.time() + 3.0
            return
        self.confirm_reset.pop(stick, None)
        clear_data(stick)
        self.correctors[stick].unload()
        self.summaries[stick] = data_summary(stick)
        self.persist()
        self.log(f"{stick.capitalize()} stick data and model deleted.")

    # ----- layout -----------------------------------------------------------
    def _panel_rect(self, stick):
        x = 16 if stick == "left" else 16 + 496 + 16
        return pygame.Rect(x, PANEL_Y, 496, PANEL_H)

    def _circle(self, stick):
        r = self._panel_rect(stick)
        return (r.x + 158, r.y + 192), 130

    def _build_buttons(self):
        idle = lambda: not self.busy()  # noqa: E731
        self.buttons = [
            Button((W - 236, 14, 220, 44),
                   lambda: "Stop correction" if self.vpad else "Start correction",
                   self.toggle_output, style=lambda: "danger" if self.vpad else "go"),
            Button((W - 392, 14, 148, 44), "Switch controller", self.switch_device,
                   enabled=lambda: idle() and not self.simulate),
            Button((W - 548, 14, 148, 44),
                   lambda: f"Auto-start: {'On' if self.settings.get('autostart') else 'Off'}",
                   self.toggle_autostart),
        ]
        for stick in STICKS:
            r = self._panel_rect(stick)
            bx, by, bw, gap = r.x + 12, r.bottom - 50, 112, 8
            has_reader = lambda: self.reader is not None  # noqa: E731
            has_model = lambda s=stick: self.correctors[s].has_model  # noqa: E731
            self.buttons += [
                Button((bx, by, bw, 38), "Calibrate",
                       lambda s=stick: self.start_calibration(s),
                       enabled=lambda: idle() and has_reader(), style="primary"),
                Button((bx + (bw + gap), by, bw, 38), "Rest refresh",
                       lambda s=stick: self.start_calibration(s, quick=True),
                       enabled=lambda s=stick: idle() and has_reader() and has_model(s)),
                Button((bx + 2 * (bw + gap), by, bw, 38), "Change mode",
                       lambda s=stick: self.cycle_mode(s),
                       enabled=lambda s=stick: idle() and has_model(s)),
                Button((bx + 3 * (bw + gap), by, bw, 38),
                       lambda s=stick: "Confirm?" if time.time() < self.confirm_reset.get(s, 0)
                       else "Reset",
                       lambda s=stick: self.reset_stick(s),
                       enabled=lambda s=stick: idle() and (has_model(s) or self.summaries[s][0] > 0),
                       style=lambda s=stick: "danger" if time.time() < self.confirm_reset.get(s, 0)
                       else "normal"),
            ]
        self.dialog_ok = Button((W // 2 - 60, 0, 120, 40), "OK", self._close_dialog,
                                style="primary")

    def _close_dialog(self):
        self.dialog = None

    # ----- events -----------------------------------------------------------
    def handle_event(self, e):
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.JOYDEVICEADDED:
            self._on_device_added(e.device_index)
        elif e.type == pygame.JOYDEVICEREMOVED:
            self._on_device_removed(e.instance_id)
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                if self.dialog:
                    self.dialog = None
                elif self.calib:
                    self.cancel_calibration()
            elif e.key in (pygame.K_RETURN, pygame.K_SPACE) and self.dialog:
                self.dialog = None
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.dialog:
                self.dialog_ok.click(e.pos)
                return True
            for b in self.buttons:
                if b.click(e.pos):
                    return True
            if self.simulate:
                for s in STICKS:
                    (cx, cy), rad = self._circle(s)
                    if math.hypot(e.pos[0] - cx, e.pos[1] - cy) <= rad:
                        self.dragging = s
                        self._drag_to(e.pos)
        elif e.type == pygame.MOUSEMOTION and self.dragging:
            self._drag_to(e.pos)
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.dragging:
            self.reader.true[self.dragging] = (0.0, 0.0)
            self.dragging = None
        return True

    def _drag_to(self, pos):
        (cx, cy), rad = self._circle(self.dragging)
        x, y = (pos[0] - cx) / rad, (pos[1] - cy) / rad
        r = math.hypot(x, y)
        if r > 1.0:
            x, y = x / r, y / r
        self.reader.true[self.dragging] = (x, y)

    # ----- main loop --------------------------------------------------------
    def tick(self, dt):
        now = time.perf_counter()
        if self.simulate and self.calib and self.pilot:
            stick, session = self.calib
            step = session.step
            if step is not None:
                self.reader.true[stick] = self.pilot.follow(step.pose, session.time_in_step, dt)

        if self.reader is not None and self.reader.attached():
            self.state = self.reader.read()
        else:
            self.state = PadState()

        if self.calib:
            stick, session = self.calib
            session.feed(now, *self.state.stick(stick))
            if session.done:
                self._finish_calibration()

        for s in STICKS:
            raw = self.state.stick(s)
            self.out[s] = self.correctors[s].correct(*raw)
            self.trails[s].append(raw)

        if self.vpad is not None:
            try:
                self.vpad.send(self.state, self.out["left"], self.out["right"])
            except Exception as e:
                self.log(f"Virtual controller error: {e}")
                self.vpad = None

        self._poll_training()

    def run(self):
        frame = 0
        running = True
        while running:
            dt = self.clock.tick(config.SAMPLE_RATE_HZ) / 1000.0
            for e in pygame.event.get():
                if not self.handle_event(e):
                    running = False
            self.tick(min(dt, 0.05))
            frame += 1
            if frame % 2 == 0:  # ~60 fps drawing, 125 Hz correction
                self.draw()
                pygame.display.flip()
        self.shutdown()

    def shutdown(self):
        self.stop_output()
        self.persist()
        if self.reader is not None:
            self.reader.close()
        pygame.quit()

    # ----- drawing ----------------------------------------------------------
    def text(self, s, pos, size=15, color=TEXT, bold=False, anchor="topleft"):
        surf = self.fonts.render(s, size, color, bold)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    def draw(self):
        self.screen.fill(BG)
        mouse = pygame.mouse.get_pos()
        self._draw_header()
        for s in STICKS:
            self._draw_stick_panel(s)
        if self.calib or self.training:
            self._draw_calibration_banner()
        else:
            self._draw_bottom()
        for b in self.buttons:
            b.draw(self.screen, self.fonts, mouse if not self.dialog else (-1, -1))
        if self.dialog:
            self._draw_dialog(mouse)

    def _draw_header(self):
        self.text("Drift Fix", (18, 8), 30, TEXT, bold=True)
        if self.reader is None:
            dot, line = DANGER, "No controller connected"
        elif not self.reader.attached():
            dot, line = DANGER, f"{self.reader.name} (disconnected)"
        else:
            how = "auto-mapped" if self.reader.mapped else "config.py mapping"
            dot, line = OUT, f"{self.reader.name}  -  {how}"
        pygame.draw.circle(self.screen, dot, (24, 56), 5)
        self.text(line, (36, 46), 14, MUTED)
        status = ("OUTPUT ON: virtual Xbox 360 pad active" if self.vpad
                  else "Output off: game sees your real pad")
        self.text(status, (W - 16, 60), 12, OUT if self.vpad else MUTED, anchor="topright")

    def _draw_stick_panel(self, stick):
        rect = self._panel_rect(stick)
        active = self.calib and self.calib[0] == stick
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=12)
        if active:
            pygame.draw.rect(self.screen, WARN, rect, width=2, border_radius=12)
        self.text(f"{stick.upper()} STICK", (rect.x + 18, rect.y + 12), 16, TEXT, bold=True)

        corr = self.correctors[stick]
        (cx, cy), rad = self._circle(stick)
        scr = self.screen
        pygame.draw.circle(scr, (22, 24, 31), (cx, cy), rad)
        pygame.draw.circle(scr, GRID, (cx, cy), rad, 2)
        pygame.draw.circle(scr, GRID, (cx, cy), rad // 2, 1)
        pygame.draw.line(scr, GRID, (cx - rad, cy), (cx + rad, cy), 1)
        pygame.draw.line(scr, GRID, (cx, cy - rad), (cx, cy + rad), 1)

        meta = corr.meta
        if corr.mode != "off" and corr.has_model:
            dz = meta.get("deadzone_model" if corr.mode == "model" else "deadzone_offset", 0)
            pygame.draw.circle(scr, (48, 70, 60), (cx, cy), max(2, int(dz * rad)), 1)
        if corr.has_model:
            ox, oy = meta.get("rest_offset", (0, 0))
            px, py = cx + ox * rad, cy + oy * rad
            pygame.draw.line(scr, (120, 80, 50), (px - 6, py), (px + 6, py), 1)
            pygame.draw.line(scr, (120, 80, 50), (px, py - 6), (px, py + 6), 1)

        if active:
            self._draw_target_guide(stick, (cx, cy), rad)

        pts = list(self.trails[stick])
        for i in range(1, len(pts)):
            a = i / len(pts)
            col = (int(70 + 120 * a), int(50 + 60 * a), int(40 * a))
            p0 = (cx + pts[i - 1][0] * rad, cy + pts[i - 1][1] * rad)
            p1 = (cx + pts[i][0] * rad, cy + pts[i][1] * rad)
            pygame.draw.line(scr, col, p0, p1, 2)

        rx, ry = self.state.stick(stick)
        ox, oy = self.out[stick]
        pygame.draw.circle(scr, RAW, (int(cx + rx * rad), int(cy + ry * rad)), 7)
        pygame.draw.circle(scr, OUT, (int(cx + ox * rad), int(cy + oy * rad)), 9, 3)

        # info column
        ix, iy = rect.x + 310, rect.y + 52
        self.text("raw", (ix, iy), 13, RAW, bold=True)
        self.text(f"x {rx:+.3f}   y {ry:+.3f}", (ix + 34, iy), 14, TEXT)
        self.text("out", (ix, iy + 22), 13, OUT, bold=True)
        self.text(f"x {ox:+.3f}   y {oy:+.3f}", (ix + 34, iy + 22), 14, TEXT)
        self.text("drift", (ix, iy + 44), 13, MUTED, bold=True)
        self.text(f"{math.hypot(rx, ry):.3f}  ->  {math.hypot(ox, oy):.3f}", (ix + 34, iy + 44), 14,
                  TEXT)

        iy += 80
        self.text("MODE", (ix, iy), 12, MUTED, bold=True)
        self.text(MODE_NAMES[corr.mode], (ix, iy + 16), 19, MODE_COLORS[corr.mode], bold=True)
        iy += 52
        n_sessions = self.summaries[stick][0]
        if corr.has_model:
            m = meta.get("metrics", {})
            off, mod = m.get("offset_mse"), m.get("model_mse")
            lines = []
            if off and mod is not None:
                if mod <= off:
                    lines.append((f"Model {100 * (1 - mod / off):.0f}% better", OUT))
                    lines.append(("than the offset fix", MUTED))
                else:
                    lines.append(("Offset fix beat the model", WARN))
                    lines.append(("(simple fixed drift)", MUTED))
            ox_, oy_ = meta.get("rest_offset", (0, 0))
            lines.append((f"Rest offset {ox_:+.3f}, {oy_:+.3f}", MUTED))
            dz = meta.get("deadzone_model" if corr.mode != "offset" else "deadzone_offset", 0)
            lines.append((f"Deadzone {dz:.3f}", MUTED))
            lines.append((f"{n_sessions} session(s), {meta.get('trained_at', '')[5:16]}", MUTED))
        elif n_sessions:
            lines = [("Data saved but no model.", WARN), ("Press Calibrate again.", MUTED)]
        else:
            lines = [("Not calibrated yet.", WARN), ("Press Calibrate and", MUTED),
                     ("follow the prompts.", MUTED)]
        for i, (s, col) in enumerate(lines):
            self.text(s, (ix, iy + i * 19), 14, col)

        # legend
        ly = rect.bottom - 74
        pygame.draw.circle(scr, RAW, (rect.x + 30, ly), 5)
        self.text("raw (what your pad says)", (rect.x + 40, ly - 9), 13, MUTED)
        pygame.draw.circle(scr, OUT, (rect.x + 222, ly), 6, 2)
        self.text("corrected (what the game gets)", (rect.x + 234, ly - 9), 13, MUTED)

    def _draw_target_guide(self, stick, centre, rad):
        _, session = self.calib
        step = session.step
        if step is None:
            return
        cx, cy = centre
        pose = step.pose
        t = session.time_in_step
        if pose[0] == "hold":
            tx, ty = angle_to_xy(pose[1])
            end = (cx + tx * rad, cy + ty * rad)
            pygame.draw.line(self.screen, WARN, (cx, cy), end, 3)
            pulse = 12 + 4 * math.sin(t * 8)
            pygame.draw.circle(self.screen, WARN, (int(end[0]), int(end[1])), int(pulse), 3)
        elif pose[0] == "sweep":
            ang = math.degrees(pose[1] * t * 2 * math.pi / 2.5)
            for k in range(6):
                a0 = ang - pose[1] * k * 9
                px, py = angle_to_xy(a0)
                pygame.draw.circle(self.screen, WARN, (int(cx + px * rad), int(cy + py * rad)),
                                   max(2, 10 - k * 1.5))
        else:
            pulse = int(8 + 3 * math.sin(t * 6))
            pygame.draw.circle(self.screen, WARN, (cx, cy), pulse, 2)
            self.text("HANDS OFF", (cx, cy + rad - 24), 14, WARN, bold=True, anchor="center")

    def _draw_calibration_banner(self):
        rect = pygame.Rect(16, BOTTOM_Y, W - 32, H - BOTTOM_Y - 16)
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=12)
        pygame.draw.rect(self.screen, WARN, rect, width=2, border_radius=12)
        bar = pygame.Rect(rect.x + 24, rect.bottom - 34, rect.w - 48, 12)
        if self.training:
            stick = self.training["stick"]
            p = self.training["progress"]
            self.text(f"Training the {stick} stick model...", (rect.x + 24, rect.y + 18), 28, TEXT,
                      bold=True)
            self.text("Hand-written neural net, full-batch gradient descent. "
                      "You can keep using the controller.", (rect.x + 24, rect.y + 62), 15, MUTED)
        else:
            stick, session = self.calib
            step = session.step
            p = session.progress
            if step is None:
                return
            self.text(f"{stick.upper()} STICK", (rect.x + 24, rect.y + 14), 14, WARN, bold=True)
            self.text(step.instruction, (rect.x + 24, rect.y + 34), 30, TEXT, bold=True)
            secs = session.step_remaining
            self.text(f"{secs:.1f}s", (rect.right - 24, rect.y + 20), 34, WARN, bold=True,
                      anchor="topright")
            nxt = session.next_step
            hint = f"Next: {nxt.instruction}" if nxt else "Last step!"
            self.text(f"{hint}      (Esc to cancel)", (rect.x + 24, rect.y + 84), 15, MUTED)
        pygame.draw.rect(self.screen, (22, 24, 31), bar, border_radius=6)
        fill = bar.copy()
        fill.w = max(6, int(bar.w * p))
        pygame.draw.rect(self.screen, WARN, fill, border_radius=6)

    def _draw_bottom(self):
        left = pygame.Rect(16, BOTTOM_Y, 496, H - BOTTOM_Y - 16)
        right = pygame.Rect(16 + 496 + 16, BOTTOM_Y, 496, H - BOTTOM_Y - 16)
        for r in (left, right):
            pygame.draw.rect(self.screen, PANEL, r, border_radius=12)

        # triggers + buttons passthrough preview
        self.text("TRIGGERS & BUTTONS (passed straight through)", (left.x + 18, left.y + 12), 12,
                  MUTED, bold=True)
        for i, (name, val) in enumerate((("LT", self.state.lt), ("RT", self.state.rt))):
            y = left.y + 40 + i * 26
            self.text(name, (left.x + 18, y - 2), 14, TEXT, bold=True)
            bar = pygame.Rect(left.x + 50, y, 180, 12)
            pygame.draw.rect(self.screen, (22, 24, 31), bar, border_radius=6)
            if val > 0.001:
                f = bar.copy()
                f.w = max(4, int(bar.w * val))
                pygame.draw.rect(self.screen, ACCENT, f, border_radius=6)
        names = [("A", "A"), ("B", "B"), ("X", "X"), ("Y", "Y"), ("LB", "LEFT_SHOULDER"),
                 ("RB", "RIGHT_SHOULDER"), ("LS", "LEFT_THUMB"), ("RS", "RIGHT_THUMB"),
                 ("Back", "BACK"), ("Start", "START"), ("Up", "DPAD_UP"), ("Down", "DPAD_DOWN"),
                 ("Left", "DPAD_LEFT"), ("Right", "DPAD_RIGHT")]
        for i, (label, key) in enumerate(names):
            on = f"XUSB_GAMEPAD_{key}" in self.state.buttons
            col, row = i % 7, i // 7
            r = pygame.Rect(left.x + 18 + col * 66, left.y + 96 + row * 26, 60, 22)
            pygame.draw.rect(self.screen, ACCENT if on else PANEL_HI, r, border_radius=5)
            self.text(label, r.center, 12, TEXT if on else MUTED, bold=True, anchor="center")

        self.text("ACTIVITY", (right.x + 18, right.y + 12), 12, MUTED, bold=True)
        for i, (ts, msg) in enumerate(list(self.messages)[-6:]):
            if len(msg) > 62:
                msg = msg[:60] + "..."
            self.text(f"{ts}  {msg}", (right.x + 18, right.y + 34 + i * 19), 13, TEXT)

    def _draw_dialog(self, mouse):
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 160))
        self.screen.blit(shade, (0, 0))
        title, lines = self.dialog
        h = 120 + 22 * len(lines)
        rect = pygame.Rect(0, 0, 640, h)
        rect.center = (W // 2, H // 2)
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=14)
        pygame.draw.rect(self.screen, ACCENT, rect, width=2, border_radius=14)
        self.text(title, (rect.x + 24, rect.y + 18), 22, TEXT, bold=True)
        for i, line in enumerate(lines):
            col = OUT if "beats" in line else WARN if ("NOT" in line or "Note" in line) else TEXT
            self.text(line, (rect.x + 24, rect.y + 58 + i * 22), 15, col)
        self.dialog_ok.rect.y = rect.bottom - 54
        self.dialog_ok.draw(self.screen, self.fonts, mouse)


def main():
    ap = argparse.ArgumentParser(description="Drift Fix desktop app")
    ap.add_argument("--simulate", action="store_true",
                    help="use a simulated drifting controller (no hardware needed)")
    args = ap.parse_args()
    App(simulate=args.simulate).run()


if __name__ == "__main__":
    main()
