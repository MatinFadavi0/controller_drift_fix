"""
reader.py

Reads the physical controller and returns a normalised PadState snapshot.

If SDL recognises the pad (Xbox, PlayStation, Switch Pro and most
third-party pads), its GameController API is used: sticks, triggers,
buttons and D-pad are mapped automatically and identically for every
pad, no config needed. Unrecognised pads fall back to the raw joystick
API with AXIS_MAP / BUTTON_MAP / HAT_INDEX from config.py.

Conventions for every consumer downstream:
    sticks   -1..+1, x: left = -1, y: UP = -1 (SDL convention)
    triggers  0..1
    buttons  frozenset of vgamepad XUSB_BUTTON member names

Call pygame.event.pump() (or drain pygame.event.get()) once per tick
before read(), otherwise SDL won't refresh the device state.
"""
import os
from dataclasses import dataclass, field

# Keep reading the controller while another window (your game) has focus.
# Must be set before SDL's joystick subsystem starts.
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame  # noqa: E402
from pygame._sdl2 import controller as sdl_controller  # noqa: E402

import config  # noqa: E402

STICKS = ("left", "right")

_CONTROLLER_BUTTONS = {
    pygame.CONTROLLER_BUTTON_A: "XUSB_GAMEPAD_A",
    pygame.CONTROLLER_BUTTON_B: "XUSB_GAMEPAD_B",
    pygame.CONTROLLER_BUTTON_X: "XUSB_GAMEPAD_X",
    pygame.CONTROLLER_BUTTON_Y: "XUSB_GAMEPAD_Y",
    pygame.CONTROLLER_BUTTON_BACK: "XUSB_GAMEPAD_BACK",
    pygame.CONTROLLER_BUTTON_GUIDE: "XUSB_GAMEPAD_GUIDE",
    pygame.CONTROLLER_BUTTON_START: "XUSB_GAMEPAD_START",
    pygame.CONTROLLER_BUTTON_LEFTSTICK: "XUSB_GAMEPAD_LEFT_THUMB",
    pygame.CONTROLLER_BUTTON_RIGHTSTICK: "XUSB_GAMEPAD_RIGHT_THUMB",
    pygame.CONTROLLER_BUTTON_LEFTSHOULDER: "XUSB_GAMEPAD_LEFT_SHOULDER",
    pygame.CONTROLLER_BUTTON_RIGHTSHOULDER: "XUSB_GAMEPAD_RIGHT_SHOULDER",
    pygame.CONTROLLER_BUTTON_DPAD_UP: "XUSB_GAMEPAD_DPAD_UP",
    pygame.CONTROLLER_BUTTON_DPAD_DOWN: "XUSB_GAMEPAD_DPAD_DOWN",
    pygame.CONTROLLER_BUTTON_DPAD_LEFT: "XUSB_GAMEPAD_DPAD_LEFT",
    pygame.CONTROLLER_BUTTON_DPAD_RIGHT: "XUSB_GAMEPAD_DPAD_RIGHT",
}


@dataclass
class PadState:
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    buttons: frozenset = field(default_factory=frozenset)

    def stick(self, which):
        return (self.lx, self.ly) if which == "left" else (self.rx, self.ry)


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def init():
    """Start the SDL subsystems we need (safe to call more than once)."""
    pygame.init()
    pygame.joystick.init()
    sdl_controller.init()


@dataclass
class DeviceInfo:
    index: int
    name: str
    guid: str
    instance_id: int
    mapped: bool  # recognised by SDL's GameController database


def list_devices(exclude_instance_ids=()):
    """All connected joysticks except the ones listed (e.g. our own
    virtual pad, which would otherwise create a feedback loop)."""
    out = []
    for i in range(pygame.joystick.get_count()):
        try:
            j = pygame.joystick.Joystick(i)
            iid = j.get_instance_id()
            if iid in exclude_instance_ids:
                continue
            out.append(DeviceInfo(i, j.get_name(), j.get_guid(), iid,
                                  sdl_controller.is_controller(i)))
        except pygame.error:
            continue
    return out


class ControllerReader:
    def __init__(self, index=0):
        if pygame.joystick.get_count() <= index:
            raise RuntimeError("No controller detected. Plug one in and retry.")
        self._joy = pygame.joystick.Joystick(index)
        self._joy.init()
        self.name = self._joy.get_name()
        self.guid = self._joy.get_guid()
        self.instance_id = self._joy.get_instance_id()
        self._ctrl = None
        if sdl_controller.is_controller(index):
            self._ctrl = sdl_controller.Controller(index)
        self.mapped = self._ctrl is not None

    def attached(self):
        try:
            if self._ctrl is not None:
                return bool(self._ctrl.attached())
            return bool(self._joy.get_init())
        except pygame.error:
            return False

    def read(self):
        if self._ctrl is not None:
            return self._read_mapped()
        return self._read_raw()

    def _read_mapped(self):
        c = self._ctrl

        def stick(axis):
            return _clamp(c.get_axis(axis) / 32767.0, -1.0, 1.0)

        def trigger(axis):
            return _clamp(c.get_axis(axis) / 32767.0, 0.0, 1.0)

        buttons = frozenset(name for btn, name in _CONTROLLER_BUTTONS.items()
                            if c.get_button(btn))
        return PadState(
            lx=stick(pygame.CONTROLLER_AXIS_LEFTX),
            ly=stick(pygame.CONTROLLER_AXIS_LEFTY),
            rx=stick(pygame.CONTROLLER_AXIS_RIGHTX),
            ry=stick(pygame.CONTROLLER_AXIS_RIGHTY),
            lt=trigger(pygame.CONTROLLER_AXIS_TRIGGERLEFT),
            rt=trigger(pygame.CONTROLLER_AXIS_TRIGGERRIGHT),
            buttons=buttons,
        )

    def _read_raw(self):
        j = self._joy
        n_axes = j.get_numaxes()
        n_buttons = j.get_numbuttons()

        def axis(name):
            idx = config.AXIS_MAP[name]
            return j.get_axis(idx) if idx < n_axes else 0.0

        def trigger(name):
            idx = config.AXIS_MAP[name]
            if idx >= n_axes:
                return 0.0
            return _clamp((j.get_axis(idx) + 1.0) / 2.0, 0.0, 1.0)

        ly, ry = axis("ly"), axis("ry")
        if config.INVERT_LY:
            ly = -ly
        if config.INVERT_RY:
            ry = -ry

        buttons = {name for idx, name in config.BUTTON_MAP.items()
                   if idx < n_buttons and j.get_button(idx)}
        if j.get_numhats() > config.HAT_INDEX:
            hx, hy = j.get_hat(config.HAT_INDEX)
            if hx == 1:
                buttons.add("XUSB_GAMEPAD_DPAD_RIGHT")
            elif hx == -1:
                buttons.add("XUSB_GAMEPAD_DPAD_LEFT")
            if hy == 1:
                buttons.add("XUSB_GAMEPAD_DPAD_UP")
            elif hy == -1:
                buttons.add("XUSB_GAMEPAD_DPAD_DOWN")

        return PadState(lx=axis("lx"), ly=ly, rx=axis("rx"), ry=ry,
                        lt=trigger("lt"), rt=trigger("rt"),
                        buttons=frozenset(buttons))

    def close(self):
        try:
            if self._ctrl is not None:
                self._ctrl.quit()
            self._joy.quit()
        except pygame.error:
            pass
