"""
virtual_pad.py

The "output" side: wraps vgamepad's virtual Xbox 360 controller. We take
the (already corrected) stick values plus a passthrough of every other
button/trigger and emit them as a virtual device that Windows and games
see as a real controller.

Requires the ViGEmBus driver on Windows (see README).
"""


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class VirtualPad:
    def __init__(self):
        import vgamepad as vg  # imported lazily so the rest of the app works without it
        self._vg = vg
        self._pad = vg.VX360Gamepad()
        self._held = set()

    def send(self, state, left_xy, right_xy):
        """state: reader.PadState (buttons/triggers pass straight through).
        left_xy / right_xy: final (x, y) per stick in SDL convention
        (up = -1); flipped here to XInput's up = +1."""
        p = self._pad
        p.left_joystick_float(x_value_float=_clamp(left_xy[0], -1.0, 1.0),
                              y_value_float=_clamp(-left_xy[1], -1.0, 1.0))
        p.right_joystick_float(x_value_float=_clamp(right_xy[0], -1.0, 1.0),
                               y_value_float=_clamp(-right_xy[1], -1.0, 1.0))
        p.left_trigger_float(value_float=_clamp(state.lt, 0.0, 1.0))
        p.right_trigger_float(value_float=_clamp(state.rt, 0.0, 1.0))

        wanted = set(state.buttons)
        for name in wanted - self._held:
            p.press_button(button=getattr(self._vg.XUSB_BUTTON, name))
        for name in self._held - wanted:
            p.release_button(button=getattr(self._vg.XUSB_BUTTON, name))
        self._held = wanted
        p.update()

    def close(self):
        try:
            self._pad.reset()
            self._pad.update()
        except Exception:
            pass
        self._pad = None


def open_virtual_pad():
    """Returns (VirtualPad, None) or (None, human-readable error)."""
    try:
        return VirtualPad(), None
    except ImportError:
        return None, "vgamepad is not installed (pip install -r requirements.txt)"
    except Exception as e:  # ViGEmBus missing / not running
        return None, (f"Could not create the virtual controller ({e}). "
                      "Is the ViGEmBus driver installed?")
