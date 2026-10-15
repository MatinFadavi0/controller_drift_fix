"""
simulator.py

A simulated worn-out controller, for trying the app without real
hardware (`python app.py --simulate`) and for the automated pipeline
test. It models the drift types the README talks about:

  * a constant resting offset
  * position-dependent error (the stick can't reach the edge in some
    directions, plus an angle-dependent wobble)
  * hysteresis: the resting point shifts towards wherever the stick was
    last pushed
  * a little sensor noise

The "true" stick position comes either from an autopilot that follows
the calibration prompts like an obedient human would, or from the GUI
(drag inside a stick circle with the mouse).
"""
import math

import numpy as np

from calibration import angle_to_xy
from reader import PadState


class DriftModel:
    def __init__(self, seed=0, offset=(0.09, -0.06), gain=(0.86, 0.95),
                 hysteresis=0.05, wobble=0.04, noise=0.003):
        self.rng = np.random.default_rng(seed)
        self.offset = offset
        self.gain = gain
        self.hysteresis = hysteresis
        self.wobble = wobble
        self.noise = noise
        self._last_dir = (0.0, 0.0)

    def measure(self, tx, ty):
        """True position (inside the unit circle) -> raw sensor reading."""
        r = math.hypot(tx, ty)
        if r > 0.6:
            self._last_dir = (tx / r, ty / r)
        ang = math.atan2(ty, tx)
        w = 1.0 + self.wobble * math.sin(3 * ang + 0.7) * r
        fade = 1.0 - 0.4 * r  # the offset shrinks a bit towards the edge
        x = tx * self.gain[0] * w + self.offset[0] * fade + self.hysteresis * self._last_dir[0]
        y = ty * self.gain[1] * w + self.offset[1] * fade + self.hysteresis * self._last_dir[1]
        x += self.rng.normal(0.0, self.noise)
        y += self.rng.normal(0.0, self.noise)
        return max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y))


class Autopilot:
    """Turns a calibration step's pose into a true stick position, with a
    human-ish reaction delay and a little hand tremor."""

    def __init__(self, seed=0, reaction=0.35):
        self.rng = np.random.default_rng(seed + 100)
        self.reaction = reaction
        self.pos = (0.0, 0.0)
        self._sweep_angle = 0.0

    def follow(self, pose, t_in_step, dt):
        kind = pose[0] if pose else "rest"
        if t_in_step < self.reaction and kind in ("hold", "sweep"):
            return self.pos  # still reacting to the new prompt
        if kind == "hold":
            target = angle_to_xy(pose[1], 0.99)
            tremor = self.rng.normal(0.0, 0.004, 2)
            target = (target[0] + tremor[0], target[1] + tremor[1])
            k = min(1.0, dt * 14.0)
            self.pos = (self.pos[0] + (target[0] - self.pos[0]) * k,
                        self.pos[1] + (target[1] - self.pos[1]) * k)
            r = math.hypot(*self.pos)
            if r > 1.0:
                self.pos = (self.pos[0] / r, self.pos[1] / r)
        elif kind == "sweep":
            if math.hypot(*self.pos) < 0.5:
                self._sweep_angle = 0.0
            self._sweep_angle += pose[1] * dt * 2 * math.pi / 2.5
            self.pos = angle_to_xy(math.degrees(self._sweep_angle), 0.99)
        else:  # rest / release: thumb off, spring snaps it home
            self.pos = (0.0, 0.0)
        return self.pos


class SimulatedReader:
    name = "Simulated drifting controller"
    guid = "simulated"
    instance_id = -1
    mapped = True

    def __init__(self, seed=0):
        self.drift = {
            "left": DriftModel(seed, offset=(-0.05, 0.07), gain=(0.93, 0.88),
                               hysteresis=0.04),
            "right": DriftModel(seed + 1),
        }
        self.true = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
        self.buttons = frozenset()

    def attached(self):
        return True

    def read(self):
        lx, ly = self.drift["left"].measure(*self.true["left"])
        rx, ry = self.drift["right"].measure(*self.true["right"])
        return PadState(lx, ly, rx, ry, 0.0, 0.0, self.buttons)

    def close(self):
        pass
