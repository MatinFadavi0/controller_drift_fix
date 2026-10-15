"""
calibration.py

The guided calibration routine, as data + a small state machine that is
driven by whoever owns the main loop (the GUI app or a terminal script).

Every raw sample is recorded, including the moments you're moving
between positions, so the feature history can be replayed faithfully at
training time. Only samples captured once you've had time to settle get
a label:

    "rest" -> true position is center (0, 0)
    "rim"  -> stick is pushed against the edge, true radius is 1.0 and the
              true direction is wherever you're actually pushing
              (worked out at training time, not the prompted angle, since
              nobody hits exactly 45 degrees)

The full routine (~1 minute) captures:
    * resting drift with nothing touched
    * the edge in 8 directions (position-dependent drift, lost range)
    * where the stick settles right after being pushed each way
      (direction-dependent drift / hysteresis)
    * slow sweeps around the edge both ways
The quick "rest refresh" (~13 s) just re-measures resting drift, e.g.
after 20 minutes of play when the controller has warmed up.
"""
import math
from dataclasses import dataclass

LABEL_NONE = ""
LABEL_REST = "rest"
LABEL_RIM = "rim"

DIRECTIONS = [
    ("RIGHT", 0), ("UP-RIGHT", 45), ("UP", 90), ("UP-LEFT", 135),
    ("LEFT", 180), ("DOWN-LEFT", 225), ("DOWN", 270), ("DOWN-RIGHT", 315),
]


@dataclass
class Step:
    instruction: str
    duration: float
    pose: tuple            # what the stick should be doing, for visual guides
    label: str = LABEL_NONE
    settle: float = 0.0    # seconds at the start of the step left unlabeled


def angle_to_xy(deg, radius=1.0):
    """Screen angle (0 = right, 90 = up) -> SDL stick coords (up = -1)."""
    a = math.radians(deg)
    return radius * math.cos(a), -radius * math.sin(a)


def full_script():
    steps = [
        Step("Let go of the stick. Don't touch it.", 3.0, ("rest",)),
        Step("Hands off -- measuring resting drift", 8.0, ("rest",), LABEL_REST),
    ]
    for name, deg in DIRECTIONS:
        steps.append(Step(f"Push {name} all the way to the edge and hold",
                          2.2, ("hold", deg), LABEL_RIM, settle=1.0))
        steps.append(Step("Let go -- let it snap back to center",
                          2.2, ("release", deg), LABEL_REST, settle=0.6))
    steps += [
        Step("Roll the stick slowly around the edge, CLOCKWISE",
             6.0, ("sweep", -1), LABEL_RIM, settle=1.2),
        Step("Keep rolling around the edge, now COUNTER-CLOCKWISE",
             6.0, ("sweep", +1), LABEL_RIM, settle=1.2),
        Step("Let go -- final resting measurement", 3.0, ("rest",), LABEL_REST,
             settle=0.6),
    ]
    return steps


def rest_script():
    return [
        Step("Let go of the stick. Don't touch it.", 3.0, ("rest",)),
        Step("Hands off -- re-measuring resting drift", 10.0, ("rest",), LABEL_REST),
    ]


class CalibrationSession:
    """Feed it (time, raw_x, raw_y) every tick; it advances through the
    steps on its own and collects rows of (step_index, x, y, label)."""

    def __init__(self, steps):
        self.steps = steps
        self.total = sum(s.duration for s in steps)
        self.rows = []
        self.index = 0
        self._step_start = None
        self._elapsed_before = 0.0
        self._now_in_step = 0.0

    @property
    def done(self):
        return self.index >= len(self.steps)

    @property
    def step(self):
        return None if self.done else self.steps[self.index]

    @property
    def next_step(self):
        i = self.index + 1
        return self.steps[i] if i < len(self.steps) else None

    @property
    def time_in_step(self):
        return self._now_in_step

    @property
    def step_remaining(self):
        return 0.0 if self.done else max(0.0, self.step.duration - self._now_in_step)

    @property
    def progress(self):
        return min(1.0, (self._elapsed_before + self._now_in_step) / self.total)

    def feed(self, now, x, y):
        if self.done:
            return
        if self._step_start is None:
            self._step_start = now
        self._now_in_step = now - self._step_start
        while not self.done and self._now_in_step >= self.step.duration:
            self._elapsed_before += self.step.duration
            self._step_start += self.step.duration
            self._now_in_step = now - self._step_start
            self.index += 1
        if self.done:
            return
        s = self.step
        label = s.label if self._now_in_step >= s.settle else LABEL_NONE
        self.rows.append((self.index, float(x), float(y), label))
