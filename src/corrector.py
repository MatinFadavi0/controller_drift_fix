"""
corrector.py

Per-stick live correction. Three modes:

  "model"  -- the hand-written MLP. It predicts a *correction* that is
              added to the raw reading (raw + residual), so anywhere the
              network has little data it degrades gracefully towards the
              raw value instead of inventing a position.
  "offset" -- the classic fix: subtract the resting offset and rescale
              so the edge still reaches full deflection. This is also the
              baseline the model has to beat during training.
  "off"    -- raw passthrough.

Both corrected modes finish with a radial deadzone (sized from the
calibration data so the stick reads exactly 0 at rest) and a rescale so
you still get the full range of travel.
"""
import math
import os

import numpy as np

import config
from features import FeatureWindow
from model import Standardizer, load_model

MODES = ("model", "offset", "off")


def radial_deadzone(x, y, deadzone):
    """Zero inside `deadzone`, rescale the rest back to 0..1, and keep the
    result inside the unit circle."""
    r = math.hypot(x, y)
    if r <= deadzone:
        return 0.0, 0.0
    scaled = min(1.0, (r - deadzone) / (1.0 - deadzone))
    return x / r * scaled, y / r * scaled


def offset_correct(x, y, offset, rim_radius):
    """Baseline correction (before the deadzone)."""
    return (x - offset[0]) / rim_radius, (y - offset[1]) / rim_radius


class StickCorrector:
    def __init__(self, stick):
        self.stick = stick
        self.mode = "off"
        self.meta = {}
        self.mlp = None
        self._x_std = None
        self._y_std = None
        self._window = FeatureWindow(config.HISTORY_LEN, config.PUSH_MEMORY,
                                     config.PUSH_THRESHOLD)

    @property
    def has_model(self):
        return self.mlp is not None

    def load(self, path=None):
        """Load this stick's model if it exists. Returns True on success."""
        path = path or config.model_path(self.stick)
        if not os.path.exists(path):
            return False
        mlp, x_std, meta = load_model(path)
        self.set_model(mlp, x_std, meta)
        return True

    def set_model(self, mlp, x_std, meta):
        self.mlp = mlp
        self._x_std = x_std
        self.meta = meta
        self._y_std = (Standardizer.from_dict(meta["target_standardizer"])
                       if "target_standardizer" in meta else None)
        self._window = FeatureWindow.from_meta(meta)
        if self.mode == "off" or self.mode not in MODES:
            self.mode = meta.get("preferred_mode", "model")

    def unload(self):
        self.mlp = None
        self.meta = {}
        self.mode = "off"

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(mode)
        if mode != "off" and not self.has_model:
            mode = "off"
        self.mode = mode

    def cycle_mode(self):
        options = list(MODES) if self.has_model else ["off"]
        i = options.index(self.mode) if self.mode in options else -1
        self.mode = options[(i + 1) % len(options)]
        return self.mode

    def reset_history(self):
        self._window.reset()

    def correct(self, x, y):
        """Feed one raw sample, get the corrected (x, y). Call every tick
        (even in 'off' mode) so the feature history stays continuous."""
        feats = self._window.push(x, y)
        if self.mode == "off" or not self.has_model:
            return x, y

        meta = self.meta
        if self.mode == "offset":
            cx, cy = offset_correct(x, y, meta.get("rest_offset", (0.0, 0.0)),
                                    meta.get("rim_radius", 1.0))
            return radial_deadzone(cx, cy, meta.get("deadzone_offset",
                                                    config.POST_CORRECTION_DEADZONE))

        pred = self.mlp.predict(self._x_std.transform(feats[None, :]))[0]
        if self._y_std is not None:
            pred = pred * self._y_std.std + self._y_std.mean
        if meta.get("residual", False):
            dx, dy = float(pred[0]), float(pred[1])
            # safety net: never apply a bigger correction than anything
            # seen during training (guards against odd inputs, e.g. very
            # fast flicks, producing a jump)
            limit = meta.get("max_residual")
            if limit is not None:
                mag = math.hypot(dx, dy)
                if mag > limit:
                    dx, dy = dx / mag * limit, dy / mag * limit
            cx, cy = x + dx, y + dy
        else:  # legacy easy_run.py model: predicts the position directly
            cx, cy = float(pred[0]), float(pred[1])
        return radial_deadzone(cx, cy, meta.get("deadzone_model",
                                                config.POST_CORRECTION_DEADZONE))
