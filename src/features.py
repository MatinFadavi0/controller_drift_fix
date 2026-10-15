"""
features.py

Builds the model's input feature vector from the raw stick stream.

Two ingredients:
  * a short rolling window of recent raw (x, y) samples, newest first,
    so the model sees where the stick is and how it's moving right now;
  * "push memory": the unit direction of the last time the stick was
    pushed well away from center. A 3-sample window forgets the stick's
    past within ~24 ms, but worn sticks often settle at a *different*
    resting spot depending on which way they were last pushed
    (hysteresis). Push memory keeps that information around for as long
    as it matters -- until the next push.

The exact same class is used to build training data (replaying recorded
calibration sessions sample by sample) and at inference time, so the
features can never drift out of sync between the two.
"""
import math
from collections import deque

import numpy as np


class FeatureWindow:
    def __init__(self, history_len=3, push_memory=True, push_threshold=0.5):
        self.history_len = int(history_len)
        self.push_memory = bool(push_memory)
        self.push_threshold = float(push_threshold)
        self.reset()

    @property
    def n_features(self):
        return 2 * self.history_len + (2 if self.push_memory else 0)

    def reset(self):
        self._hist = deque(maxlen=self.history_len)
        self._last_push = (0.0, 0.0)

    def push(self, x, y):
        """Add one raw sample and return the feature vector (1-D array)."""
        if not self._hist:
            # pad with the first real sample so the very first predictions
            # after startup aren't fed zeros
            for _ in range(self.history_len):
                self._hist.append((x, y))
        else:
            self._hist.append((x, y))

        r = math.hypot(x, y)
        if r > self.push_threshold:
            self._last_push = (x / r, y / r)

        feats = []
        for hx, hy in reversed(self._hist):  # newest first
            feats.append(hx)
            feats.append(hy)
        if self.push_memory:
            feats.extend(self._last_push)
        return np.array(feats, dtype=np.float64)

    def to_meta(self):
        return {
            "history_len": self.history_len,
            "push_memory": self.push_memory,
            "push_threshold": self.push_threshold,
        }

    @classmethod
    def from_meta(cls, meta):
        # models saved by the original easy_run.py have no feature info:
        # they used a plain 3-sample window
        return cls(
            history_len=meta.get("history_len", 3),
            push_memory=meta.get("push_memory", False),
            push_threshold=meta.get("push_threshold", 0.5),
        )
