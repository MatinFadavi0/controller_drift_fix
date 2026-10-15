"""
training.py

Turns recorded calibration sessions into a trained correction model.

Pipeline:
  1. Estimate the resting offset (median of "rest" samples) and how far
     the stick actually reaches at the edge (median radius of "rim"
     samples). These alone define the classic offset fix -- the baseline.
  2. Replay every session through FeatureWindow so each labeled sample
     gets exactly the features it would have had live.
  3. Targets: rest -> (0, 0); rim -> unit vector in the direction actually
     pushed. The network learns the *correction* (target - raw).
  4. Split train/validation by contiguous 0.2 s blocks rather than single
     samples (neighbouring samples are nearly identical, so a per-sample
     split would leak and flatter the model).
  5. Standardize inputs and outputs, train the hand-written MLP with
     full-batch gradient descent, keep the best-on-validation weights.
  6. Report validation error for: no correction, the offset baseline, and
     the model -- so you can see whether the network actually earns its
     keep. The better of the two corrections becomes the default mode.
"""
import csv
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

import config
from calibration import LABEL_REST, LABEL_RIM
from corrector import offset_correct
from features import FeatureWindow
from model import MLP, Standardizer, save_model

BLOCK_LEN = 25  # samples per validation block (0.2 s at 125 Hz)
CSV_HEADER = ["session", "step", "x", "y", "label"]


# ---------------------------------------------------------------------------
# calibration data on disk
# ---------------------------------------------------------------------------
def load_sessions(stick, path=None):
    """Returns a list of sessions, each a list of (step, x, y, label)."""
    path = path or config.data_path(stick)
    if not os.path.exists(path):
        return []
    sessions = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sessions.setdefault(row["session"], []).append(
                (int(row["step"]), float(row["x"]), float(row["y"]), row["label"]))
    return list(sessions.values())


def save_session(stick, rows, path=None):
    """Append one calibration session's rows; returns its session id."""
    path = path or config.data_path(stick)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    session_id = time.strftime("%Y%m%d-%H%M%S")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        for step, x, y, label in rows:
            w.writerow([session_id, step, f"{x:.6f}", f"{y:.6f}", label])
    return session_id


def clear_data(stick):
    for path in (config.data_path(stick), config.model_path(stick)):
        if os.path.exists(path):
            os.remove(path)


def data_summary(stick):
    """(n_sessions, n_rest_samples, n_rim_samples) on disk for a stick."""
    sessions = load_sessions(stick)
    n_rest = sum(1 for s in sessions for r in s if r[3] == LABEL_REST)
    n_rim = sum(1 for s in sessions for r in s if r[3] == LABEL_RIM)
    return len(sessions), n_rest, n_rim


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    mlp: MLP
    x_std: Standardizer
    meta: dict
    report: list = field(default_factory=list)


class TrainingError(Exception):
    pass


def _new_window():
    return FeatureWindow(config.HISTORY_LEN, config.PUSH_MEMORY, config.PUSH_THRESHOLD)


def build_dataset(sessions):
    """Returns dict of arrays for every usable labeled sample, plus the
    offset/rim-radius estimates. Raises TrainingError if data is lacking."""
    rest_raw = np.array([(x, y) for s in sessions for _, x, y, lab in s
                         if lab == LABEL_REST and math.hypot(x, y) < config.REST_MAX_RADIUS])
    if len(rest_raw) < 100:
        raise TrainingError("Not enough resting data -- run a calibration first.")
    offset = np.median(rest_raw, axis=0)

    rim = np.array([(x - offset[0], y - offset[1])
                    for s in sessions for _, x, y, lab in s if lab == LABEL_RIM]).reshape(-1, 2)
    rim = rim[np.linalg.norm(rim, axis=1) >= config.RIM_MIN_RADIUS]
    rim_radius = float(np.median(np.linalg.norm(rim, axis=1))) if len(rim) >= 50 else 1.0
    reach = _axis_reach(rim, rim_radius)

    X, T, R, kind, block = [], [], [], [], []
    for si, s in enumerate(sessions):
        window = _new_window()
        for i, (_, x, y, lab) in enumerate(s):
            feats = window.push(x, y)
            if lab == LABEL_REST:
                if math.hypot(x, y) >= config.REST_MAX_RADIUS:
                    continue  # stick was touched during a "hands off" step
                target = (0.0, 0.0)
            elif lab == LABEL_RIM:
                if math.hypot(x - offset[0], y - offset[1]) < config.RIM_MIN_RADIUS:
                    continue  # not pushed all the way out
                # true direction: undo the offset and each axis's lost range
                sx, sy = offset_correct(x, y, offset, reach)
                r = math.hypot(sx, sy)
                target = (float(sx) / r, float(sy) / r)
            else:
                continue
            X.append(feats)
            T.append(target)
            R.append((x, y))
            kind.append(lab)
            block.append(si * 1_000_000 + i // BLOCK_LEN)

    return {
        "X": np.array(X), "T": np.array(T), "raw": np.array(R),
        "kind": np.array(kind), "block": np.array(block),
        "offset": offset, "rim_radius": rim_radius, "reach": reach, "n_rim": len(rim),
    }


def _axis_reach(rim, fallback, max_angle_deg=20.0):
    """How far the stick gets along +x, -x, +y, -y: median of the
    component for edge samples within max_angle_deg of each half-axis."""
    reach = []
    if len(rim):
        ang = np.degrees(np.arctan2(rim[:, 1], rim[:, 0]))
    for axis, centre in ((0, 0.0), (0, 180.0), (1, 90.0), (1, -90.0)):
        if not len(rim):
            reach.append(fallback)
            continue
        diff = np.abs((ang - centre + 180.0) % 360.0 - 180.0)
        comp = np.abs(rim[diff <= max_angle_deg, axis])
        reach.append(float(np.median(comp)) if len(comp) >= 20 else fallback)
    return reach


def _apply_model(mlp, x_std, y_std, X, raw, max_residual):
    D = mlp.predict(x_std.transform(X)) * y_std.std + y_std.mean
    mag = np.linalg.norm(D, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_residual / np.maximum(mag, 1e-12))
    return raw + D * scale


def train_stick(sessions, stick, hidden_sizes=None, epochs=None, lr=None,
                seed=None, progress=None):
    """progress(epoch, epochs, train_loss) is called periodically."""
    hidden_sizes = hidden_sizes or config.HIDDEN_SIZES
    epochs = epochs or config.EPOCHS
    lr = lr or config.LEARNING_RATE
    seed = config.SEED if seed is None else seed

    ds = build_dataset(sessions)
    X, T, raw, kind = ds["X"], ds["T"], ds["raw"], ds["kind"]
    offset, rim_radius, reach = ds["offset"], ds["rim_radius"], ds["reach"]

    # --- block-wise train/validation split ---
    rng = np.random.default_rng(seed)
    blocks = np.unique(ds["block"])
    n_val_blocks = max(1, int(round(len(blocks) * config.VAL_SPLIT)))
    val_blocks = set(rng.choice(blocks, size=n_val_blocks, replace=False).tolist())
    is_val = np.array([b in val_blocks for b in ds["block"]])
    tr, va = ~is_val, is_val
    if va.sum() == 0 or tr.sum() == 0:
        raise TrainingError("Not enough calibration data to train.")

    # --- balance rest vs rim in the training set by oversampling ---
    tr_idx = np.flatnonzero(tr)
    n_rest = int((kind[tr_idx] == LABEL_REST).sum())
    n_rim = int((kind[tr_idx] == LABEL_RIM).sum())
    if n_rim and n_rest:
        minority = LABEL_RIM if n_rim < n_rest else LABEL_REST
        factor = int(max(n_rest, n_rim) // min(n_rest, n_rim))
        if factor > 1:
            extra = np.repeat(tr_idx[kind[tr_idx] == minority], factor - 1)
            tr_idx = np.concatenate([tr_idx, extra])

    D = T - raw  # residual targets: the correction to add to the raw reading
    x_std = Standardizer().fit(X[tr_idx])
    y_std = Standardizer().fit(D[tr_idx])
    Xtr, Ytr = x_std.transform(X[tr_idx]), y_std.transform(D[tr_idx])
    Xva, Yva = x_std.transform(X[va]), y_std.transform(D[va])

    mlp = MLP([X.shape[1], *hidden_sizes, 2], seed=seed)
    best = (math.inf, None)
    for epoch in range(1, epochs + 1):
        loss = mlp.train_step(Xtr, Ytr, lr)
        if epoch % 25 == 0 or epoch == epochs:
            val = float(np.mean((mlp.predict(Xva) - Yva) ** 2))
            if val < best[0]:
                best = (val, ([w.copy() for w in mlp.W], [b.copy() for b in mlp.b]))
        if progress and (epoch % 50 == 0 or epoch == epochs):
            progress(epoch, epochs, float(loss))
    mlp.W, mlp.b = best[1]

    # --- safety clamp: largest correction seen in training, plus margin ---
    max_residual = float(np.linalg.norm(D[tr_idx], axis=1).max() * 1.25 + 0.02)

    # --- validation error in real stick units ---
    raw_va, T_va = raw[va], T[va]
    off_va = np.column_stack(offset_correct(raw_va[:, 0], raw_va[:, 1], offset, reach))
    mod_va = _apply_model(mlp, x_std, y_std, X[va], raw_va, max_residual)
    mse = lambda P: float(np.mean((P - T_va) ** 2))  # noqa: E731
    raw_mse, offset_mse, model_mse = mse(raw_va), mse(off_va), mse(mod_va)

    # --- deadzones: just big enough that corrected rest reads exactly 0 ---
    rest = kind == LABEL_REST
    rest_model = _apply_model(mlp, x_std, y_std, X[rest], raw[rest], max_residual)
    rest_off = np.column_stack(offset_correct(raw[rest, 0], raw[rest, 1], offset, reach))

    def deadzone_for(P):
        r = np.linalg.norm(P, axis=1)
        dz = float(np.percentile(r, 99.9) * 1.3)
        return float(min(config.MAX_AUTO_DEADZONE, max(config.POST_CORRECTION_DEADZONE, dz)))

    dz_model, dz_offset = deadzone_for(rest_model), deadzone_for(rest_off)
    preferred = "model" if model_mse <= offset_mse else "offset"

    meta = {
        "stick": stick,
        "residual": True,
        **_new_window().to_meta(),
        "target_standardizer": y_std.to_dict(),
        "rest_offset": [float(offset[0]), float(offset[1])],
        "rim_radius": rim_radius,
        "axis_reach": reach,
        "max_residual": max_residual,
        "deadzone_model": dz_model,
        "deadzone_offset": dz_offset,
        "preferred_mode": preferred,
        "metrics": {"raw_mse": raw_mse, "offset_mse": offset_mse, "model_mse": model_mse},
        "n_samples": int(len(X)),
        "n_sessions": len(sessions),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report = [
        f"Samples: {len(X)} labeled ({int(rest.sum())} rest, {int((~rest).sum())} edge) "
        f"from {len(sessions)} session(s)",
        f"Resting offset: ({offset[0]:+.3f}, {offset[1]:+.3f})   edge reach "
        f"+x {reach[0]:.2f}  -x {reach[1]:.2f}  +y(down) {reach[2]:.2f}  -y(up) {reach[3]:.2f}",
        f"Validation error (MSE, lower is better):",
        f"  no correction    {raw_mse:.5f}",
        f"  offset baseline  {offset_mse:.5f}",
        f"  learned model    {model_mse:.5f}",
    ]
    if model_mse <= offset_mse:
        gain = 100.0 * (1.0 - model_mse / offset_mse) if offset_mse > 0 else 0.0
        report.append(f"Model beats the offset baseline by {gain:.1f}% -> using model.")
    else:
        report.append("Model did NOT beat the offset baseline (your drift is probably "
                      "a simple fixed offset) -> using offset mode.")
    report.append(f"Auto deadzone: model {dz_model:.3f}, offset {dz_offset:.3f}")
    if ds["n_rim"] < 50:
        report.append("Note: no edge data yet -- run the full calibration for best results.")

    return TrainResult(mlp, x_std, meta, report)


def save_result(stick, result, path=None):
    path = path or config.model_path(stick)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_model(path, result.mlp, result.x_std, result.meta)
    return path
