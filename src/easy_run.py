# -*- coding: utf-8 -*-
"""
easy_run.py

A single-file script that performs all steps sequentially:
    1. Detect connected gamepad/controller
    2. Auto-detect target stick axes (Left or Right)
    3. Auto-detect trigger axes
    4. Calibration (rest + reference points) -- automated with countdowns
    5. Model training (fast, low epoch count)
    6. Live execution of drift correction

Run command:
    python src/easy_run.py

Follow the on-screen instructions. No need to modify config.py or run
separate scripts.

Limitation: For simplicity, buttons (A/B/X/Y, etc.) pass through using
a default index mapping (controller button index = virtual controller index,
in order of A, B, X, Y, LB, RB, Back, Start, L3, R3). If buttons are swapped,
you can run src/auto_map.py later to fine-tune button mapping.
"""
import sys
import time

import numpy as np
import pygame

sys.path.insert(0, __file__.rsplit("/", 1)[0])  # allow `import model` etc.

from model import MLP, Standardizer, save_model, load_model
import vgamepad as vg


HISTORY_LEN = 3
SAMPLE_HZ = 125
DEFAULT_BUTTON_ORDER = [
    "XUSB_GAMEPAD_A", "XUSB_GAMEPAD_B", "XUSB_GAMEPAD_X", "XUSB_GAMEPAD_Y",
    "XUSB_GAMEPAD_LEFT_SHOULDER", "XUSB_GAMEPAD_RIGHT_SHOULDER",
    "XUSB_GAMEPAD_BACK", "XUSB_GAMEPAD_START",
    "XUSB_GAMEPAD_LEFT_THUMB", "XUSB_GAMEPAD_RIGHT_THUMB",
]

REFERENCE_POINTS = [
    ("center", 0.0, 0.0),
    ("right", 1.0, 0.0),
    ("up", 0.0, -1.0),
    ("left", -1.0, 0.0),
    ("down", 0.0, 1.0),
    ("up-right", 0.7071, -0.7071),
    ("down-left", -0.7071, 0.7071),
    ("center", 0.0, 0.0),
]


def countdown(msg, seconds=3):
    print(msg)
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end=" ", flush=True)
        time.sleep(1)
    print("Go!")


def record_axes(joy, seconds):
    n = joy.get_numaxes()
    samples = [[] for _ in range(n)]
    period = 1.0 / SAMPLE_HZ
    t_end = time.time() + seconds
    while time.time() < t_end:
        pygame.event.pump()
        for a in range(n):
            samples[a].append(joy.get_axis(a))
        time.sleep(period)
    return samples


def biggest_range_axis(samples, exclude):
    ranges = [
        (max(v) - min(v)) if (v and i not in exclude) else -1
        for i, v in enumerate(samples)
    ]
    idx = max(range(len(ranges)), key=lambda i: ranges[i])
    return idx, ranges[idx]


def main():
    print("=" * 60)
    print("Automatic Gamepad Drift Corrector -- Fully Automated")
    print("=" * 60)

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No gamepad found. Please connect your controller and restart.")
        return
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"Gamepad detected: {joy.get_name()}\n")

    stick_answer = input("Which stick has drift? Press Enter for Left, or type 'r' and Enter for Right: ").strip().lower()
    target = "right" if stick_answer == "r" else "left"

    # --- 1. Detect target stick X/Y axes ---
    countdown(f"\nMove ONLY the {target.upper()} stick left and right (do not move up/down or other controls).", 3)
    samples = record_axes(joy, 2.5)
    x_axis, x_range = biggest_range_axis(samples, exclude=set())
    print(f"  -> X axis detected: axis {x_axis} (range={x_range:.2f})")

    countdown("\nNow move ONLY the same stick up and down.", 3)
    samples = record_axes(joy, 2.5)
    y_axis, y_range = biggest_range_axis(samples, exclude={x_axis})
    print(f"  -> Y axis detected: axis {y_axis} (range={y_range:.2f})")

    if x_range < 0.15 or y_range < 0.15:
        print("\nWarning: Insufficient movement detected. Make sure you are moving "
              "the correct stick and run the script again.")
        return

    def read_xy():
        pygame.event.pump()
        return joy.get_axis(x_axis), joy.get_axis(y_axis)

    # --- 2. Calibration: Rest position ---
    print("\n" + "=" * 60)
    countdown("Starting calibration. Release the stick and leave it at rest position.", 3)
    print("Recording rest data (10 seconds, do not touch the controller)...")
    rows = []
    hist = []
    t_end = time.time() + 10.0
    period = 1.0 / SAMPLE_HZ
    while time.time() < t_end:
        x, y = read_xy()
        hist.append((x, y))
        if len(hist) > HISTORY_LEN:
            hist.pop(0)
        while len(hist) < HISTORY_LEN:
            hist.insert(0, hist[0])
        feats = []
        for hx, hy in reversed(hist):
            feats += [hx, hy]
        rows.append(feats + [0.0, 0.0])
        time.sleep(period)
    print(f"  {len(rows)} rest samples recorded.")

    # --- 3. Calibration: Reference points ---
    print("\nNow hold the stick at specific target points as prompted.")
    for label, tx, ty in REFERENCE_POINTS:
        countdown(f"\nMove to: {label}  (target=({tx:+.2f},{ty:+.2f})) and hold.", 2)
        t_end = time.time() + 1.5
        while time.time() < t_end:
            x, y = read_xy()
            hist.append((x, y))
            hist.pop(0)
            feats = []
            for hx, hy in reversed(hist):
                feats += [hx, hy]
            rows.append(feats + [tx, ty])
            time.sleep(period)
        print(f"  Recorded: {label}")

    print(f"\nTotal calibration samples collected: {len(rows)}")

    # --- 4. Training ---
    print("\n" + "=" * 60)
    print("Training the model...")
    data = np.array(rows, dtype=np.float64)
    n_feat = HISTORY_LEN * 2
    X = data[:, :n_feat]
    Y = data[:, n_feat:n_feat + 2]

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * 0.2))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    Xtr, Ytr, Xval, Yval = X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]

    baseline_offset = Xval[:, :2].mean(axis=0)
    baseline_mse = float(np.mean((Xval[:, :2] - baseline_offset - Yval) ** 2))

    std = Standardizer()
    Xtr_s = std.fit_transform(Xtr)
    Xval_s = std.transform(Xval)

    mlp = MLP([n_feat, 10, 2], seed=42)
    for epoch in range(1500):
        mlp.train_step(Xtr_s, Ytr, lr=0.05)
    val_mse = float(np.mean((mlp.predict(Xval_s) - Yval) ** 2))

    print(f"  Baseline error (simple offset subtraction): {baseline_mse:.5f}")
    print(f"  Trained model error:                       {val_mse:.5f}")
    if val_mse < baseline_mse:
        print(f"  Model is {100*(1-val_mse/baseline_mse):.1f}% better than baseline. Excellent.")
    else:
        print("  Model did not outperform baseline -- your drift might be a simple fixed offset."
              " Proceeding with current model.")

    save_model("models/drift_model.json", mlp, std,
               meta={"x_axis": x_axis, "y_axis": y_axis})
    print("Model saved -> models/drift_model.json")

    # --- 5. Live Correction Execution ---
    print("\n" + "=" * 60)
    input("Press Enter to start live correction (Press Ctrl+C at any time to stop)...")

    gamepad = vg.VX360Gamepad()
    held = set()
    n_buttons = joy.get_numbuttons()
    hist = []
    print("Running live drift correction... Press Ctrl+C to stop.")
    try:
        while True:
            pygame.event.pump()
            x, y = read_xy()
            hist.append((x, y))
            if len(hist) > HISTORY_LEN:
                hist.pop(0)
            while len(hist) < HISTORY_LEN:
                hist.insert(0, hist[0])
            feats = []
            for hx, hy in reversed(hist):
                feats += [hx, hy]
            feats_arr = std.transform(np.array([feats], dtype=np.float64))
            pred = mlp.predict(feats_arr)[0]
            cx = max(-1.0, min(1.0, float(pred[0])))
            cy = max(-1.0, min(1.0, float(pred[1])))
            if abs(cx) < 0.015:
                cx = 0.0
            if abs(cy) < 0.015:
                cy = 0.0

            if target == "left":
                gamepad.left_joystick_float(x_value_float=cx, y_value_float=cy)
            else:
                gamepad.right_joystick_float(x_value_float=cx, y_value_float=cy)

            # other stick + triggers passthrough (best-effort: remaining unused axes)
            other_axes = [i for i in range(joy.get_numaxes()) if i not in (x_axis, y_axis)]
            if len(other_axes) >= 2:
                if target == "left":
                    gamepad.right_joystick_float(
                        x_value_float=joy.get_axis(other_axes[0]),
                        y_value_float=joy.get_axis(other_axes[1]),
                    )
                else:
                    gamepad.left_joystick_float(
                        x_value_float=joy.get_axis(other_axes[0]),
                        y_value_float=joy.get_axis(other_axes[1]),
                    )
            if len(other_axes) >= 4:
                lt = max(0.0, min(1.0, (joy.get_axis(other_axes[2]) + 1) / 2))
                rt = max(0.0, min(1.0, (joy.get_axis(other_axes[3]) + 1) / 2))
                gamepad.left_trigger_float(value_float=lt)
                gamepad.right_trigger_float(value_float=rt)

            wanted = set()
            for i in range(min(n_buttons, len(DEFAULT_BUTTON_ORDER))):
                if joy.get_button(i):
                    wanted.add(DEFAULT_BUTTON_ORDER[i])
            for name in wanted - held:
                gamepad.press_button(button=getattr(vg.XUSB_BUTTON, name))
            for name in held - wanted:
                gamepad.release_button(button=getattr(vg.XUSB_BUTTON, name))
            held = wanted

            gamepad.update()
            time.sleep(1.0 / SAMPLE_HZ)
    except KeyboardInterrupt:
        print("\nExecution stopped.")
    finally:
        gamepad.reset()
        gamepad.update()


if __name__ == "__main__":
    main()