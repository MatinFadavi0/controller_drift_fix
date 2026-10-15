# -*- coding: utf-8 -*-
"""
main.py

Live drift correction pipeline using a previously trained model.
All tunable indices/flags now come from config.py -- edit that file
(after running diagnostics.py) instead of this one.

Usage:
    python src/main.py

IMPORTANT: HISTORY_LEN and the axis indices used here must match what
the model was actually trained with (easy_run.py). If you change
HISTORY_LEN in config.py, retrain the model.
"""
import os
import sys
import time

import numpy as np
import pygame
import vgamepad as vg

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from model import load_model
import config

# Set to True and run: prints every raw button index AND hat value as you
# press D-pad directions. Some controllers/drivers report D-pad as plain
# buttons instead of a hat (joy.get_numhats() == 0 in that case) -- if so,
# note which button index lights up for each direction and add it to
# config.BUTTON_MAP with the matching "XUSB_GAMEPAD_DPAD_..." name, no
# code change needed here.
DEBUG_INPUTS = False


def to_xinput_y(v, extra_invert):
    """Convert a raw pygame/SDL y value (up=-1) to XInput convention
    (up=+1). extra_invert flips it again on top, for pads where even
    this still comes out wrong."""
    out = -v
    return -out if extra_invert else out


def get_axis_pair(stick):
    """Return (x_axis, y_axis, invert_y) for 'left' or 'right' from config."""
    if stick == "left":
        return config.AXIS_MAP["lx"], config.AXIS_MAP["ly"], config.INVERT_LY
    return config.AXIS_MAP["rx"], config.AXIS_MAP["ry"], config.INVERT_RY


def run():
    if not os.path.exists(config.MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at '{config.MODEL_PATH}'. "
            f"Run easy_run.py once to calibrate and train the model."
        )

    mlp, std, meta = load_model(config.MODEL_PATH)

    target = config.TARGET_STICK
    other = "right" if target == "left" else "left"

    x_axis, y_axis, target_invert_y = get_axis_pair(target)
    ox_axis, oy_axis, other_invert_y = get_axis_pair(other)
    lt_axis, rt_axis = config.AXIS_MAP["lt"], config.AXIS_MAP["rt"]

    # Sanity check: does config's axis choice match what the model was
    # actually calibrated on? If easy_run.py stored x_axis/y_axis in meta
    # and they don't match config, warn loudly instead of failing silently.
    if "x_axis" in meta and (meta["x_axis"], meta["y_axis"]) != (x_axis, y_axis):
        print(f"WARNING: config.py says target '{target}' is axes "
              f"({x_axis},{y_axis}) but the model was trained on axes "
              f"({meta['x_axis']},{meta['y_axis']}). Update TARGET_STICK "
              f"or AXIS_MAP in config.py, or retrain.")

    print(f"Model loaded from {config.MODEL_PATH}")
    print(f"target={target} -> axes ({x_axis},{y_axis}), invert_y={target_invert_y}")
    print(f"passthrough stick={other} -> axes ({ox_axis},{oy_axis}), invert_y={other_invert_y}")
    print(f"triggers -> lt={lt_axis}, rt={rt_axis}")

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No controller detected.")
        return
    joy = pygame.joystick.Joystick(config.JOYSTICK_INDEX)
    joy.init()
    n_buttons = joy.get_numbuttons()
    n_hats = joy.get_numhats()
    print(f"Controller detected: {joy.get_name()} "
          f"(axes: {joy.get_numaxes()}, buttons: {n_buttons}, hats: {n_hats})")

    if DEBUG_INPUTS:
        if n_hats == 0:
            print("NOTE: this controller reports 0 hats -- D-pad is very "
                  "likely coming through as regular buttons instead.")
        print("DEBUG_INPUTS is ON: press D-pad directions / buttons. "
              "Ctrl+C to stop.")
        prev_buttons = [0] * n_buttons
        prev_hats = [(0, 0)] * max(n_hats, 1)
        try:
            while True:
                pygame.event.pump()
                for i in range(n_buttons):
                    val = joy.get_button(i)
                    if val != prev_buttons[i]:
                        print(f"button index {i}: {'pressed' if val else 'released'}")
                    prev_buttons[i] = val
                for h in range(n_hats):
                    hat = joy.get_hat(h)
                    if hat != prev_hats[h]:
                        print(f"hat {h}: {hat}")
                    prev_hats[h] = hat
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    gamepad = vg.VX360Gamepad()
    held = set()
    hist = []
    period = 1.0 / config.SAMPLE_RATE_HZ
    print("\nRunning live drift correction... Press Ctrl+C to stop.")

    try:
        while True:
            pygame.event.pump()
            raw_x, raw_y = joy.get_axis(x_axis), joy.get_axis(y_axis)

            hist.append((raw_x, raw_y))
            if len(hist) > config.HISTORY_LEN:
                hist.pop(0)
            while len(hist) < config.HISTORY_LEN:
                hist.insert(0, hist[0])

            feats = []
            for hx, hy in reversed(hist):
                feats += [hx, hy]
            feats_arr = std.transform(np.array([feats], dtype=np.float64))
            pred = mlp.predict(feats_arr)[0]

            cx = max(-1.0, min(1.0, float(pred[0])))
            cy = max(-1.0, min(1.0, float(pred[1])))
            if abs(cx) < config.POST_CORRECTION_DEADZONE:
                cx = 0.0
            if abs(cy) < config.POST_CORRECTION_DEADZONE:
                cy = 0.0
            # pygame/SDL reports "up" as -1, but XInput/vgamepad expects
            # "up" as +1 -- this base flip is ALWAYS required, independent
            # of the controller. config.INVERT_LY/INVERT_RY are an EXTRA
            # toggle on top of that, only needed if a specific pad still
            # comes out wrong after this base conversion.
            out_y = to_xinput_y(cy, target_invert_y)

            other_raw_x = joy.get_axis(ox_axis)
            other_raw_y = joy.get_axis(oy_axis)
            other_out_y = to_xinput_y(other_raw_y, other_invert_y)

            if target == "left":
                gamepad.left_joystick_float(x_value_float=cx, y_value_float=out_y)
                gamepad.right_joystick_float(x_value_float=other_raw_x, y_value_float=other_out_y)
            else:
                gamepad.right_joystick_float(x_value_float=cx, y_value_float=out_y)
                gamepad.left_joystick_float(x_value_float=other_raw_x, y_value_float=other_out_y)

            lt = max(0.0, min(1.0, (joy.get_axis(lt_axis) + 1) / 2))
            rt = max(0.0, min(1.0, (joy.get_axis(rt_axis) + 1) / 2))
            gamepad.left_trigger_float(value_float=lt)
            gamepad.right_trigger_float(value_float=rt)

            # Buttons, from config.BUTTON_MAP (index -> XUSB name)
            wanted = set()
            for idx, name in config.BUTTON_MAP.items():
                if idx < n_buttons and joy.get_button(idx):
                    wanted.add(name)

            # D-pad, from config.HAT_INDEX
            if n_hats > config.HAT_INDEX:
                hx, hy = joy.get_hat(config.HAT_INDEX)
                if hx == 1:
                    wanted.add("XUSB_GAMEPAD_DPAD_RIGHT")
                elif hx == -1:
                    wanted.add("XUSB_GAMEPAD_DPAD_LEFT")
                if hy == 1:
                    wanted.add("XUSB_GAMEPAD_DPAD_UP")
                elif hy == -1:
                    wanted.add("XUSB_GAMEPAD_DPAD_DOWN")

            for name in wanted - held:
                gamepad.press_button(button=getattr(vg.XUSB_BUTTON, name))
            for name in held - wanted:
                gamepad.release_button(button=getattr(vg.XUSB_BUTTON, name))
            held = wanted

            gamepad.update()
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        gamepad.reset()
        gamepad.update()


if __name__ == "__main__":
    run()