# -*- coding: utf-8 -*-
"""
main.py

Headless live drift correction (no window) using the models trained by
the app / easy_run.py / train.py. Each stick with a model is corrected in
the mode chosen in the app (saved in models/settings.json); a stick
without a model passes straight through. Triggers and buttons always pass
through.

Usage:
    python src/main.py                 # correct with trained models
    python src/main.py --passthrough   # no correction: sanity check the plumbing
    python src/main.py --device 1      # pick another controller index
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reader  # noqa: E402  (sets SDL hints before pygame starts)
import pygame  # noqa: E402

import config  # noqa: E402
import settings as settings_file  # noqa: E402
from corrector import StickCorrector  # noqa: E402
from reader import STICKS  # noqa: E402
from virtual_pad import open_virtual_pad  # noqa: E402


def pick_device(settings, index=None):
    devices = reader.list_devices()
    if not devices:
        return None
    if index is not None:
        return next((d for d in devices if d.index == index), None)
    want = settings.get("device_guid")
    return next((d for d in devices if d.guid == want),
                next((d for d in devices if d.index == config.JOYSTICK_INDEX), devices[0]))


def run(passthrough=False, device_index=None):
    reader.init()
    settings = settings_file.load()

    correctors = {s: StickCorrector(s) for s in STICKS}
    for s, corr in correctors.items():
        if passthrough:
            print(f"{s:>5} stick: passthrough (--passthrough)")
            continue
        if corr.load():
            saved = settings.get("modes", {}).get(s)
            if saved:
                corr.set_mode(saved)
            m = corr.meta.get("metrics", {})
            print(f"{s:>5} stick: {corr.mode} mode  (model MSE {m.get('model_mse', 0):.5f}, "
                  f"offset MSE {m.get('offset_mse', 0):.5f}, trained {corr.meta.get('trained_at')})")
        else:
            print(f"{s:>5} stick: no model -> passthrough (calibrate it in the app or easy_run.py)")

    info = pick_device(settings, device_index)
    if info is None:
        print("No controller detected. Plug it in and try again.")
        return
    # open the real controller BEFORE creating the virtual one, so we can
    # never end up reading our own output
    pad_in = reader.ControllerReader(info.index)
    print(f"Controller: {pad_in.name} "
          f"({'auto-mapped' if pad_in.mapped else 'raw mapping from config.py'})")

    pad_out, err = open_virtual_pad()
    if err:
        print(err)
        return

    period = 1.0 / config.SAMPLE_RATE_HZ
    print("\nRunning live drift correction... Press Ctrl+C to stop.")
    next_t = time.perf_counter()
    try:
        while True:
            pygame.event.pump()
            if not pad_in.attached():
                print("Controller disconnected.")
                break
            state = pad_in.read()
            left = correctors["left"].correct(state.lx, state.ly)
            right = correctors["right"].correct(state.rx, state.ry)
            pad_out.send(state, left, right)

            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()  # fell behind; don't try to catch up
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pad_out.close()
        pad_in.close()


def main():
    ap = argparse.ArgumentParser(description="Headless live drift correction")
    ap.add_argument("--passthrough", action="store_true", help="disable correction")
    ap.add_argument("--device", type=int, default=None, help="controller index")
    args = ap.parse_args()
    run(passthrough=args.passthrough, device_index=args.device)


if __name__ == "__main__":
    main()
