# -*- coding: utf-8 -*-
"""
calibrate.py

Terminal version of the guided calibration. Records one session for one
stick and appends it to data/<stick>_samples.csv (it never overwrites, so
running it again -- e.g. after 20 minutes of play -- adds more data).

Usage:
    python src/calibrate.py                  # full routine, config.TARGET_STICK
    python src/calibrate.py --stick left
    python src/calibrate.py --rest           # quick resting-drift refresh only
Then train with:  python src/train.py --stick <same stick>
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reader  # noqa: E402  (sets SDL hints before pygame starts)
import pygame  # noqa: E402

import config  # noqa: E402
from calibration import CalibrationSession, full_script, rest_script  # noqa: E402
from training import save_session  # noqa: E402


def run_session(pad_in, stick, quick=False):
    """Walk the user through one calibration session in the terminal.
    Returns the recorded rows, or None if cancelled with Ctrl+C."""
    session = CalibrationSession(rest_script() if quick else full_script())
    print(f"\nCalibrating the {stick.upper()} stick: "
          f"~{session.total:.0f} s. Follow the prompts. Ctrl+C cancels.\n")
    period = 1.0 / config.SAMPLE_RATE_HZ
    shown = -1
    tick = 0
    next_t = time.perf_counter()
    try:
        while not session.done:
            pygame.event.pump()
            session.feed(time.perf_counter(), *pad_in.read().stick(stick))
            if session.done:
                break
            if session.index != shown:
                shown = session.index
                print()
            if tick % 12 == 0 or session.time_in_step < period:  # ~10 updates/s
                print(f"\r  [{session.progress * 100:3.0f}%] {session.step.instruction}"
                      f"  ({session.step_remaining:3.1f}s)   ", end="", flush=True)
            tick += 1
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\n\nCancelled -- nothing saved.")
        return None
    print(f"\n\nDone: {len(session.rows)} samples recorded.")
    return session.rows


def main():
    ap = argparse.ArgumentParser(description="Record a calibration session")
    ap.add_argument("--stick", choices=reader.STICKS, default=config.TARGET_STICK)
    ap.add_argument("--rest", action="store_true", help="quick resting-drift refresh only")
    ap.add_argument("--device", type=int, default=config.JOYSTICK_INDEX)
    args = ap.parse_args()

    reader.init()
    try:
        pad_in = reader.ControllerReader(args.device)
    except RuntimeError as e:
        print(e)
        return
    print(f"Controller: {pad_in.name}")
    rows = run_session(pad_in, args.stick, quick=args.rest)
    if rows:
        save_session(args.stick, rows)
        print(f"Appended to {config.data_path(args.stick)}")
        print(f"Next: python src/train.py --stick {args.stick}")


if __name__ == "__main__":
    main()
