# -*- coding: utf-8 -*-
"""
easy_run.py

The whole flow in one terminal session, no config editing needed:
    1. Detect the controller (auto-mapped for any pad SDL recognises)
    2. Ask which stick drifts
    3. Guided calibration (~1 minute, follow the prompts)
    4. Train the from-scratch model and compare it to the offset fix
    5. Run live drift correction through the virtual Xbox 360 pad

Run command:
    python src/easy_run.py

Prefer a window with live visuals? Use `python src/app.py` instead -- it
does all of this plus more, and shares the same models and data.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reader  # noqa: E402  (sets SDL hints before pygame starts)

import main as live  # noqa: E402
from calibrate import run_session  # noqa: E402
from train import train_and_report  # noqa: E402
from training import save_session  # noqa: E402


def main():
    print("=" * 60)
    print("Automatic Gamepad Drift Corrector")
    print("=" * 60)

    reader.init()
    devices = reader.list_devices()
    if not devices:
        print("No gamepad found. Please connect your controller and restart.")
        return
    info = devices[0]
    if len(devices) > 1:
        for d in devices:
            print(f"  [{d.index}] {d.name}")
        choice = input("Which controller? Enter the number (default 0): ").strip()
        if choice.isdigit():
            info = next((d for d in devices if d.index == int(choice)), info)
    pad_in = reader.ControllerReader(info.index)
    how = "auto-mapped" if pad_in.mapped else "raw mapping from config.py"
    print(f"Gamepad detected: {pad_in.name} ({how})\n")

    answer = input("Which stick drifts? Enter = Left, 'r' = Right, 'b' = Both: ").strip().lower()
    sticks = {"r": ["right"], "b": ["left", "right"]}.get(answer, ["left"])

    for stick in sticks:
        rows = run_session(pad_in, stick)
        if not rows:
            return
        save_session(stick, rows)
        print()
        if train_and_report(stick) is None:
            return
    pad_in.close()

    print("\n" + "=" * 60)
    input("Press Enter to start live correction (Ctrl+C to stop)...")
    live.run(device_index=info.index)


if __name__ == "__main__":
    main()
