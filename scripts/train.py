# -*- coding: utf-8 -*-
"""
train.py

Trains one stick's correction model from every calibration session
recorded for it, prints the offset-baseline vs model comparison, and
saves models/<stick>_model.json.

Usage:
    python scripts/train.py                 # config.TARGET_STICK
    python scripts/train.py --stick left
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import config  # noqa: E402
import settings  # noqa: E402
from training import TrainingError, load_sessions, save_result, train_stick  # noqa: E402


def train_and_report(stick):
    """Train + save + print the report. Returns the TrainResult or None."""
    sessions = load_sessions(stick)
    print(f"Training {stick} stick model from {len(sessions)} session(s)...")

    def progress(epoch, epochs, loss):
        if epoch % config.PRINT_EVERY == 0 or epoch == epochs:
            print(f"  epoch {epoch:5d}/{epochs}  train loss {loss:.5f}")

    try:
        result = train_stick(sessions, stick, progress=progress)
    except TrainingError as e:
        print(f"Training failed: {e}")
        return None
    print()
    for line in result.report:
        print(line)
    print(f"Saved -> {save_result(stick, result)}")
    settings.set_mode(stick, result.meta["preferred_mode"])
    return result


def main():
    ap = argparse.ArgumentParser(description="Train a stick's drift correction model")
    ap.add_argument("--stick", choices=("left", "right"), default=config.TARGET_STICK)
    args = ap.parse_args()
    train_and_report(args.stick)


if __name__ == "__main__":
    main()
