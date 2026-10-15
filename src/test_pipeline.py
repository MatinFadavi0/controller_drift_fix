"""
test_pipeline.py

End-to-end check of calibration -> training -> live correction, using
the simulated drifting stick (no hardware needed). It runs the full
guided calibration with an autopilot "player", trains a model, then
plays back fresh movements and measures the error against the TRUE
stick position for: no correction, the offset fix, and the model.

Usage:
    python src/test_pipeline.py
"""
import math
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from calibration import CalibrationSession, angle_to_xy, full_script  # noqa: E402
from corrector import StickCorrector  # noqa: E402
from simulator import Autopilot, DriftModel  # noqa: E402
from training import load_sessions, save_result, save_session, train_stick  # noqa: E402

DT = 1.0 / config.SAMPLE_RATE_HZ


def run_calibration(drift, seed):
    session = CalibrationSession(full_script())
    pilot = Autopilot(seed)
    t = 0.0
    while not session.done:
        step = session.step
        true = pilot.follow(step.pose, session.time_in_step, DT)
        session.feed(t, *drift.measure(*true))
        t += DT
    return session.rows


def evaluation_path():
    """Fresh movements (true positions) the model never saw."""
    path = []
    for deg in range(0, 360, 30):  # push out and let go, all around
        tx, ty = angle_to_xy(deg + 10)
        path += [(tx * k / 10, ty * k / 10, "move") for k in range(10)]
        path += [(tx, ty, "rim")] * 60
        path += [(0.0, 0.0, "move")] * 15
        path += [(0.0, 0.0, "rest")] * 100
    for k in range(400):  # slow circles at half deflection
        path.append((*angle_to_xy(k * 3.6, 0.5), "mid"))
    return path


def evaluate(drift_seed, sessions_rows):
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "s.csv")
        for rows in sessions_rows:
            save_session("right", rows, path=data)
        t0 = time.time()
        result = train_stick(load_sessions("right", path=data), "right")
        train_s = time.time() - t0
        model_file = os.path.join(tmp, "m.json")
        save_result("right", result, path=model_file)
        for line in result.report:
            print("   ", line)
        print(f"    (trained in {train_s:.1f}s)")

        errors = {}
        for mode in ("off", "offset", "model"):
            drift = DriftModel(drift_seed)
            corr = StickCorrector("right")
            corr.load(model_file)
            corr.set_mode(mode)
            err = {"rest": [], "rim": [], "mid": []}
            rest_nonzero = 0
            for tx, ty, kind in evaluation_path():
                cx, cy = corr.correct(*drift.measure(tx, ty))
                if kind == "move":
                    continue
                err[kind].append(math.hypot(cx - tx, cy - ty))
                if kind == "rest" and (cx, cy) != (0.0, 0.0):
                    rest_nonzero += 1
            errors[mode] = {k: float(np.mean(v)) for k, v in err.items()}
            errors[mode]["rest_nonzero"] = rest_nonzero
        return result, errors


def main():
    rows = [run_calibration(DriftModel(1), seed=1)]
    print(f"Calibration captured {len(rows[0])} samples.")
    result, errors = evaluate(1, rows)

    print("\n    mean error vs TRUE position   rest     edge     half-way   rest samples != 0")
    for mode, e in errors.items():
        print(f"    {mode:>8}                    {e['rest']:.4f}   {e['rim']:.4f}   "
              f"{e['mid']:.4f}     {e['rest_nonzero']}")

    m, o, r = errors["model"], errors["offset"], errors["off"]
    assert m["rest_nonzero"] == 0, "corrected stick should read exactly 0 at rest"
    assert m["rest"] < r["rest"] * 0.1, "model should remove resting drift"
    assert m["rim"] < r["rim"], "model should improve edge accuracy"
    assert m["mid"] < r["mid"], "model should improve half-way accuracy"
    assert m["rim"] < o["rim"], "model should beat offset fix at the edge (position-dependent drift)"
    print("\nPASS: calibration -> training -> correction works end to end.")


if __name__ == "__main__":
    main()
