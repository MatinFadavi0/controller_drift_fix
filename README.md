# Controller Drift Correction (from-scratch ML)

Fixes analog stick drift in software by learning the drift pattern of
your specific controller and correcting for it in real time, using a
neural network **written entirely from scratch** — no scikit-learn,
PyTorch, TensorFlow, Keras, or any other ML framework. `numpy` is used
only as a linear-algebra library (matrix multiply, elementwise math);
every forward pass, backward pass (backprop), and gradient-descent
update is hand-derived and hand-coded in `src/model.py`.

It reads your real controller, runs each drifting stick through a small
learned correction model, and re-emits everything as a virtual Xbox 360
controller that games see instead of your real pad.

```
Real controller (SDL GameController API, auto-mapped)
        |
        v
Feature window (last N raw samples + direction of last push)
        |
        v
Hand-written MLP  ---- predicts the correction (no framework)
        |
        v
raw + correction -> radial deadzone (sized from your data) -> (x, y)
        |
        v
Virtual controller (ViGEmBus / vgamepad)
        |
        v
Your game reads the virtual pad as if it were real
```

---

## Why not just subtract an offset?

A basic fix is "log the resting value, subtract it, add a deadzone."
That only works if the drift is a constant offset. In practice it
often isn't:

- **Position-dependent drift** — worn potentiometers drift more at
  some points in the stick's travel than others, and often can't reach
  the edge in some directions any more.
- **Direction-dependent drift (hysteresis)** — the stick settles at a
  different resting spot after being pushed left than after being
  pushed right.
- **Drift that changes over a session** — heat and wear make a fixed
  calibration go stale.

Training measures this for you: it reports the validation error of the
plain offset fix next to the learned model and automatically uses
whichever is better, so you're not just taking it on faith.

---

## Requirements

- **Windows** (the virtual controller uses ViGEmBus, a Windows kernel
  driver).
- Python 3.9+
- [ViGEmBus driver](https://github.com/ViGEm/ViGEmBus/releases) —
  install this first; it lets `vgamepad` create a virtual Xbox 360
  controller that Windows/games recognize.
- Your controller connected (Xbox, PlayStation, Switch Pro and most
  third-party pads are mapped automatically).

```bash
pip install -r requirements.txt
```

---

## Quick start: the app

```bash
python src/app.py
```

1. **Calibrate** the drifting stick (about a minute). The app tells you
   what to do and shows a target guide inside the stick circle:
   - hands off, so it can measure resting drift
   - push to the edge in 8 directions and let go after each, which
     captures edge reach *and* where the stick settles after each push
   - roll slowly around the edge both ways
2. Training starts automatically (a few seconds) and a report shows the
   error with no correction, with the offset fix, and with the model.
3. Press **Start correction**. Point your game at the virtual Xbox 360
   controller.

In the stick view, the **orange dot** is what your pad reports and the
**green ring** is what the game gets. Let go of the stick and the green
ring should sit dead center while the orange dot drifts.

Other controls:

| Button | What it does |
| --- | --- |
| **Rest refresh** | 13-second re-measure of resting drift; adds to your data and retrains. Do it after 15–20 min of play to also capture warm-controller drift. |
| **Change mode** | Learned model → Offset fix → Off, per stick. |
| **Reset** | Deletes that stick's data and model (click twice). |
| **Switch controller** | Cycles through connected pads (never picks the app's own virtual pad). |
| **Auto-start** | Start correction as soon as the app opens. |

Calibrate both sticks if both drift — each gets its own model.

No controller handy? `python src/app.py --simulate` runs the app with a
simulated worn stick (drag inside a circle to move it).

### Making the game use the corrected pad

If the game sees both your real pad and the virtual one, it may read
the wrong one. Options:

- Disable/hide the physical pad in the game's controller settings.
- **PlayStation pads:** many PC games don't read DualShock/DualSense
  natively at all, so they only see the virtual Xbox pad. Easy.
- **HidHide** (by the ViGEm/Nefarius folks) hides the physical pad from
  games. Add your Python interpreter to HidHide's allow-list so this app
  can still read it. With a venv, the process that opens the pad is the
  *base* Python install's `python.exe`, not `.venv\Scripts\python.exe`.

### Using DS4Windows at the same time

DS4Windows already turns a DS4 into a virtual Xbox pad (and usually
hides the real DS4 via HidHide). Two ways to combine them:

- **Recommended:** close DS4Windows and let this app read the DS4
  directly. SDL supports it natively. Allow Python in HidHide (see
  above) if the DS4 is hidden.
- **Keep DS4Windows:** this app will read DS4Windows' virtual pad as its
  input and create a *second* virtual pad with the correction. Games
  then see two Xbox pads, so make sure the game uses the second one.

---

## Terminal tools

Everything the app does is also available from the terminal, using the
same data and model files.

```bash
python src/easy_run.py                  # guided: calibrate -> train -> run live
python src/calibrate.py --stick right   # record a session (appends to data/)
python src/calibrate.py --stick right --rest   # quick resting refresh
python src/train.py --stick right       # train + baseline report
python src/main.py                      # headless live correction, both sticks
python src/main.py --passthrough        # sanity check the plumbing, no correction
python src/diagnostics.py               # raw axis/button indices (for unrecognised pads)
```

`main.py` uses each stick's model in the mode you last picked in the
app; sticks without a model pass straight through.

### Unrecognised controllers

If the app says **"config.py mapping"** instead of **"auto-mapped"**,
SDL doesn't know your pad. Run `python src/diagnostics.py`, note which
indices move, and update `AXIS_MAP` / `BUTTON_MAP` / `INVERT_LY` /
`INVERT_RY` in `src/config.py`.

---

## How the model works (`src/model.py`, `src/training.py`)

- **Architecture**: input layer → configurable hidden layers (tanh
  activation) → linear output layer (2 units). Default `[n_inputs, 12, 2]`.
- **Input features** (`src/features.py`):
  - a sliding window of the last `HISTORY_LEN` raw `(x, y)` samples
    (default 3, i.e. current + 2 previous), newest first;
  - **push memory**: the unit direction of the last time the stick was
    pushed past `PUSH_THRESHOLD`. A 24 ms window forgets which way the
    stick was pushed the moment it comes to rest, but that's exactly
    what hysteresis depends on. Push memory keeps it until the next push.
- **Labels**: "rest" samples have true position (0, 0). "Edge" samples
  have true radius 1 in the direction actually pushed, after undoing the
  resting offset and each half-axis's lost range.
- **Residual output**: the network predicts the *correction*
  (`target − raw`), and the output is `raw + correction`. Where it has
  little data it falls back towards the raw reading instead of making
  up a position. The correction is also capped at the largest one seen
  in training, so odd inputs (very fast flicks) can't cause a jump.
- **Forward pass**: plain matrix multiplies (`X @ W + b`), tanh on
  hidden layers, linear on the output layer.
- **Backward pass**: manually derived backpropagation —
  `dL/dy_hat = 2*(y_hat - Y) / (n * n_outputs)` for MSE loss, then the
  chain rule layer by layer (`tanh' = 1 - tanh(x)^2`), all coded
  explicitly in `train_step`. There's no autograd anywhere.
- **Optimizer**: full-batch vanilla gradient descent (`W -= lr * dW`).
  No Adam, no momentum, no framework optimizer — just the update rule.
  The weights with the best validation error are kept.
- **Feature scaling**: a small `Standardizer` class does
  `(x - mean) / std` on inputs *and* targets, fit on the training data
  and saved alongside the model so inference uses identical scaling.
- **Honest validation**: train/validation are split by contiguous
  0.2 s blocks, not single samples. Neighbouring samples are nearly
  identical, so a per-sample split would leak and flatter the model.
- **Deadzone**: sized from your own resting data (just above the
  residual noise left after correction), then the remaining range is
  rescaled so you keep full travel.

### Correctness checks

Because there's no framework catching bugs in the manual gradients,
`src/test_model.py` includes a **numerical gradient check**: it
perturbs individual weights by a tiny epsilon, measures the resulting
change in loss, and compares that finite-difference estimate against
the analytic gradient computed by `train_step`.

```bash
python src/test_model.py      # gradient check + training sanity test
python src/test_pipeline.py   # full calibrate -> train -> correct on a simulated worn stick
```

The gradient check should report a max relative error around `1e-10`.
(It caught a real off-by-factor-of-2 bug during development — the
loss averages over `n * n_outputs` elements, not just `n`, so the
output-layer gradient needs that same denominator.)

`test_pipeline.py` runs the whole guided calibration against a
simulated stick with offset, lost edge range, angle-dependent error and
hysteresis. It then checks against the *true* stick position that the
corrected stick reads exactly 0 at rest after pushes in every
direction, and that the model beats the offset fix at the edge.

---

## Project layout

```
controller_drift_fix/
├── README.md
├── requirements.txt
├── data/                  # calibration CSVs, one per stick (gitignored)
├── models/                # trained models + settings.json (gitignored)
└── src/
    ├── app.py             # the desktop app (start here)
    ├── easy_run.py        # guided terminal flow: calibrate -> train -> run
    ├── main.py            # headless live correction loop
    ├── calibrate.py       # terminal calibration session
    ├── train.py           # terminal training + baseline report
    ├── diagnostics.py     # raw axis/button indices for unrecognised pads
    ├── config.py          # fallback mapping, hyperparameters, paths
    ├── settings.py        # remembered controller / modes / auto-start
    ├── reader.py          # reads the real controller (auto-mapped)
    ├── virtual_pad.py     # writes the virtual controller (vgamepad)
    ├── features.py        # history window + push memory -> features
    ├── calibration.py     # the guided calibration routine
    ├── training.py        # dataset building, training, baseline report
    ├── corrector.py       # live per-stick correction (model/offset/off)
    ├── model.py           # the hand-written MLP, standardizer, save/load
    ├── simulator.py       # simulated worn stick (demo + tests)
    ├── test_model.py      # numerical gradient check + training sanity test
    └── test_pipeline.py   # end-to-end test on the simulated stick
```

---

## Limitations & things worth extending

- **No online/adaptive retraining.** Use **Rest refresh** when drift
  changes. A natural extension: detect idle moments automatically
  (an untouched stick is far stiller than a resting thumb) and retrain
  in the background.
- **Startup hysteresis.** Until you first push the stick after starting
  the app, the model can't know which way it was last pushed, so you
  get the average resting correction for that moment.
- **Windows-only as written.** `vgamepad`/ViGEmBus are Windows
  components. A Linux port would swap `virtual_pad.py` to use
  `python-uinput` to create a `/dev/uinput` virtual device instead —
  the model, features, calibration and training are OS-agnostic.
- **Small network, small data.** This is intentionally a low-capacity
  model for a low-dimensional, low-frequency problem. A few thousand
  calibration samples is plenty. Don't add hidden layers/units
  expecting it to help; it's more likely to overfit.
