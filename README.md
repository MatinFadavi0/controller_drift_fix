# Controller Drift Correction (from-scratch ML)

Fixes analog stick drift in software by learning the drift pattern of
your specific controller and correcting for it in real time, using a
neural network **written entirely from scratch** — no scikit-learn,
PyTorch, TensorFlow, Keras, or any other ML framework. `numpy` is used
only as a linear-algebra library (matrix multiply, elementwise math);
every forward pass, backward pass (backprop), and gradient-descent
update is hand-derived and hand-coded in `src/model.py`.

It intercepts your real controller's input, runs it through a small
learned correction model, and re-emits it as a virtual Xbox 360
controller that games see instead of your real pad.

```
Real controller (pygame/XInput)
        |
        v
History window (last N raw samples, for hysteresis)
        |
        v
Hand-written MLP  ---- (regression/classification math, no framework)
        |
        v
Corrected (x, y)
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
  some points in the stick's travel than others.
- **Direction-dependent drift (hysteresis)** — the error right after
  moving left differs from the error right after moving right, even
  at the same resting spot.
- **Drift that changes over a session** — heat and wear make a fixed
  calibration go stale.

`train.py` actually measures this for you: it computes the naive
"subtract the mean offset" baseline MSE and reports whether the
learned model beats it, so you're not just taking it on faith.

---

## Requirements

- **Windows** (the virtual-controller step uses ViGEmBus, a Windows
  kernel driver; this project is Windows-only as written).
- Python 3.9+
- [ViGEmBus driver](https://github.com/ViGEm/ViGEmBus/releases) —
  install this first, it's what lets `vgamepad` create a virtual Xbox
  360 controller that Windows/games recognize.
- Your real controller connected.

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. Find your controller's axis/button layout

Controller axis ordering isn't standardized across pads/drivers. Run:

```bash
python src/diagnostics.py
```

Wiggle each stick and trigger, press each button, and note which
printed index moves. Update `AXIS_MAP` and `BUTTON_MAP` near the top of
`src/config.py` to match. Also set `INVERT_LY` / `INVERT_RY` if a
stick's up/down feels backwards.

Set `TARGET_STICK` in `config.py` to `"left"` or `"right"` — whichever
one is drifting.

### 2. Collect calibration data

Two capture modes, run from the project root:

```bash
# Capture drift while the stick is untouched (true position = center).
# Do this a few times: right after startup, and again after 15-20 min
# of play, to also capture thermal drift.
python src/calibrate.py rest --seconds 45

# Capture drift at known reference positions (center, 8 directions,
# 2 deflection depths). You'll be prompted to hold each position.
python src/calibrate.py reference
```

Both append to CSV files in `data/`. Run `reference` a couple of times
if you want more data — it appends, it doesn't overwrite.

### 3. Train the correction model

```bash
python src/train.py
```

This builds a train/validation split, standardizes the features,
trains the hand-written MLP with full-batch gradient descent, and
prints the naive-baseline MSE alongside the model's validation MSE so
you can see the actual improvement. Saves weights to
`models/drift_model.json`.

If the model doesn't beat the baseline: you probably need more
`reference` data, or the network is overfitting a tiny dataset — try
lowering `HIDDEN_SIZES` in `config.py` (e.g. `[6]` instead of `[12]`).

### 4. Run it live

```bash
python src/main.py
```

Real controller in, corrected virtual controller out. Point your game
at the new virtual Xbox 360 controller instead of your physical pad
(most games auto-detect whichever pad is "active"; if both are
plugged in, you may need to disable/hide the physical one in the
game's controller settings, or unplug it if wired).

Sanity check the plumbing without the model:

```bash
python src/main.py --passthrough
```

---

## How the model works (`src/model.py`)

- **Architecture**: input layer → configurable hidden layers (tanh
  activation) → linear output layer (2 units: corrected x, y).
- **Input features**: a sliding window of the last `HISTORY_LEN` raw
  `(x, y)` samples (default 3, i.e. 6 numbers: current + 2 previous).
  Feeding recent history rather than just the current sample is what
  lets the model pick up on hysteresis, since it can see which
  direction the stick was just moving in.
- **Forward pass**: plain matrix multiplies (`X @ W + b`), tanh on
  hidden layers, linear on the output layer.
- **Backward pass**: manually derived backpropagation —
  `dL/dy_hat = 2*(y_hat - Y) / (n * n_outputs)` for MSE loss, then the
  chain rule layer by layer (`tanh' = 1 - tanh(x)^2`), all coded
  explicitly in `train_step`. There's no autograd anywhere.
- **Optimizer**: full-batch vanilla gradient descent (`W -= lr * dW`).
  No Adam, no momentum, no framework optimizer — just the update rule.
- **Feature scaling**: a small `Standardizer` class does
  `(x - mean) / std`, fit on the training data and saved alongside the
  model so inference uses identical scaling.

### Correctness check

Because there's no framework catching bugs in the manual gradients,
`src/test_model.py` includes a **numerical gradient check**: it
perturbs individual weights by a tiny epsilon, measures the resulting
change in loss, and compares that finite-difference estimate against
the analytic gradient computed by `train_step`. Run it:

```bash
python src/test_model.py
```

It should report a max relative error around `1e-10` and confirm the
training loop actually drives loss down on a synthetic task. (This
check caught a real off-by-factor-of-2 bug during development — the
loss averages over `n * n_outputs` elements, not just `n`, so the
output-layer gradient needs that same denominator. Worth knowing if
you extend the math yourself.)

---

## Project layout

```
controller_drift_fix/
├── README.md
├── requirements.txt
├── data/                  # calibration CSVs land here (gitignored)
├── models/                # trained weights land here (gitignored)
└── src/
    ├── config.py          # axis/button mapping, hyperparameters
    ├── diagnostics.py      # discover your pad's axis/button indices
    ├── reader.py           # reads the real controller (pygame)
    ├── virtual_pad.py      # writes the virtual controller (vgamepad)
    ├── features.py         # rolling history window -> feature vector
    ├── calibrate.py        # rest + reference-point data collection
    ├── model.py            # the hand-written MLP, standardizer, save/load
    ├── train.py            # trains the model, reports baseline comparison
    ├── test_model.py       # numerical gradient check + sanity training test
    └── main.py             # the live correction loop
```

---

## Limitations & things worth extending

- **One stick at a time.** The pipeline corrects `TARGET_STICK` only.
  If both sticks drift, run the whole calibrate → train flow twice
  with different `MODEL_PATH`/`TARGET_STICK` settings and load both
  models in `main.py`.
- **No online/adaptive retraining.** The model is trained once from a
  calibration session. If drift worsens significantly over weeks, redo
  calibration and retraining. A natural extension: periodically log
  live rest-position samples during idle moments and do a short
  incremental `train_step` pass to track drift as it evolves.
- **Windows-only as written.** `vgamepad`/ViGEmBus are Windows
  components. A Linux port would swap `virtual_pad.py` to use
  `python-uinput` to create a `/dev/uinput` virtual device instead —
  the model and feature code are OS-agnostic and would carry over
  unchanged.
- **Small network, small data.** This is intentionally a low-capacity
  model for a low-dimensional, low-frequency problem — a few hundred
  to a few thousand calibration samples is plenty. Don't add hidden
  layers/units expecting it to help; it's more likely to overfit.
