# Controller Drift Fix

Fixes analog stick drift in software. It learns how your controller drifts
and corrects it in real time, then outputs a virtual Xbox 360 controller
for your games.

The correction model is a small neural network written from scratch:
numpy only, hand-written backprop and gradient descent, no ML frameworks.

## Features

- Guided one-minute calibration with on-screen prompts
- Handles drift a simple offset can't: position-dependent drift, lost
  edge range, and hysteresis (resting point changes with the last push direction)
- Compares itself against a plain offset fix and uses whichever is better
- Live view of raw vs corrected stick
- Corrects both sticks; works with Xbox, PlayStation, Switch Pro and most other pads

## Requirements

- Windows
- Python 3.9+
- [ViGEmBus driver](https://github.com/ViGEm/ViGEmBus/releases)

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

1. Click **Calibrate** on the drifting stick and follow the prompts.
2. Training runs automatically and shows the results.
3. Click **Start correction** and set your game to use the virtual Xbox 360 controller.

Use **Rest refresh** after the controller has warmed up to capture
changes in drift. Try it without a controller: `python app.py --simulate`.

If your game sees both the real and virtual pad, hide the real one with
[HidHide](https://github.com/nefarius/HidHide) (allow Python in its list).
If you use DS4Windows, close it first.

### Command line

```bash
python scripts/easy_run.py                 # calibrate, train and run in one go
python scripts/calibrate.py --stick right  # record calibration data
python scripts/train.py --stick right      # train a model
python scripts/main.py                     # run correction without the window
```

## How it works

Each stick sample goes through a small MLP (tanh hidden layer, linear
output). Its inputs are the last few raw readings plus the direction of
the last push. It predicts a correction that is added to the raw value.
A deadzone sized from your calibration data then makes sure the stick
reads exactly zero at rest.

See [src/model.py](src/model.py) for the network and [src/training.py](src/training.py) for training.

## Tests

```bash
python tests/test_model.py      # numerical gradient check
python tests/test_pipeline.py   # end-to-end test on a simulated drifting stick
```
