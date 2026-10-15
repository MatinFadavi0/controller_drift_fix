"""
config.py

All the tunable knobs live here. Controller axis/button indices vary by
OS, driver, and even cable/wireless mode, so DO NOT assume the defaults
below are correct for your pad. Run `python src/diagnostics.py` first,
wiggle every stick/button, and update AXIS_MAP / BUTTON_MAP accordingly.
"""

# ---------------------------------------------------------------------------
# Which physical joystick device to open (index into pygame's joystick list)
# ---------------------------------------------------------------------------
JOYSTICK_INDEX = 0

# ---------------------------------------------------------------------------
# Axis mapping: pygame axis index -> logical name.
# Common SDL2/XInput layout for an Xbox-style pad on Windows:
#   0 = Left stick X    (-1 left  .. +1 right)
#   1 = Left stick Y    (-1 up    .. +1 down)   <-- note: often inverted
#   2 = Right stick X
#   3 = Right stick Y
#   4 = Left trigger     (-1 released .. +1 fully pressed)
#   5 = Right trigger
# Verify with diagnostics.py -- this is the #1 thing that differs per pad.
# ---------------------------------------------------------------------------
AXIS_MAP = {
    "lx": 0,
    "ly": 1,
    "rx": 2,
    "ry": 3,
    "lt": 4,
    "rt": 5,
}

# Some pads report left-stick Y inverted relative to what feels "natural".
# Flip sign here if pushing the stick up gives you +1 instead of -1 (or
# whichever convention you want to standardize on before correction).
INVERT_LY = False
INVERT_RY = False

# ---------------------------------------------------------------------------
# Button mapping: pygame button index -> XUSB button flag name
# (names match vgamepad.XUSB_BUTTON members). Adjust after diagnostics.py.
# ---------------------------------------------------------------------------
BUTTON_MAP = {
    0: "XUSB_GAMEPAD_A",
    1: "XUSB_GAMEPAD_B",
    2: "XUSB_GAMEPAD_X",
    3: "XUSB_GAMEPAD_Y",
    9: "XUSB_GAMEPAD_LEFT_SHOULDER",
    10: "XUSB_GAMEPAD_RIGHT_SHOULDER",
    4: "XUSB_GAMEPAD_BACK",
    6: "XUSB_GAMEPAD_START",
    7: "XUSB_GAMEPAD_LEFT_THUMB",
    8: "XUSB_GAMEPAD_RIGHT_THUMB",
    11: "XUSB_GAMEPAD_DPAD_UP",
    12: "XUSB_GAMEPAD_DPAD_DOWN",
    13: "XUSB_GAMEPAD_DPAD_LEFT",
    14: "XUSB_GAMEPAD_DPAD_RIGHT",

}

# D-pad via pygame "hat" 0 -> (x, y) each in {-1, 0, 1}
HAT_INDEX = 0

# ---------------------------------------------------------------------------
# Which stick are we correcting? "left" or "right". The pipeline only
# corrects one stick at a time to keep the model small and the calibration
# routine short. Run the whole flow twice (with different --stick flags
# and model filenames) if both sticks drift.
# ---------------------------------------------------------------------------
TARGET_STICK = "right"  # "left" or "right"

# ---------------------------------------------------------------------------
# History window: how many past raw samples (including current) feed the
# model. 3 means [current, t-1, t-2] for both x and y -> 6 input features.
# This is what lets the model learn direction-dependent drift (hysteresis).
# ---------------------------------------------------------------------------
HISTORY_LEN = 3

# ---------------------------------------------------------------------------
# Sampling / loop rate for calibration capture and the live correction loop
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = 125

# ---------------------------------------------------------------------------
# Small deadzone applied AFTER correction, purely to stop residual model
# noise (not raw drift) from causing tiny virtual-stick jitter.
# ---------------------------------------------------------------------------
POST_CORRECTION_DEADZONE = 0.015

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
DATA_DIR = "data"
MODELS_DIR = "models"
REST_CSV = f"{DATA_DIR}/rest_samples.csv"
REFERENCE_CSV = f"{DATA_DIR}/reference_samples.csv"
MODEL_PATH = f"{MODELS_DIR}/drift_model.json"

# ---------------------------------------------------------------------------
# MLP architecture / training hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_SIZES = [12]      # one hidden layer of 12 units; add more ints for more layers
LEARNING_RATE = 0.05
EPOCHS = 3000
VAL_SPLIT = 0.2
SEED = 42
PRINT_EVERY = 200
