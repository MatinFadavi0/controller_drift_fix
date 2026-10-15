"""
config.py

All the tunable knobs live here. Most controllers are recognised by SDL's
GameController database, in which case sticks/triggers/buttons are mapped
automatically and AXIS_MAP / BUTTON_MAP below are NOT used. They only
matter for unrecognised pads (the app tells you which mode it's in). For
those, run `python scripts/diagnostics.py`, wiggle every stick/button, and
update AXIS_MAP / BUTTON_MAP accordingly.
"""
import os

# ---------------------------------------------------------------------------
# Which physical joystick device to open by default (index into pygame's
# joystick list). The GUI app lets you pick from a list instead.
# ---------------------------------------------------------------------------
JOYSTICK_INDEX = 0

# ---------------------------------------------------------------------------
# Raw-joystick fallback only: pygame axis index -> logical name.
# Common SDL2/XInput layout for an Xbox-style pad on Windows:
#   0 = Left stick X    (-1 left  .. +1 right)
#   1 = Left stick Y    (-1 up    .. +1 down)
#   2 = Right stick X
#   3 = Right stick Y
#   4 = Left trigger     (-1 released .. +1 fully pressed)
#   5 = Right trigger
# ---------------------------------------------------------------------------
AXIS_MAP = {
    "lx": 0,
    "ly": 1,
    "rx": 2,
    "ry": 3,
    "lt": 4,
    "rt": 5,
}

# Raw-joystick fallback only: flip a stick's Y if up/down comes out
# backwards. Everything downstream uses the SDL convention (up = -1).
INVERT_LY = False
INVERT_RY = False

# ---------------------------------------------------------------------------
# Raw-joystick fallback only: pygame button index -> XUSB button flag name
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

# D-pad via pygame "hat" 0 -> (x, y) each in {-1, 0, 1} (raw fallback only)
HAT_INDEX = 0

# ---------------------------------------------------------------------------
# Default stick for the command-line tools (calibrate.py / train.py).
# The GUI app and main.py handle both sticks, each with its own model.
# ---------------------------------------------------------------------------
TARGET_STICK = "right"  # "left" or "right"

# ---------------------------------------------------------------------------
# Feature window: how many past raw samples (including current) feed the
# model. 3 means [current, t-1, t-2] for both x and y.
#
# A 3-sample window is only 24 ms at 125 Hz, so once the stick is back at
# rest it can no longer see which way it was last pushed. PUSH_MEMORY adds
# 2 more features: the unit direction of the last time the stick was
# pushed past PUSH_THRESHOLD. That is what lets the model learn
# direction-dependent resting drift (hysteresis).
# ---------------------------------------------------------------------------
HISTORY_LEN = 3
PUSH_MEMORY = True
PUSH_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Sampling / loop rate for calibration capture and the live correction loop
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = 125

# ---------------------------------------------------------------------------
# Smallest radial deadzone applied AFTER correction. Training measures how
# much residual noise is left at rest and raises this per stick if needed
# (capped at MAX_AUTO_DEADZONE), so the corrected stick reads exactly 0
# when you let go. The remaining range is rescaled so you keep full travel.
# ---------------------------------------------------------------------------
POST_CORRECTION_DEADZONE = 0.015
MAX_AUTO_DEADZONE = 0.12

# ---------------------------------------------------------------------------
# Calibration sample filters (in raw stick units)
# ---------------------------------------------------------------------------
REST_MAX_RADIUS = 0.6   # "rest" samples further out than this = you touched it
RIM_MIN_RADIUS = 0.7    # "edge" samples closer than this = not pushed fully

# ---------------------------------------------------------------------------
# File locations (absolute, so scripts work from any working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SETTINGS_PATH = os.path.join(MODELS_DIR, "settings.json")


def data_path(stick):
    """Calibration samples for one stick (appended to across sessions)."""
    return os.path.join(DATA_DIR, f"{stick}_samples.csv")


def model_path(stick):
    """Trained correction model for one stick."""
    return os.path.join(MODELS_DIR, f"{stick}_model.json")


# ---------------------------------------------------------------------------
# MLP architecture / training hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_SIZES = [12]      # one hidden layer of 12 units; add more ints for more layers
LEARNING_RATE = 0.1
EPOCHS = 3000
VAL_SPLIT = 0.2
SEED = 42
PRINT_EVERY = 200
