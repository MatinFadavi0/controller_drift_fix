"""
settings.py

Small JSON file (models/settings.json) shared by the app and the
command-line tools: chosen controller, per-stick correction mode and the
app's auto-start flag.
"""
import json
import os

import config


def load():
    try:
        with open(config.SETTINGS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save(data):
    os.makedirs(os.path.dirname(config.SETTINGS_PATH), exist_ok=True)
    with open(config.SETTINGS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def set_mode(stick, mode):
    data = load()
    data.setdefault("modes", {})[stick] = mode
    save(data)
