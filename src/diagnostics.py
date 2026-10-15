"""
diagnostics.py

Run this FIRST. It prints live raw axis/button/hat values so you can
figure out which pygame index corresponds to which physical control on
YOUR controller, then fill those into config.py (AXIS_MAP / BUTTON_MAP).

Usage:
    python src/diagnostics.py
"""
import time
import pygame


def main():
    pygame.init()
    pygame.joystick.init()

    count = pygame.joystick.get_count()
    if count == 0:
        print("No joystick/controller detected. Plug it in and try again.")
        return

    print(f"Found {count} joystick(s):")
    for i in range(count):
        j = pygame.joystick.Joystick(i)
        j.init()
        print(f"  [{i}] {j.get_name()}  "
              f"axes={j.get_numaxes()} buttons={j.get_numbuttons()} hats={j.get_numhats()}")

    idx = 0
    if count > 1:
        idx = int(input("Which index do you want to inspect? "))

    joy = pygame.joystick.Joystick(idx)
    joy.init()
    print("\nMove every stick/trigger and press every button. Ctrl+C to quit.")
    print("Watch which index changes when you move/press each control.\n")

    try:
        while True:
            pygame.event.pump()

            axes = [round(joy.get_axis(a), 3) for a in range(joy.get_numaxes())]
            buttons = [joy.get_button(b) for b in range(joy.get_numbuttons())]
            hats = [joy.get_hat(h) for h in range(joy.get_numhats())]

            pressed = [i for i, v in enumerate(buttons) if v]

            line = f"axes={axes}  pressed_buttons={pressed}  hats={hats}"
            print("\r" + line + " " * 10, end="", flush=True)

            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
