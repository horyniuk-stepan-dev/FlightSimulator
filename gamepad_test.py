"""Quick gamepad axis diagnostic — shows raw values for all axes and buttons."""
import pygame
import time

pygame.init()
pygame.joystick.init()

count = pygame.joystick.get_count()
print(f"Joysticks found: {count}")
if count == 0:
    print("No gamepad detected!")
    exit(1)

joy = pygame.joystick.Joystick(0)
joy.init()
print(f"Name: {joy.get_name()}")
print(f"Axes: {joy.get_numaxes()}")
print(f"Buttons: {joy.get_numbuttons()}")
print(f"Hats: {joy.get_numhats()}")
print()
print("Move sticks and press triggers. Press Ctrl+C to stop.")
print("=" * 70)

try:
    while True:
        pygame.event.pump()
        axes = [joy.get_axis(i) for i in range(joy.get_numaxes())]
        buttons = [joy.get_button(i) for i in range(joy.get_numbuttons())]

        # Only show axes with significant deflection
        axis_str = "  ".join(
            f"A{i}:{v:+.2f}" if abs(v) > 0.1 else f"A{i}: .  "
            for i, v in enumerate(axes)
        )
        pressed = [i for i, b in enumerate(buttons) if b]
        btn_str = f"  BTN: {pressed}" if pressed else ""

        print(f"\r{axis_str}{btn_str}      ", end="", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nDone.")
    joy.quit()
    pygame.quit()
