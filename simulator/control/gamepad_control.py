"""
Gamepad Control — reads DualShock 4 (or any SDL-compatible gamepad) via pygame.

Axis mapping (DualShock 4 / DirectInput on Windows):
  Axis 0 — Left stick X   (strafe left/right)
  Axis 1 — Left stick Y   (forward/backward, inverted)
  Axis 2 — Right stick X  (yaw)
  Axis 3 — Right stick Y  (pitch, inverted)
  Axis 4 — L2 trigger     (-1 released … +1 fully pressed)
  Axis 5 — R2 trigger     (-1 released … +1 fully pressed)

If the controller exposes fewer axes (e.g. some DirectInput drivers
merge triggers into one axis), the code gracefully falls back to 0.0.
"""

import math
import logging

import pygame

log = logging.getLogger(__name__)


class GamepadControl:
    """
    Reads a single gamepad and exposes flight-control values.

    This is NOT a full ``CommandSource``; it is used *inside*
    ``ManualControl`` as an additional input channel that coexists
    with the keyboard/mouse listeners.
    """

    # ---- Axis indices (DualShock 4 via SDL / DirectInput) ----
    AXIS_LEFT_X = 0
    AXIS_LEFT_Y = 1
    AXIS_RIGHT_X = 2
    AXIS_RIGHT_Y = 3
    AXIS_L2 = 4
    AXIS_R2 = 5

    # ---- Button indices ----
    BTN_OPTIONS = 9   # "Options" on DS4

    def __init__(
        self,
        deadzone: float = 0.15,
        yaw_sensitivity: float = 2.5,
        pitch_sensitivity: float = 1.5,
    ):
        self.deadzone = deadzone
        self.yaw_sensitivity = yaw_sensitivity
        self.pitch_sensitivity = pitch_sensitivity

        # Initialise only the joystick subsystem so we don't conflict
        # with the OpenCV window / pynput event loop.
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError("No gamepad detected")

        self._joy = pygame.joystick.Joystick(0)
        self._joy.init()
        log.info(
            "Gamepad connected: %s  (axes=%d, buttons=%d)",
            self._joy.get_name(),
            self._joy.get_numaxes(),
            self._joy.get_numbuttons(),
        )

        self._exit_pressed = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._joy.get_name()

    @property
    def exit_pressed(self) -> bool:
        """True if the Options (Start) button was pressed."""
        return self._exit_pressed

    def poll(
        self,
        speed_xy: float,
        speed_z: float,
        dt: float,
    ) -> dict:
        """
        Read the current gamepad state and return raw flight inputs.

        Returns a dict with keys:
            v_forward, v_right, v_up  — body-frame linear velocities (m/s)
            yaw_delta                 — yaw change this frame (radians)
            pitch_delta               — pitch change this frame (radians)
            active                    — True if any stick/trigger is deflected
        """
        # Pump the pygame event queue so axis values refresh.
        pygame.event.pump()

        left_x = self._axis(self.AXIS_LEFT_X)
        left_y = self._axis(self.AXIS_LEFT_Y)
        right_x = self._axis(self.AXIS_RIGHT_X)
        right_y = self._axis(self.AXIS_RIGHT_Y)

        # Triggers: raw range is -1 (released) … +1 (pressed).
        # Normalise to 0…1.
        l2_raw = self._axis(self.AXIS_L2)
        r2_raw = self._axis(self.AXIS_R2)
        l2 = max(0.0, (l2_raw + 1.0) / 2.0)
        r2 = max(0.0, (r2_raw + 1.0) / 2.0)

        # Apply deadzone + small threshold for triggers
        l2 = l2 if l2 > 0.05 else 0.0
        r2 = r2 if r2 > 0.05 else 0.0

        # Check exit button
        if self._joy.get_numbuttons() > self.BTN_OPTIONS:
            if self._joy.get_button(self.BTN_OPTIONS):
                self._exit_pressed = True

        # Movement (left stick)
        v_forward = -left_y * speed_xy   # up on stick = forward
        v_right = left_x * speed_xy

        # Altitude (triggers)
        v_up = (r2 - l2) * speed_z

        # Camera (right stick) — delta per frame
        yaw_delta = -right_x * self.yaw_sensitivity * dt
        pitch_delta = right_y * self.pitch_sensitivity * dt

        active = (
            abs(left_x) > 0.0
            or abs(left_y) > 0.0
            or abs(right_x) > 0.0
            or abs(right_y) > 0.0
            or l2 > 0.0
            or r2 > 0.0
        )

        return {
            "v_forward": v_forward,
            "v_right": v_right,
            "v_up": v_up,
            "yaw_delta": yaw_delta,
            "pitch_delta": pitch_delta,
            "active": active,
        }

    def poll_realistic(self) -> dict:
        """
        Read gamepad in RC Mode 2 layout and return normalised inputs.

        RC Mode 2 mapping (standard for most real drones):
          Left stick Y  → throttle  (-1 down … +1 up)
          Left stick X  → yaw rate  (-1 left … +1 right)
          Right stick Y → pitch     (-1 back … +1 forward)
          Right stick X → roll      (-1 left … +1 right)

        All values are normalised to [-1, 1] (after deadzone & curve).
        ``ManualControl`` applies speed scaling and inertia.
        """
        pygame.event.pump()

        left_x = self._axis(self.AXIS_LEFT_X)
        left_y = self._axis(self.AXIS_LEFT_Y)
        right_x = self._axis(self.AXIS_RIGHT_X)
        right_y = self._axis(self.AXIS_RIGHT_Y)

        # Check exit button
        if self._joy.get_numbuttons() > self.BTN_OPTIONS:
            if self._joy.get_button(self.BTN_OPTIONS):
                self._exit_pressed = True

        active = (
            abs(left_x) > 0.0
            or abs(left_y) > 0.0
            or abs(right_x) > 0.0
            or abs(right_y) > 0.0
        )

        return {
            "throttle": -left_y,    # up on stick = positive = climb
            "yaw": left_x,          # right on stick = positive = turn right
            "pitch": -right_y,      # up on stick = positive = forward
            "roll": -right_x,       # right on stick = positive = strafe right (inverted for correct mapping)
            "active": active,
        }

    def cleanup(self):
        """Release the joystick subsystem."""
        if self._joy:
            self._joy.quit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _axis(self, index: int) -> float:
        """Read an axis with deadzone and quadratic response curve."""
        if index >= self._joy.get_numaxes():
            return 0.0
        raw = self._joy.get_axis(index)
        if abs(raw) < self.deadzone:
            return 0.0
        # Re-scale so that the usable range starts at 0 after deadzone
        sign = 1.0 if raw > 0 else -1.0
        magnitude = (abs(raw) - self.deadzone) / (1.0 - self.deadzone)
        # Quadratic curve for finer control at low deflections
        return sign * magnitude * magnitude
