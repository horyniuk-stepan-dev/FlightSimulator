"""
Manual Control — polls keyboard state (see key_state.py) for WASD flight.
Optionally integrates a gamepad (DualShock 4 or any SDL-compatible controller)
via ``GamepadControl`` as a parallel input source.

Gamepad modes:
  ``arcade``    — left stick = movement, right stick = camera, triggers = altitude.
                  Instant velocity response.
  ``realistic`` — RC Mode 2: left stick = throttle + yaw, right stick = pitch + roll.
                  Inertia model: sticks control acceleration, drone coasts & decelerates.
"""
import logging
import math
import threading

from simulator.control.command_source import CommandSource, CommandVector
from simulator.control.key_state import make_key_state
from simulator.physics.drone_state import DroneState

log = logging.getLogger(__name__)


class ManualControl(CommandSource):
    """
    Keyboard, mouse, and (optionally) gamepad control for the drone.

    When a gamepad is connected and ``enable_gamepad=True``, both the
    keyboard/mouse and the gamepad work simultaneously.  If the gamepad
    sticks/triggers are deflected, their values take priority; otherwise
    keyboard input is used.
    """

    # --- Realistic-mode physics tuning ---
    # Aerodynamic drag coefficient (1/s).  Terminal velocity = speed_xy
    # because accel is computed as speed_xy * SIM_DRAG at full stick.
    # Higher = snappier stop, lower = more floaty / drift.
    # 0.8 gives a realistic "heavy drone" feel where it drifts after releasing the stick.
    SIM_DRAG = 0.8
    # Yaw rate at full left-stick X deflection (rad/s)
    SIM_YAW_RATE = 1.8
    # Climb rate at full throttle (multiplier of speed_z)
    SIM_THROTTLE_SCALE = 1.0

    def __init__(
        self,
        speed_xy: float = 5.0,
        speed_z: float = 2.0,
        enable_gamepad: bool = True,
        gamepad_mode: str = "arcade",
    ):
        self.speed_xy = speed_xy
        self.speed_z = speed_z
        self._gamepad_mode = gamepad_mode

        self._keys_pressed = set()
        self._lock = threading.Lock()
        self._finished = False
        
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        
        # Smoothed velocities (used in realistic mode)
        self._smooth_vx = 0.0
        self._smooth_vy = 0.0
        self._smooth_vz = 0.0
        
        # Mouse state
        self._mouse_pressed = False
        self._last_mouse_x = None
        self._last_mouse_y = None

        # Polled keyboard state. Deliberately NOT a pynput global hook: the
        # WH_KEYBOARD_LL hook starves the OpenCV preview window's message queue
        # while a key is held, which froze the picture at a full 30 FPS.
        # See simulator/control/key_state.py.
        self._key_state = make_key_state()

        # Mouse look is served by cv2.setMouseCallback (see self.on_mouse),
        # registered by main.py — also hook-free.

        # Gamepad (optional)
        self._gamepad = None
        if enable_gamepad:
            try:
                from simulator.control.gamepad_control import GamepadControl
                self._gamepad = GamepadControl()
                mode_label = "SIM" if gamepad_mode == "realistic" else "ARCADE"
                print(f"🎮 Gamepad connected: {self._gamepad.name}  [{mode_label}]")
            except RuntimeError:
                log.info("No gamepad detected — using keyboard/mouse only.")
            except ImportError:
                log.warning(
                    "pygame not installed — gamepad support unavailable. "
                    "Install with: pip install pygame-ce"
                )

    def on_mouse(self, event, x, y, flags, param):
        """OpenCV mouse callback — drag with the left button to look around.

        Runs on the HighGUI thread during cv2.waitKey, so it touches the same
        state under self._lock.
        """
        import cv2

        if event == cv2.EVENT_LBUTTONDOWN:
            with self._lock:
                self._mouse_pressed = True
                self._last_mouse_x = x
                self._last_mouse_y = y
            return

        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONDOWN):
            with self._lock:
                self._mouse_pressed = False
                self._last_mouse_x = None
                self._last_mouse_y = None
            return

        if event != cv2.EVENT_MOUSEMOVE:
            return

        with self._lock:
            if (
                not self._mouse_pressed
                or self._last_mouse_x is None
                or self._last_mouse_y is None
            ):
                return
            dx = x - self._last_mouse_x
            dy = y - self._last_mouse_y
            # Ignore huge jumps (window resize, pointer warp)
            if abs(dx) < 100 and abs(dy) < 100:
                self.target_yaw -= dx * 0.005
                self.target_pitch += dy * 0.005
                # Clamp pitch to prevent looking fully upside down or looping
                self.target_pitch = max(
                    -math.pi / 2.1, min(math.pi / 2.1, self.target_pitch)
                )
            self._last_mouse_x = x
            self._last_mouse_y = y

    def _poll_keys(self) -> set:
        """Read the current key set and handle ESC. Called once per frame."""
        keys = self._key_state.pressed()
        # Kept as an attribute purely so external code (diagnostics in main.py)
        # can inspect it; it is no longer written from another thread.
        self._keys_pressed = keys
        if "esc" in keys:
            self._finished = True
        return keys

    # ------------------------------------------------------------------
    # get_command — dispatches to arcade or realistic path
    # ------------------------------------------------------------------

    def get_command(self, state: DroneState, dt: float) -> CommandVector:
        if (
            self._gamepad is not None
            and self._gamepad_mode == "realistic"
        ):
            return self._get_command_realistic(state, dt)
        return self._get_command_arcade(state, dt)

    # ------------------------------------------------------------------
    # Arcade mode (instant velocity, camera on right stick)
    # ------------------------------------------------------------------

    def _get_command_arcade(self, state: DroneState, dt: float) -> CommandVector:
        cmd = CommandVector()
        
        keys = self._poll_keys()
        with self._lock:
            target_yaw = self.target_yaw
            target_pitch = self.target_pitch

        # ---- Keyboard input (body-frame) ----
        kb_forward = 0.0
        kb_right = 0.0
        kb_up = 0.0

        if any(k in keys for k in ('w', 'ц')): kb_forward += self.speed_xy
        if any(k in keys for k in ('s', 'і', 'ы')): kb_forward -= self.speed_xy
        if any(k in keys for k in ('d', 'в')): kb_right += self.speed_xy
        if any(k in keys for k in ('a', 'ф')): kb_right -= self.speed_xy

        if 'space' in keys: kb_up += self.speed_z
        if 'shift' in keys: kb_up -= self.speed_z

        # ---- Gamepad input ----
        gp_forward = 0.0
        gp_right = 0.0
        gp_up = 0.0
        gamepad_active = False

        if self._gamepad is not None:
            gp = self._gamepad.poll(self.speed_xy, self.speed_z, dt)
            gamepad_active = gp["active"]

            if gamepad_active:
                gp_forward = gp["v_forward"]
                gp_right = gp["v_right"]
                gp_up = gp["v_up"]

                # Accumulate yaw/pitch deltas from right stick
                with self._lock:
                    self.target_yaw += gp["yaw_delta"]
                    self.target_pitch += gp["pitch_delta"]
                    self.target_pitch = max(
                        -math.pi / 2.1,
                        min(math.pi / 2.1, self.target_pitch),
                    )
                    target_yaw = self.target_yaw
                    target_pitch = self.target_pitch

            if self._gamepad.exit_pressed:
                self._finished = True

        # ---- Merge: gamepad takes priority if active, else keyboard ----
        if gamepad_active:
            v_forward = gp_forward
            v_right = gp_right
            v_up = gp_up
        else:
            v_forward = kb_forward
            v_right = kb_right
            v_up = kb_up

        # Convert local body velocities to world velocities using yaw
        yaw = state.yaw
        
        target_vx = v_right * math.cos(yaw) - v_forward * math.sin(yaw)
        target_vy = v_right * math.sin(yaw) + v_forward * math.cos(yaw)
        target_vz = v_up
        
        # Arcade-style instant response (no smoothing)
        cmd.vx = target_vx
        cmd.vy = target_vy
        cmd.vz = target_vz
        cmd.yaw_rate = target_yaw 
        cmd.pitch_rate = target_pitch

        return cmd

    # ------------------------------------------------------------------
    # Realistic mode (RC Mode 2 — tilt-based physics)
    # ------------------------------------------------------------------

    def _get_command_realistic(self, state: DroneState, dt: float) -> CommandVector:
        """
        RC Mode 2 with real drone tilt physics.

        Left stick:
          Y → throttle (climb/descend rate)
          X → yaw rate (rotate drone around vertical axis)

        Right stick:
          Y → pitch angle  (tilt nose down → fly forward)
          X → roll angle   (tilt right → fly right)

        The sticks control the drone's body **tilt angles**.
        Horizontal acceleration is a *consequence* of the tilt:
          a_horizontal = g · tan(tilt_angle)
        The camera tilts together with the drone body, so the user sees
        the horizon shift when pitching/rolling — exactly like a real FPV feed.
        """
        cmd = CommandVector()

        # ---- Poll gamepad (realistic layout) ----
        gp = self._gamepad.poll_realistic()

        if self._gamepad.exit_pressed:
            self._finished = True

        # ---- Also read keyboard as fallback ----
        keys = self._poll_keys()

        kb_pitch = 0.0
        kb_roll = 0.0
        kb_throttle = 0.0
        kb_yaw = 0.0

        if any(k in keys for k in ('w', 'ц')): kb_pitch += 1.0
        if any(k in keys for k in ('s', 'і', 'ы')): kb_pitch -= 1.0
        if any(k in keys for k in ('d', 'в')): kb_roll += 1.0
        if any(k in keys for k in ('a', 'ф')): kb_roll -= 1.0
        if 'space' in keys: kb_throttle += 1.0
        if 'shift' in keys: kb_throttle -= 1.0

        # Merge: gamepad priority
        if gp["active"]:
            pitch_input = gp["pitch"]
            roll_input = gp["roll"]
            throttle_input = gp["throttle"]
            yaw_input = gp["yaw"]
        else:
            pitch_input = kb_pitch
            roll_input = kb_roll
            throttle_input = kb_throttle
            yaw_input = kb_yaw

        # ---- Yaw rate → accumulate heading ----
        with self._lock:
            self.target_yaw += yaw_input * self.SIM_YAW_RATE * dt

        # ---- Target tilt angles from right stick ----
        # Max tilt ≈ 35° (typical for DJI-style drones)
        MAX_TILT = math.radians(35.0)

        target_pitch_angle = pitch_input * MAX_TILT   # forward stick → nose down
        target_roll_angle = roll_input * MAX_TILT     # right stick → right tilt

        # Smooth the current angles toward targets (simulates servo/motor response)
        # Rate ≈ 5/s → smoother, heavier feel typical of a real camera drone
        rate = min(1.0, 5.0 * dt)
        self._current_pitch = getattr(self, '_current_pitch', 0.0)
        self._current_roll = getattr(self, '_current_roll', 0.0)
        self._current_pitch += (target_pitch_angle - self._current_pitch) * rate
        self._current_roll += (target_roll_angle - self._current_roll) * rate

        # ---- Horizontal acceleration from tilt ----
        # Real physics shape: a = G_eff · tan(angle).
        # G_eff is scaled so that at MAX_TILT the terminal velocity equals speed_xy:
        #   terminal_v = G_eff · tan(MAX_TILT) / drag = speed_xy
        #   G_eff = speed_xy · drag / tan(MAX_TILT)
        # This preserves the natural tan() response curve while matching the
        # simulation's speed scale.
        g_eff = self.speed_xy * self.SIM_DRAG / math.tan(MAX_TILT)
        accel_forward = g_eff * math.tan(self._current_pitch)
        accel_right = g_eff * math.tan(self._current_roll)

        # Convert body-frame acceleration to world frame using yaw
        yaw = self.target_yaw
        ax_world = accel_right * math.cos(yaw) - accel_forward * math.sin(yaw)
        ay_world = accel_right * math.sin(yaw) + accel_forward * math.cos(yaw)

        # ---- Integrate velocity with drag ----
        drag = self.SIM_DRAG
        self._smooth_vx += (ax_world - drag * self._smooth_vx) * dt
        self._smooth_vy += (ay_world - drag * self._smooth_vy) * dt

        # Throttle → climb rate (with light smoothing)
        target_vz = throttle_input * self.speed_z * self.SIM_THROTTLE_SCALE
        self._smooth_vz += (target_vz - self._smooth_vz) * min(1.0, 5.0 * dt)

        # Clamp horizontal speed
        max_v = self.speed_xy * 1.2
        speed_h = math.hypot(self._smooth_vx, self._smooth_vy)
        if speed_h > max_v:
            scale = max_v / speed_h
            self._smooth_vx *= scale
            self._smooth_vy *= scale

        cmd.vx = self._smooth_vx
        cmd.vy = self._smooth_vy
        cmd.vz = self._smooth_vz
        cmd.yaw_rate = self.target_yaw
        # Camera pitch/roll follow the drone body
        cmd.pitch_rate = self._current_pitch
        cmd.roll = self._current_roll

        return cmd

    # ------------------------------------------------------------------

    def is_finished(self) -> bool:
        return self._finished

    @property
    def mode_name(self) -> str:
        if self._gamepad is not None:
            if self._gamepad_mode == "realistic":
                return "GAMEPAD (SIM)"
            return "GAMEPAD"
        return "MANUAL"
