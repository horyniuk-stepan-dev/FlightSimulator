"""
Manual Control — uses pynput to read keyboard input for WASD flight.
"""
import math
import threading

from pynput import keyboard, mouse

from simulator.control.command_source import CommandSource, CommandVector
from simulator.physics.drone_state import DroneState


class ManualControl(CommandSource):
    """
    Keyboard and mouse control for the drone.
    """

    def __init__(self, speed_xy: float = 5.0, speed_z: float = 2.0):
        self.speed_xy = speed_xy
        self.speed_z = speed_z

        self._keys_pressed = set()
        self._lock = threading.Lock()
        self._finished = False
        
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        
        # Smoothed velocities
        self._smooth_vx = 0.0
        self._smooth_vy = 0.0
        self._smooth_vz = 0.0
        
        # Mouse state
        self._mouse_pressed = False
        self._last_mouse_x = None
        self._last_mouse_y = None

        # Start keyboard listener in background
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()
        
        # Start mouse listener
        self._mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click
        )
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

    def on_mouse(self, event, x, y, flags, param):
        """OpenCV mouse callback (no longer used)"""
        pass

    def _on_mouse_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            with self._lock:
                self._mouse_pressed = pressed
                if not pressed:
                    self._last_mouse_x = None
                    self._last_mouse_y = None
                else:
                    self._last_mouse_x = x
                    self._last_mouse_y = y

    def _on_mouse_move(self, x, y):
        with self._lock:
            if self._mouse_pressed and self._last_mouse_x is not None and self._last_mouse_y is not None:
                dx = x - self._last_mouse_x
                dy = y - self._last_mouse_y
                # Ignore huge jumps
                if abs(dx) < 100 and abs(dy) < 100:
                    self.target_yaw -= dx * 0.005
                    self.target_pitch += dy * 0.005
                    # Clamp pitch to prevent looking fully upside down or looping
                    self.target_pitch = max(-math.pi/2.1, min(math.pi/2.1, self.target_pitch))
                self._last_mouse_x = x
                self._last_mouse_y = y

    def _on_press(self, key):
        with self._lock:
            self._keys_pressed.add(self._get_key_name(key))

    def _on_release(self, key):
        key_name = self._get_key_name(key)
        with self._lock:
            if key_name in self._keys_pressed:
                self._keys_pressed.remove(key_name)
            if key == keyboard.Key.esc:
                self._finished = True

    def _get_key_name(self, key):
        try:
            return key.char.lower()
        except AttributeError:
            return key.name

    def get_command(self, state: DroneState, dt: float) -> CommandVector:
        cmd = CommandVector()
        
        with self._lock:
            keys = self._keys_pressed.copy()
            target_yaw = self.target_yaw
            target_pitch = self.target_pitch

        # Local body velocities
        v_forward = 0.0
        v_right = 0.0
        v_up = 0.0

        if any(k in keys for k in ('w', 'ц')): v_forward += self.speed_xy
        if any(k in keys for k in ('s', 'і', 'ы')): v_forward -= self.speed_xy
        if any(k in keys for k in ('d', 'в')): v_right += self.speed_xy
        if any(k in keys for k in ('a', 'ф')): v_right -= self.speed_xy

        if 'space' in keys: v_up += self.speed_z
        if 'shift' in keys: v_up -= self.speed_z

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

    def is_finished(self) -> bool:
        return self._finished

    @property
    def mode_name(self) -> str:
        return "MANUAL"
