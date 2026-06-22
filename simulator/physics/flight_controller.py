"""
Flight controller — wraps RotorPy Multirotor with cmd_vel control abstraction.

Uses RotorPy's built-in velocity controller: the Multirotor class with
control_abstraction='cmd_vel' already implements a full SE3-based velocity
tracking controller internally (velocity → desired force → attitude → motor speeds).

Performance optimizations:
- Kinematic mode: inline euler→quaternion (avoids scipy.Rotation overhead)
- Reduced array copies in hot path
"""
import math

import numpy as np

from simulator.physics.drone_state import DroneState
from simulator.config import PhysicsConfig


def _euler_zxy_to_quat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Convert ZXY Euler angles to quaternion [x, y, z, w].
    
    Inline formula avoids scipy.spatial.transform.Rotation.from_euler overhead.
    ZXY intrinsic = composition q = qz * qx * qy (Hamilton product).
    
    When roll=0 (most common case), qy=identity, so q = qz * qx:
      qz = [0, 0, sin(yaw/2), cos(yaw/2)]
      qx = [sin(pitch/2), 0, 0, cos(pitch/2)]
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    
    if roll == 0.0:
        # Fast path: qz * qx (Hamilton product, roll=0 means qy=identity)
        w = cy * cp
        x = cy * sp
        y = sy * sp
        z = sy * cp
    else:
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        # Full qz * qx * qy Hamilton product
        # First: qzx = qz * qx
        w_zx = cy * cp
        x_zx = cy * sp
        y_zx = sy * sp
        z_zx = sy * cp
        
        # Then: q = qzx * qy where qy = [x=0, y=sr, z=0, w=cr]
        # Hamilton product: q1*q2
        # w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        # x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        # y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        # z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        w = w_zx * cr - y_zx * sr
        x = x_zx * cr - z_zx * sr
        y = w_zx * sr + y_zx * cr
        z = x_zx * sr + z_zx * cr
    
    return np.array([x, y, z, w], dtype=np.float64)


def _make_quad_params(cfg: PhysicsConfig) -> dict:
    """
    Build RotorPy quad_params dict for a larger drone (DJI-style).
    The default RotorPy params are for a tiny Crazyflie (30g).
    We scale up to a realistic survey drone (~3.5 kg).
    """
    d = cfg.arm_length_m  # arm length

    return {
        # Inertial properties (scaled for ~1.0kg drone)
        'mass': cfg.mass_kg,
        'Ixx': 0.01,
        'Iyy': 0.01,
        'Izz': 0.02,
        'Ixy': 0.0,
        'Iyz': 0.0,
        'Ixz': 0.0,

        # Geometry
        'num_rotors': 4,
        'rotor_pos': {
            'r1': d * np.array([0.707, 0.707, 0]),
            'r2': d * np.array([0.707, -0.707, 0]),
            'r3': d * np.array([-0.707, -0.707, 0]),
            'r4': d * np.array([-0.707, 0.707, 0]),
        },
        'rotor_directions': np.array([1, -1, 1, -1]),
        'rI': np.array([0, 0, 0]),

        # Frame drag
        'c_Dx': 0.05,
        'c_Dy': 0.05,
        'c_Dz': 0.1,

        # Rotor properties (scaled for ~10" props, enough to lift 1.0kg)
        'k_eta': 9.8e-06,
        'k_m': 1.5e-07,
        'k_d': 0.0,
        'k_z': 0.0,
        'k_h': 0.0,
        'k_flap': 0.0,

        # Motor properties
        'tau_m': 0.05,
        'rotor_speed_min': 0,
        'rotor_speed_max': 1500,
        'motor_noise_std': 0.0,

        # Lower-level controller gains
        'k_w': 50,
        'k_v': cfg.k_v,
        'kp_att': cfg.kp_att,
        'kd_att': cfg.kd_att,
    }


# Pre-allocated zero arrays to avoid repeated allocation in hot path
_ZEROS3 = np.zeros(3, dtype=np.float64)


class FlightController:
    """
    Wraps RotorPy Multirotor with velocity command interface.

    The controller receives world-frame velocity commands (vx, vy, vz)
    and RotorPy's built-in cmd_vel controller handles the rest:
    velocity error → desired force → desired attitude → motor speeds.
    """

    def __init__(self, cfg: PhysicsConfig, initial_position: np.ndarray | None = None):
        """
        Initialize the flight controller with RotorPy.

        Args:
            cfg: Physics configuration.
            initial_position: Starting position [x, y, z] in meters.
        """
        from rotorpy.vehicles.multirotor import Multirotor
        from rotorpy.controllers.quadrotor_control import SE3Control

        self._quad_params = _make_quad_params(cfg)

        pos = np.array(initial_position, dtype=np.float64) if initial_position is not None else np.array([0.0, 0.0, 100.0])

        # Compute hover rotor speed: thrust = mass * g, each rotor provides 1/4
        hover_thrust_per_rotor = self._quad_params['mass'] * 9.81 / 4.0
        hover_speed = np.sqrt(hover_thrust_per_rotor / self._quad_params['k_eta'])
        hover_speed = min(hover_speed, self._quad_params['rotor_speed_max'] * 0.95)

        initial_state = {
            'x': pos.copy(),
            'v': np.zeros(3),
            'q': np.array([0.0, 0.0, 0.0, 1.0]),  # identity quaternion [i,j,k,w]
            'w': np.zeros(3),
            'wind': np.zeros(3),
            'rotor_speeds': np.full(4, hover_speed),
        }

        self._vehicle = Multirotor(
            self._quad_params,
            initial_state=initial_state,
            control_abstraction='cmd_motor_speeds',
            aero=True,
            enable_ground=True,
        )

        self._controller = SE3Control(self._quad_params)
        
        # In SE3Control, kd_pos acts as the velocity gain if we fake the position error to 0.
        self._controller.kd_pos = cfg.k_v

        self._state = initial_state.copy()
        self._time = 0.0

        print(f"[FlightController] Initialized at position {pos}, "
              f"mass={cfg.mass_kg}kg, hover_speed={hover_speed:.1f} rad/s")

    def step(self, velocity_cmd: np.ndarray, target_yaw: float, target_pitch: float, dt: float, kinematic: bool = False) -> DroneState:
        """
        Advance physics by dt seconds with the given velocity command.

        Args:
            velocity_cmd: Desired velocity [vx, vy, vz] in world frame (m/s).
            target_yaw: Desired yaw angle in radians.
            dt: Time step in seconds.
            kinematic: If True, bypasses physics and integrates velocity directly (arcade mode).

        Returns:
            Updated DroneState.
        """
        if kinematic:
            # Direct integration — no physics simulation
            self._state['x'] += velocity_cmd * dt
            self._state['v'] = velocity_cmd
            # Inline euler→quaternion (avoids scipy.Rotation.from_euler overhead)
            self._state['q'] = _euler_zxy_to_quat(target_yaw, target_pitch, 0.0)
            self._state['w'] = _ZEROS3
            self._time += dt
            return DroneState.from_rotorpy_state_nocopy(self._state, self._time)

        # Build flat outputs for SE3Control.
        # We set 'x' to current state['x'] so pos_err = 0.
        # This makes the controller effectively a velocity controller with gain kd_pos.
        flat_output = {
            'x': self._state['x'].copy(),
            'x_dot': velocity_cmd.astype(np.float64),
            'x_ddot': _ZEROS3,
            'x_dddot': _ZEROS3,
            'x_ddddot': _ZEROS3,
            'yaw': target_yaw,
            'yaw_dot': 0.0
        }

        control = self._controller.update(self._time, self._state, flat_output)

        # Step the physics using the motor speeds output by SE3Control
        self._state = self._vehicle.step(self._state, control, dt)
        self._time += dt

        return DroneState.from_rotorpy_state(self._state, self._time)

    def get_state(self) -> DroneState:
        """Get current drone state without stepping."""
        return DroneState.from_rotorpy_state(self._state, self._time)

    def reset(self, position: np.ndarray | None = None):
        """Reset the drone to initial or given position."""
        if position is not None:
            self._state['x'] = position.copy()
        self._state['v'] = np.zeros(3)
        self._state['q'] = np.array([0.0, 0.0, 0.0, 1.0])
        self._state['w'] = np.zeros(3)
        self._time = 0.0
