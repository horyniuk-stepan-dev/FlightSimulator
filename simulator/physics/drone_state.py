"""
Drone state dataclass — represents the complete state of the simulated drone.

Performance optimizations:
- Inline math for speed_3d (avoids np.linalg.norm dispatch overhead)
- from_rotorpy_state_nocopy() for read-only usage (avoids 5x .copy())
"""
from dataclasses import dataclass, field

import math
import numpy as np

# Pre-allocated for from_rotorpy_state_nocopy fallback
_ZEROS4 = np.zeros(4, dtype=np.float64)


@dataclass
class DroneState:
    """Complete state of the simulated drone."""
    # Position in local coordinates (meters, x=east, y=north, z=up)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Velocity in world frame (m/s)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Quaternion [i, j, k, w] (scipy/RotorPy convention)
    quaternion: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    # Angular velocity in body frame (rad/s)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Rotor speeds (rad/s)
    rotor_speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))
    # Simulation time (seconds)
    time: float = 0.0

    @property
    def yaw(self) -> float:
        """Extract yaw angle from quaternion (radians)."""
        q = self.quaternion
        # Fast formula directly from quaternion [x, y, z, w]
        return math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1]**2 + q[2]**2))

    @property
    def speed(self) -> float:
        """Horizontal speed magnitude (m/s)."""
        v = self.velocity
        return math.sqrt(v[0] * v[0] + v[1] * v[1])

    @property
    def speed_3d(self) -> float:
        """Total 3D speed (m/s)."""
        v = self.velocity
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    @property
    def altitude(self) -> float:
        """Altitude (z coordinate, meters)."""
        return float(self.position[2])

    def to_rotorpy_state(self) -> dict:
        """Convert to RotorPy state dictionary format."""
        return {
            'x': self.position.copy(),
            'v': self.velocity.copy(),
            'q': self.quaternion.copy(),
            'w': self.angular_velocity.copy(),
            'wind': np.zeros(3),
            'rotor_speeds': self.rotor_speeds.copy(),
        }

    @staticmethod
    def from_rotorpy_state(state: dict, time: float = 0.0) -> 'DroneState':
        """Create DroneState from RotorPy state dictionary (with copies for safety)."""
        return DroneState(
            position=state['x'].copy(),
            velocity=state['v'].copy(),
            quaternion=state['q'].copy(),
            angular_velocity=state['w'].copy(),
            rotor_speeds=state.get('rotor_speeds', _ZEROS4).copy(),
            time=time,
        )

    @staticmethod
    def from_rotorpy_state_nocopy(state: dict, time: float = 0.0) -> 'DroneState':
        """Create DroneState without copying arrays.
        
        Use when the source state dict won't be modified before the DroneState
        is consumed (e.g. kinematic mode where state is written fresh each step).
        Saves ~0.2ms per call by avoiding 5 array copies.
        """
        return DroneState(
            position=state['x'],
            velocity=state['v'],
            quaternion=state['q'],
            angular_velocity=state['w'],
            rotor_speeds=state.get('rotor_speeds', _ZEROS4),
            time=time,
        )
