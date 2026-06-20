"""
Command Source interface — defines how control inputs are provided to the drone.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from simulator.physics.drone_state import DroneState


@dataclass
class CommandVector:
    """Control command sent to the flight controller."""
    vx: float = 0.0      # Velocity in world X (East) m/s
    vy: float = 0.0      # Velocity in world Y (North) m/s
    vz: float = 0.0      # Velocity in world Z (Up) m/s
    yaw_rate: float = 0.0  # target yaw
    pitch_rate: float = 0.0 # target pitch

    def to_numpy(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])


class CommandSource(ABC):
    """Abstract base class for all control sources (Manual, Auto)."""

    @abstractmethod
    def get_command(self, state: DroneState, dt: float) -> CommandVector:
        """
        Compute the command for the current timestep.

        Args:
            state: Current drone state.
            dt: Time step in seconds.

        Returns:
            CommandVector with desired velocities.
        """
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        """Return True if the mission/session is complete."""
        pass

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Return the name of the control mode (e.g., 'MANUAL', 'AUTO')."""
        pass
