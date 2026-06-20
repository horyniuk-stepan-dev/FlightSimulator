"""
Semi-Auto Control — manual flight but with a drawn survey path to follow.
"""
import math

from simulator.control.command_source import CommandSource, CommandVector
from simulator.control.manual_control import ManualControl
from simulator.physics.drone_state import DroneState
from simulator.planning.survey_planner import Waypoint


class SemiAutoControl(CommandSource):
    """
    Combines ManualControl for flight with waypoint tracking for the HUD.
    """

    def __init__(self, manual_control: ManualControl, waypoints: list[Waypoint], arrival_threshold_m: float = 20.0):
        self.manual_control = manual_control
        self.waypoints = waypoints
        self.arrival_threshold = arrival_threshold_m
        
        self.current_wp_idx = 0

    def get_command(self, state: DroneState, dt: float) -> CommandVector:
        cmd = self.manual_control.get_command(state, dt)

        if self.current_wp_idx < len(self.waypoints):
            target = self.waypoints[self.current_wp_idx]
            
            dx = target.x - state.position[0]
            dy = target.y - state.position[1]
            dz = target.z - state.position[2]
            
            dist_xy = math.hypot(dx, dy)

            # Check if arrived (using horizontal distance and generous Z distance for manual flight)
            if dist_xy < self.arrival_threshold and abs(dz) < 50.0:
                self.current_wp_idx += 1
                
        return cmd

    def is_finished(self) -> bool:
        return self.manual_control.is_finished()

    @property
    def mode_name(self) -> str:
        return "SEMI-AUTO"
    
    @property
    def progress_str(self) -> str:
        return f"{self.current_wp_idx}/{len(self.waypoints)}"
