"""
Auto Pilot — follows a list of waypoints using a simple P-controller.
"""
import math

import numpy as np

from simulator.control.command_source import CommandSource, CommandVector
from simulator.physics.drone_state import DroneState
from simulator.planning.survey_planner import Waypoint


class AutoPilot(CommandSource):
    """
    Automatically flies the drone through a list of waypoints.
    """

    def __init__(self, waypoints: list[Waypoint], speed_m_s: float = 5.0, arrival_threshold_m: float = 150.0):
        self.waypoints = waypoints
        self.speed = speed_m_s
        self.arrival_threshold = arrival_threshold_m
        
        self.current_wp_idx = 0
        self._finished = len(waypoints) == 0
        
        # Smoothing state
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vz = 0.0
        self.current_yaw = None
        
        self.max_accel = 300.0  # m/s^2 (tighter turns to prevent overshooting)
        self.max_yaw_rate = 3.0  # rad/s (faster camera panning to match tight turns)

    def get_command(self, state: DroneState, dt: float) -> CommandVector:
        cmd = CommandVector()

        if self._finished or self.current_wp_idx >= len(self.waypoints):
            self._finished = True
            return cmd

        # Current target
        target = self.waypoints[self.current_wp_idx]
        
        # Calculate vector to target
        dx = target.x - state.position[0]
        dy = target.y - state.position[1]
        dz = target.z - state.position[2]
        
        dist_xy = math.hypot(dx, dy)
        dist_z = abs(dz)
        dist_total = math.hypot(dist_xy, dz)

        # Check if arrived
        if dist_total < self.arrival_threshold:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                self._finished = True
                return cmd
            target = self.waypoints[self.current_wp_idx]
            dx = target.x - state.position[0]
            dy = target.y - state.position[1]
            dz = target.z - state.position[2]
            dist_xy = math.hypot(dx, dy)
            dist_total = math.hypot(dist_xy, dz)

        # Proportional controller for velocity
        vx_desired = dx
        vy_desired = dy
        vz_desired = dz

        # Scale to max speed
        v_desired_mag = dist_total
        if v_desired_mag > self.speed:
            scale = self.speed / v_desired_mag
            vx_desired *= scale
            vy_desired *= scale
            vz_desired *= scale

        # Smooth acceleration
        dvx = vx_desired - self.current_vx
        dvy = vy_desired - self.current_vy
        dvz = vz_desired - self.current_vz
        
        dv_mag = math.hypot(math.hypot(dvx, dvy), dvz)
        max_dv = self.max_accel * dt
        if dv_mag > max_dv:
            scale = max_dv / dv_mag
            dvx *= scale
            dvy *= scale
            dvz *= scale
            
        self.current_vx += dvx
        self.current_vy += dvy
        self.current_vz += dvz

        cmd.vx = self.current_vx
        cmd.vy = self.current_vy
        cmd.vz = self.current_vz

        # Steer nose smoothly towards velocity vector
        vel_xy_mag = math.hypot(self.current_vx, self.current_vy)
        if vel_xy_mag > 0.5:
            desired_yaw = math.atan2(-self.current_vx, self.current_vy)
            if self.current_yaw is None:
                self.current_yaw = desired_yaw
                
            # Shortest angular distance
            dyaw = (desired_yaw - self.current_yaw + math.pi) % (2 * math.pi) - math.pi
            
            max_dyaw = self.max_yaw_rate * dt
            if dyaw > max_dyaw:
                dyaw = max_dyaw
            elif dyaw < -max_dyaw:
                dyaw = -max_dyaw
                
            self.current_yaw += dyaw
            self.current_yaw = (self.current_yaw + math.pi) % (2 * math.pi) - math.pi
            
        cmd.yaw_rate = self.current_yaw if self.current_yaw is not None else 0.0

        return cmd

    def is_finished(self) -> bool:
        return self._finished

    @property
    def mode_name(self) -> str:
        return "AUTO"
    
    @property
    def progress_str(self) -> str:
        return f"{self.current_wp_idx}/{len(self.waypoints)}"
