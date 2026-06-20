"""
Survey Planner — generates a boustrophedon (lawnmower) flight path.
"""
import math
from dataclasses import dataclass

import numpy as np

from simulator.camera.camera_model import CameraModel


@dataclass
class Waypoint:
    x: float
    y: float
    z: float


class SurveyPlanner:
    """Generates a sweep pattern over a bounding box."""

    @staticmethod
    def generate_path(
        bounds_local: tuple[float, float, float, float],
        altitude_m: float,
        camera: CameraModel,
        overlap_percent: float,
        grid_angle_deg: float,
    ) -> list[Waypoint]:
        """
        Generate a lawnmower path over the given area.

        Args:
            bounds_local: (x_min, y_min, x_max, y_max) in meters.
            altitude_m: Flight altitude in meters.
            camera: Camera model to compute footprint.
            overlap_percent: Overlap between adjacent sweeps (0 to 100).
            grid_angle_deg: Angle of the grid (0 = North-South sweeps).

        Returns:
            List of Waypoints.
        """
        x_min, y_min, x_max, y_max = bounds_local

        # Compute camera footprint width and height
        footprint_w, footprint_h = camera.footprint_meters(altitude_m)
        
        # Calculate line spacing based on overlap
        overlap_ratio = max(0.0, min(overlap_percent / 100.0, 0.99))
        line_spacing = footprint_w * (1.0 - overlap_ratio)
        
        # Make sure line_spacing is reasonable
        line_spacing = max(line_spacing, 1.0)

        # For simplicity, if grid_angle_deg is 0, we do North-South sweeps (vary Y, step X).
        # We will generate the grid in a local unrotated frame, then rotate it.
        
        # Center of the bounds
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        # Width and height of the area (subtract margin so camera never sees outside the map)
        width = max(0.0, (x_max - x_min) - footprint_w)
        height = max(0.0, (y_max - y_min) - footprint_h)

        # Number of lines
        num_lines = int(math.ceil(width / line_spacing))
        
        waypoints = []
        angle_rad = math.radians(grid_angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        for i in range(num_lines + 1):
            # X coordinate in unrotated frame (centered)
            local_x = -width / 2.0 + i * line_spacing
            
            # Y coordinates for the ends of the sweep
            y_start = -height / 2.0
            y_end = height / 2.0
            
            # Alternate direction
            if i % 2 == 1:
                y_start, y_end = y_end, y_start

            # Rotate and translate back to map coordinates
            # Point 1
            x1 = cx + local_x * cos_a - y_start * sin_a
            y1 = cy + local_x * sin_a + y_start * cos_a
            waypoints.append(Waypoint(x1, y1, altitude_m))

            # Point 2
            x2 = cx + local_x * cos_a - y_end * sin_a
            y2 = cy + local_x * sin_a + y_end * cos_a
            waypoints.append(Waypoint(x2, y2, altitude_m))

        return waypoints
