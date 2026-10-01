"""
Survey Planner — generates a boustrophedon (lawnmower) flight path.
"""
import math
from dataclasses import dataclass

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
        margin_altitude_m: float | None = None,
        max_pitch_deg: float = 0.0,
        max_roll_deg: float = 0.0,
    ) -> list[Waypoint]:
        """
        Generate a lawnmower path over the given area.

        Args:
            bounds_local: (x_min, y_min, x_max, y_max) in meters.
            altitude_m: Flight altitude in meters.
            camera: Camera model to compute footprint.
            overlap_percent: Overlap between adjacent sweeps (0 to 100).
            grid_angle_deg: Angle of the grid (0 = North-South sweeps).
            margin_altitude_m: Highest altitude during flight for edge safety margins.
            max_pitch_deg: Maximum pitch magnitude (degrees) during flight.
            max_roll_deg: Maximum roll magnitude (degrees) during flight.

        Returns:
            List of Waypoints.
        """
        x_min, y_min, x_max, y_max = bounds_local

        # Compute camera footprint width and height at flight altitude
        footprint_w, footprint_h = camera.footprint_meters(altitude_m)
        
        # Calculate line spacing based on overlap
        overlap_ratio = max(0.0, min(overlap_percent / 100.0, 0.99))
        line_spacing = footprint_w * (1.0 - overlap_ratio)
        line_spacing = max(line_spacing, 1.0)

        # Calculate bounding margins at maximum altitude and attitude tilt
        margin_alt = (
            max(altitude_m, margin_altitude_m)
            if margin_altitude_m is not None
            else altitude_m
        )
        fov_v = camera.fov_vertical_rad
        fov_h = camera.fov_horizontal_rad
        pitch_rad = math.radians(abs(max_pitch_deg))
        roll_rad = math.radians(abs(max_roll_deg))

        # Ground extent from nadir to FOV edges at max altitude/tilt.
        half_pitch_angle = min(pitch_rad + fov_v / 2.0, math.pi / 2.0 - 0.01)
        half_roll_angle = min(roll_rad + fov_h / 2.0, math.pi / 2.0 - 0.01)

        # Since the drone turns by 90°/180° at the ends of sweeps, the camera footprint
        # rotates (swapping width and height extents relative to the sweep line).
        # We use the maximum extent across both axes plus safety buffer
        # to ensure the projected image corners never exit DEM/map coverage.
        max_extent_angle = max(half_pitch_angle, half_roll_angle)
        margin_extent = 2.0 * margin_alt * math.tan(max_extent_angle)
        safety_buffer = 60.0  # meters buffer against dynamic turn overshoot

        # Center of the bounds
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        # Width and height of the area (subtract margin so camera never sees outside the map)
        width = max(0.0, (x_max - x_min) - margin_extent - 2.0 * safety_buffer)
        height = max(0.0, (y_max - y_min) - margin_extent - 2.0 * safety_buffer)

        # Number of lines and actual spacing so sweeps stay strictly within [-width/2, width/2]
        num_lines = int(math.ceil(width / line_spacing)) if (line_spacing > 0 and width > 0) else 0
        actual_line_spacing = (width / num_lines) if num_lines > 0 else 0.0

        waypoints = []
        angle_rad = math.radians(grid_angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        for i in range(num_lines + 1):
            # X coordinate in unrotated frame (centered)
            local_x = -width / 2.0 + i * actual_line_spacing
            
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
