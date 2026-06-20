"""
Camera model — defines virtual camera parameters and ground footprint calculations.
Synchronized with the localization project's GSDCalculator.
"""
import math
from dataclasses import dataclass

from simulator.config import CameraConfig


@dataclass
class CameraModel:
    """Virtual nadir camera model for the drone simulator."""
    image_width_px: int
    image_height_px: int
    focal_length_mm: float
    sensor_width_mm: float

    @classmethod
    def from_config(cls, cfg: CameraConfig) -> 'CameraModel':
        return cls(
            image_width_px=cfg.image_width_px,
            image_height_px=cfg.image_height_px,
            focal_length_mm=cfg.focal_length_mm,
            sensor_width_mm=cfg.sensor_width_mm,
        )

    @property
    def sensor_height_mm(self) -> float:
        """Sensor height based on aspect ratio."""
        return self.sensor_width_mm * self.image_height_px / self.image_width_px

    @property
    def fov_horizontal_rad(self) -> float:
        """Horizontal field of view in radians."""
        return 2.0 * math.atan(self.sensor_width_mm / (2.0 * self.focal_length_mm))

    @property
    def fov_vertical_rad(self) -> float:
        """Vertical field of view in radians."""
        return 2.0 * math.atan(self.sensor_height_mm / (2.0 * self.focal_length_mm))

    def footprint_meters(self, altitude_m: float) -> tuple[float, float]:
        """
        Ground footprint size at given altitude.

        Returns:
            (width_m, height_m): Ground coverage in meters.
        """
        w = altitude_m * self.sensor_width_mm / self.focal_length_mm
        h = altitude_m * self.sensor_height_mm / self.focal_length_mm
        return w, h

    def gsd_m_per_px(self, altitude_m: float) -> float:
        """Ground Sample Distance — meters per pixel at given altitude."""
        if self.focal_length_mm <= 0 or self.image_width_px <= 0:
            return 0.0
        return (altitude_m * self.sensor_width_mm) / \
               (self.focal_length_mm * self.image_width_px)

    def px_per_meter(self, altitude_m: float) -> float:
        """Pixels per meter at given altitude."""
        gsd = self.gsd_m_per_px(altitude_m)
        return 1.0 / gsd if gsd > 1e-9 else 0.0
