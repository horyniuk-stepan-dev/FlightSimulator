"""
Simulator configuration dataclass.
All parameters for the drone flight simulator.
"""

from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    """Virtual camera parameters — synchronized with the localization project's GSDCalculator."""

    image_width_px: int = 1280
    image_height_px: int = 720
    focal_length_mm: float = 13.2
    sensor_width_mm: float = 8.8


@dataclass
class PhysicsConfig:
    """Drone physics parameters."""

    # Smaller drone params (~1.0 kg) for better agility
    mass_kg: float = 1.0
    arm_length_m: float = 0.15
    # RotorPy control gains overrides
    k_v: float = 4.0  # Velocity P-gain (kd_pos in SE3Control)
    kp_att: float = 1000.0  # Attitude P-gain
    kd_att: float = 60.0  # Attitude D-gain


@dataclass
class SimulatorConfig:
    """Top-level simulator configuration."""

    # Terrain — bounding box (WGS84 lat/lon)
    # 7x7 km area around Kyiv
    lat_min: float = 50.4185
    lon_min: float = 30.4710
    lat_max: float = 50.4815
    lon_max: float = 30.5690
    zoom: int = 17  # Reduced zoom to keep VRAM usage normal for a large map
    geotiff_path: str = ""  # If provided, skip tile download

    # Flight parameters
    altitude_m: float = 1000.0
    speed_m_s: float = 5.0
    overlap_percent: float = 70.0
    grid_angle_deg: float = 0.0

    # Mode: "manual", "auto", "semi", or "record"
    mode: str = "manual"

    # Render
    target_fps: int = 30
    physics_substeps: int = 20  # Physics steps per render frame

    # Components
    camera: CameraConfig = field(default_factory=CameraConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    # Tile cache directory
    cache_dir: str = ".tile_cache"

    # Telemetry logging
    telemetry_file: str = "telemetry.csv"
    telemetry_interval: int = 15
