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
    # Епоха супутникових знімків (Esri Wayback): "YYYY", "YYYY-MM" або
    # "YYYY-MM-DD". Порожньо → поточний Esri.WorldImagery (стара поведінка).
    # Дозволяє зняти той самий район у різні періоди (напр. літо vs зима).
    map_date: str = ""
    # Синтетичний сезон поверх ортофото: "" (як є) або "winter".
    # Справжньої зими в Esri World Imagery немає — мозаїка навмисно збирається
    # з безсніжних leaf-on знімків, тому зимовий вигляд синтезується фільтром.
    season: str = ""
    season_strength: float = 0.85
    # Якщо задано — зберегти зменшене прев'ю обробленої карти в цей PNG і вийти
    season_preview: str = ""

    # Flight parameters
    altitude_m: float = 1000.0
    speed_m_s: float = 5.0
    overlap_percent: float = 70.0
    grid_angle_deg: float = 0.0

    # Mode: "manual", "auto", "semi", or "record"
    mode: str = "manual"

    # Gamepad support (DualShock 4 / any SDL-compatible controller)
    enable_gamepad: bool = True
    # Gamepad control style: "arcade" (instant velocity) or "realistic" (RC Mode 2 with inertia)
    gamepad_mode: str = "arcade"

    # Render
    target_fps: int = 30
    physics_substeps: int = 20  # Physics steps per render frame
    enable_hillshade: bool = True  # Apply 3D hillshade shading from elevation raster

    # Performance / headless
    # fast: run the loop at maximum speed (no real-time throttle). The recorded
    # video is still written at target_fps, so playback speed stays correct.
    fast: bool = False
    # no_display: do not open the live preview window (headless run).
    no_display: bool = False

    # Components
    camera: CameraConfig = field(default_factory=CameraConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    # Tile cache directory
    cache_dir: str = ".tile_cache"

    # Telemetry logging
    telemetry_file: str = "telemetry.csv"
    telemetry_interval: int = 15

    # Video and Calibration logging
    video_file: str = ""
    calib_file: str = ""
    # Ground-truth експорт ПО СЛОТАХ для validate_vs_telemetry.py (Етап 0.2).
    # Порожньо → не пишеться. Потребує --calib-file (спільний CalibrationLogger).
    gt_file: str = ""
    # МУСИТЬ збігатися з database.frame_step системи локалізації:
    # слот БД S = кадр відео S * frame_step
    frame_step: int = 30
    # Якорі: поріг швидкості зміни напрямку руху (°/слот БД), мін. інтервал
    # додаткових якорів та макс. інтервал між якорями на прямих ділянках
    anchor_turn_rate_deg: float = 3.0
    anchor_spacing_slots: int = 15
    anchor_max_spacing_slots: int = 60
    # Стратегія відбору кадрів — МУСИТЬ збігатися з database.keyframe_* системи
    # локалізації. "overlap": якорі ставляться лише на слоти, які локалізатор
    # залишить keyframe-ами (інакше він снепить їх сам, а при порозі 0.5 снап
    # може бути на півкадру руху). "step" — стара поведінка.
    keyframe_criterion: str = "overlap"
    keyframe_max_overlap: float = 0.5
    keyframe_max_gap_frames: int = 60
    # Heading-hold (градуси): сталий курс камери на весь політ, як у гімбала
    # реального дрона. None → ніс слідує за вектором швидкості (стара поведінка)
    heading_hold_deg: float | None = None
