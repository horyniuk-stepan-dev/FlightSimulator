"""Command-line arguments for the flight simulator (simulator/main.py)."""

import argparse

from simulator.config import SimulatorConfig


def parse_args() -> SimulatorConfig:
    parser = argparse.ArgumentParser(description="Drone Flight Simulator")

    # Bounding box
    parser.add_argument(
        "--lat_min",
        type=float,
        default=48.39950223106055,
        help="Minimum latitude (South)",
    )
    parser.add_argument(
        "--lon_min",
        type=float,
        default=26.102003186115134,
        help="Minimum longitude (West)",
    )
    parser.add_argument(
        "--lat_max",
        type=float,
        default=48.44349548338706,
        help="Maximum latitude (North)",
    )
    parser.add_argument(
        "--lon_max",
        type=float,
        default=26.257266910369705,
        help="Maximum longitude (East)",
    )
    parser.add_argument("--zoom", type=int, default=17, help="Map tile zoom level")
    parser.add_argument(
        "--season",
        type=str,
        choices=["", "winter"],
        default="",
        help="Синтетичний сезон поверх ортофото. winter = leaf-off + "
        "десатурація + сніг на гладких ділянках + зимове світло. "
        "Геометрія сцени не змінюється, тільки фотометрика",
    )
    parser.add_argument(
        "--season-strength",
        dest="season_strength",
        type=float,
        default=0.85,
        help="Сила сезонного фільтра [0.0 — вимкнено, 1.0 — максимум]",
    )
    parser.add_argument(
        "--season-preview",
        dest="season_preview",
        type=str,
        default="",
        help="Зберегти зменшене прев'ю обробленої карти в цей PNG і вийти "
        "(щоб підібрати --season-strength без запису польоту)",
    )
    parser.add_argument(
        "--map-date",
        dest="map_date",
        type=str,
        default="",
        help="Епоха супутникових знімків Esri Wayback: YYYY, YYYY-MM або "
        "YYYY-MM-DD (береться найновіший реліз не пізніше цієї дати). "
        "Список реально різних епох для точки: "
        "python -m simulator.terrain.wayback --lat <LAT> --lon <LON>",
    )

    # Flight params
    parser.add_argument(
        "--altitude",
        dest="altitude_m",
        type=float,
        default=1000.0,
        help="Flight altitude (m)",
    )
    parser.add_argument(
        "--altitude-reference",
        dest="altitude_reference",
        choices=["base", "msl"],
        default="base",
        help="Datum of --altitude and scenario altitudes: 'base' = above the "
        "map's lowest DEM point (simulator Z=0, default); 'msl' = above sea "
        "level from the DEM. Від чого рахується висота польоту",
    )
    parser.add_argument(
        "--altitude-offset",
        dest="altitude_offset_m",
        type=float,
        default=0.0,
        help="Metres added to --altitude and every scenario altitude "
        "(e.g. lift a survey over high terrain)",
    )
    parser.add_argument(
        "--min-clearance",
        dest="min_clearance_m",
        type=float,
        default=50.0,
        help="Minimum camera height above the terrain under the drone when "
        "recording: checked along the planned route before the flight and at "
        "every frame (0 = off)",
    )
    parser.add_argument(
        "--flat-spacing",
        dest="flat_spacing",
        action="store_true",
        help="Survey line spacing from the flight height above Z=0 (old "
        "behaviour) instead of the lowest height above the terrain",
    )
    parser.add_argument(
        "--speed",
        dest="speed_m_s",
        type=float,
        default=150.0,
        help="Flight speed (m/s)",
    )
    parser.add_argument(
        "--overlap",
        dest="overlap_percent",
        type=float,
        default=30.0,
        help="Survey overlap (percent)",
    )
    parser.add_argument(
        "--grid-angle",
        dest="grid_angle_deg",
        type=float,
        default=0.0,
        help="Survey grid angle (deg)",
    )

    # Control
    parser.add_argument(
        "--mode",
        type=str,
        choices=["manual", "auto", "semi", "record"],
        default="manual",
        help="Control mode",
    )
    parser.add_argument(
        "--no-gamepad",
        dest="enable_gamepad",
        action="store_false",
        help="Disable gamepad/controller support (DualShock 4, etc.)",
    )
    parser.add_argument(
        "--gamepad-mode",
        dest="gamepad_mode",
        type=str,
        choices=["arcade", "realistic"],
        default="arcade",
        help="Gamepad control style: 'arcade' (instant velocity, camera on right stick) "
        "or 'realistic' (RC Mode 2 — throttle+yaw on left stick, pitch+roll on right, "
        "inertia & drag physics)",
    )

    # Performance / output rate
    parser.add_argument(
        "--fps",
        dest="target_fps",
        type=int,
        default=30,
        help="Recorded video frame rate / playback FPS (default 30). "
        "Частота кадрів запису відео.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run the simulation as fast as possible (no real-time throttle). "
        "Обробка на максимальній швидкості; відео все одно пишеться у --fps.",
    )
    parser.add_argument(
        "--no-display",
        "--headless",
        dest="no_display",
        action="store_true",
        help="Do not open the live preview window (headless). "
        "Не показувати вікно з польотом дрона.",
    )
    parser.add_argument(
        "--no-hillshade",
        dest="enable_hillshade",
        action="store_false",
        help="Disable 3D hillshade shading from elevation raster. "
        "Вимкнути світлотіньове 3D-рельефування.",
    )
    parser.add_argument(
        "--renderer",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Rendering backend. 'cpu' is the deterministic reference path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many recorded frames (0 = full mission).",
    )
    parser.add_argument(
        "--yes",
        dest="assume_yes",
        action="store_true",
        help="Do not ask for interactive confirmation before record mode.",
    )
    parser.add_argument(
        "--scenario",
        dest="scenario_file",
        type=str,
        default="",
        help="JSON scenario with deterministic altitude and camera attitude keyframes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed recorded in the manifest (default: 0).",
    )

    # File overrides
    parser.add_argument(
        "--geotiff",
        dest="geotiff_path",
        type=str,
        default="",
        help="Path to existing GeoTIFF (skips download)",
    )
    parser.add_argument(
        "--elevation",
        dest="elevation_path",
        type=str,
        default="",
        help="Georeferenced elevation GeoTIFF used with --geotiff.",
    )
    parser.add_argument(
        "--elevation-format",
        choices=["auto", "terrarium", "meters"],
        default="auto",
        help="DEM encoding: auto detects RGB Terrarium or one-band metre heights.",
    )
    parser.add_argument(
        "--manifest-file",
        type=str,
        default="",
        help="Dataset manifest path. Defaults beside --video-file when recording.",
    )

    # Telemetry
    parser.add_argument(
        "--telemetry-file",
        type=str,
        default="telemetry.csv",
        help="Path to save telemetry CSV",
    )
    parser.add_argument(
        "--telemetry-interval",
        type=int,
        default=15,
        help="Save telemetry every N frames",
    )

    # Video and Calibration Output
    parser.add_argument(
        "--video-file",
        type=str,
        default="",
        help="Path to save flight video (e.g., flight.mp4)",
    )
    parser.add_argument(
        "--calib-file",
        type=str,
        default="",
        help="Path to save calibration JSON",
    )
    parser.add_argument(
        "--gt-file",
        dest="gt_file",
        type=str,
        default="",
        help="Шлях до ground_truth.json — GT по КОЖНОМУ слоту для "
        "validate_vs_telemetry.py (Етап 0.2). Потребує --calib-file",
    )
    parser.add_argument(
        "--frame-gt-file",
        dest="frame_gt_file",
        type=str,
        default="",
        help="JSONL ground truth for every recorded frame. Defaults to "
        "<video>.frames.jsonl when --video-file is set.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=30,
        help="Крок слотів; для запису відео з калібруванням автоматично "
        "береться з чинної конфігурації DroneLocalization",
    )
    parser.add_argument(
        "--keyframe-criterion",
        dest="keyframe_criterion",
        choices=["overlap", "step"],
        default="overlap",
        help="Стратегія відбору; для запису відео з калібруванням автоматично "
        "береться з DroneLocalization, а слоти обчислює його візуальний селектор",
    )
    parser.add_argument(
        "--keyframe-max-overlap",
        dest="keyframe_max_overlap",
        type=float,
        default=0.5,
        help="Поріг перекриття; для запису відео з калібруванням "
        "береться з DroneLocalization",
    )
    parser.add_argument(
        "--keyframe-max-gap-frames",
        dest="keyframe_max_gap_frames",
        type=int,
        default=60,
        help="Найбільша кількість пропущених слотів; для запису відео з "
        "калібруванням береться з DroneLocalization",
    )
    parser.add_argument(
        "--anchor-spacing-slots",
        dest="anchor_spacing_slots",
        type=int,
        default=15,
        help="Мінімальний інтервал між якорями калібрування (у слотах БД)",
    )
    parser.add_argument(
        "--anchor-turn-rate-deg",
        dest="anchor_turn_rate_deg",
        type=float,
        default=3.0,
        help="Поріг швидкості зміни напрямку руху (°/слот БД): вище — розворот, "
        "якорі ставляться на його межах та апексі",
    )
    parser.add_argument(
        "--heading-hold-deg",
        dest="heading_hold_deg",
        type=float,
        default=None,
        help="Сталий курс камери у градусах (heading-hold, як гімбал реального "
        "дрона): кадри всіх ніг серпантину матимуть однакову орієнтацію. "
        "Без прапорця ніс слідує за вектором швидкості",
    )
    parser.add_argument(
        "--anchor-max-spacing-slots",
        dest="anchor_max_spacing_slots",
        type=int,
        default=60,
        help="Максимальний інтервал між якорями (у слотах БД) на прямих ділянках",
    )

    args = parser.parse_args()
    return SimulatorConfig(**vars(args))
