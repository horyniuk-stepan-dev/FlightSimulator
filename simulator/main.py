"""
Drone Flight Simulator — Main Entry Point.
Ties together physics, rendering, planning, and control.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from simulator.config import SimulatorConfig
from simulator.terrain.orthophoto_map import OrthophotoMap
from simulator.camera.camera_model import CameraModel
from simulator.camera.camera_renderer import CameraRenderer
from simulator.physics.flight_controller import FlightController
from simulator.control.manual_control import ManualControl
from simulator.control.auto_pilot import AutoPilot
from simulator.control.semi_auto_control import SemiAutoControl
from simulator.planning.survey_planner import SurveyPlanner
from simulator.planning.scenario import FlightScenario, ProfiledCommandSource
from simulator.display.hud import HUD
from simulator.display.frame_sink import DisplaySink
from simulator.display.video_sink import VideoWriterSink


from simulator.physics.telemetry_logger import TelemetryLogger
from simulator.physics.calibration_logger import CalibrationLogger
from simulator.physics.frame_ground_truth import FrameGroundTruthLogger
from simulator.dataset_manifest import write_dataset_manifest


def _configure_console_encoding() -> None:
    """Prevent Windows console encoding from aborting dataset finalisation."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                # Captured/test streams can reject a runtime encoding change.
                pass


def _database_selector_runtime() -> tuple[Path, Path]:
    """Find the sibling localization project and its own model-enabled Python.

    The simulator must use DatabaseBuilder's image-feature selection code,
    rather than infer keyframes from its ground-truth camera matrices. Paths
    may be overridden when the repositories are not adjacent.
    """
    default_root = Path(__file__).resolve().parents[2] / "DroneLocalization"
    root = Path(os.environ.get("DRONE_LOCALIZATION_ROOT", default_root)).resolve()
    default_python = root / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    python = Path(os.environ.get("DRONE_LOCALIZATION_PYTHON", default_python)).resolve()
    script = Path(__file__).resolve().parent / "localizer_keyframe_cli.py"
    if not script.is_file() or not python.is_file():
        raise RuntimeError(
            "Cannot run the database image selector: expected "
            f"{script} and {python}. Set DRONE_LOCALIZATION_ROOT and/or "
            "DRONE_LOCALIZATION_PYTHON to the localization installation."
        )
    return root, python


def _call_database_selector(
    root: Path, python: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["DRONE_LOCALIZATION_ROOT"] = str(root)
    simulator_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (simulator_root, env.get("PYTHONPATH", "")))
    )
    # Per-frame model logs can be large on long recordings. Keep them off the
    # parent process's memory while preserving a useful error tail on failure.
    with tempfile.TemporaryFile(mode="w+b") as diagnostic_stream:
        result = subprocess.run(
            [str(python), "-m", "simulator.localizer_keyframe_cli", *args],
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=diagnostic_stream,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode:
            diagnostic_stream.seek(0, os.SEEK_END)
            diagnostic_stream.seek(max(0, diagnostic_stream.tell() - 4096))
            details = diagnostic_stream.read().decode("utf-8", errors="replace").strip()
            if not details:
                details = result.stdout.strip()
            raise RuntimeError(
                f"Database image selector failed (exit {result.returncode}): "
                f"{details[-3000:]}"
            )
    return result


def _database_selector_settings(root: Path, python: Path) -> dict:
    result = _call_database_selector(root, python, "--print-config")
    try:
        settings = json.loads(result.stdout)
        frame_step = settings["frame_step"]
        criterion = settings["criterion"]
        max_overlap = settings["max_overlap"]
        max_gap_frames = settings["max_gap_frames"]
        if (
            type(frame_step) is not int or frame_step < 1
            or criterion not in ("step", "overlap")
            or not 0.0 <= float(max_overlap) <= 1.0
            or type(max_gap_frames) is not int or max_gap_frames < 0
        ):
            raise ValueError("invalid selector settings")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Database selector returned invalid settings JSON") from exc
    return settings


def _select_encoded_video_keyframes(
    *, root: Path, python: Path, video_file: str, output_file: str,
    expected_frames: int, expected_settings: dict,
) -> dict:
    _call_database_selector(
        root, python, "--video", str(Path(video_file).resolve()),
        "--output", str(Path(output_file).resolve()),
    )
    try:
        data = json.loads(Path(output_file).read_text(encoding="utf-8"))
        selected = data["selected_slots"]
        featureless = data.get("featureless_selected_slots", [])
        frame_step = int(expected_settings["frame_step"])
        total_slots = (expected_frames + frame_step - 1) // frame_step
        if (
            data["frame_step"] != frame_step
            or data["source_total_frames"] != expected_frames
            or data["total_slots"] != total_slots
            or data["selection_settings"] != expected_settings["selection_settings"]
            or not isinstance(selected, list)
            or not selected
            or any(type(slot) is not int or not 0 <= slot < total_slots for slot in selected)
            or selected != sorted(set(selected))
            or not isinstance(featureless, list)
            or any(type(slot) is not int for slot in featureless)
            or not set(featureless).issubset(selected)
            or not isinstance(data["video_sha256"], str)
            or len(data["video_sha256"]) != 64
        ):
            raise ValueError("selection contract does not match recorded video/config")
        usable = sorted(set(selected) - set(featureless))
        if not usable:
            raise ValueError("selector found no keyframes with local features")
        data["usable_anchor_slots"] = usable
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Database selector returned an invalid keyframe sidecar") from exc
    return data


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


def main():
    _configure_console_encoding()
    print("=== Drone Flight Simulator ===")
    cfg = parse_args()
    if cfg.target_fps <= 0:
        raise ValueError("--fps must be positive")
    if cfg.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")
    selector_runtime: tuple[Path, Path] | None = None
    selector_settings: dict | None = None
    if cfg.video_file and cfg.calib_file:
        # Resolve the effective DB settings before producing a single frame.
        # A missing selector must stop the run early, not leave apparently
        # valid calibration that the database cannot use.
        selector_runtime = _database_selector_runtime()
        selector_settings = _database_selector_settings(*selector_runtime)
        cfg.frame_step = selector_settings["frame_step"]
        cfg.keyframe_criterion = selector_settings["criterion"]
        cfg.keyframe_max_overlap = selector_settings["max_overlap"]
        cfg.keyframe_max_gap_frames = selector_settings["max_gap_frames"]
        cfg.keyframe_file = str(Path(cfg.video_file).with_suffix(".keyframes.json"))
        print(
            "Calibration will use the database image selector "
            f"(step={cfg.frame_step}, criterion={cfg.keyframe_criterion}, "
            f"overlap={cfg.keyframe_max_overlap}, gap={cfg.keyframe_max_gap_frames})."
        )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.video_file and not cfg.frame_gt_file:
        cfg.frame_gt_file = str(Path(cfg.video_file).with_suffix(".frames.jsonl"))
    scenario = FlightScenario.load(cfg.scenario_file) if cfg.scenario_file else None
    if scenario is not None:
        cfg.altitude_m = scenario.altitude_m.value_at(0.0)
        if cfg.max_frames <= 0:
            cfg.max_frames = int(math.ceil(scenario.duration_s * cfg.target_fps))
        print(f"Loaded scenario '{scenario.name}' ({scenario.duration_s:.1f}s)")

    # 1. Setup Terrain
    if cfg.geotiff_path and Path(cfg.geotiff_path).exists():
        print(f"Using provided GeoTIFF: {cfg.geotiff_path}")
        geotiff_path = cfg.geotiff_path
        if cfg.elevation_path and not Path(cfg.elevation_path).exists():
            raise FileNotFoundError(f"Elevation GeoTIFF not found: {cfg.elevation_path}")
        elevation_path = cfg.elevation_path or None
    else:
        print("Downloading tiles...")
        from simulator.terrain.tile_loader import download_tiles, download_elevation

        geotiff_path = download_tiles(
            lat_min=cfg.lat_min,
            lon_min=cfg.lon_min,
            lat_max=cfg.lat_max,
            lon_max=cfg.lon_max,
            zoom=cfg.zoom,
            map_date=cfg.map_date,
            cache_dir=cfg.cache_dir,
        )

        elevation_path = download_elevation(
            lat_min=cfg.lat_min,
            lon_min=cfg.lon_min,
            lat_max=cfg.lat_max,
            lon_max=cfg.lon_max,
            zoom=min(cfg.zoom, 15),  # Terrain tiles max out at zoom 15 on AWS
            cache_dir=cfg.cache_dir,
        )

    ortho_map = OrthophotoMap(
        geotiff_path,
        elevation_path=elevation_path,
        elevation_format=cfg.elevation_format,
    )
    if cfg.enable_hillshade and ortho_map.elevation is not None:
        ortho_map.apply_hillshade(blend_factor=0.35, z_factor=2.0)

    if cfg.season:
        ortho_map.apply_season(cfg.season, strength=cfg.season_strength)

    if cfg.season_preview:
        import cv2 as _cv2

        _h, _w = ortho_map.image.shape[:2]
        _scale = min(1.0, 1600.0 / max(_h, _w))
        _prev = _cv2.resize(ortho_map.image, (int(_w * _scale), int(_h * _scale)),
                            interpolation=_cv2.INTER_AREA)
        _cv2.imwrite(cfg.season_preview, _prev)
        print(f"Прев'ю карти збережено: {cfg.season_preview} ({_prev.shape[1]}x{_prev.shape[0]})")
        return

    bounds = ortho_map.get_bounds_local()
    print(f"Map Bounds (local meters): {bounds}")

    # 2. Camera setup
    camera = CameraModel.from_config(cfg.camera)
    renderer = CameraRenderer(camera, ortho_map, renderer=cfg.renderer)
    hud = HUD(ortho_map)

    # 3. Physics setup
    # Start in the center of the map
    initial_pos = [0.0, 0.0, cfg.altitude_m]
    flight_ctrl = FlightController(cfg.physics, initial_position=initial_pos)

    # 4. Control setup
    if cfg.mode in ("auto", "semi", "record"):
        print(f"Generating survey path for {cfg.mode} mode...")
        max_alt = cfg.altitude_m
        max_pitch = 0.0
        max_roll = 0.0
        if scenario is not None:
            if scenario.altitude_m.keyframes:
                max_alt = max(k.value for k in scenario.altitude_m.keyframes)
            if scenario.pitch_deg.keyframes:
                max_pitch = max(abs(k.value) for k in scenario.pitch_deg.keyframes)
            if scenario.roll_deg.keyframes:
                max_roll = max(abs(k.value) for k in scenario.roll_deg.keyframes)

        if scenario is not None and scenario.route_xy:
            waypoints = scenario.route_waypoints(bounds, camera.footprint_meters)
            print(f"Scenario route: {len(waypoints)} waypoints.")
        else:
            waypoints = SurveyPlanner.generate_path(
                bounds_local=bounds,
                altitude_m=cfg.altitude_m,
                camera=camera,
                overlap_percent=cfg.overlap_percent,
                grid_angle_deg=cfg.grid_angle_deg,
                margin_altitude_m=max_alt,
                max_pitch_deg=max_pitch,
                max_roll_deg=max_roll,
            )
            print(f"Generated {len(waypoints)} waypoints.")

        if cfg.mode in ("auto", "record"):
            hold_rad = (
                math.radians(cfg.heading_hold_deg)
                if cfg.heading_hold_deg is not None
                else None
            )
            command_source = AutoPilot(
                waypoints,
                speed_m_s=cfg.speed_m_s,
                hold_heading_rad=hold_rad,
                # A scenario owns the altitude: the route must keep flying
                # instead of stalling at a leg end while off survey altitude.
                horizontal_only=scenario is not None,
            )
        else:
            manual = ManualControl(
                speed_xy=cfg.speed_m_s,
                speed_z=cfg.speed_m_s * 0.5,
                enable_gamepad=cfg.enable_gamepad,
                gamepad_mode=cfg.gamepad_mode,
            )
            command_source = SemiAutoControl(manual, waypoints)

        if waypoints:
            flight_ctrl.reset(
                position=np.array([waypoints[0].x, waypoints[0].y, waypoints[0].z])
            )
    else:
        print("Starting in Manual mode (Keyboard: WASD + Space/Shift)")
        command_source = ManualControl(
            speed_xy=cfg.speed_m_s,
            speed_z=cfg.speed_m_s * 0.5,
            enable_gamepad=cfg.enable_gamepad,
            gamepad_mode=cfg.gamepad_mode,
        )

    if scenario is not None:
        command_source = ProfiledCommandSource(command_source, scenario)
        initial_state = flight_ctrl.get_state()
        initial_pitch = scenario.pitch_deg.value_at(0.0) or 0.0
        initial_roll = scenario.roll_deg.value_at(0.0) or 0.0
        initial_yaw = scenario.yaw_deg.value_at(0.0) or 0.0
        flight_ctrl.reset(
            position=initial_state.position,
            yaw=math.radians(initial_yaw),
            pitch=math.radians(initial_pitch),
            roll=math.radians(initial_roll),
        )

    # 5. Display Sinks and Telemetry
    window_name = "Drone Simulator"
    sinks = []
    if cfg.no_display:
        print("Headless mode: live preview window disabled.")
        if not cfg.video_file and not cfg.calib_file:
            print(
                "  Note: no --video-file / --calib-file given, so only telemetry "
                "will be produced."
            )
    else:
        sinks.append(DisplaySink(window_name))

    if cfg.video_file:
        video_sink = VideoWriterSink(
            cfg.video_file,
            cfg.target_fps,
            camera.image_width_px,
            camera.image_height_px,
        )
    else:
        video_sink = None

    telemetry_logger = TelemetryLogger(
        output_file=cfg.telemetry_file, log_interval_frames=cfg.telemetry_interval
    )
    print(
        f"Telemetry will be saved to: {cfg.telemetry_file} every {cfg.telemetry_interval} frames"
    )

    if cfg.calib_file:
        calib_logger = CalibrationLogger(
            cfg.calib_file,
            ortho_map,
            frame_step=cfg.frame_step,
            turn_rate_deg_per_slot=cfg.anchor_turn_rate_deg,
            min_anchor_spacing_slots=cfg.anchor_spacing_slots,
            max_anchor_spacing_slots=cfg.anchor_max_spacing_slots,
            keyframe_criterion=cfg.keyframe_criterion,
            keyframe_max_overlap=cfg.keyframe_max_overlap,
            keyframe_max_gap_frames=cfg.keyframe_max_gap_frames,
            generator_metadata={
                "tool": "FlightSimulator",
                "renderer": renderer.renderer_name,
                "fps": cfg.target_fps,
                "video_file": cfg.video_file,
                "gt_file": cfg.gt_file,
                "frame_gt_file": cfg.frame_gt_file,
                "scenario": scenario.source_path if scenario is not None else None,
                "keyframe_file": cfg.keyframe_file or None,
            },
        )
    else:
        calib_logger = None

    if cfg.frame_gt_file:
        frame_gt_logger = FrameGroundTruthLogger(
            cfg.frame_gt_file,
            ortho_map,
            width=camera.image_width_px,
            height=camera.image_height_px,
            fps=cfg.target_fps,
            renderer=renderer.renderer_name,
            focal_length_mm=camera.focal_length_mm,
            sensor_width_mm=camera.sensor_width_mm,
            scenario=scenario.source_path if scenario is not None else None,
        )
    else:
        frame_gt_logger = None

    if not cfg.no_display:
        # Need a tiny delay to ensure window is created before polling properties
        cv2.waitKey(1)

        if cfg.mode == "manual":
            # command_source.on_mouse is obsolete but keeping it prevents breaking
            if hasattr(command_source, "on_mouse"):
                cv2.setMouseCallback(window_name, command_source.on_mouse)

    if cfg.mode == "record" and not cfg.assume_yes:
        try:
            ans = input("Start recording? (y/n): ").strip().lower()
        except (UnicodeDecodeError, UnicodeEncodeError):
            ans = "n"
        if ans not in ("y", "yes", "у", "так"):
            print("Recording cancelled.")
            sys.exit(0)

    # 6. Main Loop
    # "offline" == run the loop at maximum speed with a fixed timestep, with no
    # wall-clock pacing. Enabled by --fast or the legacy "record" mode. The output
    # video is still written at cfg.target_fps, so playback speed is unaffected.
    offline = cfg.fast or cfg.mode == "record"
    target_dt = 1.0 / cfg.target_fps
    physics_dt = 1.0 / 100.0  # Fixed 100 Hz physics step for stability

    speed_desc = "MAX SPEED" if offline else f"real-time {cfg.target_fps} FPS"
    print(
        f"Starting simulation loop ({speed_desc}; video written @ "
        f"{cfg.target_fps} FPS)..."
    )

    next_frame_wall_time = time.perf_counter()
    # Лічильник кадрів У ЛОКСТЕПІ з video_sink: саме цей індекс бачитиме
    # DatabaseBuilder у відеофайлі, і саме від нього рахуються слоти якорів
    video_frame_idx = 0
    run_status = "running"

    try:
        while not command_source.is_finished():
            if not offline:
                now_wall = time.perf_counter()
                remaining = next_frame_wall_time - now_wall
                if remaining > 0.002:
                    time.sleep(remaining - 0.001)
                    continue
                if remaining > 0:
                    continue
            next_frame_wall_time += target_dt
            actual_dt = target_dt
            actual_fps = cfg.target_fps

            # Frame n is sampled at exactly n/fps. Physics advances only after
            # that frame has been rendered and recorded.
            state = flight_ctrl.get_state()

            # Log telemetry
            telemetry_logger.log(
                state,
                frame_index=video_frame_idx,
                timestamp=video_frame_idx / cfg.target_fps,
            )

            # Render frame
            gsd = camera.gsd_m_per_px(state.altitude)
            _t0 = time.perf_counter()
            raw_frame = renderer.render(state)
            _t_render = time.perf_counter() - _t0

            current_wp_idx = getattr(command_source, "current_wp_idx", 0)
            last_P = getattr(renderer, "last_P", None)
            H1 = last_P[:, [0, 1, 3]] if last_P is not None else None

            if calib_logger:
                # Викликаємо на КОЖЕН кадр (навіть без last_P) — логер веде
                # вікно стабільності курсу; frame_idx = індекс кадру у відео
                calib_logger.log(
                    H1,
                    camera.image_width_px,
                    camera.image_height_px,
                    state,
                    frame_idx=video_frame_idx,
                    surface_valid_fraction=renderer.last_surface_valid_fraction,
                )
            if frame_gt_logger:
                frame_gt_logger.log(
                    H1,
                    state,
                    frame_index=video_frame_idx,
                    surface_valid_fraction=renderer.last_surface_valid_fraction,
                )

            # Display preview — only built when a window is actually open. The
            # recorded video always receives the clean raw_frame below, never the
            # HUD overlay, so skipping the HUD here costs the video nothing.
            _t1 = time.perf_counter()
            if sinks:
                if any(sink.is_closed() for sink in sinks):
                    print("\nWindow closed by user.")
                    break
                if offline:
                    hud_frame = raw_frame
                else:
                    progress = getattr(command_source, "progress_str", "")
                    waypoints = getattr(command_source, "waypoints", [])

                    hud_frame = hud.render(
                        raw_frame,
                        state,
                        mode_name=command_source.mode_name,
                        fps=actual_fps,
                        gsd=gsd,
                        progress=progress,
                        P_matrix=getattr(renderer, "last_P", None),
                        waypoints=waypoints,
                        current_wp_idx=current_wp_idx,
                    )

                for sink in sinks:
                    sink.consume(hud_frame)
            _t2 = time.perf_counter()

            if video_sink:
                video_sink.consume(raw_frame)
            video_frame_idx += 1
            if cfg.max_frames > 0 and video_frame_idx >= cfg.max_frames:
                print(f"Reached --max-frames={cfg.max_frames}.")
                break

            # Window event pump + ESC / close handling — only when a window is
            # open. Skipping cv2.waitKey(1) in headless mode removes a ~1 ms/frame
            # stall, which is a big part of what lets processing run at full speed.
            if sinks:
                key = cv2.waitKey(1) & 0xFF
                _t3 = time.perf_counter()
                if video_frame_idx % 10 == 0:
                    _keys = sorted(getattr(command_source, "_keys_pressed", set()))
                    print(
                        f"render={_t_render * 1000:5.1f}  "
                        f"hud+show={(_t2 - _t1) * 1000:5.1f}  "
                        f"waitKey={(_t3 - _t2) * 1000:5.1f}  "
                        f"fps={actual_fps:4.1f}  "
                        f"pos={state.position[0]:7.1f},{state.position[1]:7.1f},"
                        f"{state.position[2]:6.1f}  "
                        f"yaw={math.degrees(state.yaw):6.1f}  keys={_keys}"
                    )
                if key == 27:  # ESC
                    print("\nESC pressed. Exiting...")
                    break

                # Check if user closed the window via 'X' button
                if any(sink.is_closed() for sink in sinks):
                    print("\nWindow closed by user.")
                    break

            cmd = command_source.get_command(state, actual_dt)
            cmd_v = cmd.to_numpy()
            remaining_physics = target_dt
            while remaining_physics > 1e-12:
                step_dt = min(physics_dt, remaining_physics)
                flight_ctrl.step(
                    cmd_v,
                    cmd.yaw_rate,
                    cmd.pitch_rate,
                    step_dt,
                    kinematic=True,
                    roll=cmd.roll,
                )
                remaining_physics -= step_dt

        run_status = "complete"
    except KeyboardInterrupt:
        run_status = "interrupted"
        print("\nInterrupted by user.")
    except Exception:
        run_status = "failed"
        raise
    finally:
        finalization_error = None
        for sink in sinks:
            if hasattr(sink, "cleanup"):
                sink.cleanup()
        cv2.destroyAllWindows()
        telemetry_logger.close()
        if frame_gt_logger:
            frame_gt_logger.close()
        if video_sink:
            video_sink.cleanup()
        if calib_logger:
            try:
                if selector_runtime is not None and selector_settings is not None:
                    if video_frame_idx <= 0:
                        raise RuntimeError("No recorded video frames for image keyframe selection")
                    selected = _select_encoded_video_keyframes(
                        root=selector_runtime[0],
                        python=selector_runtime[1],
                        video_file=cfg.video_file,
                        output_file=cfg.keyframe_file,
                        expected_frames=video_frame_idx,
                        expected_settings=selector_settings,
                    )
                    calib_logger.set_exact_keyframe_slots(
                        selected["usable_anchor_slots"],
                        provenance={
                            "sidecar": cfg.keyframe_file,
                            "video_sha256": selected["video_sha256"],
                        },
                    )
                    print(
                        "Image selector kept "
                        f"{len(selected['selected_slots'])}/{selected['total_slots']} slots "
                        f"({len(selected['usable_anchor_slots'])} with local features)."
                    )
                calib_logger.close()
                if cfg.gt_file:
                    calib_logger.dump_ground_truth(cfg.gt_file, fps=cfg.target_fps)
            except Exception as exc:
                run_status = "failed"
                finalization_error = exc
                print(f"Calibration finalization failed: {exc}", file=sys.stderr)
        manifest_path = cfg.manifest_file
        if not manifest_path and cfg.video_file:
            manifest_path = str(Path(cfg.video_file).with_suffix(".manifest.json"))
        if manifest_path:
            manifest = write_dataset_manifest(
                manifest_path,
                status=run_status,
                config=cfg,
                renderer=renderer.renderer_name,
                frame_count=video_frame_idx,
            )
            if run_status == "complete" and manifest["status"] != "complete" and finalization_error is None:
                finalization_error = RuntimeError(
                    "Dataset manifest validation failed: "
                    + "; ".join(manifest["validation_errors"][:3])
                )
        print("Simulation ended.")
        if finalization_error is not None:
            raise RuntimeError("Simulator recording did not produce a valid dataset") from finalization_error


if __name__ == "__main__":
    main()
