"""
Drone Flight Simulator — Main Entry Point.
Ties together physics, rendering, planning, and control.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from simulator.config import SimulatorConfig
from simulator.terrain.tile_loader import download_tiles
from simulator.terrain.orthophoto_map import OrthophotoMap
from simulator.camera.camera_model import CameraModel
from simulator.camera.camera_renderer import CameraRenderer
from simulator.physics.flight_controller import FlightController
from simulator.control.manual_control import ManualControl
from simulator.control.auto_pilot import AutoPilot
from simulator.control.semi_auto_control import SemiAutoControl
from simulator.planning.survey_planner import SurveyPlanner
from simulator.display.hud import HUD
from simulator.display.frame_sink import DisplaySink
from simulator.display.video_sink import VideoWriterSink


from simulator.physics.telemetry_logger import TelemetryLogger
from simulator.physics.calibration_logger import CalibrationLogger


def parse_args() -> SimulatorConfig:
    parser = argparse.ArgumentParser(description="Drone Flight Simulator")

    # Bounding box
    parser.add_argument(
        "--lat_min",
        type=float,
        default=48.364964,
        help="Minimum latitude (South)",
    )
    parser.add_argument(
        "--lon_min",
        type=float,
        default=26.062876,
        help="Minimum longitude (West)",
    )
    parser.add_argument(
        "--lat_max",
        type=float,
        default=48.402230,
        help="Maximum latitude (North)",
    )
    parser.add_argument(
        "--lon_max",
        type=float,
        default=26.116208,
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

    # File overrides
    parser.add_argument(
        "--geotiff",
        dest="geotiff_path",
        type=str,
        default="",
        help="Path to existing GeoTIFF (skips download)",
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
        "--frame-step",
        type=int,
        default=30,
        help="Frame step to sync calibration indices with Topometric Database "
        "(МУСИТЬ збігатися з database.frame_step системи локалізації)",
    )
    parser.add_argument(
        "--keyframe-criterion",
        dest="keyframe_criterion",
        choices=["overlap", "step"],
        default="overlap",
        help="Стратегія відбору кадрів для якорів. overlap — ставити якорі лише "
        "на слоти, які локалізатор залишить keyframe-ами (МУСИТЬ збігатися з "
        "database.keyframe_criterion); step — стара поведінка",
    )
    parser.add_argument(
        "--keyframe-max-overlap",
        dest="keyframe_max_overlap",
        type=float,
        default=0.5,
        help="Поріг перекриття для criterion=overlap "
        "(МУСИТЬ збігатися з database.keyframe_max_overlap)",
    )
    parser.add_argument(
        "--keyframe-max-gap-frames",
        dest="keyframe_max_gap_frames",
        type=int,
        default=60,
        help="Скільки слотів поспіль можна пропустити до примусового keyframe "
        "(МУСИТЬ збігатися з database.keyframe_max_gap_frames)",
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
    print("=== Drone Flight Simulator ===")
    cfg = parse_args()

    # 1. Setup Terrain
    if cfg.geotiff_path and Path(cfg.geotiff_path).exists():
        print(f"Using provided GeoTIFF: {cfg.geotiff_path}")
        geotiff_path = cfg.geotiff_path
        elevation_path = None
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
        )

        elevation_path = download_elevation(
            lat_min=cfg.lat_min,
            lon_min=cfg.lon_min,
            lat_max=cfg.lat_max,
            lon_max=cfg.lon_max,
            zoom=min(cfg.zoom, 15),  # Terrain tiles max out at zoom 15 on AWS
        )

    ortho_map = OrthophotoMap(geotiff_path, elevation_path=elevation_path)
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
    renderer = CameraRenderer(camera, ortho_map)
    hud = HUD(ortho_map)

    # 3. Physics setup
    # Start in the center of the map
    initial_pos = [0.0, 0.0, cfg.altitude_m]
    flight_ctrl = FlightController(cfg.physics, initial_position=initial_pos)

    # 4. Control setup
    if cfg.mode in ("auto", "semi", "record"):
        print(f"Generating survey path for {cfg.mode} mode...")
        waypoints = SurveyPlanner.generate_path(
            bounds_local=bounds,
            altitude_m=cfg.altitude_m,
            camera=camera,
            overlap_percent=cfg.overlap_percent,
            grid_angle_deg=cfg.grid_angle_deg,
        )
        print(f"Generated {len(waypoints)} waypoints.")

        if cfg.mode in ("auto", "record"):
            hold_rad = (
                math.radians(cfg.heading_hold_deg)
                if cfg.heading_hold_deg is not None
                else None
            )
            command_source = AutoPilot(
                waypoints, speed_m_s=cfg.speed_m_s, hold_heading_rad=hold_rad
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
        )
    else:
        calib_logger = None

    if not cfg.no_display:
        # Need a tiny delay to ensure window is created before polling properties
        cv2.waitKey(1)

        if cfg.mode == "manual":
            # command_source.on_mouse is obsolete but keeping it prevents breaking
            if hasattr(command_source, "on_mouse"):
                cv2.setMouseCallback(window_name, command_source.on_mouse)

    if cfg.mode == "record":
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

    prev_time = time.perf_counter()
    accumulated_time = 0.0
    # Лічильник кадрів У ЛОКСТЕПІ з video_sink: саме цей індекс бачитиме
    # DatabaseBuilder у відеофайлі, і саме від нього рахуються слоти якорів
    video_frame_idx = 0

    try:
        while not command_source.is_finished():
            if offline:
                # Offline rendering: run as fast as possible, simulating exactly target_dt per frame
                actual_dt = target_dt
                actual_fps = cfg.target_fps
            else:
                curr_time = time.perf_counter()
                actual_dt = curr_time - prev_time

                # Improved frame pacing: sleep for bulk of wait, spin-wait only the last ~2ms.
                if actual_dt < target_dt:
                    remaining = target_dt - actual_dt
                    if remaining > 0.003:
                        time.sleep(remaining - 0.002)
                    continue

                prev_time = curr_time
                actual_fps = 1.0 / actual_dt if actual_dt > 0 else 0.0

            accumulated_time += actual_dt
            # Prevent "spiral of death" if rendering becomes very slow
            if not offline and accumulated_time > 0.1:
                accumulated_time = 0.1

            # Step Physics (multiple fixed substeps for stability)
            state = flight_ctrl.get_state()
            cmd = command_source.get_command(state, actual_dt)
            cmd_v = cmd.to_numpy()
            target_yaw = cmd.yaw_rate  # We stored target_yaw in yaw_rate
            target_pitch = cmd.pitch_rate
            target_roll = cmd.roll
            is_kinematic = True

            while accumulated_time >= physics_dt:
                state = flight_ctrl.step(
                    cmd_v,
                    target_yaw,
                    target_pitch,
                    physics_dt,
                    kinematic=is_kinematic,
                    roll=target_roll,
                )
                accumulated_time -= physics_dt

            # Log telemetry
            telemetry_logger.log(state)

            # Render frame
            gsd = camera.gsd_m_per_px(state.altitude)
            _t0 = time.perf_counter()
            raw_frame = renderer.render(state)
            _t_render = time.perf_counter() - _t0

            current_wp_idx = getattr(command_source, "current_wp_idx", 0)

            if calib_logger:
                # Викликаємо на КОЖЕН кадр (навіть без last_P) — логер веде
                # вікно стабільності курсу; frame_idx = індекс кадру у відео
                last_P = getattr(renderer, "last_P", None)
                H1 = last_P[:, [0, 1, 3]] if last_P is not None else None
                calib_logger.log(
                    H1,
                    camera.image_width_px,
                    camera.image_height_px,
                    state,
                    frame_idx=video_frame_idx,
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

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        for sink in sinks:
            if hasattr(sink, "cleanup"):
                sink.cleanup()
        cv2.destroyAllWindows()
        telemetry_logger.close()
        if calib_logger:
            calib_logger.close()
            if cfg.gt_file:
                calib_logger.dump_ground_truth(cfg.gt_file, fps=cfg.target_fps)
        for sink in sinks:
            sink.cleanup()
        if video_sink:
            video_sink.cleanup()
        print("Simulation ended.")


if __name__ == "__main__":
    main()
