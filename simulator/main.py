"""
Drone Flight Simulator — Main Entry Point.
Ties together physics, rendering, planning, and control.
"""

import argparse
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


from simulator.physics.telemetry_logger import TelemetryLogger


def parse_args() -> SimulatorConfig:
    parser = argparse.ArgumentParser(description="Drone Flight Simulator")

    # Bounding box
    parser.add_argument(
        "--lat-min", type=float, default=47.81778199390838, help="Min Latitude (WGS84)"
    )
    parser.add_argument(
        "--lon-min", type=float, default=34.922599133780636, help="Min Longitude (WGS84)"
    )
    parser.add_argument(
        "--lat-max", type=float, default=47.83464232085366, help="Max Latitude (WGS84)"
    )
    parser.add_argument(
        "--lon-max", type=float, default=34.94308924682522, help="Max Longitude (WGS84)"
    )
    parser.add_argument("--zoom", type=int, default=17, help="Map tile zoom level")

    # Flight params
    parser.add_argument(
        "--altitude", type=float, default=1000.0, help="Flight altitude (m)"
    )
    parser.add_argument("--speed", type=float, default=150.0, help="Flight speed (m/s)")
    parser.add_argument(
        "--overlap", type=float, default=50.0, help="Survey overlap (percent)"
    )
    parser.add_argument(
        "--grid-angle", type=float, default=0.0, help="Survey grid angle (deg)"
    )

    # Control
    parser.add_argument(
        "--mode",
        type=str,
        choices=["manual", "auto", "semi"],
        default="manual",
        help="Control mode",
    )

    # File overrides
    parser.add_argument(
        "--geotiff",
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

    args = parser.parse_args()

    cfg = SimulatorConfig()
    cfg.lat_min = args.lat_min
    cfg.lon_min = args.lon_min
    cfg.lat_max = args.lat_max
    cfg.lon_max = args.lon_max
    cfg.zoom = args.zoom
    cfg.altitude_m = args.altitude
    cfg.speed_m_s = args.speed
    cfg.overlap_percent = args.overlap
    cfg.grid_angle_deg = args.grid_angle
    cfg.mode = args.mode
    cfg.geotiff_path = args.geotiff

    # Attach telemetry args dynamically to cfg for convenience
    cfg.telemetry_file = args.telemetry_file
    cfg.telemetry_interval = args.telemetry_interval

    return cfg


def main():
    print("=== Drone Flight Simulator ===")
    cfg = parse_args()

    # 1. Load Terrain
    if cfg.geotiff_path and Path(cfg.geotiff_path).exists():
        geotiff_path = cfg.geotiff_path
        print(f"Using provided GeoTIFF: {geotiff_path}")
    else:
        print("Downloading tiles...")
        geotiff_path = download_tiles(
            cfg.lat_min,
            cfg.lon_min,
            cfg.lat_max,
            cfg.lon_max,
            zoom=cfg.zoom,
            cache_dir=cfg.cache_dir,
        )

    ortho_map = OrthophotoMap(geotiff_path)
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
    if cfg.mode == "auto" or cfg.mode == "semi":
        print(f"Generating survey path for {cfg.mode} mode...")
        waypoints = SurveyPlanner.generate_path(
            bounds_local=bounds,
            altitude_m=cfg.altitude_m,
            camera=camera,
            overlap_percent=cfg.overlap_percent,
            grid_angle_deg=cfg.grid_angle_deg,
        )
        print(f"Generated {len(waypoints)} waypoints.")

        if cfg.mode == "auto":
            command_source = AutoPilot(waypoints, speed_m_s=cfg.speed_m_s)
        else:
            manual = ManualControl(speed_xy=cfg.speed_m_s, speed_z=cfg.speed_m_s * 0.5)
            command_source = SemiAutoControl(manual, waypoints)

        if waypoints:
            flight_ctrl.reset(
                position=np.array([waypoints[0].x, waypoints[0].y, waypoints[0].z])
            )
    else:
        print("Starting in Manual mode (Keyboard: WASD + Space/Shift)")
        command_source = ManualControl(
            speed_xy=cfg.speed_m_s, speed_z=cfg.speed_m_s * 0.5
        )

    # 5. Display Sinks and Telemetry
    window_name = "Drone Simulator"
    sinks = [DisplaySink(window_name)]
    telemetry_logger = TelemetryLogger(
        output_file=cfg.telemetry_file, log_interval_frames=cfg.telemetry_interval
    )
    print(
        f"Telemetry will be saved to: {cfg.telemetry_file} every {cfg.telemetry_interval} frames"
    )

    # Need a tiny delay to ensure window is created before polling properties
    import cv2

    cv2.waitKey(1)

    if cfg.mode == "manual":
        # command_source.on_mouse is obsolete but keeping it prevents breaking
        if hasattr(command_source, "on_mouse"):
            cv2.setMouseCallback(window_name, command_source.on_mouse)

    # 6. Main Loop
    target_dt = 1.0 / cfg.target_fps
    physics_dt = 1.0 / 100.0  # Fixed 100 Hz physics step for stability

    print(f"Starting simulation loop (Target FPS: {cfg.target_fps})...")

    prev_time = time.perf_counter()
    accumulated_time = 0.0

    try:
        while not command_source.is_finished():
            curr_time = time.perf_counter()
            actual_dt = curr_time - prev_time

            # Improved frame pacing: time.sleep is imprecise on Windows (15ms resolution),
            # so we only sleep if there's a large margin, then spin-wait the rest.
            if actual_dt < target_dt:
                sleep_time = target_dt - actual_dt - 0.015
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            prev_time = curr_time
            actual_fps = 1.0 / actual_dt if actual_dt > 0 else 0.0

            accumulated_time += actual_dt
            # Prevent "spiral of death" if rendering becomes very slow
            if accumulated_time > 0.1:
                accumulated_time = 0.1

            # Step Physics (multiple fixed substeps for stability)
            state = flight_ctrl.get_state()
            cmd = command_source.get_command(state, actual_dt)
            cmd_v = cmd.to_numpy()
            target_yaw = cmd.yaw_rate  # We stored target_yaw in yaw_rate
            target_pitch = cmd.pitch_rate
            is_kinematic = True

            while accumulated_time >= physics_dt:
                state = flight_ctrl.step(
                    cmd_v, target_yaw, target_pitch, physics_dt, kinematic=is_kinematic
                )
                accumulated_time -= physics_dt

            # Log telemetry
            telemetry_logger.log(state)

            # Render frame
            gsd = camera.gsd_m_per_px(state.altitude)
            raw_frame = renderer.render(state)

            # HUD
            progress = getattr(command_source, "progress_str", "")
            waypoints = getattr(command_source, "waypoints", [])
            current_wp_idx = getattr(command_source, "current_wp_idx", 0)

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

            # Consume
            for sink in sinks:
                sink.consume(hud_frame)

            # OpenCV waitKey
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            # Check if user closed the window via 'X' button
            if any(sink.is_closed() for sink in sinks):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        for sink in sinks:
            sink.cleanup()
        print("Simulation ended.")


if __name__ == "__main__":
    main()
