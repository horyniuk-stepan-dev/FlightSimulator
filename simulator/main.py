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
        "--frame-step",
        type=int,
        default=30,
        help="Frame step to sync calibration indices with Topometric Database",
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
        )

        elevation_path = download_elevation(
            lat_min=cfg.lat_min,
            lon_min=cfg.lon_min,
            lat_max=cfg.lat_max,
            lon_max=cfg.lon_max,
            zoom=min(cfg.zoom, 15),  # Terrain tiles max out at zoom 15 on AWS
        )

    ortho_map = OrthophotoMap(geotiff_path, elevation_path=elevation_path)
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
    
    if cfg.video_file:
        video_sink = VideoWriterSink(cfg.video_file, cfg.target_fps, camera.image_width_px, camera.image_height_px)
    else:
        video_sink = None
        
    telemetry_logger = TelemetryLogger(
        output_file=cfg.telemetry_file, log_interval_frames=cfg.telemetry_interval
    )
    print(
        f"Telemetry will be saved to: {cfg.telemetry_file} every {cfg.telemetry_interval} frames"
    )
    
    if cfg.calib_file:
        calib_logger = CalibrationLogger(cfg.calib_file, ortho_map, cfg.telemetry_interval, cfg.frame_step)
    else:
        calib_logger = None

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
    target_dt = 1.0 / cfg.target_fps
    physics_dt = 1.0 / 100.0  # Fixed 100 Hz physics step for stability

    print(f"Starting simulation loop (Target FPS: {cfg.target_fps})...")

    prev_time = time.perf_counter()
    accumulated_time = 0.0

    try:
        while not command_source.is_finished():
            if cfg.mode == "record":
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
            if cfg.mode != "record" and accumulated_time > 0.1:
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
            
            current_wp_idx = getattr(command_source, "current_wp_idx", 0)
            
            if calib_logger and hasattr(renderer, "last_P") and renderer.last_P is not None:
                H1 = renderer.last_P[:, [0, 1, 3]]
                lat, lon = ortho_map.local_to_gps(state.position[0], state.position[1])
                calib_logger.log(H1, camera.image_width_px, camera.image_height_px, state, lat, lon, current_wp_idx)

            # HUD
            if cfg.mode == "record":
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

            # Consume
            for sink in sinks:
                sink.consume(hud_frame)
                
            if video_sink:
                video_sink.consume(raw_frame)

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
        telemetry_logger.close()
        if calib_logger:
            calib_logger.close()
        for sink in sinks:
            sink.cleanup()
        if video_sink:
            video_sink.cleanup()
        print("Simulation ended.")


if __name__ == "__main__":
    main()
