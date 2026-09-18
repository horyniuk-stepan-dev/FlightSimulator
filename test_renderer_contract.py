"""End-to-end contracts for pixels, DEM georeferencing, HUD, and scenarios."""

import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from simulator.camera.camera_model import CameraModel
from simulator.camera.camera_renderer import CameraRenderer
from simulator.control.command_source import CommandSource, CommandVector
from simulator.display.hud import HUD
from simulator.physics.drone_state import DroneState
from simulator.physics.frame_ground_truth import FrameGroundTruthLogger
from simulator.physics.telemetry_logger import TelemetryLogger
from simulator.planning.scenario import FlightScenario, ProfiledCommandSource
from simulator.terrain.orthophoto_map import OrthophotoMap
from simulator.terrain.surface_projection import (
    ProjectionGeometryError,
    project_image_pixels_to_surface,
)


class SyntheticMap:
    def __init__(self):
        yy, xx = np.mgrid[0:300, 0:400]
        self.image = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
        self.res_x = self.res_y = 1.0
        self.width = 400
        self.height = 300
        self.elevation = None
        self.base_elevation = 0.0

    @staticmethod
    def local_to_pixel(x, y):
        return x + 200.0, 150.0 - y

    @staticmethod
    def local_to_gps(x, y):
        return 50.0 + y * 1e-5, 30.0 + x * 1e-5

    @staticmethod
    def pixel_to_local(col, row):
        return col - 200.0, 150.0 - row

    @staticmethod
    def local_to_mercator(x, y):
        return 3_000_000.0 + x, 6_000_000.0 + y


def test_cpu_renderer_projects_map_centre_to_image_centre():
    terrain = SyntheticMap()
    camera = CameraModel(128, 72, 13.2, 8.8)
    renderer = CameraRenderer(camera, terrain, renderer="cpu")
    frame = renderer.render(DroneState(position=np.array([0.0, 0.0, 100.0])))
    expected = terrain.image[150, 200]
    assert np.max(np.abs(frame[36, 64].astype(int) - expected.astype(int))) <= 2
    assert renderer.last_surface_valid_fraction == 1.0


def test_hud_does_not_mutate_clean_frame():
    terrain = SyntheticMap()
    clean = np.zeros((720, 1280, 3), dtype=np.uint8)
    before = clean.copy()
    rendered = HUD(terrain).render(clean, DroneState(), "TEST", 30.0, 1.0)
    assert np.array_equal(clean, before)
    assert rendered is not clean
    assert np.count_nonzero(rendered) > 0


def test_dem_sampling_uses_geotransforms_instead_of_size_ratio():
    terrain = OrthophotoMap.__new__(OrthophotoMap)
    terrain._transform = Affine.translation(1000.0, 2000.0) * Affine.scale(2.0, -2.0)
    terrain._crs = "EPSG:3857"
    terrain._elevation_transform = Affine.translation(900.0, 2100.0) * Affine.scale(10.0, -10.0)
    terrain._elevation_inv_transform = ~terrain._elevation_transform
    terrain._elevation_crs = "EPSG:3857"
    terrain._map_to_elevation_crs = None
    yy, xx = np.mgrid[0:50, 0:60]
    terrain.elevation = (xx + 100 * yy).astype(np.float32)

    cols = np.array([20.0])
    rows = np.array([30.0])
    dem_cols, dem_rows = terrain.map_pixels_to_elevation_pixels(cols, rows)
    assert dem_cols[0] == 14.0
    assert dem_rows[0] == 16.0
    assert terrain.sample_elevation_at_map_pixels(cols, rows)[0] == 1614.0


def test_single_band_dem_in_metres_is_auto_detected():
    transform = Affine.translation(1000.0, 2000.0) * Affine.scale(2.0, -2.0)
    with tempfile.TemporaryDirectory() as directory:
        ortho_path = Path(directory) / "ortho.tif"
        dem_path = Path(directory) / "dem.tif"
        with rasterio.open(
            ortho_path,
            "w",
            driver="GTiff",
            width=10,
            height=10,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=transform,
        ) as dataset:
            dataset.write(np.zeros((3, 10, 10), dtype=np.uint8))
        heights = np.full((10, 10), 123.5, dtype=np.float32)
        heights[0, 0] = -9999.0
        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            width=10,
            height=10,
            count=1,
            dtype="float32",
            nodata=-9999.0,
            crs="EPSG:3857",
            transform=transform,
        ) as dataset:
            dataset.write(heights, 1)

        terrain = OrthophotoMap(str(ortho_path), str(dem_path))
        assert terrain.elevation_format_actual == "meters"
        assert terrain.base_elevation == 123.5
        assert terrain.sample_elevation_at_map_pixels(np.array([5]), np.array([5]))[0] == 123.5
        assert np.isnan(
            terrain.sample_elevation_at_map_pixels(np.array([0]), np.array([0]))[0]
        )


def test_local_coordinates_correct_web_mercator_ground_scale():
    terrain = OrthophotoMap.__new__(OrthophotoMap)
    terrain._center_x = 1000.0
    terrain._center_y = 2000.0
    terrain.ground_scale = 0.5
    assert terrain.local_to_mercator(50.0, -25.0) == (1100.0, 1950.0)
    assert terrain.mercator_to_local(1100.0, 1950.0) == (50.0, -25.0)


class IdleSource(CommandSource):
    def get_command(self, state, dt):
        return CommandVector()

    def is_finished(self):
        return False

    @property
    def mode_name(self):
        return "IDLE"


class FinishedSource(IdleSource):
    def is_finished(self):
        return True


def test_scenario_reaches_exact_next_frame_altitude():
    scenario = FlightScenario.load(Path("scenarios/ascent_descent.json"))
    source = ProfiledCommandSource(IdleSource(), scenario)
    state = DroneState(position=np.array([0.0, 0.0, 500.0]), time=0.0)
    dt = 1.0 / 30.0
    command = source.get_command(state, dt)
    expected = scenario.altitude_m.value_at(dt)
    assert math.isclose(state.altitude + command.vz * dt, expected, abs_tol=1e-12)


def test_scenario_duration_is_not_cut_short_by_base_route():
    scenario = FlightScenario.load(Path("scenarios/ascent_descent.json"))
    source = ProfiledCommandSource(FinishedSource(), scenario)
    assert not source.is_finished()
    source.get_command(
        DroneState(position=np.array([0.0, 0.0, 500.0]), time=0.0), 1.0 / 30.0
    )
    assert not source.is_finished()


def test_telemetry_carries_video_frame_identity():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "telemetry.csv"
        logger = TelemetryLogger(str(path), log_interval_frames=1)
        logger.log(DroneState(time=1.25), frame_index=37, timestamp=37 / 30)
        logger.close()
        row = next(csv.DictReader(path.open(encoding="utf-8")))
        assert row["frame_index"] == "37"
        assert math.isclose(float(row["timestamp"]), 37 / 30, abs_tol=1e-9)


def test_sparse_telemetry_starts_at_video_frame_zero():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "telemetry.csv"
        logger = TelemetryLogger(str(path), log_interval_frames=15)
        for frame_index in range(16):
            logger.log(
                DroneState(time=frame_index / 30),
                frame_index=frame_index,
                timestamp=frame_index / 30,
            )
        logger.close()
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        assert [int(row["frame_index"]) for row in rows] == [0, 15]


def test_frame_ground_truth_has_one_atomic_row_per_video_frame():
    terrain = SyntheticMap()
    width, height = 128, 72
    H1 = np.array(
        [[1.0, 0.0, width / 2.0], [0.0, -1.0, height / 2.0], [0.0, 0.0, 1.0]]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "frame_ground_truth.jsonl"
        logger = FrameGroundTruthLogger(
            str(path), terrain, width=width, height=height, fps=30, renderer="cpu"
        )
        for frame_index in range(3):
            logger.log(
                H1,
                DroneState(
                    position=np.array([0.0, 0.0, 100.0]),
                    time=frame_index / 30,
                ),
                frame_index=frame_index,
                surface_valid_fraction=1.0,
            )
        logger.close()
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert [row["frame_index"] for row in rows] == [0, 1, 2]
        assert [row["timestamp_s"] for row in rows] == [0.0, 1 / 30, 2 / 30]
        assert all(row["ground_center_valid"] for row in rows)
        assert rows[0]["ground_center_local_m"] == [0.0, 0.0]
        assert not path.with_name(f".{path.name}.part").exists()


def test_surface_projection_rejects_rays_behind_camera():
    terrain = SyntheticMap()
    state = DroneState(position=np.array([0.0, 0.0, 100.0]))
    behind_camera = np.diag([1.0, 1.0, -1.0])
    try:
        project_image_pixels_to_surface(
            behind_camera, np.array([[64.0, 36.0]]), terrain, state
        )
    except ProjectionGeometryError:
        return
    raise AssertionError("A ray behind the camera was accepted as ground truth")


if __name__ == "__main__":
    test_cpu_renderer_projects_map_centre_to_image_centre()
    test_hud_does_not_mutate_clean_frame()
    test_dem_sampling_uses_geotransforms_instead_of_size_ratio()
    test_single_band_dem_in_metres_is_auto_detected()
    test_local_coordinates_correct_web_mercator_ground_scale()
    test_scenario_reaches_exact_next_frame_altitude()
    test_scenario_duration_is_not_cut_short_by_base_route()
    test_telemetry_carries_video_frame_identity()
    test_sparse_telemetry_starts_at_video_frame_zero()
    test_frame_ground_truth_has_one_atomic_row_per_video_frame()
    test_surface_projection_rejects_rays_behind_camera()
    print("renderer/scenario contracts: OK")
