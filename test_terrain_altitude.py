"""Altitude datum (base / msl), DEM spike removal and terrain clearance.

Run: python -m pytest -q test_terrain_altitude.py
"""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from affine import Affine

import run_vocab_regions as rvr
from simulator.camera.camera_model import CameraModel
from simulator.planning.altitude import (
    TerrainClearanceError,
    altitude_shift_m,
    camera_clearance_m,
    positions_along_route,
    predicted_route_clearance,
    relief_top_above_base,
)
from simulator.planning.scenario import FlightScenario
from simulator.planning.survey_planner import SurveyPlanner
from simulator.terrain.orthophoto_map import (
    OrthophotoMap,
    dem_spike_threshold_m,
    remove_elevation_spikes,
)

ROOT = Path(__file__).resolve().parent


def smooth_terrain(h=200, w=240):
    yy, xx = np.mgrid[0:h, 0:w]
    return (150.0 + 40.0 * np.sin(xx / 30.0) + 25.0 * np.cos(yy / 20.0)).astype(np.float32)


def test_dem_pits_and_spikes_are_replaced_and_gaps_kept():
    terrain = smooth_terrain()
    dem = terrain.copy()
    dem[10, 20] = -2404.9  # single-pixel void (Kaniv reservoir)
    dem[50:52, 30:66] = -1819.6  # 2 x 36 line of voids
    dem[100:103, 100:104] = -900.0  # 3 x 4 cluster (steppe)
    dem[150, 150] = 5000.0  # positive spike
    dem[0:3, 0:3] = np.nan  # no coverage must stay no coverage

    cleaned, replaced = remove_elevation_spikes(dem)

    assert replaced == 1 + 72 + 12 + 1
    assert np.isnan(cleaned[0:3, 0:3]).all()
    finite = np.isfinite(cleaned)
    assert cleaned[finite].min() >= terrain.min() - 1.0
    assert cleaned[finite].max() <= terrain.max() + 1.0
    untouched = np.ones(dem.shape, dtype=bool)
    for sl in ((10, 20), (slice(50, 52), slice(30, 66)), (slice(100, 103), slice(100, 104)), (150, 150)):
        untouched[sl] = False
    untouched[0:3, 0:3] = False
    assert np.array_equal(cleaned[untouched], dem[untouched])
    # the replacement is the neighbourhood's terrain, not a constant
    assert abs(cleaned[10, 20] - terrain[10, 20]) < 10.0


def test_steep_real_relief_is_not_touched():
    yy, xx = np.mgrid[0:120, 0:120]
    # 300 m hill rising 6 m per pixel and an 80 m terrace step
    hill = 400.0 + np.clip(300.0 - 6.0 * np.hypot(xx - 60, yy - 60), 0.0, None)
    hill[:, :20] -= 80.0
    cleaned, replaced = remove_elevation_spikes(hill.astype(np.float32))
    assert replaced == 0
    assert np.array_equal(cleaned, hill.astype(np.float32))


def test_spike_threshold_scales_with_coarse_pixels():
    assert dem_spike_threshold_m(3.2) == 100.0  # z15 Terrarium
    assert dem_spike_threshold_m(None) == 100.0
    assert dem_spike_threshold_m(30.0) == pytest.approx(180.0)  # SRTM 1"


def write_rasters(directory, dem_m, pixel_m=10.0, origin=(3_000_000.0, 6_200_000.0)):
    """EPSG:3857 orthophoto + one-band DEM covering the same area."""
    h, w = dem_m.shape
    transform = Affine.translation(*origin) * Affine.scale(pixel_m, -pixel_m)
    ortho = Path(directory) / "ortho.tif"
    dem = Path(directory) / "dem.tif"
    yy, xx = np.mgrid[0:h, 0:w]
    image = np.stack([(xx * 7) % 256, (yy * 5) % 256, ((xx + yy) * 3) % 256]).astype(np.uint8)
    with rasterio.open(ortho, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8",
                       crs="EPSG:3857", transform=transform) as ds:
        ds.write(image)
    with rasterio.open(dem, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32",
                       crs="EPSG:3857", transform=transform) as ds:
        ds.write(dem_m.astype(np.float32), 1)
    return ortho, dem


def test_orthophoto_base_plane_ignores_dem_pits(tmp_path):
    dem = smooth_terrain(60, 60)
    dem[30, 30] = -986.8
    ortho, dem_path = write_rasters(tmp_path, dem)
    terrain = OrthophotoMap(str(ortho), str(dem_path))
    assert terrain.elevation_raw_min == pytest.approx(-986.8, abs=0.1)
    assert terrain.elevation_spikes_removed == 1
    expected = np.delete(dem.ravel(), 30 * 60 + 30).min()
    assert terrain.base_elevation == pytest.approx(expected, abs=0.5)


class HillMap:
    """Local metres == map pixels; a 300 m Gaussian hill at (0, 0) on 150 m ground."""

    base_elevation = 150.0
    elevation = np.zeros((1, 1))

    @staticmethod
    def local_to_pixel(x, y):
        return x, y

    @staticmethod
    def sample_elevation_at_map_pixels(cols, rows):
        cols = np.asarray(cols, dtype=float)
        rows = np.asarray(rows, dtype=float)
        h = 150.0 + 300.0 * np.exp(-(cols**2 + rows**2) / (2 * 400.0**2))
        return np.where(np.abs(cols) > 5000, np.nan, h)


def test_altitude_shift_for_both_references():
    terrain = HillMap()
    assert altitude_shift_m("base", 25.0, terrain) == 25.0
    assert altitude_shift_m("msl", 25.0, terrain) == -125.0
    with pytest.raises(ValueError, match="elevation raster"):
        altitude_shift_m("msl", 0.0, SimpleNamespace(elevation=None, base_elevation=0.0))
    with pytest.raises(ValueError):
        altitude_shift_m("agl", 0.0, terrain)


def test_scenario_altitude_offset_shifts_only_altitudes():
    scenario = FlightScenario.load(ROOT / "scenarios" / "vlad_vocab_survey.json")
    shifted = scenario.with_altitude_offset(-94.0)
    assert [k.value for k in shifted.altitude_m.keyframes] == [
        k.value - 94.0 for k in scenario.altitude_m.keyframes
    ]
    assert scenario.altitude_m.value_at(0.0) == 700.0  # original untouched
    assert shifted.pitch_deg is scenario.pitch_deg and shifted.route_xy == scenario.route_xy


def test_clearance_over_a_hill():
    terrain = HillMap()
    route = [(-2000.0, 0.0), (2000.0, 0.0)]
    # 400 m above Z=0 (550 m above sea level): 100 m over the hilltop
    report = predicted_route_clearance(terrain, route, 100.0, lambda t: 400.0, 40.0)
    assert report.min_clearance_m == pytest.approx(100.0, abs=1.0)
    assert report.time_s == pytest.approx(20.0, abs=0.5) and abs(report.x_m) < 60
    state = SimpleNamespace(position=np.array([0.0, 0.0, 400.0]), altitude=400.0)
    assert camera_clearance_m(terrain, state) == pytest.approx(100.0)
    state.position = np.array([9000.0, 0.0, 400.0])
    assert camera_clearance_m(terrain, state) is None  # off the DEM
    assert relief_top_above_base(terrain, (-1000, -1000, 1000, 1000)) == pytest.approx(300.0, abs=1.0)


def test_route_positions_hover_at_the_end():
    xy = positions_along_route([(0, 0), (100, 0), (100, 100)], 10.0, [0, 5, 15, 100])
    assert np.allclose(xy, [[0, 0], [50, 0], [100, 50], [100, 100]])


def test_survey_spacing_from_lowest_height_adds_lines():
    camera = CameraModel(1280, 720, 13.2, 8.8)
    bounds = (-4000.0, -4000.0, 4000.0, 4000.0)
    flat = SurveyPlanner.generate_path(bounds, 1000.0, camera, 30.0, 0.0)
    relief = SurveyPlanner.generate_path(bounds, 1000.0, camera, 30.0, 0.0, spacing_altitude_m=737.0)
    assert len(relief) > len(flat)
    xs = sorted({round(w.x, 3) for w in relief})
    width_at_lowest = 737.0 * 8.8 / 13.2
    assert max(np.diff(xs)) <= width_at_lowest * 0.7 + 1e-6
    with pytest.raises(ValueError, match="positive"):
        SurveyPlanner.generate_path(bounds, 1000.0, camera, 30.0, 0.0, spacing_altitude_m=-5.0)


def test_vocab_recorder_flies_above_sea_level():
    args = rvr.parse_args(["--only", "foothills", "--altitude-offset", "600"])
    [task] = rvr.build_tasks(args)
    cmd = rvr.build_command(task, args)
    assert cmd[cmd.index("--altitude-reference") + 1] == "msl"
    assert cmd[cmd.index("--altitude-offset") + 1] == "600"
    assert rvr.parse_args([]).altitude_reference == "msl"


def run_main(monkeypatch, tmp_path, *extra):
    from simulator import main as sim_main

    yy, xx = np.mgrid[0:200, 0:200]
    dem = 150.0 + 300.0 * np.exp(-((xx - 100) ** 2 + (yy - 100) ** 2) / (2 * 40.0**2))
    ortho, dem_path = write_rasters(tmp_path, dem)
    frame_gt = tmp_path / "frames.jsonl"
    argv = [
        "simulator.main", "--mode", "record", "--yes", "--no-display", "--fast",
        "--renderer", "cpu", "--geotiff", str(ortho), "--elevation", str(dem_path),
        "--speed", "50", "--max-frames", "2", "--no-gamepad",
        "--telemetry-file", str(tmp_path / "telemetry.csv"),
        "--frame-gt-file", str(frame_gt), *extra,
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sim_main.cv2, "destroyAllWindows", lambda: None)  # headless OpenCV
    sim_main.main()
    return [json.loads(line) for line in frame_gt.read_text(encoding="utf-8").splitlines()]


def test_main_refuses_a_survey_into_the_hill(monkeypatch, tmp_path):
    # hilltop 450 m above sea level; a 400 m flight would enter it
    with pytest.raises(TerrainClearanceError, match="at least 100 m"):
        run_main(monkeypatch, tmp_path, "--altitude-reference", "msl", "--altitude", "400")
    # 480 m clears the top by 30 m, less than the default 50 m clearance
    with pytest.raises(TerrainClearanceError, match="min-clearance 50"):
        run_main(monkeypatch, tmp_path, "--altitude-reference", "msl", "--altitude", "480")


def test_main_flies_msl_altitude_above_the_hill(monkeypatch, tmp_path):
    frames = run_main(monkeypatch, tmp_path, "--altitude-reference", "msl", "--altitude", "900")
    first = frames[0]
    lowest = 150.0 + 300.0 * math.exp(-(100**2 + 100**2) / (2 * 40.0**2))  # DEM corner
    assert first["camera_altitude_msl_m"] == pytest.approx(900.0, abs=0.01)
    assert first["coordinate_system"]["base_elevation_m"] == pytest.approx(lowest, abs=0.01)
    assert first["camera_altitude_base_m"] == pytest.approx(900.0 - lowest, abs=0.01)
    assert first["camera_agl_m"] > 450.0 - 1.0
    assert math.isclose(
        first["camera_altitude_base_m"] + first["coordinate_system"]["base_elevation_m"],
        first["camera_altitude_msl_m"],
    )
