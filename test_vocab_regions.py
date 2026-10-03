"""run_vocab_regions.py: region maps fit the vocabulary scenario and avoid the test area.

Run: python -m pytest -q test_vocab_regions.py   (numpy only)
"""

import argparse
import math

import pytest

import run_vocab_regions as rvr
from simulator.planning.scenario import FlightScenario


def footprint(altitude_m):
    # camera of the recordings: 1280x720, focal 13.2 mm, sensor width 8.8 mm
    width = altitude_m * 8.8 / 13.2
    return width, width * 720 / 1280


def test_bbox_is_square_in_ground_metres():
    lat_min, lon_min, lat_max, lon_max = rvr.region_bbox(50.0, 30.0, 4300.0)
    height = (lat_max - lat_min) * 111_320
    width = (lon_max - lon_min) * 111_320 * math.cos(math.radians(50.0))
    assert height == pytest.approx(8600, rel=1e-6)
    assert width == pytest.approx(8600, rel=1e-6)


def test_default_regions_fit_the_scenario_and_avoid_the_test_area():
    scenario = FlightScenario.load(rvr.ROOT_DIR / rvr.DEFAULT_SCENARIO)
    half = rvr.parse_args([]).half_size_m
    scenario.route_waypoints((-half, -half, half, half), footprint)  # raises near the edge
    for name, (lat, lon) in rvr.REGIONS.items():
        assert rvr.overlapping_area(rvr.region_bbox(lat, lon, half)) is None, name


def test_test_area_and_bad_region_specs_are_refused():
    with pytest.raises(SystemExit, match="overlaps"):
        rvr.build_tasks(rvr.parse_args(["--region", "home=48.42,26.18"]))
    for bad in ("x", "x=1", "x=95,10", "a b=1,2"):
        with pytest.raises(argparse.ArgumentTypeError):
            rvr.parse_region(bad)


def test_command_carries_bbox_scenario_and_outputs():
    args = rvr.parse_args(["--only", "steppe"])
    [task] = rvr.build_tasks(args)
    cmd = rvr.build_command(task, args)
    assert cmd[cmd.index("--scenario") + 1] == rvr.DEFAULT_SCENARIO
    assert float(cmd[cmd.index("--lat_min") + 1]) < 48.2 < float(cmd[cmd.index("--lat_max") + 1])
    assert cmd[cmd.index("--zoom") + 1] == "18"
    assert cmd[cmd.index("--video-file") + 1].endswith("output/vocab_steppe/video.mp4")


def test_gpu_renderer_and_memory_estimate():
    args = rvr.parse_args(["--only", "polissia"])
    [task] = rvr.build_tasks(args)
    cmd = rvr.build_command(task, args)
    assert cmd[cmd.index("--renderer") + 1] == "gpu" and args.jobs == 1
    # the simulator's own log for this bbox: "this will download 8281 tiles"
    assert rvr.tile_count(task["bbox"], 18) == (91, 91)
    assert rvr.map_gpu_gb(task["bbox"], 18) == pytest.approx(6.51, abs=0.01)


def test_recorded_renderer_reads_the_manifest(tmp_path):
    assert rvr.recorded_renderer(tmp_path) == "unknown"
    (tmp_path / "manifest.json").write_text('{"renderer": "gpu"}', encoding="utf-8")
    assert rvr.recorded_renderer(tmp_path) == "gpu"
