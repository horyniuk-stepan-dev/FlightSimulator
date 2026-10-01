"""Scenario routes: their own horizontal path instead of the survey lawnmower.

Run: python -m pytest -q test_scenario_route.py   (numpy only)
"""

import json

import pytest

from simulator.planning.scenario import FlightScenario

BOUNDS = (-5000.0, -2000.0, 5000.0, 2000.0)


def footprint(altitude_m):
    return altitude_m * 2 / 3, altitude_m * 3 / 8  # 1280x720 sensor, f = 1.5 x width


def write(tmp_path, route, altitude=((0.0, 1000.0), (10.0, 2000.0)), pitch=((0.0, 0.0),)):
    data = {
        "version": 1,
        "duration_s": 10.0,
        "profiles": {
            "altitude_m": [{"time_s": t, "value": v} for t, v in altitude],
            "pitch_deg": [{"time_s": t, "value": v} for t, v in pitch],
        },
    }
    if route is not None:
        data["route"] = [{"x_m": x, "y_m": y} for x, y in route]
    path = tmp_path / "s.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_route_becomes_waypoints_at_start_altitude(tmp_path):
    scenario = FlightScenario.load(write(tmp_path, [(-3000, 0), (0, 500), (3000, -500)]))
    waypoints = scenario.route_waypoints(BOUNDS, footprint)
    assert [(w.x, w.y, w.z) for w in waypoints] == [
        (-3000, 0, 1000.0),
        (0, 500, 1000.0),
        (3000, -500, 1000.0),
    ]


def test_without_route_the_scenario_keeps_the_survey_path(tmp_path):
    assert FlightScenario.load(write(tmp_path, None)).route_xy == ()


def test_route_near_the_edge_at_highest_altitude_and_tilt_is_refused(tmp_path):
    # margin at 2000 m: half footprint diagonal 765 m + 2000 * tan(20 deg) = 728 m
    path = write(tmp_path, [(0, 0), (0, 600)], pitch=((0.0, 0.0), (5.0, 20.0)))
    with pytest.raises(ValueError, match="too close to the map edge"):
        FlightScenario.load(path).route_waypoints(BOUNDS, footprint)
    FlightScenario.load(write(tmp_path, [(0, 0), (0, 600)])).route_waypoints(BOUNDS, footprint)


def test_route_needs_two_finite_points(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        FlightScenario.load(write(tmp_path, [(0, 0)]))
    with pytest.raises(ValueError, match="non-finite"):
        FlightScenario.load(write(tmp_path, [(0, 0), (float("nan"), 0)]))
