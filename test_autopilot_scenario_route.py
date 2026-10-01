"""Scenario runs own the altitude; the survey route must keep flying.

Before the fix AutoPilot measured waypoint arrival in 3-D against the survey
altitude. A scenario holding the drone >150 m above/below that altitude could
never "arrive", so the route stalled at the first leg end and the drone hovered
(the 2026-09-25 combined_scale_tilt query hovered from t~30 s to t~72 s).

Run: python -m pytest -q test_autopilot_scenario_route.py   (numpy only)
"""

import math

import numpy as np

from simulator.control.auto_pilot import AutoPilot
from simulator.physics.drone_state import DroneState
from simulator.planning.survey_planner import Waypoint

FPS = 30.0
ROUTE = [Waypoint(0.0, 0.0, 1000.0), Waypoint(0.0, 3000.0, 1000.0), Waypoint(450.0, 3000.0, 1000.0)]


def _fly(pilot, altitude_m, seconds):
    """Kinematic integration as in main.py (kinematic=True); altitude is held
    by the caller, exactly as ProfiledCommandSource overrides vz."""
    pos = np.array([ROUTE[0].x, ROUTE[0].y, altitude_m], float)
    speeds, commands = [], []
    for i in range(int(seconds * FPS)):
        state = DroneState(position=pos.copy(), time=i / FPS)
        cmd = pilot.get_command(state, 1.0 / FPS)
        commands.append((cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate))
        pos[0] += cmd.vx / FPS
        pos[1] += cmd.vy / FPS
        speeds.append(math.hypot(cmd.vx, cmd.vy))
    return pos, speeds, commands


def test_route_continues_while_scenario_holds_other_altitude():
    pilot = AutoPilot(ROUTE, speed_m_s=150.0, horizontal_only=True)
    pos, speeds, _ = _fly(pilot, altitude_m=2400.0, seconds=30.0)
    assert pilot.current_wp_idx >= 2, "leg end must be reached at 2400 m"
    assert pos[0] > 100.0, "drone must turn onto the next leg"
    cruise = speeds[int(3 * FPS) : int(15 * FPS)]
    assert min(cruise) > 145.0, "horizontal cruise speed must not shrink with |dz|"


def test_default_keeps_3d_arrival_for_non_scenario_flights():
    # Documents the behaviour the flag exists for; reference flights (no
    # scenario) keep the original 3-D logic untouched.
    pilot = AutoPilot(ROUTE, speed_m_s=150.0)
    pos, _, _ = _fly(pilot, altitude_m=2400.0, seconds=60.0)
    # Starting on waypoint 0 but 1400 m off its altitude: never "arrives".
    assert pilot.current_wp_idx == 0
    assert math.hypot(pos[0], pos[1]) < 1.0


def test_constant_altitude_commands_identical():
    # batch.py references use a constant scenario at the survey altitude:
    # horizontal_only must not change a single command there.
    _, _, a = _fly(AutoPilot(ROUTE, speed_m_s=150.0), altitude_m=1000.0, seconds=40.0)
    _, _, b = _fly(AutoPilot(ROUTE, speed_m_s=150.0, horizontal_only=True), 1000.0, 40.0)
    assert a == b


if __name__ == "__main__":
    test_route_continues_while_scenario_holds_other_altitude()
    test_default_keeps_3d_arrival_for_non_scenario_flights()
    test_constant_altitude_commands_identical()
    print("OK")
