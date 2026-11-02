"""Behavioural tests: swapping a component must change flight behaviour the
way physics says it should."""

import numpy as np
import pytest
from simulator.fpv import build_preset
from simulator.fpv.betaflight import ActualRates, BetaflightController, BetaflightRates, Sticks
from simulator.fpv.quad_sim import FpvQuadSim
from simulator.fpv.report import static_report
from simulator.fpv.rotorpy_adapter import to_rotorpy_quad_params


def test_actual_rates_endpoints():
    r = ActualRates(center_dps=70, max_dps=670, expo=0.54)
    assert r(0.0) == 0.0
    assert r(1.0) == pytest.approx(670)
    assert r(-1.0) == pytest.approx(-670)
    assert r(0.01) == pytest.approx(0.7, rel=0.05)  # centre sensitivity 70 deg/s per full stick


def test_betaflight_rates_full_stick():
    # rc_rate 1.0, super 0.7, no expo -> 200 / (1 - 0.7) = 666.7 deg/s
    assert BetaflightRates(1.0, 0.7, 0.0)(1.0) == pytest.approx(666.67, rel=1e-3)


def test_airmode_keeps_authority_at_zero_throttle():
    fc = BetaflightController(1 / 2000)
    out = fc.mix(np.array([0.2, 0.0, 0.0]), throttle=0.0)
    assert out.max() - out.min() == pytest.approx(0.4)  # full differential preserved
    fc.p.airmode = False
    out = fc.mix(np.array([0.2, 0.0, 0.0]), throttle=0.0)
    assert out.max() - out.min() == pytest.approx(0.2)  # half clipped away


def test_bigger_battery_heavier_longer_hover():
    small = static_report(build_preset("freestyle_5in_6s", battery="6s_1100"))
    big = static_report(build_preset("freestyle_5in_6s", battery="6s_1300"))
    assert big["mass_g"] > small["mass_g"]
    assert big["hover_time_min"] > small["hover_time_min"]


def test_fewer_cells_less_thrust():
    s6 = static_report(build_preset("freestyle_5in_6s"))
    s4 = static_report(build_preset("freestyle_5in_6s", battery="4s_1500"))
    assert s4["max_thrust_g"] < 0.75 * s6["max_thrust_g"]


def test_higher_pitch_more_thrust_more_current():
    lo = static_report(build_preset("freestyle_5in_6s", prop="5x3_tri"))
    hi = static_report(build_preset("freestyle_5in_6s", prop="5x5_tri"))
    assert hi["full_throttle_current_a"] > lo["full_throttle_current_a"]
    assert hi["full_throttle_rpm"] < lo["full_throttle_rpm"]


def test_payload_raises_hover_throttle():
    a = static_report(build_preset("longrange_7in_6s"))
    b = static_report(build_preset("longrange_7in_6s", payload_g=400))
    assert b["hover_throttle_pct"] > a["hover_throttle_pct"]
    assert b["thrust_to_weight"] < a["thrust_to_weight"]


def test_hover_is_stable_in_acro():
    b = build_preset("freestyle_5in_6s")
    hover = static_report(b)["hover_throttle_pct"] / 100
    sim = FpvQuadSim(b)
    sim.state.pos[2] = 10.0
    sim.on_ground = False
    sim.motors.omega[:] = sim.motors.steady_omega(
        sim.motors.duty_from_output(np.array([hover]))[0], sim.battery.voltage
    )
    log = sim.run(lambda t: Sticks(throttle=hover), 1.0)
    assert abs(log["pos"][-1, 2] - 10.0) < 1.0
    assert np.abs(log["gyro"]).max() < 5.0


def test_roll_direction_matches_betaflight():
    b = build_preset("freestyle_5in_6s")
    sim = FpvQuadSim(b)
    sim.state.pos[2] = 10.0
    sim.on_ground = False
    sim.motors.omega[:] = 1200.0
    log = sim.run(lambda t: Sticks(roll=0.5, throttle=0.3), 0.2)
    assert log["gyro"][-1, 0] > 100  # +roll stick -> +roll rate
    assert log["att"][-1][0] > 0  # right wing down


def test_rotorpy_export_hovers():
    p = to_rotorpy_quad_params(build_preset("freestyle_5in_6s"))
    hover_w = np.sqrt(p["mass"] * 9.81 / 4 / p["k_eta"])
    assert hover_w < 0.5 * p["rotor_speed_max"]
