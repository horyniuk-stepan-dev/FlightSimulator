"""
Build report: static powertrain numbers + closed-loop flight tests.
Shows directly how each component choice changes behaviour.
"""

from __future__ import annotations

import math

import numpy as np

from .betaflight import Sticks
from .components import DroneBuild
from .powertrain import Battery, Motors
from .quad_sim import ESC_EFFICIENCY, FpvQuadSim, G


def static_report(b: DroneBuild) -> dict:
    m = Motors(b)
    bat = Battery(b)
    weight_n = b.mass_kg * G
    # full throttle, iterate battery sag
    v = bat.ocv()
    for _ in range(20):
        w = m.steady_omega(1.0, v)
        i = 4 * m.steady_current(1.0, v, w) / ESC_EFFICIENCY + 0.6
        v = 0.5 * v + 0.5 * max(bat.ocv() - i * bat.r_pack, 0.5 * bat.cells)
    t_max = 4 * m.k_thrust * w**2
    i_max = i
    v_full = v
    rpm_full = w * 60 / (2 * math.pi)
    # hover duty by bisection
    lo, hi = 0.0, 1.0
    for _ in range(40):
        d = 0.5 * (lo + hi)
        vh = bat.ocv()
        for _ in range(5):
            wh = m.steady_omega(d, vh)
            ih = 4 * d * m.steady_current(d, vh, wh) / ESC_EFFICIENCY + 0.6  # battery side: I_batt = duty * I_motor
            vh = bat.ocv() - ih * bat.r_pack
        if 4 * m.k_thrust * wh**2 > weight_n:
            hi = d
        else:
            lo = d
    hover_out = (d - m.idle) / (1 - m.idle)
    usable_ah = 0.8 * bat.capacity_ah
    hover_min = usable_ah / max(ih, 1e-3) * 60
    # top speed: forward tilt where vertical thrust = weight, rest vs drag
    tw = t_max / weight_n
    if tw > 1:
        horiz = math.sqrt(t_max**2 - weight_n**2)
        v_top = math.sqrt(2 * horiz / (1.225 * b.frame.drag_area_m2))
    else:
        v_top = 0.0
    tip_mach = w * b.prop.diameter_in * 0.0254 / 2 / 343
    if tip_mach > 0.8:
        b.warnings.append(f"prop tip Mach {tip_mach:.2f} at full throttle: too much KV x cells for this prop")
    if i_max > bat.max_current:
        b.warnings.append(f"full-throttle draw {i_max:.0f} A exceeds battery C-rating ({bat.max_current:.0f} A)")
    return {
        "mass_g": b.mass_kg * 1000,
        "thrust_to_weight": tw,
        "max_thrust_g": t_max / G * 1000,
        "full_throttle_rpm": rpm_full,
        "full_throttle_current_a": i_max,
        "full_throttle_sag_v": bat.ocv() - v_full,
        "battery_c_limit_a": bat.max_current,
        "hover_throttle_pct": 100 * hover_out,
        "hover_current_a": ih,
        "hover_time_min": hover_min,
        "top_speed_kmh_est": v_top * 3.6,
        "inertia_kgm2": b.inertia,
        "tip_mach_full": tip_mach,
    }


def _rise_time(t, y, target, t0):
    mask = t >= t0
    tt, yy = t[mask], y[mask]
    i10 = np.argmax(yy >= 0.1 * target)
    i90 = np.argmax(yy >= 0.9 * target)
    if yy.max() < 0.9 * target:
        return float("nan"), float(yy.max() / target - 1) * 100
    return float(tt[i90] - tt[i10]) * 1000, float(yy.max() / target - 1) * 100


def dynamic_report(b: DroneBuild, hover_out: float, **sim_kw) -> dict:
    """Hover-trimmed tests: roll-rate step (acro), punch-out, angle-mode tilt."""
    res = {}
    # 1) roll-rate step to full stick
    sim = FpvQuadSim(b, **sim_kw)
    sim.state.pos[2] = 20.0
    sim.on_ground = False
    sim.motors.omega[:] = sim.motors.steady_omega(
        sim.motors.duty_from_output(np.array([hover_out]))[0], sim.battery.voltage
    )
    t_step = 0.3
    log = sim.run(lambda t: Sticks(roll=1.0 if t >= t_step else 0.0, throttle=hover_out), 0.8)
    target = sim.fc.rates[0](1.0)
    rt, ov = _rise_time(log["t"], log["gyro"][:, 0], target, t_step)
    res["roll_step_target_dps"] = target
    res["roll_rise_ms"] = rt
    res["roll_overshoot_pct"] = ov
    tracking = log["gyro"][log["t"] > t_step + 0.25, 0]
    res["roll_rate_reached_dps"] = float(np.mean(tracking))

    # 2) punch-out: hover -> full throttle for 1 s (acro, level)
    sim = FpvQuadSim(b, **sim_kw)
    sim.state.pos[2] = 20.0
    sim.on_ground = False
    sim.motors.omega[:] = sim.motors.steady_omega(
        sim.motors.duty_from_output(np.array([hover_out]))[0], sim.battery.voltage
    )
    log = sim.run(lambda t: Sticks(throttle=hover_out if t < 0.2 else 1.0), 1.2)
    res["punch_climb_m_1s"] = float(log["pos"][-1, 2] - 20.0)
    res["punch_min_vbat"] = float(log["vbat"].min())
    res["punch_peak_amps"] = float(log["amps"].max())

    # 3) angle mode: full forward stick for 3 s, how fast does it go
    sim = FpvQuadSim(b, **sim_kw)
    sim.fc.mode.mode = "angle"
    sim.state.pos[2] = 50.0
    sim.on_ground = False
    sim.motors.omega[:] = sim.motors.steady_omega(
        sim.motors.duty_from_output(np.array([hover_out]))[0], sim.battery.voltage
    )
    log = sim.run(lambda t: Sticks(pitch=1.0, throttle=min(1.0, hover_out * 1.5)), 3.0)
    res["angle_pitch_deg"] = float(log["att"][-1][1])
    res["angle_fwd_speed_kmh_3s"] = float(np.linalg.norm(log["vel"][-1, :2]) * 3.6)
    return res


def full_report(b: DroneBuild, **sim_kw) -> dict:
    s = static_report(b)
    d = dynamic_report(b, s["hover_throttle_pct"] / 100, **sim_kw) if s["thrust_to_weight"] > 1 else {}
    return {"static": s, "dynamic": d, "warnings": list(b.warnings)}
