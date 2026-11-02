"""
Export a DroneBuild as a RotorPy quad_params dict, so the existing
simulator.physics.FlightController (velocity-command survey mode) can fly
with the same mass / inertia / motor limits as the FPV build.

RotorPy's motor model is first-order (tau_m) with T = k_eta * w^2, so battery
sag and current limits are not represented there; use FpvQuadSim for those.
"""

from __future__ import annotations

import math

import numpy as np

from .components import DroneBuild
from .powertrain import Battery, Motors


def to_rotorpy_quad_params(b: DroneBuild, k_v: float = 4.0, kp_att: float = 1000.0, kd_att: float = 60.0) -> dict:
    m = Motors(b)
    bat = Battery(b)
    w_max = m.steady_omega(1.0, bat.ocv() * 0.9)  # ~ loaded full throttle
    ixx, iyy, izz = b.inertia
    d = b.arm_m
    # mechanical time constant of motor+prop around hover
    tau_m = m.J * m.R / m.kt**2
    return {
        "mass": b.mass_kg,
        "Ixx": ixx,
        "Iyy": iyy,
        "Izz": izz,
        "Ixy": 0.0,
        "Iyz": 0.0,
        "Ixz": 0.0,
        "num_rotors": 4,
        "rotor_pos": {
            "r1": d * np.array([0.707, 0.707, 0]),
            "r2": d * np.array([0.707, -0.707, 0]),
            "r3": d * np.array([-0.707, -0.707, 0]),
            "r4": d * np.array([-0.707, 0.707, 0]),
        },
        "rotor_directions": np.array([1, -1, 1, -1]),
        "rI": np.array([0, 0, 0]),
        "c_Dx": b.frame.drag_area_m2 * 0.5 * 1.225,
        "c_Dy": b.frame.drag_area_m2 * 0.5 * 1.225,
        "c_Dz": b.frame.drag_area_m2 * 1.225,
        "k_eta": m.k_thrust,
        "k_m": m.k_torque,
        "k_d": 0.0,
        "k_z": 0.0,
        "k_h": 0.0,
        "k_flap": 0.0,
        "tau_m": max(tau_m, 0.005),
        "rotor_speed_min": 0,
        "rotor_speed_max": float(w_max),
        "motor_noise_std": 0.0,
        "k_w": 50,
        "k_v": k_v,
        "kp_att": kp_att,
        "kd_att": kd_att,
        "_max_rpm": w_max * 60 / (2 * math.pi),
    }
