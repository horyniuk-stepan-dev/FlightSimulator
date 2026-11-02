"""
Electro-mechanical powertrain: battery -> ESC duty -> BLDC motor -> prop.

Per motor (DC-equivalent BLDC model, what eCalc-style tools use):
    V_m   = duty * V_batt
    I     = (V_m - omega / Kv) / R                (Kv in rad/s per volt)
    tau_m = (I - I0) / Kv                         (Kt = 1/Kv in SI)
    Q     = Cp/(2pi) * rho * n^2 * D^5,  T = Ct * rho * n^2 * D^4
    J * d(omega)/dt = tau_m - Q
Battery: V = OCV(SoC) - I_total * R_pack, capacity drains with current.

So KV, winding resistance, prop size/pitch, cell count, C-rating/IR and mass
all show up in thrust, response time, sag and flight time.
"""

from __future__ import annotations

import math

import numpy as np

from .components import INCH, RHO_AIR, DroneBuild

RPM_TO_RADS = 2 * math.pi / 60

# Per-cell open-circuit voltage vs state of charge (approximate curves).
_SOC = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
_OCV_LIPO = np.array([3.30, 3.50, 3.60, 3.70, 3.78, 3.87, 4.00, 4.08, 4.20])
_OCV_LIION = np.array([2.80, 3.10, 3.30, 3.45, 3.60, 3.75, 3.95, 4.05, 4.20])


class Battery:
    def __init__(self, b: DroneBuild):
        bat = b.battery
        self.cells = bat.cells
        self.capacity_ah = bat.capacity_mah / 1000.0
        self.r_pack = bat.cells * bat.cell_resistance_ohm + 0.004  # + leads/connector
        self.max_current = bat.capacity_mah / 1000.0 * bat.c_rating
        self._ocv = _OCV_LIION if "li-ion" in bat.name.lower() else _OCV_LIPO
        self.used_ah = 0.0
        self.voltage = self.ocv()
        self.current = 0.0

    @property
    def soc(self) -> float:
        return max(0.0, 1.0 - self.used_ah / self.capacity_ah)

    def ocv(self) -> float:
        return self.cells * float(np.interp(self.soc, _SOC, self._ocv))

    def update(self, current_a: float, dt: float) -> float:
        self.current = current_a
        self.used_ah += max(current_a, 0.0) * dt / 3600.0
        self.voltage = max(self.ocv() - current_a * self.r_pack, 0.5 * self.cells)
        return self.voltage


class Motors:
    """Four motors+props, vectorised. Order = Betaflight quad-X:
    1 RR, 2 FR, 3 RL, 4 FL (index 0..3)."""

    def __init__(self, b: DroneBuild):
        m, p = b.motor, b.prop
        self.kv = m.kv * RPM_TO_RADS
        self.kt = 1.0 / self.kv
        self.R = m.resistance_ohm
        self.I0 = m.no_load_current_a
        self.i_max = min(m.max_current_a * 1.15, b.electronics.esc_max_current_a * 1.2)
        D = p.diameter_in * INCH
        # T = kT * omega^2, Q = kQ * omega^2   (n = omega / 2pi)
        self.k_thrust = p.ct * RHO_AIR * D**4 / (4 * math.pi**2)
        self.k_torque = p.cp / (2 * math.pi) * RHO_AIR * D**5 / (4 * math.pi**2)
        self.J = b.prop_inertia
        self.idle = b.electronics.motor_idle
        self.omega = np.zeros(4)
        self.current = np.zeros(4)
        self.omega_dot = np.zeros(4)

    def duty_from_output(self, out: np.ndarray) -> np.ndarray:
        """Betaflight motor output 0..1 -> ESC duty with dshot idle."""
        return self.idle + (1.0 - self.idle) * np.clip(out, 0.0, 1.0)

    def step(self, duty: np.ndarray, v_batt: float, dt: float) -> None:
        emf = self.omega / self.kv
        i = (duty * v_batt - emf) / self.R
        i = np.clip(i, -0.5 * self.i_max, self.i_max)  # ESC limit; neg = active braking
        tau = self.kt * (i - self.I0 * np.sign(self.omega + 1e-9))
        q = self.k_torque * self.omega**2
        self.omega_dot = (tau - q) / self.J
        self.omega = np.maximum(self.omega + self.omega_dot * dt, 0.0)
        self.current = i

    @property
    def thrust(self) -> np.ndarray:
        return self.k_thrust * self.omega**2

    @property
    def drag_torque(self) -> np.ndarray:
        return self.k_torque * self.omega**2

    # ---- steady-state helpers (for build reports) -------------------------
    def steady_omega(self, duty: float, v: float) -> float:
        """Solve kQ w^2 + Kt/(Kv R) w - Kt (dV/R - I0) = 0, respecting current limit."""
        a = self.k_torque
        bq = self.kt / (self.kv * self.R)
        c = -self.kt * (duty * v / self.R - self.I0)
        w = (-bq + math.sqrt(bq * bq - 4 * a * c)) / (2 * a)
        i = (duty * v - w / self.kv) / self.R
        if i > self.i_max:  # current-limited: tau = Kt (Imax - I0) = kQ w^2
            w = math.sqrt(self.kt * (self.i_max - self.I0) / a)
        return max(w, 0.0)

    def steady_current(self, duty: float, v: float, w: float) -> float:
        return min((duty * v - w / self.kv) / self.R, self.i_max)
