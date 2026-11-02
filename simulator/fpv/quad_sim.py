"""
6-DOF FPV quad simulation driven by a DroneBuild + BetaflightController.

World frame ENU-like, z up. Body frame FLU. Quaternion [x, y, z, w].
Includes: per-motor thrust & drag torque, motor spin-up reaction torque,
gyroscopic body term, quadratic air drag, battery sag, ground contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .betaflight import BetaflightController, PidProfile, Sticks
from .components import RHO_AIR, DroneBuild
from .powertrain import Battery, Motors

G = 9.81
ESC_EFFICIENCY = 0.92
# spin direction seen from top: +1 CCW, -1 CW (order RR, FR, RL, FL)
SPIN = np.array([-1.0, +1.0, +1.0, -1.0])


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


@dataclass
class QuadState:
    pos: np.ndarray
    vel: np.ndarray
    q: np.ndarray
    w: np.ndarray  # body rates rad/s

    def roll_pitch_deg(self) -> tuple[float, float]:
        """Betaflight-convention attitude: +roll right-down, +pitch nose-down."""
        R = _quat_to_rot(self.q)
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
        return roll, pitch


class FpvQuadSim:
    def __init__(
        self,
        build: DroneBuild,
        dt: float = 1 / 2000,
        profile: PidProfile | None = None,
        rates=None,
        gyro_noise_dps: float = 0.0,
        seed: int = 0,
    ):
        self.b = build
        self.dt = dt
        self.mass = build.mass_kg
        self.I = np.diag(build.inertia)
        self.I_inv = np.linalg.inv(self.I)
        a = build.arm_m / math.sqrt(2)
        # positions (x fwd, y left): RR, FR, RL, FL
        self.r = np.array([[-a, -a, 0], [a, -a, 0], [-a, a, 0], [a, a, 0]])
        self.motors = Motors(build)
        self.battery = Battery(build)
        self.fc = BetaflightController(dt, profile, rates)
        self.cda = build.frame.drag_area_m2
        self.noise = gyro_noise_dps
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.state = QuadState(np.array([0.0, 0.0, 0.0]), np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3))
        self.on_ground = True

    def gyro_dps(self) -> np.ndarray:
        w = self.state.w
        g = np.degrees(np.array([w[0], w[1], -w[2]]))
        if self.noise:
            g = g + self.rng.normal(0, self.noise, 3)
        return g

    def step(self, sticks: Sticks) -> None:
        dt = self.dt
        s = self.state
        out = self.fc.update(sticks, self.gyro_dps(), s.roll_pitch_deg())
        duty = self.motors.duty_from_output(out) if self.fc.mode.armed else np.zeros(4)
        # implicit battery/motor voltage solve (avoids sag oscillation on high-IR packs)
        m, bat = self.motors, self.battery
        emf = m.omega / m.kv
        v = (bat.ocv() + bat.r_pack * np.sum(duty * emf) / m.R - bat.r_pack * 0.6) / (
            1.0 + bat.r_pack * np.sum(duty**2) / m.R
        )
        bat.voltage = max(v, 0.5 * bat.cells)
        self.motors.step(duty, bat.voltage, dt)
        i_total = float(np.sum(duty * self.motors.current)) / ESC_EFFICIENCY + 0.6
        bat.current = i_total
        bat.voltage = max(bat.ocv() - i_total * bat.r_pack, 0.5 * bat.cells)  # with ESC-clipped currents
        bat.used_ah += max(i_total, 0.0) * dt / 3600.0

        T = self.motors.thrust
        # torques in body frame
        tau = np.zeros(3)
        tau[0] = np.sum(self.r[:, 1] * T)
        tau[1] = -np.sum(self.r[:, 0] * T)
        tau[2] = np.sum(-SPIN * self.motors.drag_torque) - np.sum(SPIN * self.motors.J * self.motors.omega_dot)
        # gyroscopic effect of spinning props
        h_rot = np.array([0.0, 0.0, np.sum(SPIN * self.motors.J * self.motors.omega)])
        w = s.w
        w_dot = self.I_inv @ (tau - np.cross(w, self.I @ w + h_rot))
        s.w = w + w_dot * dt
        dq = np.concatenate([0.5 * s.w * dt, [1.0]])
        s.q = _quat_mul(s.q, dq)
        s.q /= np.linalg.norm(s.q)

        R = _quat_to_rot(s.q)
        f = R @ np.array([0.0, 0.0, T.sum()])
        f[2] -= self.mass * G
        f -= 0.5 * RHO_AIR * self.cda * np.linalg.norm(s.vel) * s.vel
        s.vel = s.vel + f / self.mass * dt
        s.pos = s.pos + s.vel * dt
        if s.pos[2] <= 0.0:
            s.pos[2] = 0.0
            if s.vel[2] < 0:
                s.vel[:] = 0.0
            if self.on_ground:
                s.w[:] = 0.0
        else:
            self.on_ground = False
        self.t += dt

    def run(self, sticks_fn, duration: float, record_every: int = 1) -> dict:
        n = int(duration / self.dt)
        log = {k: [] for k in ("t", "sp", "gyro", "pos", "vel", "vbat", "amps", "thr_out", "att")}
        for k in range(n):
            st = sticks_fn(self.t)
            self.step(st)
            if k % record_every == 0:
                log["t"].append(self.t)
                log["sp"].append(self.fc.setpoint.copy())
                log["gyro"].append(self.gyro_dps())
                log["pos"].append(self.state.pos.copy())
                log["vel"].append(self.state.vel.copy())
                log["vbat"].append(self.battery.voltage)
                log["amps"].append(self.battery.current)
                log["thr_out"].append(self.motors.omega.copy())
                log["att"].append(self.state.roll_pitch_deg())
        return {k: np.array(v) for k, v in log.items()}
