"""
Betaflight-style flight controller logic (simplified port, not a line-by-line copy).

Pipeline per PID loop, same order as Betaflight:
  sticks -> rates (Actual / Betaflight) -> rate setpoint [deg/s]
  gyro -> PT1 gyro LPF
  rate error -> P + I (+ iterm_relax, iterm limit) + D-on-measurement (PT1 D-term LPF) + feedforward
  pidSum (limited) -> quad-X mixer with airmode -> motor outputs 0..1

Angle mode wraps an outer attitude-P loop that produces the rate setpoint.
PID gains use Betaflight's own scaling constants, so numbers like P=45 I=80
D=30 mean roughly the same thing as in the configurator.

Axis convention (Betaflight): +roll = right wing down, +pitch = nose down
(stick forward), +yaw = nose right. Body frame here is FLU (x fwd, y left,
z up): roll_rate = wx, pitch_rate = wy, yaw_rate = -wz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Betaflight scaling constants (pid.h)
PTERM_SCALE = 0.032029
ITERM_SCALE = 0.244381
DTERM_SCALE = 0.000529
FEEDFORWARD_SCALE = 0.013754
PID_MIXER_SCALING = 1.0 / 1000.0


# ---------------------------------------------------------------- rates ----
@dataclass
class ActualRates:
    center_dps: float = 70.0
    max_dps: float = 670.0
    expo: float = 0.54

    def __call__(self, x: float) -> float:
        x = max(-1.0, min(1.0, x))
        ax = abs(x)
        expof = ax * (x**5 * self.expo + x * (1 - self.expo))
        stick_movement = max(0.0, self.max_dps - self.center_dps)
        return x * self.center_dps + stick_movement * expof


@dataclass
class BetaflightRates:
    rc_rate: float = 1.0
    super_rate: float = 0.7
    expo: float = 0.0

    def __call__(self, x: float) -> float:
        x = max(-1.0, min(1.0, x))
        ax = abs(x)
        rc_rate = self.rc_rate
        if rc_rate > 2.0:
            rc_rate += 14.54 * (rc_rate - 2.0)
        if self.expo:
            x = x * ax**3 * self.expo + x * (1 - self.expo)
        rate = 200.0 * rc_rate * x
        if self.super_rate:
            rate *= 1.0 / max(0.01, min(1.0, 1.0 - ax * self.super_rate))
        return rate


# --------------------------------------------------------------- filters ---
class PT1:
    def __init__(self, cutoff_hz: float, dt: float, n: int = 3):
        rc = 1.0 / (2 * math.pi * cutoff_hz)
        self.k = dt / (rc + dt)
        self.y = np.zeros(n)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.y = self.y + self.k * (x - self.y)
        return self.y


# ------------------------------------------------------------------- PID ---
@dataclass
class PidProfile:
    # [roll, pitch, yaw]
    P: tuple = (45, 47, 45)
    I: tuple = (80, 84, 80)
    D: tuple = (30, 34, 0)
    F: tuple = (120, 125, 120)
    iterm_limit: float = 400.0
    pidsum_limit: float = 500.0
    pidsum_limit_yaw: float = 400.0
    gyro_lpf_hz: float = 250.0
    rc_smoothing_hz: float = 30.0  # setpoint smoothing (Betaflight rc_smoothing)
    dterm_lpf_hz: float = 100.0
    airmode: bool = True
    iterm_relax: bool = True  # Betaflight iterm_relax = RP, setpoint mode
    iterm_relax_cutoff_hz: float = 15.0
    angle_strength: float = 50.0  # Betaflight angle_p_gain-ish
    angle_limit_deg: float = 55.0


@dataclass
class FlightModeState:
    mode: str = "acro"  # "acro" | "angle"
    armed: bool = True


@dataclass
class Sticks:
    """Normalised RC input: roll/pitch/yaw in [-1, 1], throttle in [0, 1]."""

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.0


class BetaflightController:
    # Quad-X mixer, Betaflight motor order 1 RR, 2 FR, 3 RL, 4 FL.
    # Columns: roll, pitch, yaw contributions (+ = more thrust).
    MIX = np.array(
        [
            [-1.0, +1.0, -1.0],  # RR  (CW)
            [-1.0, -1.0, +1.0],  # FR  (CCW)
            [+1.0, +1.0, +1.0],  # RL  (CCW)
            [+1.0, -1.0, -1.0],  # FL  (CW)
        ]
    )

    def __init__(self, dt: float, profile: PidProfile | None = None, rates: tuple | None = None):
        self.dt = dt
        self.p = profile or PidProfile()
        self.rates = rates or (ActualRates(), ActualRates(), ActualRates(center_dps=70, max_dps=500))
        self.mode = FlightModeState()
        self.gyro_lpf = PT1(self.p.gyro_lpf_hz, dt)
        self.dterm_lpf = PT1(self.p.dterm_lpf_hz, dt)
        self.sp_lpf = PT1(self.p.rc_smoothing_hz, dt)
        self.relax_lpf = PT1(self.p.iterm_relax_cutoff_hz, dt)
        self.iterm = np.zeros(3)
        self.prev_gyro = np.zeros(3)
        self.prev_sp = np.zeros(3)
        self.setpoint = np.zeros(3)
        self.kp = PTERM_SCALE * np.array(self.p.P, float)
        self.ki = ITERM_SCALE * np.array(self.p.I, float)
        self.kd = DTERM_SCALE * np.array(self.p.D, float)
        self.kf = FEEDFORWARD_SCALE * np.array(self.p.F, float) / 100.0

    def rate_setpoint(self, s: Sticks, att_rp_deg: tuple[float, float]) -> np.ndarray:
        sp = np.array([self.rates[0](s.roll), self.rates[1](s.pitch), self.rates[2](s.yaw)])
        if self.mode.mode == "angle":
            lim = self.p.angle_limit_deg
            gain = self.p.angle_strength * 0.1  # (deg/s) per deg of error
            for ax, stick in ((0, s.roll), (1, s.pitch)):
                err = stick * lim - att_rp_deg[ax]
                sp[ax] = gain * err
        return sp

    def update(self, s: Sticks, gyro_dps: np.ndarray, att_rp_deg: tuple[float, float]) -> np.ndarray:
        """Return 4 motor outputs in 0..1."""
        if not self.mode.armed:
            self.iterm[:] = 0
            return np.zeros(4)
        dt = self.dt
        g = self.gyro_lpf(gyro_dps)
        sp = self.sp_lpf(self.rate_setpoint(s, att_rp_deg)).copy()
        err = sp - g

        p_term = self.kp * err
        # Betaflight zeroes/limits I-term growth at low throttle unless airmode
        i_err = err.copy()
        if self.p.iterm_relax:
            # suppress I-term growth while the stick is moving fast (roll/pitch)
            hpf = np.abs(sp - self.relax_lpf(sp))
            relax = np.clip(1.0 - hpf / 40.0, 0.0, 1.0)  # ITERM_RELAX_SETPOINT_THRESHOLD
            i_err[:2] *= relax[:2]
        if self.p.airmode or s.throttle > 0.05:
            self.iterm = np.clip(self.iterm + self.ki * i_err * dt, -self.p.iterm_limit, self.p.iterm_limit)
        d_term = self.dterm_lpf(-self.kd * (g - self.prev_gyro) / dt)
        ff = self.kf * (sp - self.prev_sp) / dt
        self.prev_gyro = g.copy()
        self.prev_sp = sp
        self.setpoint = sp

        pid = p_term + self.iterm + d_term + ff
        lim = np.array([self.p.pidsum_limit, self.p.pidsum_limit, self.p.pidsum_limit_yaw])
        pid = np.clip(pid, -lim, lim) * PID_MIXER_SCALING
        return self.mix(pid, s.throttle)

    def mix(self, pid: np.ndarray, throttle: float) -> np.ndarray:
        mix = self.MIX @ pid
        lo, hi = mix.min(), mix.max()
        rng = hi - lo
        if rng > 1.0:  # not enough authority: scale the differential down
            mix /= rng
            lo, hi = mix.min(), mix.max()
        if self.p.airmode:
            # shift throttle so the full differential fits into 0..1
            throttle = min(max(throttle, -lo), 1.0 - hi)
        return np.clip(throttle + mix, 0.0, 1.0)
