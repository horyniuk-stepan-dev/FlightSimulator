"""Deterministic time profiles for repeatable scale and viewpoint tests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulator.control.command_source import CommandSource, CommandVector
from simulator.physics.drone_state import DroneState
from simulator.planning.survey_planner import Waypoint


@dataclass(frozen=True)
class Keyframe:
    time_s: float
    value: float


class LinearProfile:
    def __init__(self, items, *, name: str, required: bool = False):
        if not items:
            if required:
                raise ValueError(f"Scenario profile '{name}' is required")
            self.keyframes = []
            return
        keyframes = [
            Keyframe(float(item["time_s"]), float(item["value"])) for item in items
        ]
        if any(not math.isfinite(k.time_s) or not math.isfinite(k.value) for k in keyframes):
            raise ValueError(f"Scenario profile '{name}' contains non-finite values")
        if keyframes[0].time_s != 0.0:
            raise ValueError(f"Scenario profile '{name}' must start at time_s=0")
        if any(b.time_s <= a.time_s for a, b in zip(keyframes, keyframes[1:])):
            raise ValueError(f"Scenario profile '{name}' times must strictly increase")
        self.keyframes = keyframes

    def value_at(self, time_s: float) -> float | None:
        if not self.keyframes:
            return None
        times = [k.time_s for k in self.keyframes]
        values = [k.value for k in self.keyframes]
        return float(np.interp(float(time_s), times, values))


@dataclass
class FlightScenario:
    name: str
    duration_s: float
    altitude_m: LinearProfile
    pitch_deg: LinearProfile
    roll_deg: LinearProfile
    yaw_deg: LinearProfile
    source_path: str
    # Optional horizontal route in local map metres (x east, y north, map centre
    # at 0,0). Without it the drone flies the survey lawnmower of the references.
    route_xy: tuple[tuple[float, float], ...] = ()

    @classmethod
    def load(cls, path: str | Path) -> "FlightScenario":
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError("Scenario version must be 1")
        duration = float(data["duration_s"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Scenario duration_s must be finite and positive")
        profiles = data.get("profiles", {})
        route_xy: tuple[tuple[float, float], ...] = ()
        if data.get("route") is not None:
            route = data["route"]
            if not isinstance(route, list) or len(route) < 2:
                raise ValueError("Scenario route needs at least two points")
            points = []
            for item in route:
                x, y = float(item["x_m"]), float(item["y_m"])
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError("Scenario route contains non-finite coordinates")
                points.append((x, y))
            route_xy = tuple(points)
        result = cls(
            name=str(data.get("name") or source.stem),
            duration_s=duration,
            altitude_m=LinearProfile(
                profiles.get("altitude_m"), name="altitude_m", required=True
            ),
            pitch_deg=LinearProfile(profiles.get("pitch_deg"), name="pitch_deg"),
            roll_deg=LinearProfile(profiles.get("roll_deg"), name="roll_deg"),
            yaw_deg=LinearProfile(profiles.get("yaw_deg"), name="yaw_deg"),
            source_path=str(source.resolve()),
            route_xy=route_xy,
        )
        for profile_name in ("altitude_m", "pitch_deg", "roll_deg", "yaw_deg"):
            profile = getattr(result, profile_name)
            if profile.keyframes and profile.keyframes[-1].time_s > duration:
                raise ValueError(
                    f"Scenario profile '{profile_name}' exceeds duration_s={duration}"
                )
        if min(k.value for k in result.altitude_m.keyframes) <= 0:
            raise ValueError("Scenario altitude_m must remain positive")
        return result

    def max_tilt_deg(self) -> float:
        values = [abs(k.value) for p in (self.pitch_deg, self.roll_deg) for k in p.keyframes]
        return max(values, default=0.0)

    def route_waypoints(self, bounds_local, footprint_at) -> list[Waypoint]:
        """The scenario route as waypoints, refused if any point is near the map edge.

        A point is accepted when the camera footprint at the scenario's highest
        altitude (half diagonal), shifted by its largest pitch/roll, stays on the
        map. ``footprint_at(altitude_m) -> (width_m, height_m)``.
        """
        max_altitude = max(k.value for k in self.altitude_m.keyframes)
        width, height = footprint_at(max_altitude)
        margin = math.hypot(width, height) / 2 + max_altitude * math.tan(
            math.radians(self.max_tilt_deg())
        )
        x_min, y_min, x_max, y_max = bounds_local
        outside = [
            (x, y)
            for x, y in self.route_xy
            if not (x_min + margin <= x <= x_max - margin and y_min + margin <= y <= y_max - margin)
        ]
        if outside:
            raise ValueError(
                f"Scenario route points too close to the map edge (margin {margin:.0f} m; "
                f"usable x {x_min + margin:.0f}..{x_max - margin:.0f}, "
                f"y {y_min + margin:.0f}..{y_max - margin:.0f}): {outside}"
            )
        start_altitude = self.altitude_m.value_at(0.0)
        return [Waypoint(x, y, start_altitude) for x, y in self.route_xy]


class ProfiledCommandSource(CommandSource):
    """Apply scenario state at the next video timestamp to a base route controller."""

    def __init__(self, base: CommandSource, scenario: FlightScenario):
        self.base = base
        self.scenario = scenario
        self._time = 0.0

    def get_command(self, state: DroneState, dt: float) -> CommandVector:
        command = self.base.get_command(state, dt)
        next_time = min(self.scenario.duration_s, float(state.time) + float(dt))
        target_altitude = self.scenario.altitude_m.value_at(next_time)
        command.vz = (target_altitude - state.altitude) / dt

        pitch = self.scenario.pitch_deg.value_at(next_time)
        roll = self.scenario.roll_deg.value_at(next_time)
        yaw = self.scenario.yaw_deg.value_at(next_time)
        if pitch is not None:
            command.pitch_rate = math.radians(pitch)
        if roll is not None:
            command.roll = math.radians(roll)
        if yaw is not None:
            command.yaw_rate = math.radians(yaw)
        self._time = next_time
        return command

    def is_finished(self) -> bool:
        # A scenario owns its requested duration. If the route controller
        # reaches its final waypoint first, its zero command simply becomes a
        # hover while altitude/attitude profiles continue to their end.
        return self._time >= self.scenario.duration_s

    @property
    def mode_name(self) -> str:
        return f"{self.base.mode_name}+SCENARIO"

    @property
    def progress_str(self) -> str:
        return f"{self._time:.1f}/{self.scenario.duration_s:.1f}s"

    def __getattr__(self, name):
        return getattr(self.base, name)
