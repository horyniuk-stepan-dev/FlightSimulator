"""Flight altitude datum and terrain clearance.

The simulator flies in heights above its Z=0 plane: the lowest DEM height of the
map (``OrthophotoMap.base_elevation``). Scenario profiles and ``--altitude`` are
read in one of two references:

- ``base``: above that plane (the original behaviour, default);
- ``msl``: above mean sea level as the DEM gives it, i.e. the same number a
  barometric altimeter or an aviation chart would show. Needs an elevation
  raster.

``--altitude-offset`` is added in either reference. Over relief, the camera's
height above the ground (AGL) is the flight height minus the terrain under the
drone, so a flight that is safe over the lowest point can hit a hill. The
helpers here predict that along the planned route before rendering starts, and
check it at every recorded frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ALTITUDE_REFERENCES = ("base", "msl")


class TerrainClearanceError(RuntimeError):
    """The camera is (or would be) too close to, or inside, the terrain."""


def altitude_shift_m(reference: str, offset_m: float, ortho_map) -> float:
    """Metres to add to configured altitudes to get heights above the Z=0 plane."""
    if reference not in ALTITUDE_REFERENCES:
        raise ValueError(f"altitude reference must be one of {ALTITUDE_REFERENCES}, got {reference!r}")
    shift = float(offset_m)
    if reference == "msl":
        if getattr(ortho_map, "elevation", None) is None:
            raise ValueError(
                "--altitude-reference msl needs an elevation raster "
                "(--elevation, or the DEM downloaded with the map tiles)"
            )
        shift -= float(ortho_map.base_elevation)
    return shift


def terrain_height_above_base(ortho_map, x, y) -> np.ndarray:
    """Terrain height above the Z=0 plane at local points (metres); NaN off the DEM.

    Without a DEM the terrain is the Z=0 plane itself (zeros).
    """
    xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
    ys = np.atleast_1d(np.asarray(y, dtype=np.float64))
    if getattr(ortho_map, "elevation", None) is None:
        return np.zeros(xs.shape, dtype=np.float64)
    pixels = [ortho_map.local_to_pixel(float(a), float(b)) for a, b in zip(xs, ys)]
    cols = np.array([p[0] for p in pixels], dtype=np.float64)
    rows = np.array([p[1] for p in pixels], dtype=np.float64)
    heights = np.asarray(ortho_map.sample_elevation_at_map_pixels(cols, rows), dtype=np.float64)
    return heights - float(getattr(ortho_map, "base_elevation", 0.0))


def camera_clearance_m(ortho_map, state) -> float | None:
    """Camera height above the terrain right under it; None off the DEM."""
    h = terrain_height_above_base(ortho_map, state.position[0], state.position[1])[0]
    if not np.isfinite(h):
        return None
    return float(state.altitude) - float(h)


def relief_top_above_base(ortho_map, bounds_local, samples: int = 256) -> float:
    """Highest terrain above the Z=0 plane inside ``bounds_local`` (grid sampled)."""
    if getattr(ortho_map, "elevation", None) is None:
        return 0.0
    x_min, y_min, x_max, y_max = bounds_local
    xs = np.linspace(x_min, x_max, int(samples))
    ys = np.linspace(y_min, y_max, int(samples))
    gx, gy = np.meshgrid(xs, ys)
    heights = terrain_height_above_base(ortho_map, gx.ravel(), gy.ravel())
    return float(np.nanmax(heights)) if np.isfinite(heights).any() else 0.0


def positions_along_route(route_xy, speed_m_s: float, times_s) -> np.ndarray:
    """Points reached at ``times_s`` flying the polyline at constant speed (then hovering)."""
    route = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)
    times = np.asarray(times_s, dtype=np.float64)
    if len(route) == 1:
        return np.repeat(route, len(times), axis=0)
    seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    dist = np.clip(times * float(speed_m_s), 0.0, cum[-1])
    return np.column_stack([np.interp(dist, cum, route[:, 0]), np.interp(dist, cum, route[:, 1])])


@dataclass(frozen=True)
class ClearanceReport:
    min_clearance_m: float
    time_s: float
    x_m: float
    y_m: float
    flight_height_m: float
    terrain_height_m: float


def predicted_route_clearance(
    ortho_map,
    route_xy,
    speed_m_s: float,
    altitude_at,
    duration_s: float,
    step_s: float = 0.5,
) -> ClearanceReport | None:
    """Lowest predicted camera height above the terrain along a planned flight.

    ``altitude_at(t)`` gives the flight height above the Z=0 plane. The route is
    flown at constant speed from its first point, like the autopilot does;
    turn dynamics are ignored, so keep a margin (``--min-clearance``).
    None without a DEM or when the whole route is off the DEM.
    """
    if getattr(ortho_map, "elevation", None) is None or speed_m_s <= 0:
        return None
    n = max(2, math.ceil(float(duration_s) / float(step_s)) + 1)
    times = np.linspace(0.0, float(duration_s), n)
    xy = positions_along_route(route_xy, speed_m_s, times)
    terrain = terrain_height_above_base(ortho_map, xy[:, 0], xy[:, 1])
    heights = np.array([float(altitude_at(t)) for t in times])
    clearance = heights - terrain
    if not np.isfinite(clearance).any():
        return None
    i = int(np.nanargmin(clearance))
    return ClearanceReport(
        min_clearance_m=float(clearance[i]),
        time_s=float(times[i]),
        x_m=float(xy[i, 0]),
        y_m=float(xy[i, 1]),
        flight_height_m=float(heights[i]),
        terrain_height_m=float(terrain[i]),
    )
