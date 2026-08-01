"""Parallax displacement mapping — THE single source of the terrain formula.

The renderer does not draw a plain projection of the Z=0 plane: it displaces
each sample towards the nadir point in proportion to the terrain height under
it, which is what makes hills look like hills. Anything that needs to know
*which ground point a camera pixel actually shows* must apply the very same
displacement, or it will be describing a different image than the one on disk.

That was exactly the CalibrationLogger bug (audit 2026-08-01): anchors were
built from ``H1 = P[:, [0, 1, 3]]``, the homography of the flat Z=0 plane, while
the recorded frames carried terrain parallax. The error is zero at the nadir
point and grows with radius: ``r * h_rel / altitude``. With 178 m of relief at
1000 m altitude that is ~19 m at mid-frame and ~68 m at the frame corner —
systematic, invisible to the anchor's own residuals (all five reference points
share the same wrong assumption), and large enough to poison a pose graph.

The formula therefore lives here once, and both callers inject their own
sampler: the renderer a CuPy one over the whole pixel grid, the logger a NumPy
one over five points.
"""

from __future__ import annotations

import numpy as np

# Above this altitude the parallax shift converges in a single pass.
_SINGLE_PASS_ALTITUDE_M = 200.0
# Guards against division blow-up when the drone is on the ground.
_MIN_ALTITUDE_M = 10.0


def passes_for_altitude(altitude: float) -> int:
    """Iteration count: high flight needs one pass, low flight needs three."""
    return 1 if float(altitude) > _SINGLE_PASS_ALTITUDE_M else 3


def apply_parallax(
    u0,
    v0,
    drone_col: float,
    drone_row: float,
    altitude: float,
    sample_h_abs,
    base_elevation: float,
    num_passes: int | None = None,
):
    """Displace flat-plane coordinates onto the terrain surface.

    Works with NumPy or CuPy arrays alike — only arithmetic is used, and the
    elevation lookup is delegated to ``sample_h_abs``.

    Args:
        u0, v0: flat Z=0 intersection of each camera ray, in map-pixel
            coordinates (column, row). Scalars or arrays.
        drone_col, drone_row: nadir point in the same map-pixel coordinates.
        altitude: height above the Z=0 base plane, in the same units as the
            elevation raster.
        sample_h_abs: ``f(u, v) -> h_abs`` absolute elevation lookup, taking the
            same map-pixel coordinates.
        base_elevation: elevation of the Z=0 plane (the raster's minimum).
        num_passes: override for the iteration count; ``None`` picks it from
            the altitude.

    Returns:
        ``(u, v)`` displaced map-pixel coordinates, same type as the inputs.

    A point at the nadir is its own answer at any relief, and zero relief
    returns the inputs unchanged — both are relied upon by the tests.
    """
    if num_passes is None:
        num_passes = passes_for_altitude(altitude)
    denom = max(float(altitude), _MIN_ALTITUDE_M)

    u, v = u0, v0
    for _ in range(int(num_passes)):
        frac = (sample_h_abs(u, v) - base_elevation) / denom
        u = u0 - frac * (u0 - drone_col)
        v = v0 - frac * (v0 - drone_row)
    return u, v


def sample_bilinear(img: np.ndarray, rows, cols) -> np.ndarray:
    """Bilinear lookup with edge clamping — NumPy twin of
    ``map_coordinates(order=1, mode="nearest")``.

    Written out rather than imported so that the five-point CPU path carries no
    SciPy dependency and cannot drift from the GPU path through a version
    difference in the interpolator.
    """
    h, w = img.shape[:2]
    r = np.clip(np.asarray(rows, dtype=np.float64), 0.0, h - 1.0)
    c = np.clip(np.asarray(cols, dtype=np.float64), 0.0, w - 1.0)

    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    fr = r - r0
    fc = c - c0

    return (
        img[r0, c0] * (1.0 - fr) * (1.0 - fc)
        + img[r1, c0] * fr * (1.0 - fc)
        + img[r0, c1] * (1.0 - fr) * fc
        + img[r1, c1] * fr * fc
    )


def elevation_sampler(elevation: np.ndarray, map_w: int, map_h: int):
    """``f(u, v) -> h_abs`` for map-pixel coordinates, NumPy side.

    The elevation raster is assumed to cover the same ground extent as the
    orthophoto but at its own resolution, so map-pixel coordinates are rescaled
    by the size ratio — the same assumption the GPU path makes.
    """
    elev_h, elev_w = elevation.shape[:2]
    row_scale = elev_h / float(map_h)
    col_scale = elev_w / float(map_w)

    def _sample(u, v):
        return sample_bilinear(elevation, np.asarray(v) * row_scale, np.asarray(u) * col_scale)

    return _sample
