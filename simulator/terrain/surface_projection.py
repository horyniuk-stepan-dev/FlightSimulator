"""Shared image-pixel to terrain projection used by render metadata exports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.terrain.parallax import apply_parallax, elevation_sampler


class ProjectionGeometryError(ValueError):
    """The camera homography cannot be inverted for the requested pixels."""


class SurfaceCoverageError(ValueError):
    """The requested surface point is outside valid DEM coverage."""


@dataclass(frozen=True)
class SurfaceProjection:
    local_xy: np.ndarray
    mercator_xy: np.ndarray
    gps: list[list[float]]
    surface_height_absolute_m: np.ndarray
    surface_height_base_m: np.ndarray
    camera_agl_m: float


def _camera_agl(ortho_map, state, *, strict_terrain: bool) -> float:
    agl = float(state.altitude)
    if getattr(ortho_map, "elevation", None) is None:
        return agl
    if not hasattr(ortho_map, "local_to_pixel"):
        return agl

    col, row = ortho_map.local_to_pixel(
        float(state.position[0]), float(state.position[1])
    )
    if hasattr(ortho_map, "sample_elevation_at_map_pixels"):
        height = ortho_map.sample_elevation_at_map_pixels(
            np.array([col]), np.array([row])
        )[0]
    else:
        sample = elevation_sampler(
            ortho_map.elevation, ortho_map.width, ortho_map.height
        )
        height = sample(np.array([col]), np.array([row]))[0]
    if not np.isfinite(height):
        if strict_terrain:
            raise SurfaceCoverageError(
                "Camera position is outside valid DEM coverage"
            )
        return agl
    return agl - float(height - getattr(ortho_map, "base_elevation", 0.0))


def project_image_pixels_to_surface(
    H1: np.ndarray,
    image_xy: np.ndarray,
    ortho_map,
    state,
    *,
    strict_terrain: bool = True,
) -> SurfaceProjection:
    """Back-project image pixels and apply the renderer's terrain displacement.

    ``H1`` maps the simulator's local Z=0 plane into image pixels. The inverse
    gives the flat-plane ray intersection. When a DEM is present, the exact
    same iterative displacement as the renderer moves that intersection onto
    the visible terrain surface.
    """
    H1 = np.asarray(H1, dtype=np.float64)
    points = np.asarray(image_xy, dtype=np.float64)
    if H1.shape != (3, 3) or points.ndim != 2 or points.shape[1] != 2:
        raise ProjectionGeometryError("Expected H1=(3,3) and image_xy=(N,2)")
    if not np.all(np.isfinite(H1)) or not np.all(np.isfinite(points)):
        raise ProjectionGeometryError("Projection contains non-finite values")
    try:
        inverse = np.linalg.inv(H1)
    except np.linalg.LinAlgError as exc:
        raise ProjectionGeometryError("Camera homography is singular") from exc

    homogeneous = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
    local_h = (inverse @ homogeneous.T).T
    denominators = local_h[:, 2]
    if (
        not np.all(np.isfinite(local_h))
        or np.any(np.abs(denominators) < 1e-9)
        or not np.all(denominators > 0)
    ):
        raise ProjectionGeometryError(
            "Requested pixels cross the camera horizon or project to infinity"
        )
    local_xy = local_h[:, :2] / denominators[:, None]

    elevation = getattr(ortho_map, "elevation", None)
    if elevation is not None and hasattr(ortho_map, "local_to_pixel"):
        cols, rows = zip(
            *(ortho_map.local_to_pixel(float(x), float(y)) for x, y in local_xy)
        )
        drone_col, drone_row = ortho_map.local_to_pixel(
            float(state.position[0]), float(state.position[1])
        )
        if hasattr(ortho_map, "sample_elevation_at_map_pixels"):
            sample = ortho_map.sample_elevation_at_map_pixels
        else:
            sample = elevation_sampler(elevation, ortho_map.width, ortho_map.height)
        displaced_cols, displaced_rows = apply_parallax(
            np.asarray(cols, dtype=np.float64),
            np.asarray(rows, dtype=np.float64),
            drone_col,
            drone_row,
            float(state.altitude),
            sample,
            float(getattr(ortho_map, "base_elevation", 0.0)),
        )
        if not (
            np.all(np.isfinite(displaced_cols))
            and np.all(np.isfinite(displaced_rows))
        ):
            if strict_terrain:
                raise SurfaceCoverageError(
                    "Projected image pixel is outside valid DEM coverage"
                )
        else:
            local_xy = np.array(
                [
                    ortho_map.pixel_to_local(float(col), float(row))
                    for col, row in zip(displaced_cols, displaced_rows)
                ],
                dtype=np.float64,
            )

    if hasattr(ortho_map, "local_to_mercator"):
        mercator_xy = np.array(
            [ortho_map.local_to_mercator(float(x), float(y)) for x, y in local_xy],
            dtype=np.float64,
        )
    else:
        mercator_xy = local_xy + np.array(
            [ortho_map._center_x, ortho_map._center_y], dtype=np.float64
        )
    gps = [
        list(ortho_map.local_to_gps(float(x), float(y))) for x, y in local_xy
    ]
    base_elevation = float(getattr(ortho_map, "base_elevation", 0.0))
    if elevation is None or not hasattr(ortho_map, "local_to_pixel"):
        surface_height_absolute = np.full(len(local_xy), base_elevation)
    else:
        final_cols, final_rows = zip(
            *(ortho_map.local_to_pixel(float(x), float(y)) for x, y in local_xy)
        )
        if hasattr(ortho_map, "sample_elevation_at_map_pixels"):
            surface_height_absolute = ortho_map.sample_elevation_at_map_pixels(
                np.asarray(final_cols), np.asarray(final_rows)
            )
        else:
            sample = elevation_sampler(elevation, ortho_map.width, ortho_map.height)
            surface_height_absolute = sample(
                np.asarray(final_cols), np.asarray(final_rows)
            )
        if strict_terrain and not np.all(np.isfinite(surface_height_absolute)):
            raise SurfaceCoverageError(
                "Final projected point is outside valid DEM coverage"
            )
    return SurfaceProjection(
        local_xy=local_xy,
        mercator_xy=mercator_xy,
        gps=gps,
        surface_height_absolute_m=np.asarray(
            surface_height_absolute, dtype=np.float64
        ),
        surface_height_base_m=np.asarray(
            surface_height_absolute - base_elevation, dtype=np.float64
        ),
        camera_agl_m=_camera_agl(
            ortho_map, state, strict_terrain=strict_terrain
        ),
    )
