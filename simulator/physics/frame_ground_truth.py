"""Atomic, per-video-frame ground-truth JSONL writer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from simulator.terrain.surface_projection import (
    ProjectionGeometryError,
    SurfaceCoverageError,
    project_image_pixels_to_surface,
)


class FrameGroundTruthLogger:
    """Write one self-contained truth record for every attempted video frame."""

    def __init__(
        self,
        output_file: str,
        ortho_map,
        *,
        width: int,
        height: int,
        fps: float,
        renderer: str,
        focal_length_mm: float | None = None,
        sensor_width_mm: float | None = None,
        scenario: str | None = None,
        strict_terrain: bool = True,
    ):
        self.output_path = Path(output_file).resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = self.output_path.with_name(
            f".{self.output_path.name}.part"
        )
        self.ortho_map = ortho_map
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.renderer = str(renderer)
        self.focal_length_mm = (
            None if focal_length_mm is None else float(focal_length_mm)
        )
        self.sensor_width_mm = (
            None if sensor_width_mm is None else float(sensor_width_mm)
        )
        self.scenario = scenario
        self.strict_terrain = bool(strict_terrain)
        self.frames_written = 0
        self._stream = self.temporary_path.open("w", encoding="utf-8", newline="\n")
        self._closed = False

    def log(
        self,
        H1: np.ndarray | None,
        state,
        *,
        frame_index: int,
        surface_valid_fraction: float,
    ) -> None:
        if self._closed:
            raise RuntimeError("FrameGroundTruthLogger is already closed")
        if int(frame_index) != self.frames_written:
            raise ValueError(
                f"Frame GT must be sequential: got {frame_index}, "
                f"expected {self.frames_written}"
            )

        record = {
            "version": "frame-gt-1.0",
            "frame_index": int(frame_index),
            "timestamp_s": int(frame_index) / self.fps,
            "frame_size": [self.width, self.height],
            "camera_position_world": np.asarray(
                state.position, dtype=float
            ).tolist(),
            "camera_orientation_xyzw": np.asarray(
                state.quaternion, dtype=float
            ).tolist(),
            "camera_altitude_base_m": float(state.altitude),
            "surface_valid_fraction": float(surface_valid_fraction),
            "renderer": self.renderer,
            "scenario": self.scenario,
            "coordinate_system": {
                "world_xy": "local ground-plane metres, Mercator scale corrected at map centre",
                "world_z": "height above simulator base plane",
                "vertical_datum": "inherited from source DEM; not inferred by simulator",
                "reference_gps": getattr(self.ortho_map, "reference_gps", None),
            },
            "ground_center_valid": False,
        }
        if self.focal_length_mm is not None and self.sensor_width_mm is not None:
            focal_px = self.focal_length_mm * self.width / self.sensor_width_mm
            record["camera_intrinsics"] = {
                "fx_px": focal_px,
                "fy_px": focal_px,
                "cx_px": self.width / 2.0,
                "cy_px": self.height / 2.0,
                "focal_length_mm": self.focal_length_mm,
                "sensor_width_mm": self.sensor_width_mm,
            }
        if H1 is not None:
            try:
                result = project_image_pixels_to_surface(
                    H1,
                    np.array([[self.width / 2.0, self.height / 2.0]]),
                    self.ortho_map,
                    state,
                    strict_terrain=self.strict_terrain,
                )
                local = result.local_xy[0]
                mercator = result.mercator_xy[0]
                record.update(
                    ground_center_valid=True,
                    ground_center_local_m=[float(local[0]), float(local[1])],
                    ground_center_mercator_m=[
                        float(mercator[0]), float(mercator[1])
                    ],
                    ground_center_gps=result.gps[0],
                    ground_center_height_absolute_m=float(
                        result.surface_height_absolute_m[0]
                    ),
                    ground_center_height_base_m=float(
                        result.surface_height_base_m[0]
                    ),
                    camera_agl_m=float(result.camera_agl_m),
                )
            except (ProjectionGeometryError, SurfaceCoverageError) as exc:
                record["invalid_reason"] = str(exc)
        else:
            record["invalid_reason"] = "renderer did not produce a projection matrix"

        self._stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        self._stream.write("\n")
        self.frames_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        os.replace(self.temporary_path, self.output_path)
        self._closed = True
        print(
            f"[FrameGroundTruthLogger] Saved {self.frames_written} frame records "
            f"to {self.output_path}"
        )

    def abort(self) -> None:
        """Close the partial stream without presenting it as a complete artifact."""
        if self._closed:
            return
        self._stream.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False
