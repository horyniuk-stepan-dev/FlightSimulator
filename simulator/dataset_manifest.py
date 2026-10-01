"""Atomic manifest and post-recording consistency checks."""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

import cv2


def _fourcc_name(value: float) -> str:
    integer = int(round(value))
    return "".join(chr((integer >> (8 * offset)) & 0xFF) for offset in range(4))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def _portable_path(path: Path, start: Path) -> str:
    """Prefer a relative path, but keep an absolute path across Windows drives."""
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return str(path)


def write_dataset_manifest(
    path: str,
    *,
    status: str,
    config,
    renderer: str,
    frame_count: int,
) -> dict:
    """Validate the recorded sidecars and atomically write their identity."""
    manifest_path = Path(path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    files = {}
    errors = []

    configured = {
        "video": getattr(config, "video_file", ""),
        "calibration": getattr(config, "calib_file", ""),
        "keyframes": getattr(config, "keyframe_file", ""),
        "ground_truth": getattr(config, "gt_file", ""),
        "frame_ground_truth": getattr(config, "frame_gt_file", ""),
        "telemetry": getattr(config, "telemetry_file", ""),
        "scenario": getattr(config, "scenario_file", ""),
    }
    for role, raw in configured.items():
        if not raw:
            continue
        file_path = Path(raw).resolve()
        if not file_path.exists():
            errors.append(f"{role} file missing: {file_path}")
            continue
        files[role] = {
            "path": _portable_path(file_path, manifest_path.parent),
            "size_bytes": file_path.stat().st_size,
            "sha256": _sha256(file_path),
        }

    video_path = Path(config.video_file).resolve() if config.video_file else None
    if video_path and video_path.exists():
        capture = cv2.VideoCapture(str(video_path))
        decoded_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        decoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
        decoded_fourcc = _fourcc_name(capture.get(cv2.CAP_PROP_FOURCC))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture.release()
        files["video"].update(
            frames=decoded_frames,
            fps=decoded_fps,
            width=width,
            height=height,
            codec=decoded_fourcc,
            duration_s=(decoded_frames / decoded_fps if decoded_fps > 0 else None),
        )
        if decoded_frames != frame_count:
            errors.append(
                f"video contains {decoded_frames} frames, recorder produced {frame_count}"
            )
        if abs(decoded_fps - float(config.target_fps)) > 1e-6:
            errors.append(f"video FPS {decoded_fps} != configured {config.target_fps}")

    if config.gt_file and Path(config.gt_file).exists():
        gt = json.loads(Path(config.gt_file).read_text(encoding="utf-8"))
        slots = gt.get("slots", [])
        expected_slots = len(range(0, frame_count, int(config.frame_step)))
        files["ground_truth"]["slots"] = len(slots)
        if len(slots) != expected_slots:
            errors.append(
                f"ground truth contains {len(slots)} slots, expected {expected_slots}"
            )

    keyframe_file = getattr(config, "keyframe_file", "")
    if keyframe_file and Path(keyframe_file).exists():
        try:
            selection = json.loads(Path(keyframe_file).read_text(encoding="utf-8"))
            selected = selection["selected_slots"]
            if (
                not isinstance(selected, list)
                or not selected
                or any(type(slot) is not int for slot in selected)
                or selected != sorted(set(selected))
            ):
                errors.append("keyframe slots must be nonempty, sorted, unique integers")
                selected = []
            if int(selection["frame_step"]) != int(config.frame_step):
                errors.append("keyframe frame_step differs from simulator frame_step")
            expected_slots = len(range(0, frame_count, int(config.frame_step)))
            if int(selection["total_slots"]) != expected_slots:
                errors.append(
                    f"keyframe sidecar contains {selection['total_slots']} candidate slots, "
                    f"expected {expected_slots}"
                )
            if int(selection["source_total_frames"]) != frame_count:
                errors.append("keyframe sidecar video frame count differs from recording")
            if any(slot < 0 or slot >= expected_slots for slot in selected):
                errors.append("keyframe sidecar contains an out-of-range slot")
            featureless = selection.get("featureless_selected_slots", [])
            if (
                not isinstance(featureless, list)
                or any(type(slot) is not int for slot in featureless)
                or featureless != sorted(set(featureless))
                or not set(featureless).issubset(selected)
            ):
                errors.append("featureless keyframe slots must be a sorted subset of selected slots")
                featureless = []
            if video_path and video_path.exists():
                if selection["video_sha256"] != files["video"]["sha256"]:
                    errors.append("keyframe sidecar video SHA-256 differs from recording")
            if config.calib_file and Path(config.calib_file).exists():
                calibration = json.loads(Path(config.calib_file).read_text(encoding="utf-8"))
                anchor_ids = [anchor["frame_id"] for anchor in calibration["anchors"]]
                if not anchor_ids or anchor_ids != sorted(set(anchor_ids)):
                    errors.append("calibration anchor slots must be nonempty, sorted and unique")
                if not set(anchor_ids).issubset(selected):
                    missing = sorted(set(anchor_ids) - set(selected))
                    errors.append(f"calibration anchors are absent from keyframes: {missing}")
                if set(anchor_ids).intersection(featureless):
                    errors.append("calibration anchors include keyframes without local features")
                metadata = calibration.get("keyframe_selection", {})
                if metadata.get("source") != "database_image_selector":
                    errors.append("calibration is not marked as image-selected")
                if int(metadata.get("frame_step", -1)) != int(config.frame_step):
                    errors.append("calibration frame_step differs from keyframes")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"invalid keyframe/calibration contract: {exc}")

    frame_gt_file = getattr(config, "frame_gt_file", "")
    if frame_gt_file and Path(frame_gt_file).exists():
        row_count = 0
        valid_centers = 0
        with Path(frame_gt_file).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(
                        f"frame ground truth line {line_number} is invalid JSON: {exc}"
                    )
                    continue
                if record.get("frame_index") != row_count:
                    errors.append(
                        "frame ground truth is not contiguous at row "
                        f"{row_count}: got index {record.get('frame_index')}"
                    )
                expected_timestamp = row_count / float(config.target_fps)
                timestamp = record.get("timestamp_s")
                if not isinstance(timestamp, (int, float)) or abs(
                    float(timestamp) - expected_timestamp
                ) > 1e-9:
                    errors.append(
                        f"frame ground truth timestamp mismatch at frame {row_count}"
                    )
                valid_centers += bool(record.get("ground_center_valid"))
                row_count += 1
        files["frame_ground_truth"].update(
            rows=row_count, valid_ground_centers=valid_centers
        )
        if row_count != frame_count:
            errors.append(
                f"frame ground truth contains {row_count} rows, expected {frame_count}"
            )

    if config.telemetry_file and Path(config.telemetry_file).exists():
        with Path(config.telemetry_file).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        files["telemetry"]["rows"] = len(rows)
        expected_indices = list(
            range(0, frame_count, max(1, int(config.telemetry_interval)))
        )
        actual_indices = [int(row["frame_index"]) for row in rows]
        if actual_indices != expected_indices:
            errors.append(
                f"telemetry frame indices {actual_indices} != expected {expected_indices}"
            )
        for row, frame_index in zip(rows, expected_indices):
            expected_timestamp = frame_index / float(config.target_fps)
            if abs(float(row["timestamp"]) - expected_timestamp) > 1e-8:
                errors.append(
                    f"telemetry timestamp mismatch at frame {frame_index}"
                )

    final_status = "invalid" if errors and status == "complete" else status
    data = {
        "version": 2,
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": final_status,
        "validation_errors": errors,
        "frame_count": int(frame_count),
        "renderer": renderer,
        "time_model": {
            "frame_timestamp": "frame_index / fps",
            "fixed_output_fps": float(config.target_fps),
            "physics_step_max_s": 0.01,
        },
        "coordinate_contract": {
            "world_xy": "local ground-plane metres, Mercator scale corrected at map centre",
            "world_z": "height above simulator base plane",
            "dem_vertical_datum": "source raster datum; not inferred",
        },
        "git": _git_state(root),
        "config": asdict(config),
        "files": files,
    }

    payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    fd, temporary = tempfile.mkstemp(
        dir=manifest_path.parent, prefix=".tmp_manifest_", suffix=".part"
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(
        f"Dataset manifest: {manifest_path} "
        f"(status={final_status}, validation_errors={len(errors)})"
    )
    return data
