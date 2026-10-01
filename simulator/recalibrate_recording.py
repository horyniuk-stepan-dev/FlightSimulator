"""Place recorded calibration anchors on the image-selected database slots.

This is an offline repair path for recordings made before the visual selector was
integrated into the simulator. It reads the recorded video, per-slot ground truth,
the original calibration, and a visual-keyframe sidecar. It never opens a map
database or estimates a new affine from neighbouring frames.

The two outputs are explicit staging paths. The dataset manifest is deliberately
left unchanged: the caller must promote both staged outputs and refresh the
manifest only after the resulting dataset has been validated as a whole.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from simulator.physics.calibration_logger import CalibrationLogger


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return data


def _slot_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    if any(type(slot) is not int or slot < 0 for slot in value):
        raise ValueError(f"{label} must contain nonnegative integer slots")
    if value != sorted(set(value)):
        raise ValueError(f"{label} must be sorted and unique")
    return value


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _recorded_frame_count(video: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open recorded video: {video}")
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if frame_count < 1:
            raise ValueError(f"Recorded video has no frames: {video}")
        return frame_count
    finally:
        capture.release()


def _selection_contract(
    *, video: Path, ground_truth: dict[str, Any], selection: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> tuple[list[int], set[int], dict[int, dict[str, Any]], int, dict[str, Any]]:
    selected = _slot_list(selection.get("selected_slots"), "selected_slots")
    featureless_raw = selection.get("featureless_selected_slots", [])
    if not isinstance(featureless_raw, list) or any(
        type(slot) is not int or slot < 0 for slot in featureless_raw
    ) or featureless_raw != sorted(set(featureless_raw)):
        raise ValueError("featureless_selected_slots must be sorted, unique nonnegative integers")
    featureless = set(featureless_raw)
    if not featureless.issubset(selected):
        raise ValueError("featureless_selected_slots must be a subset of selected_slots")

    if "source_video" not in selection or Path(selection["source_video"]).resolve() != video:
        raise ValueError("Visual selection belongs to a different video path")
    actual_sha = _sha256(video)
    if selection.get("video_sha256") != actual_sha:
        raise ValueError("Visual selection video SHA-256 differs from the recorded video")

    frame_step = selection.get("frame_step")
    if type(frame_step) is not int or frame_step < 1:
        raise ValueError("Visual selection has an invalid frame_step")
    if ground_truth.get("frame_step") != frame_step:
        raise ValueError("Ground-truth frame_step differs from visual selection")
    frame_count = _recorded_frame_count(video)
    if selection.get("source_total_frames") != frame_count:
        raise ValueError("Visual selection frame count differs from the recorded video")
    total_slots = len(range(0, frame_count, frame_step))
    if selection.get("total_slots") != total_slots:
        raise ValueError("Visual selection slot count differs from the recorded video")
    if selected[-1] >= total_slots:
        raise ValueError("Visual selection contains an out-of-range slot")

    raw_gt_slots = ground_truth.get("slots")
    if not isinstance(raw_gt_slots, list) or len(raw_gt_slots) != total_slots:
        raise ValueError("Ground truth must contain one record for every video slot")
    gt_slots: dict[int, dict[str, Any]] = {}
    for expected, record in enumerate(raw_gt_slots):
        if not isinstance(record, dict) or record.get("slot") != expected:
            raise ValueError(f"Ground truth is not contiguous at slot {expected}")
        if record.get("video_frame") != expected * frame_step:
            raise ValueError(f"Ground truth video-frame mapping is wrong at slot {expected}")
        gt_slots[expected] = record

    settings = selection.get("selection_settings")
    if not isinstance(settings, dict):
        raise ValueError("Visual selection is missing selection_settings")
    if settings.get("criterion") != "overlap":
        raise ValueError("Visual selection must use DatabaseBuilder overlap criterion")
    if manifest is not None:
        cfg = manifest.get("config", {})
        if cfg.get("frame_step") != frame_step or manifest.get("frame_count") != frame_count:
            raise ValueError("Dataset manifest frame count or frame_step differs from video")
        manifest_sha = manifest.get("files", {}).get("video", {}).get("sha256")
        if manifest_sha and manifest_sha != actual_sha:
            raise ValueError("Dataset manifest video SHA-256 differs from recorded video")
    return selected, featureless, gt_slots, frame_step, settings


def recalibrate_recording(
    *,
    video_path: str | Path,
    ground_truth_path: str | Path,
    calibration_path: str | Path,
    keyframes_path: str | Path,
    output_calibration_path: str | Path,
    output_ground_truth_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write staged calibration and GT, using only selected video/GT slots.

    Existing complete QA is preserved for unchanged anchors whose affine still
    matches the recorded ground truth. Newly selected anchors retain the exact
    per-slot GT affine and RMSE; missing five-point QA remains explicitly empty.
    """
    video = Path(video_path).resolve()
    gt_path = Path(ground_truth_path).resolve()
    calibration = Path(calibration_path).resolve()
    keyframes = Path(keyframes_path).resolve()
    out_calibration = Path(output_calibration_path).resolve()
    out_gt = Path(output_ground_truth_path).resolve()
    if out_calibration == out_gt:
        raise ValueError("Calibration and ground-truth outputs must be different files")
    if out_calibration in {calibration, gt_path} or out_gt in {calibration, gt_path}:
        raise ValueError("Outputs must be separate staging files, not the source artifacts")
    if not video.is_file():
        raise FileNotFoundError(video)

    gt = _load_object(gt_path, "Ground truth")
    old_calibration = _load_object(calibration, "Calibration")
    selection = _load_object(keyframes, "Visual keyframe sidecar")
    if manifest_path is None:
        possible_manifest = gt_path.parent / "manifest.json"
        manifest_path = possible_manifest if possible_manifest.is_file() else None
    manifest = _load_object(Path(manifest_path).resolve(), "Dataset manifest") if manifest_path else None
    selected, featureless, gt_slots, frame_step, settings = _selection_contract(
        video=video, ground_truth=gt, selection=selection, manifest=manifest,
    )
    if old_calibration.get("frame_size") != gt.get("frame_size"):
        raise ValueError("Calibration and ground-truth frame sizes differ")
    if old_calibration.get("projection") != gt.get("projection"):
        raise ValueError("Calibration and ground-truth projections differ")

    original_anchors: dict[int, dict[str, Any]] = {}
    for anchor in old_calibration.get("anchors", []):
        slot = anchor.get("frame_id")
        if type(slot) is not int or slot in original_anchors:
            raise ValueError("Original calibration has invalid or duplicate anchor slots")
        original_anchors[slot] = anchor

    eligible = [slot for slot in selected if slot not in featureless]
    if not eligible:
        raise ValueError("No selected slot has local features for a calibration anchor")
    for slot in eligible:
        record = gt_slots[slot]
        affine = np.asarray(record.get("affine"), dtype=np.float64)
        if affine.shape != (2, 3) or not np.isfinite(affine).all():
            raise ValueError(f"Invalid GT affine at selected slot {slot}")
        if not isinstance(record.get("heading_deg"), (int, float)) or not math.isfinite(
            float(record["heading_deg"])
        ):
            raise ValueError(f"Invalid GT heading at selected slot {slot}")
        if not isinstance(record.get("rmse_m"), (int, float)) or not math.isfinite(
            float(record["rmse_m"])
        ) or float(record["rmse_m"]) < 0:
            raise ValueError(f"Invalid GT RMSE at selected slot {slot}")

    cfg = manifest.get("config", {}) if manifest is not None else {}
    selected_indices = CalibrationLogger.select_anchor_indices(
        slots=eligible,
        headings_rad=[math.radians(float(gt_slots[slot]["heading_deg"])) for slot in eligible],
        turn_rate_deg_per_slot=float(cfg.get("anchor_turn_rate_deg", 3.0)),
        min_spacing_slots=int(cfg.get("anchor_spacing_slots", 15)),
        max_spacing_slots=int(cfg.get("anchor_max_spacing_slots", 60)),
    )
    if not selected_indices:
        raise ValueError("Anchor selection returned no anchors")

    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    anchors: list[dict[str, Any]] = []
    reasons_by_slot: dict[int, str] = {}
    preserved_qa = 0
    for index in sorted(selected_indices):
        slot = eligible[index]
        record = gt_slots[slot]
        reason = selected_indices[index]
        reasons_by_slot[slot] = reason
        affine = record["affine"]
        old = original_anchors.get(slot)
        if old is not None and np.array_equal(
            np.asarray(old.get("affine_matrix"), dtype=np.float64),
            np.asarray(affine, dtype=np.float64),
        ):
            qa = copy.deepcopy(old.get("qa_data", {}))
            preserved_qa += 1
        else:
            qa = {
                "rmse_m": float(record["rmse_m"]),
                "inliers_count": 0,
                "points_2d": [],
                "points_gps": [],
                "points_metric": [],
                "transform_type": "simulator_ground_truth_lsq5_from_slot_gt",
                "projection_mode": gt["projection"]["mode"],
                "created_at": now_iso,
                "updated_at": now_iso,
                "notes": (
                    f"Simulator GT anchor [{reason}] at db_slot={slot}, "
                    f"video_frame={record['video_frame']}; selected by the database image "
                    "selector. Affine and RMSE come from ground_truth.json; original "
                    "five-point QA coordinates were not stored for this slot."
                ),
                "quality_flag": "warning",
            }
        anchors.append({"frame_id": slot, "affine_matrix": copy.deepcopy(affine), "qa_data": qa})

    new_calibration = copy.deepcopy(old_calibration)
    new_calibration["anchors"] = anchors
    keyframe_metadata = dict(new_calibration.get("keyframe_selection") or {})
    keyframe_metadata.update({
        "source": "database_image_selector",
        "criterion": settings["criterion"],
        "max_overlap": settings.get("max_overlap"),
        "max_gap_frames": settings.get("max_gap_frames"),
        "frame_step": frame_step,
        "sidecar_sha256": _sha256(keyframes),
        "video_sha256": selection["video_sha256"],
    })
    new_calibration["keyframe_selection"] = keyframe_metadata

    new_gt = copy.deepcopy(gt)
    for record in new_gt["slots"]:
        slot = record["slot"]
        record["is_anchor"] = slot in reasons_by_slot
        record["anchor_reason"] = reasons_by_slot.get(slot)
    new_gt["keyframe_selection"] = {
        "source": "database_image_selector",
        "frame_step": frame_step,
        "sidecar_sha256": keyframe_metadata["sidecar_sha256"],
    }

    # Each destination is written atomically. These are staged files; the caller
    # promotes them together after independent validation and then updates the
    # dataset manifest's hashes/status. The manifest is intentionally untouched.
    _atomic_json(out_calibration, new_calibration)
    _atomic_json(out_gt, new_gt)
    return {
        "anchor_count": len(anchors),
        "preserved_qa_count": preserved_qa,
        "new_anchor_count": len(anchors) - preserved_qa,
        "selected_slot_count": len(selected),
        "featureless_selected_count": len(featureless),
        "anchor_slots": [anchor["frame_id"] for anchor in anchors],
        "output_calibration": str(out_calibration),
        "output_ground_truth": str(out_gt),
        "manifest_updated": False,
        "manifest_reason": "Outputs are staged; refresh manifest after both are promoted and validated.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--keyframes", required=True)
    parser.add_argument("--output-calibration", required=True)
    parser.add_argument("--output-ground-truth", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args(argv)
    result = recalibrate_recording(
        video_path=args.video,
        ground_truth_path=args.ground_truth,
        calibration_path=args.calibration,
        keyframes_path=args.keyframes,
        output_calibration_path=args.output_calibration,
        output_ground_truth_path=args.output_ground_truth,
        manifest_path=args.manifest,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
