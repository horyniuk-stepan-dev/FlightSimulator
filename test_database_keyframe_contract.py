"""Image-selected calibration slots must match the database selection sidecar."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from simulator import main as simulator_main
from simulator.physics.calibration_logger import CalibrationLogger


class _Map:
    _center_x = 0.0
    _center_y = 0.0


def _candidate(slot: int) -> dict:
    return {
        "slot": slot,
        "video_frame": slot * 30,
        "heading": 0.0,
        "yaw_deg": 0.0,
        "alt": 1000.0,
        "camera_agl": 1000.0,
        "camera_position": [float(slot), 0.0, 1000.0],
        "camera_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "timestamp": float(slot),
        "ground_center_local": [float(slot), 0.0],
        "ground_center_mercator": [float(slot), 0.0],
        "ground_center_gps": [0.0, 0.0],
        "surface_valid_fraction": 1.0,
        "M": np.array([[0.5, 0.0, float(slot)], [0.0, -0.5, 0.0]]),
        "rmse": 0.0,
        "median": 0.0,
        "max": 0.0,
        "pts_px": [[640.0, 360.0]] * 5,
        "pts_gps": [[0.0, 0.0]] * 5,
        "pts_mercator": [[float(slot), 0.0]] * 5,
    }


def test_exact_slots_override_geometric_prediction_and_keep_target_matrices(tmp_path):
    path = tmp_path / "calibration.json"
    logger = CalibrationLogger(
        str(path), _Map(), frame_step=30, keyframe_criterion="overlap",
        min_anchor_spacing_slots=1, max_anchor_spacing_slots=4,
    )
    logger._frame_size = (1280, 720)
    logger._candidates = [_candidate(i) for i in range(16)]
    logger.set_exact_keyframe_slots(
        list(range(1, 16, 2)),
        provenance={"sidecar": "video.keyframes.json", "video_sha256": "a" * 64},
    )

    logger.close()
    logger.dump_ground_truth(str(tmp_path / "ground_truth.json"))

    calibration = json.loads(path.read_text(encoding="utf-8"))
    anchors = calibration["anchors"]
    assert anchors
    assert all(anchor["frame_id"] % 2 == 1 for anchor in anchors)
    assert [anchor["frame_id"] for anchor in anchors] == sorted(
        {anchor["frame_id"] for anchor in anchors}
    )
    for anchor in anchors:
        slot = anchor["frame_id"]
        assert anchor["affine_matrix"] == _candidate(slot)["M"].tolist()
    assert calibration["keyframe_selection"]["source"] == "database_image_selector"
    assert calibration["keyframe_selection"]["video_sha256"] == "a" * 64

    truth = json.loads((tmp_path / "ground_truth.json").read_text(encoding="utf-8"))
    assert {s["slot"] for s in truth["slots"] if s["is_anchor"]} == {
        anchor["frame_id"] for anchor in anchors
    }


def test_exact_slots_reject_empty_or_duplicate(tmp_path):
    logger = CalibrationLogger(str(tmp_path / "calibration.json"), _Map())
    logger._candidates = [_candidate(i) for i in range(3)]
    with pytest.raises(ValueError):
        logger.set_exact_keyframe_slots([])
    with pytest.raises(ValueError):
        logger.set_exact_keyframe_slots([1, 1])
    with pytest.raises(ValueError):
        logger.set_exact_keyframe_slots([9])


def test_encoded_video_selector_sidecar_filters_featureless_slots(tmp_path, monkeypatch):
    settings = {
        "frame_step": 2,
        "criterion": "overlap",
        "max_overlap": 0.5,
        "max_gap_frames": 60,
        "selection_settings": {"frame_step": 2, "criterion": "overlap"},
    }
    sidecar = tmp_path / "video.keyframes.json"
    payload = {
        "frame_step": 2,
        "source_total_frames": 10,
        "total_slots": 5,
        "selection_settings": settings["selection_settings"],
        "selected_slots": [0, 1, 2, 3, 4],
        "featureless_selected_slots": [2],
        "video_sha256": "b" * 64,
    }

    def fake_call(*_args):
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(simulator_main, "_call_database_selector", fake_call)
    result = simulator_main._select_encoded_video_keyframes(
        root=tmp_path, python=tmp_path / "python", video_file="video.mp4",
        output_file=str(sidecar), expected_frames=10, expected_settings=settings,
    )
    assert result["usable_anchor_slots"] == [0, 1, 3, 4]

    payload["source_total_frames"] = 9
    with pytest.raises(RuntimeError, match="invalid keyframe sidecar"):
        simulator_main._select_encoded_video_keyframes(
            root=tmp_path, python=tmp_path / "python", video_file="video.mp4",
            output_file=str(sidecar), expected_frames=10, expected_settings=settings,
        )
