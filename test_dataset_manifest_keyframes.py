"""The recording manifest must reject calibration outside measured image keyframes."""

import hashlib
import json

import cv2
import numpy as np

from simulator.config import SimulatorConfig
from simulator.dataset_manifest import write_dataset_manifest


def _tiny_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 24))
    assert writer.isOpened()
    for value in (0, 80, 160):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()


def test_manifest_checks_image_selected_anchor_contract(tmp_path):
    video = tmp_path / "video.mp4"
    calibration = tmp_path / "calibration.json"
    keyframes = tmp_path / "video.keyframes.json"
    manifest = tmp_path / "manifest.json"
    _tiny_video(video)

    cfg = SimulatorConfig()
    cfg.video_file = str(video)
    cfg.calib_file = str(calibration)
    cfg.keyframe_file = str(keyframes)
    cfg.frame_step = 1
    cfg.target_fps = 30
    cfg.telemetry_file = ""
    cfg.gt_file = ""
    cfg.frame_gt_file = ""

    keyframes.write_text(
        json.dumps(
            {
                "frame_step": 1,
                "source_total_frames": 3,
                "total_slots": 3,
                "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "selected_slots": [0, 2],
                "featureless_selected_slots": [],
            }
        ),
        encoding="utf-8",
    )
    calibration.write_text(
        json.dumps(
            {
                "keyframe_selection": {
                    "source": "database_image_selector",
                    "frame_step": 1,
                },
                "anchors": [{"frame_id": 0}, {"frame_id": 2}],
            }
        ),
        encoding="utf-8",
    )

    result = write_dataset_manifest(
        str(manifest), status="complete", config=cfg, renderer="cpu", frame_count=3
    )
    assert result["status"] == "complete", result["validation_errors"]
    assert "keyframes" in result["files"]

    calibration.write_text(
        json.dumps(
            {
                "keyframe_selection": {
                    "source": "database_image_selector",
                    "frame_step": 1,
                },
                "anchors": [{"frame_id": 1}],
            }
        ),
        encoding="utf-8",
    )
    rejected = write_dataset_manifest(
        str(manifest), status="complete", config=cfg, renderer="cpu", frame_count=3
    )
    assert rejected["status"] == "invalid"
    assert any("absent from keyframes" in error for error in rejected["validation_errors"])
