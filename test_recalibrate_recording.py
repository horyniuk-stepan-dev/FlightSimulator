"""Previously recorded ground truth can be anchored to measured image slots."""

import hashlib
import json

import cv2
import numpy as np

from simulator.recalibrate_recording import recalibrate_recording


def test_recalibration_uses_each_target_slots_own_affine(tmp_path):
    video = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 24))
    assert writer.isOpened()
    for index in range(6):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()

    gt_path = tmp_path / "ground_truth.json"
    calibration_path = tmp_path / "calibration.json"
    keyframes_path = tmp_path / "video.keyframes.json"
    candidate_calibration = tmp_path / "calibration.candidate.json"
    candidate_gt = tmp_path / "ground_truth.candidate.json"
    projection = {"mode": "WEB_MERCATOR", "reference_gps": None}
    slots = [
        {
            "slot": index,
            "video_frame": index,
            "affine": [[0.5, 0.0, float(index * 10)], [0.0, -0.5, 0.0]],
            "heading_deg": 0.0,
            "rmse_m": 1.0,
            "is_anchor": index == 0,
            "anchor_reason": "first" if index == 0 else None,
        }
        for index in range(6)
    ]
    gt_path.write_text(
        json.dumps({"frame_step": 1, "frame_size": [32, 24], "projection": projection, "slots": slots}),
        encoding="utf-8",
    )
    old_qa = {"rmse_m": 1.0, "points_2d": [[16, 12]], "points_gps": [[50, 30]]}
    calibration_path.write_text(
        json.dumps(
            {
                "version": "2.4",
                "projection": projection,
                "frame_size": [32, 24],
                "keyframe_selection": {"criterion": "overlap", "frame_step": 1},
                "anchors": [{"frame_id": 0, "affine_matrix": slots[0]["affine"], "qa_data": old_qa}],
            }
        ),
        encoding="utf-8",
    )
    keyframes_path.write_text(
        json.dumps(
            {
                "source_video": str(video.resolve()),
                "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "source_total_frames": 6,
                "total_slots": 6,
                "frame_step": 1,
                "selected_slots": [0, 2, 4],
                "featureless_selected_slots": [],
                "selection_settings": {
                    "criterion": "overlap",
                    "frame_step": 1,
                    "max_overlap": 0.5,
                    "max_gap_frames": 60,
                },
            }
        ),
        encoding="utf-8",
    )

    result = recalibrate_recording(
        video_path=video,
        ground_truth_path=gt_path,
        calibration_path=calibration_path,
        keyframes_path=keyframes_path,
        output_calibration_path=candidate_calibration,
        output_ground_truth_path=candidate_gt,
    )
    assert result["anchor_slots"] == [0, 4]
    anchors = json.loads(candidate_calibration.read_text(encoding="utf-8"))["anchors"]
    assert anchors[0]["qa_data"] == old_qa
    assert anchors[1]["affine_matrix"] == slots[4]["affine"]
    assert anchors[1]["qa_data"]["points_2d"] == []
    assert "not stored" in anchors[1]["qa_data"]["notes"]
    refreshed_gt = json.loads(candidate_gt.read_text(encoding="utf-8"))["slots"]
    assert [record["slot"] for record in refreshed_gt if record["is_anchor"]] == [0, 4]
    original = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert len(original["anchors"]) == 1
    assert original["anchors"][0]["frame_id"] == 0
