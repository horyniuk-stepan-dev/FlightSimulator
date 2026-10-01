"""The read-only video scan follows the database frame processor's decisions."""

from queue import Queue
import os
from pathlib import Path
import sys

import numpy as np
import pytest

LOCALIZER_ROOT = Path(
    os.environ.get(
        "DRONE_LOCALIZATION_ROOT",
        Path(__file__).resolve().parent.parent / "DroneLocalization",
    )
).resolve()
if str(LOCALIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCALIZER_ROOT))
pytest.importorskip("faiss", reason="run this selector test with DroneLocalization's environment")

from simulator import database_keyframe_scan as keyframe_scan
from src.database.video_frame_source import EOF_INDEX


class _Source:
    frame_step = 1
    total_frames = 5
    num_frames = 5
    width = 100
    height = 100

    def start_prefetch(self):
        queue = Queue()
        for slot in range(self.num_frames):
            frame = np.full((self.height, self.width, 3), slot, dtype=np.uint8)
            queue.put((slot, (frame, frame)))
        queue.put((EOF_INDEX, None))
        return queue

    def raise_if_failed(self):
        pass

    def join(self, timeout=5):
        pass

    def release(self):
        pass


class _Extractor:
    def extract_local_features(self, image, mask):
        count = 0 if image[0, 0, 0] == 4 else 4
        return {
            "keypoints": np.zeros((count, 2), dtype=np.float32),
            "descriptors": np.zeros((count, 128), dtype=np.float32),
        }


class _Masker:
    def get_mask_batch(self, images):
        return [np.full(image.shape[:2], 255, dtype=np.uint8) for image in images]


def test_scan_keeps_image_selected_slots_and_marks_featureless(monkeypatch):
    # Each adjacent image is displaced 30 px; 50% overlap is crossed after
    # two slots. FrameProcessor is the same implementation used by the builder.
    monkeypatch.setattr(
        keyframe_scan.keyframe_selector,
        "compute_inter_frame_homography",
        lambda *_args, **_kwargs: np.array(
            [[1.0, 0.0, 30.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    )
    result = keyframe_scan.scan_with_components(
        _Source(),
        _Extractor(),
        object(),
        _Masker(),
        {
            "database": {
                "frame_step": 1,
                "keyframe_criterion": "overlap",
                "keyframe_max_overlap": 0.5,
                "yolo_batch_size": 2,
            }
        },
    )
    assert result.selected_slots == [0, 2, 4]
    assert result.featureless_selected_slots == [4]
    assert result.total_slots == 5
