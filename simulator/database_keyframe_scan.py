"""Select the database's image-driven keyframes without creating a database.

The simulator uses this on the *encoded* reference video before writing its
calibration anchors.  It deliberately runs the same decoder, local feature
extractor, mask, matcher and :class:`FrameProcessor` decision path as
``DatabaseBuilder``.  No HDF5 or LanceDB file is opened.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import get_cfg
from src.database import keyframe_selector
from src.database.frame_processor import FrameProcessor
from src.database.video_frame_source import EOF_INDEX, VideoFrameSource


def selection_settings(config: dict) -> dict[str, Any]:
    """Effective config keys that can affect the selected DB slot IDs."""
    g = lambda path, default: get_cfg(config, path, default)
    return {
        "frame_step": int(g("database.frame_step", 30)),
        "criterion": str(g("database.keyframe_criterion", "step")),
        "max_overlap": float(g("database.keyframe_max_overlap", 0.5)),
        "max_gap_frames": int(g("database.keyframe_max_gap_frames", 0)),
        "min_translation_px": float(g("database.keyframe_min_translation_px", 0.0)),
        "min_rotation_deg": float(g("database.keyframe_min_rotation_deg", 1.5)),
        "always_save_first": bool(g("database.keyframe_always_save_first", True)),
        "required_frame_ids": [int(x) for x in (g("database.required_frame_ids", []) or [])],
        "inter_frame_min_matches": int(g("database.inter_frame_min_matches", 15)),
        "inter_frame_ransac_thresh": float(g("database.inter_frame_ransac_thresh", 3.0)),
        "homography_backend": str(g("homography.backend", "opencv")),
        "use_mad_ransac": bool(g("homography.use_mad_ransac", True)),
        "mad_k_factor": float(g("homography.mad_k_factor", 2.5)),
        "masking_strategy": str(g("preprocessing.masking_strategy", "yolo")),
        "yolo_batch_size": int(g("database.yolo_batch_size", 1)),
        "local_extractor": str(g("models.local_extractor", "aliked")),
        "fallback_extractor": str(g("localization.fallback_extractor", "aliked")),
        "ratio_threshold": float(g("localization.ratio_threshold", 0.75)),
        "max_local_edge": int(g("localization.max_local_edge", 1600)),
        "use_decord": bool(g("database.use_decord", True)),
        "decode_batch_size": int(g("database.decode_batch_size", 32)),
        "prefetch_queue_size": int(g("database.prefetch_queue_size", 32)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _NullWriter:
    """FrameProcessor storage hooks; the decision state remains unchanged."""

    def write_pose(self, frame_id: int, pose: np.ndarray) -> None:
        pass

    def save_frame_data(self, frame_id: int, features: dict, pose: np.ndarray) -> None:
        pass


class _LocalOnlyAdapter:
    """Skip DINO descriptors, which do not participate in keyframe decisions."""

    def __init__(self, extractor):
        self.extractor = extractor

    def extract_features(self, image: np.ndarray, mask: np.ndarray) -> dict:
        return self.extractor.extract_local_features(image, mask)


@dataclass(frozen=True)
class ScanResult:
    selected_slots: list[int]
    featureless_selected_slots: list[int]
    source_total_frames: int
    total_slots: int
    frame_width: int
    frame_height: int
    frame_step: int


def scan_with_components(
    source,
    feature_extractor,
    matcher,
    masking_strategy,
    config: dict,
    *,
    progress_callback: Callable[[int], None] | None = None,
) -> ScanResult:
    """Run DatabaseBuilder's keyframe decision path with injected collaborators.

    This narrow seam allows deterministic tests with fake models, while the
    production wrapper below constructs the exact DB components.
    """
    settings = selection_settings(config)
    if settings["frame_step"] != int(source.frame_step):
        raise ValueError("source frame_step differs from database.frame_step")
    criterion = settings["criterion"]
    use_selection = settings["min_translation_px"] > 0 or criterion == "overlap"

    def compute_h(fa: dict, fb: dict) -> np.ndarray | None:
        return keyframe_selector.compute_inter_frame_homography(
            matcher,
            fa,
            fb,
            min_matches=settings["inter_frame_min_matches"],
            ransac_thresh=settings["inter_frame_ransac_thresh"],
            homography_backend=settings["homography_backend"],
            use_mad_ransac=settings["use_mad_ransac"],
            mad_k_factor=settings["mad_k_factor"],
        )

    def significant(H: np.ndarray, width: int, height: int) -> bool:
        return keyframe_selector.is_significant_motion(
            H,
            width,
            height,
            min_translation_px=settings["min_translation_px"],
            min_rotation_deg=settings["min_rotation_deg"],
        )

    processor = FrameProcessor(
        feature_extractor=_LocalOnlyAdapter(feature_extractor),
        db_writer=_NullWriter(),
        compute_inter_frame_h=compute_h,
        is_significant_motion=significant,
        draw_keypoints=lambda *args: None,
        config=config,
        width=source.width,
        height=source.height,
        num_frames=source.num_frames,
        use_keyframe_selection=use_selection,
        always_save_first=settings["always_save_first"],
        keyframe_criterion=criterion,
        overlap_gate=lambda H, w, h: keyframe_selector.is_overlap_below(
            H, w, h, max_overlap=settings["max_overlap"]
        ),
        keyframe_max_gap_frames=settings["max_gap_frames"],
        forced_frame_ids=set(settings["required_frame_ids"]),
        progress_callback=progress_callback,
    )
    batch_size = max(1, settings["yolo_batch_size"])
    featureless_selected: list[int] = []
    pending: list[tuple] = []
    processed_count = 0
    frame_queue = source.start_prefetch()
    try:
        while True:
            idx, data = frame_queue.get()
            if idx != EOF_INDEX and data is not None:
                bgr, rgb = data
                pending.append((idx, bgr, rgb))
                if len(pending) < batch_size:
                    continue
            if not pending:
                break

            masks = masking_strategy.get_mask_batch([item[2] for item in pending])
            if len(masks) != len(pending):
                raise RuntimeError("Masking strategy returned a different batch size")
            for (slot, bgr, rgb), static_mask in zip(pending, masks):
                previous_saved = processor.saved_count
                processor.process(slot, bgr, rgb, static_mask)
                processed_count += 1
                if processor.saved_count > previous_saved and len(processor.prev_features["keypoints"]) == 0:
                    featureless_selected.append(int(slot))
            pending = []
            if idx == EOF_INDEX:
                break
        source.raise_if_failed()
    finally:
        source.join(timeout=5)
        source.release()

    if processed_count != source.num_frames:
        raise RuntimeError(
            f"Decoded {processed_count} sampled slots, expected {source.num_frames}; "
            "refusing incomplete keyframe selection"
        )
    if not processor.frame_index_map:
        raise RuntimeError("No keyframes selected from reference video")
    if len(featureless_selected) == len(processor.frame_index_map):
        raise RuntimeError("No local features in any selected keyframe")

    return ScanResult(
        selected_slots=processor.frame_index_map,
        featureless_selected_slots=featureless_selected,
        source_total_frames=int(source.total_frames),
        total_slots=int(source.num_frames),
        frame_width=int(source.width),
        frame_height=int(source.height),
        frame_step=int(source.frame_step),
    )


def scan_video_keyframes(
    video_path: str | Path,
    config: dict,
    *,
    progress_callback: Callable[[int], None] | None = None,
) -> ScanResult:
    """Scan a finished MP4 using the same local-vision path as DatabaseBuilder."""
    from src.localization.matcher import FeatureMatcher
    from src.models.model_manager import ModelManager
    from src.models.wrappers.feature_extractor import FeatureExtractor
    from src.models.wrappers.masking_strategy import create_masking_strategy

    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference video not found: {path}")
    settings = selection_settings(config)
    source = VideoFrameSource(
        str(path),
        frame_step=settings["frame_step"],
        use_decord=settings["use_decord"],
        decode_batch_size=settings["decode_batch_size"],
        prefetch_size=settings["prefetch_queue_size"],
    )
    try:
        manager = ModelManager(config=config)
        masking_strategy = create_masking_strategy(
            settings["masking_strategy"], manager, manager.device
        )
        # DatabaseBuilder branches on fallback_extractor, even when
        # models.local_extractor names a different model. Mirror that exact
        # choice so descriptor dimensions and matching stay identical.
        local_model = (
            manager.load_xfeat()
            if settings["fallback_extractor"] == "xfeat"
            else manager.load_local_extractor()
        )
        extractor = FeatureExtractor(local_model, None, manager.device, config=config)
        matcher = FeatureMatcher(model_manager=manager, config=config)
        return scan_with_components(
            source,
            extractor,
            matcher,
            masking_strategy,
            config,
            progress_callback=progress_callback,
        )
    finally:
        source.release()
