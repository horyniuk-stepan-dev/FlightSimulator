"""Export image-driven DB slots from the simulator, using localization code.

This reads the finished reference video, never ``database.h5``.  FlightSimulator
uses the resulting slot list to place calibration anchors before a project is
created. The simulator launches this module with DroneLocalization's Python
environment, so it uses the existing builder's models and decision classes.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

LOCALIZER_ROOT = Path(
    os.environ.get(
        "DRONE_LOCALIZATION_ROOT",
        Path(__file__).resolve().parents[2] / "DroneLocalization",
    )
).resolve()
if not LOCALIZER_ROOT.is_dir():
    raise RuntimeError(f"DroneLocalization source directory not found: {LOCALIZER_ROOT}")
if str(LOCALIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCALIZER_ROOT))

from config import APP_CONFIG, CONFIG_LOADED_FROM, get_cfg
from simulator.database_keyframe_scan import (
    scan_video_keyframes,
    selection_settings,
    sha256_file,
)


def effective_config() -> dict:
    """Apply the same hardware speed overrides as application startup."""
    from src.utils.hardware_profile import HardwareProfile

    config = copy.deepcopy(APP_CONFIG)
    hardware = HardwareProfile.detect()
    hardware.apply_torch_backends(
        deterministic=bool(get_cfg(config, "models.performance.deterministic", False))
    )
    if get_cfg(config, "models.performance.auto_tune", False):
        overrides = hardware.auto_tune(config)
        if overrides:
            hardware.apply_overrides(config, overrides)
    return config


def config_payload(config: dict) -> dict:
    settings = selection_settings(config)
    return {
        "frame_step": settings["frame_step"],
        "criterion": settings["criterion"],
        "max_overlap": settings["max_overlap"],
        "max_gap_frames": settings["max_gap_frames"],
        "selection_settings": settings,
        "config_source": CONFIG_LOADED_FROM,
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_path = stream.name
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-config", action="store_true", help="Print effective DB selection settings as JSON")
    parser.add_argument("--video", type=Path, help="Finished reference video to scan")
    parser.add_argument("--output", type=Path, help="Output JSON sidecar path")
    args = parser.parse_args(argv)

    if args.print_config:
        if args.video or args.output:
            parser.error("--print-config cannot be combined with --video or --output")
    elif args.video is None or args.output is None:
        parser.error("--video and --output are required unless --print-config is used")

    try:
        # ModelManager and HardwareProfile log to stdout in some environments.
        # Reserve stdout for the machine-readable one-line JSON contract.
        with contextlib.redirect_stdout(sys.stderr):
            config = effective_config()
            if args.print_config:
                payload = config_payload(config)
            else:
                video = args.video.resolve(strict=True)
                if not video.is_file():
                    raise ValueError(f"Video path is not a file: {video}")
                last_progress = -1

                def progress(percent: int) -> None:
                    nonlocal last_progress
                    bucket = int(percent) // 10
                    if bucket > last_progress:
                        last_progress = bucket
                        print(f"Keyframe scan: {min(100, bucket * 10)}%", file=sys.stderr)

                result = scan_video_keyframes(video, config, progress_callback=progress)
                payload = {
                    "version": 1,
                    "source_video": str(video),
                    "video_sha256": sha256_file(video),
                    "source_total_frames": result.source_total_frames,
                    "total_slots": result.total_slots,
                    "frame_width": result.frame_width,
                    "frame_height": result.frame_height,
                    "frame_step": result.frame_step,
                    "selected_slots": result.selected_slots,
                    "featureless_selected_slots": result.featureless_selected_slots,
                    "selection_settings": selection_settings(config),
                    "config_source": CONFIG_LOADED_FROM,
                }
                write_json_atomic(args.output.resolve(), payload)
        # The full slot list is already in the sidecar. Keep stdout bounded
        # for recordings with many thousands of selected frames.
        stdout_payload = payload if args.print_config else {
            "output": str(args.output.resolve()),
            "selected_count": len(payload["selected_slots"]),
            "total_slots": payload["total_slots"],
        }
        print(json.dumps(stdout_payload, ensure_ascii=False, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(f"Keyframe selection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
