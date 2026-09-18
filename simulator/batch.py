"""Batch dataset factory for reference layers and query scenarios."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _run(
    output: Path,
    *,
    scenario: Path,
    reference: bool,
    args,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if args.resume and existing.get("status") == "complete":
            print(f"[batch] already complete: {output}")
            return manifest
        raise FileExistsError(
            f"Refusing to overwrite existing run: {output}. Use a new output directory."
        )
    existing_payload = [path for path in output.iterdir() if path.name != "scenario.json"]
    if existing_payload:
        raise FileExistsError(
            f"Refusing to overwrite incomplete run: {output}. Use a new output directory."
        )

    scenario_copy = output / "scenario.json"
    if scenario.resolve() != scenario_copy.resolve():
        shutil.copy2(scenario, scenario_copy)
    scenario = scenario_copy

    command = [
        sys.executable,
        "-m", "simulator.main",
        "--mode", "record",
        "--scenario", str(scenario),
        "--geotiff", str(args.geotiff),
        "--renderer", args.renderer,
        "--fps", str(args.fps),
        "--frame-step", str(args.frame_step),
        "--speed", str(args.speed),
        "--seed", str(args.seed),
        "--video-file", str(output / "video.mp4"),
        "--frame-gt-file", str(output / "frame_ground_truth.jsonl"),
        "--telemetry-file", str(output / "telemetry.csv"),
        "--manifest-file", str(manifest),
        "--no-display", "--fast", "--yes",
    ]
    if reference:
        command.extend(
            (
                "--calib-file", str(output / "calibration.json"),
                "--gt-file", str(output / "ground_truth.json"),
            )
        )
    if args.elevation:
        command.extend(
            (
                "--elevation", str(args.elevation),
                "--elevation-format", args.elevation_format,
            )
        )
    if args.no_hillshade:
        command.append("--no-hillshade")
    if args.max_frames:
        command.extend(("--max-frames", str(args.max_frames)))
    print(f"[batch] generating {output.name} from {scenario.name}")
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
    result = json.loads(manifest.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise RuntimeError(
            f"Dataset run did not validate: {output} ({result.get('validation_errors')})"
        )
    return manifest


def _constant_scenario(path: Path, altitude: float, duration: float) -> None:
    data = {
        "version": 1,
        "name": f"reference_{altitude:g}m",
        "duration_s": duration,
        "profiles": {
            "altitude_m": [{"time_s": 0.0, "value": altitude}],
            "pitch_deg": [{"time_s": 0.0, "value": 0.0}],
            "roll_deg": [{"time_s": 0.0, "value": 0.0}],
            "yaw_deg": [{"time_s": 0.0, "value": 0.0}],
        },
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible simulator dataset")
    parser.add_argument("scenarios", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--geotiff", type=Path, required=True)
    parser.add_argument("--elevation", type=Path)
    parser.add_argument(
        "--elevation-format",
        choices=["auto", "terrarium", "meters"],
        default="auto",
    )
    parser.add_argument("--reference-altitude", type=float, action="append", default=[])
    parser.add_argument("--reference-duration", type=float, default=30.0)
    parser.add_argument("--renderer", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-step", type=int, default=30)
    parser.add_argument("--speed", type=float, default=50.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-hillshade", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.geotiff.exists():
        parser.error(f"GeoTIFF not found: {args.geotiff}")
    if args.elevation and not args.elevation.exists():
        parser.error(f"Elevation GeoTIFF not found: {args.elevation}")
    if not args.scenarios and not args.reference_altitude:
        parser.error("provide scenarios and/or --reference-altitude")

    args.out.mkdir(parents=True, exist_ok=True)
    manifests = []
    for altitude in args.reference_altitude:
        run_dir = args.out / "references" / f"layer_{altitude:g}m"
        run_dir.mkdir(parents=True, exist_ok=True)
        scenario = run_dir / "scenario.json"
        if not scenario.exists():
            _constant_scenario(scenario, altitude, args.reference_duration)
        manifests.append(
            _run(run_dir, scenario=scenario, reference=True, args=args)
        )

    for scenario in args.scenarios:
        if not scenario.exists():
            parser.error(f"Scenario not found: {scenario}")
        manifests.append(
            _run(
                args.out / "queries" / scenario.stem,
                scenario=scenario,
                reference=False,
                args=args,
            )
        )

    index = {
        "version": 1,
        "runs": [str(path.relative_to(args.out)) for path in manifests],
    }
    (args.out / "dataset.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    print(f"[batch] complete: {len(manifests)} run(s) in {args.out}")


if __name__ == "__main__":
    main()
