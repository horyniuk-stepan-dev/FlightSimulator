"""Record short survey flights over several regions for a VLAD vocabulary.

The vocabulary must come from imagery OUTSIDE the localization test area: built from
the test area it is map-specific and makes the test look better than it is (AnyLoc,
arXiv 2308.00688, Table V: aerial R@1 domain-specific 76.2 vs map-specific 62.9).
Each region gets a ~8.6 x 8.6 km map and flies scenarios/vlad_vocab_survey.json
(5 min, 600-2000 m, small pitch/roll), so one video spans the GSD of all layers.

    python run_vocab_regions.py                      # the four default regions
    python run_vocab_regions.py --only steppe
    python run_vocab_regions.py --region delta=45.45,29.60

Rendering is forced onto the GPU (--renderer gpu: the simulator stops instead of
silently falling back to the CPU) and regions are recorded one after another: the
renderer keeps the whole z18 map on the GPU as float32 (~6 GB per region here), and
two parallel jobs on 2026-10-01 rendered 13 + 3 fps against ~33 fps for one job.

writes output/vocab_<region>/video.mp4 (+ the usual GT files) and prints the
build_vlad_vocab.py command for DroneLocalization at the end. Regions that overlap
the Bochkivtsi test maps (incl. the topnew neighbour) are refused. A folder that
already holds a recording (manifest.json) is skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from run_parallel_bochkivtsi import PYTHON_EXE, ROOT_DIR, get_current_frame

# Region centres (lat, lon): rural landscapes away from the test area.
REGIONS: dict[str, tuple[float, float]] = {
    "polissia": (51.25, 28.55),  # Zhytomyr oblast: forest, bog, small fields
    "steppe": (48.20, 32.90),  # Kirovohrad oblast: large open fields
    "dnipro": (49.70, 31.50),  # near Kaniv: river, reservoir, hills, forest
    "foothills": (48.42, 24.85),  # Ivano-Frankivsk oblast: Carpathian foothills
}

# Bochkivtsi test maps and the topnew neighbour, padded by ~10 km.
EXCLUDED_AREAS: dict[str, tuple[float, float, float, float]] = {
    "bochkivtsi test area": (48.27, 25.92, 48.54, 26.40),  # lat_min, lon_min, lat_max, lon_max
}

DEFAULT_SCENARIO = "scenarios/vlad_vocab_survey.json"
M_PER_DEG_LAT = 111_320.0


def tile_count(bbox: tuple[float, float, float, float], zoom: int) -> tuple[int, int]:
    """Web-Mercator tiles (x, y) the simulator downloads for a bbox."""
    lat_min, lon_min, lat_max, lon_max = bbox
    n = 2**zoom

    def tile_x(lon: float) -> int:
        return int(math.floor((lon + 180.0) / 360.0 * n))

    def tile_y(lat: float) -> int:
        lat_rad = math.radians(lat)
        return int(math.floor((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n))

    return tile_x(lon_max) - tile_x(lon_min) + 1, tile_y(lat_min) - tile_y(lat_max) + 1


def map_gpu_gb(bbox: tuple[float, float, float, float], zoom: int) -> float:
    """GPU memory of the map texture: the renderer uploads it as float32 BGR."""
    tiles_x, tiles_y = tile_count(bbox, zoom)
    return tiles_x * 256 * tiles_y * 256 * 3 * 4 / 1e9


def region_bbox(lat: float, lon: float, half_m: float) -> tuple[float, float, float, float]:
    """(lat_min, lon_min, lat_max, lon_max) of a square of +-half_m ground metres."""
    dlat = half_m / M_PER_DEG_LAT
    dlon = half_m / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def overlapping_area(bbox: tuple[float, float, float, float]) -> str | None:
    """Name of an excluded area the bbox intersects, or None."""
    lat_min, lon_min, lat_max, lon_max = bbox
    for name, (a_lat_min, a_lon_min, a_lat_max, a_lon_max) in EXCLUDED_AREAS.items():
        if lat_min < a_lat_max and lat_max > a_lat_min and lon_min < a_lon_max and lon_max > a_lon_min:
            return name
    return None


def recorded_renderer(output_dir: Path) -> str:
    """Renderer named in the recording's manifest ("gpu"/"cpu"), or "unknown"."""
    try:
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    return str(manifest.get("renderer", "unknown"))


def parse_region(text: str) -> tuple[str, tuple[float, float]]:
    name, sep, coords = text.partition("=")
    lat_text, comma, lon_text = coords.partition(",")
    if not sep or not comma or not name.strip().replace("_", "").isalnum():
        raise argparse.ArgumentTypeError(f"expected NAME=LAT,LON, got {text!r}")
    lat, lon = float(lat_text), float(lon_text)
    if not (-85 < lat < 85 and -180 <= lon <= 180):
        raise argparse.ArgumentTypeError(f"coordinates out of range: {text!r}")
    return name.strip(), (lat, lon)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--region",
        action="append",
        type=parse_region,
        default=[],
        metavar="NAME=LAT,LON",
        help="add or replace a region centre (repeatable)",
    )
    parser.add_argument("--only", action="append", default=[], metavar="NAME", help="record only these")
    parser.add_argument("--half-size-m", type=float, default=4300.0, help="map half-size (default 4300)")
    parser.add_argument("--zoom", type=int, default=18, help="tile zoom (default 18, like the 500/1000 m layers)")
    parser.add_argument("--speed", type=float, default=70.0, help="m/s (default 70: the route in ~306 s)")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel recordings (default 1: each z18 map takes ~6 GB of GPU memory)",
    )
    parser.add_argument(
        "--renderer",
        choices=["gpu", "auto", "cpu"],
        default="gpu",
        help="simulator renderer (default gpu: fail instead of falling back to the CPU)",
    )
    parser.add_argument("--force", action="store_true", help="re-record folders that hold a recording")
    return parser.parse_args(argv)


def build_tasks(args) -> list[dict]:
    regions = dict(REGIONS)
    regions.update(dict(args.region))
    unknown = set(args.only) - set(regions)
    if unknown:
        raise SystemExit(f"unknown region(s): {sorted(unknown)}; known: {sorted(regions)}")
    tasks = []
    for name, (lat, lon) in regions.items():
        if args.only and name not in args.only:
            continue
        bbox = region_bbox(lat, lon, args.half_size_m)
        area = overlapping_area(bbox)
        if area:
            raise SystemExit(f"region {name!r} overlaps the {area}; a vocabulary from it is map-specific")
        tasks.append({"name": f"vocab_{name}", "bbox": bbox, "output_dir": ROOT_DIR / "output" / f"vocab_{name}"})
    return tasks


def build_command(task: dict, args) -> list[str]:
    out_dir = task["output_dir"]
    lat_min, lon_min, lat_max, lon_max = task["bbox"]
    return [
        PYTHON_EXE, "-u", "-m", "simulator.main",
        "--mode", "record",
        "--scenario", str(args.scenario),
        "--speed", str(args.speed),
        "--zoom", str(args.zoom),
        "--lat_min", f"{lat_min:.6f}", "--lon_min", f"{lon_min:.6f}",
        "--lat_max", f"{lat_max:.6f}", "--lon_max", f"{lon_max:.6f}",
        "--video-file", (out_dir / "video.mp4").as_posix(),
        "--calib-file", (out_dir / "calibration.json").as_posix(),
        "--gt-file", (out_dir / "ground_truth.json").as_posix(),
        "--telemetry-file", (out_dir / "telemetry.csv").as_posix(),
        "--manifest-file", (out_dir / "manifest.json").as_posix(),
        "--renderer", args.renderer, "--fast", "--no-display", "--yes",
    ]  # fmt: skip


def main(argv=None) -> int:
    args = parse_args(argv)
    tasks = build_tasks(args)
    pending = []
    for task in tasks:
        if (task["output_dir"] / "manifest.json").exists() and not args.force:
            print(f"skip {task['name']}: already recorded (use --force to re-record)")
            task["exit_code"] = 0
        else:
            pending.append(task)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    running: list[dict] = []
    last_print = 0.0
    print(f"Recording {len(pending)} region(s), {args.jobs} at a time, renderer {args.renderer}")
    print(f"python: {PYTHON_EXE}")
    for task in pending:
        task["gpu_gb"] = map_gpu_gb(task["bbox"], args.zoom)
        print(f"  {task['name']}: map texture ~{task['gpu_gb']:.1f} GB on the GPU")
    if args.jobs > 1 and pending:
        biggest = sorted((t["gpu_gb"] for t in pending), reverse=True)[: args.jobs]
        print(
            f"WARNING: {args.jobs} parallel jobs need up to ~{sum(biggest):.1f} GB of GPU memory "
            "for the maps alone; past the card's memory Windows pages it to system RAM and "
            "rendering slows down several times."
        )
    try:
        while pending or running:
            while pending and len(running) < max(1, args.jobs):
                task = pending.pop(0)
                out_dir = task["output_dir"]
                out_dir.mkdir(parents=True, exist_ok=True)
                task["log"] = open(out_dir / "generation.log", "w", encoding="utf-8", buffering=1)
                task["proc"] = subprocess.Popen(
                    build_command(task, args),
                    cwd=str(ROOT_DIR),
                    env=env,
                    stdout=task["log"],
                    stderr=subprocess.STDOUT,
                )
                task["start"] = time.time()
                lat_min, lon_min, lat_max, lon_max = task["bbox"]
                print(
                    f"[{datetime.now():%H:%M:%S}] start {task['name']} "
                    f"(lat {lat_min:.4f}..{lat_max:.4f}, lon {lon_min:.4f}..{lon_max:.4f})"
                )
                running.append(task)
            time.sleep(5)
            for task in list(running):
                code = task["proc"].poll()
                if code is None:
                    continue
                task["exit_code"] = code
                task["log"].close()
                running.remove(task)
                minutes = (time.time() - task["start"]) / 60
                print(
                    f"[{datetime.now():%H:%M:%S}] {task['name']}: exit {code} after {minutes:.1f} min, "
                    f"renderer {recorded_renderer(task['output_dir'])}"
                )
            if running and time.time() - last_print >= 15:
                last_print = time.time()
                states = []
                for task in running:
                    frame, state = get_current_frame(task["output_dir"])
                    states.append(f"{task['name']}: {frame} frames" if frame > 0 else f"{task['name']}: {state}")
                print(f"[{datetime.now():%H:%M:%S}] " + " | ".join(states))
    except KeyboardInterrupt:
        print("\nStopping...")
        for task in running:
            task["proc"].terminate()
        return 1
    done = [t for t in tasks if t.get("exit_code") == 0 and (t["output_dir"] / "video.mp4").exists()]
    failed = [t["name"] for t in tasks if t.get("exit_code") not in (0, None)]
    if failed:
        print(f"FAILED: {failed} (see output/<name>/generation.log)")
    if done:
        videos = " ".join(f'"{(t["output_dir"] / "video.mp4").resolve()}"' for t in done)
        print("\nVocabulary (run in the DroneLocalization folder):")
        print(
            f"python scripts\\build_vlad_vocab.py --video {videos} "
            "--output models\\vlad_vocab_v1_c32_p256.npz --max-frames 3000 --pca-dim 256"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
