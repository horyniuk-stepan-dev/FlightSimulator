"""
Script to restart and run video generation for bochkivtsi, bochkivtsi_500m, and bochkivtsi_2000m
simultaneously in parallel with real-time progress monitoring.

Without arguments it reproduces the original recordings (150 m/s, nose follows
the velocity vector) into the original folders. For a new set of layers:

    python run_parallel_bochkivtsi.py --tag hh --heading-hold-deg 0 \
        --speed 500=50 --speed 1000=100 --speed 2000=150

writes output/bochkivtsi_hh_{500,1000,2000}m. A folder that already holds a
recording (manifest.json) is never overwritten unless --force is given.
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = str(ROOT_DIR / ".venv" / "Scripts" / "python.exe")

TASKS = [
    {
        "name": "bochkivtsi",
        "altitude": 1000,
        "speed": 150.0,
        "est_frames": 22603,
        "output_dir": ROOT_DIR / "output" / "bochkivtsi",
    },
    {
        "name": "bochkivtsi_500m",
        "altitude": 500,
        "speed": 150.0,
        "est_frames": 45446,
        "output_dir": ROOT_DIR / "output" / "bochkivtsi_500m",
    },
    {
        "name": "bochkivtsi_2000m",
        "altitude": 2000,
        "speed": 150.0,
        "est_frames": 10250,
        "output_dir": ROOT_DIR / "output" / "bochkivtsi_2000m",
    },
]


def get_current_frame(output_dir: Path) -> tuple[int, str]:
    """Attempt to detect current frame count from telemetry or part file."""
    # After the flight the telemetry stops growing while the keyframe scan runs,
    # so a finished video must be reported as such, not as a frozen frame count.
    log_file = output_dir / "generation.log"
    if log_file.is_file():
        try:
            content = log_file.read_text(encoding="utf-8", errors="replace")
            if "Simulation ended." in content:
                return -1, "completed"
            if "[VideoWriterSink] Video saved to" in content:
                return -1, "flight done, keyframe scan / ground truth"
        except Exception:
            pass

    # First check telemetry.csv
    telem = output_dir / "telemetry.csv"
    if telem.is_file():
        try:
            with open(telem, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                # Read last 2KB
                read_size = min(size, 2048)
                f.seek(max(0, size - read_size))
                lines = f.read().decode("utf-8", errors="replace").strip().splitlines()
                if len(lines) > 1:
                    last_line = lines[-1].split(",")
                    if last_line and last_line[0].isdigit():
                        return int(last_line[0]), "simulating"
        except Exception:
            pass

    # Check log for keyframe selector or completion
    log_file = output_dir / "generation.log"
    if log_file.is_file():
        try:
            content = log_file.read_text(encoding="utf-8", errors="replace")
            if "Simulation ended." in content:
                return -1, "completed"
            elif "Image selector kept" in content or "Starting keyframe selection" in content:
                return -1, "keyframe scan"
            elif "[VideoWriterSink] Video saved to" in content:
                return -1, "video saved / analyzing"
        except Exception:
            pass

    return 0, "starting"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Record the Bochkivtsi reference layers in parallel.")
    parser.add_argument("--tag", default="", help="output/bochkivtsi_<tag>_<alt>m instead of the original folders")
    parser.add_argument(
        "--heading-hold-deg",
        type=float,
        default=None,
        help="constant camera heading (survey gimbal); legs keep one orientation",
    )
    parser.add_argument(
        "--speed",
        action="append",
        default=[],
        metavar="ALT=M_S",
        help="flight speed per layer altitude, e.g. 500=50 (default 150 m/s)",
    )
    parser.add_argument(
        "--only",
        action="append",
        type=int,
        default=[],
        metavar="ALT",
        help="record only this layer altitude (repeatable), e.g. --only 2000",
    )
    parser.add_argument(
        "--zoom",
        action="append",
        default=[],
        metavar="ALT=Z",
        help="map tile zoom per layer altitude, e.g. 500=18 (default: simulator's 17); "
        "each +1 quadruples texture memory",
    )
    parser.add_argument(
        "--anchor-spacing-m",
        type=float,
        default=None,
        help="GT anchor spacing along straight legs in metres, converted per layer to DB slots "
        "(1 slot = 1 s at the default frame_step 30 / 30 fps); default: simulator's 60 slots",
    )
    parser.add_argument("--force", action="store_true", help="overwrite folders with a recording")
    return parser.parse_args(argv)


def build_tasks(args):
    speeds = {}
    for item in args.speed:
        altitude, _, value = item.partition("=")
        speeds[int(float(altitude))] = float(value)
    zooms = {}
    for item in args.zoom:
        altitude, _, value = item.partition("=")
        zooms[int(float(altitude))] = int(value)
    known = {base["altitude"] for base in TASKS}
    unknown = (set(speeds) | set(zooms) | set(args.only)) - known
    if unknown:
        raise SystemExit(f"unknown layer altitude(s): {sorted(unknown)}; known: {sorted(known)}")
    tasks = []
    for base in TASKS:
        if args.only and base["altitude"] not in args.only:
            continue
        task = dict(base)
        task["speed"] = speeds.get(task["altitude"], base["speed"])
        task["zoom"] = zooms.get(task["altitude"])
        task["anchor_slots"] = (
            max(1, round(args.anchor_spacing_m / task["speed"]))
            if args.anchor_spacing_m is not None
            else None
        )
        # Frame count scales inversely with speed over the same route.
        task["est_frames"] = int(base["est_frames"] * base["speed"] / task["speed"])
        if args.tag:
            task["name"] = f"bochkivtsi_{args.tag}_{task['altitude']}m"
            task["output_dir"] = ROOT_DIR / "output" / task["name"]
        tasks.append(task)
    return tasks


def main(argv=None):
    args = parse_args(argv)
    tasks = build_tasks(args)
    existing = [t["name"] for t in tasks if (t["output_dir"] / "manifest.json").exists()]
    if existing and not args.force:
        print(f"Refusing to overwrite existing recordings: {existing} (use --tag or --force)")
        sys.exit(1)

    print("=" * 80)
    print("FLIGHT SIMULATOR - PARALLEL VIDEO GENERATION LAUNCHER")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {PYTHON_EXE}")
    print("=" * 80)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"

    procs = {}
    log_handles = {}

    for task in tasks:
        out_dir = task["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        # Clean up any leftover temporary files
        for tmp in out_dir.glob(".*.part"):
            try:
                tmp.unlink()
            except Exception:
                pass

        log_path = out_dir / "generation.log"
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        log_handles[task["name"]] = log_file

        cmd = [
            PYTHON_EXE,
            "-u",
            "-m",
            "simulator.main",
            "--mode",
            "record",
            "--altitude",
            str(task["altitude"]),
            "--speed",
            str(task["speed"]),
            "--video-file",
            str((out_dir / "video.mp4").as_posix()),
            "--calib-file",
            str((out_dir / "calibration.json").as_posix()),
            "--gt-file",
            str((out_dir / "ground_truth.json").as_posix()),
            "--telemetry-file",
            str((out_dir / "telemetry.csv").as_posix()),
            "--manifest-file",
            str((out_dir / "manifest.json").as_posix()),
            "--renderer",
            "auto",
            "--fast",
            "--no-display",
            "--yes",
        ]
        if args.heading_hold_deg is not None:
            cmd += ["--heading-hold-deg", str(args.heading_hold_deg)]
        if task.get("zoom") is not None:
            cmd += ["--zoom", str(task["zoom"])]
        if task.get("anchor_slots") is not None:
            # Fill anchors may sit no closer than half the spacing to another anchor.
            cmd += [
                "--anchor-max-spacing-slots",
                str(task["anchor_slots"]),
                "--anchor-spacing-slots",
                str(max(1, task["anchor_slots"] // 2)),
            ]

        print(
            f"Launching {task['name']} (altitude: {task['altitude']}m, "
            f"speed: {task['speed']:g} m/s, heading hold: {args.heading_hold_deg}, "
            f"zoom: {task.get('zoom') or 'default'}, "
            f"anchor spacing: {task.get('anchor_slots') or 'default'} slots, "
            f"log: {log_path.name})..."
        )
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        procs[task["name"]] = proc
        task["proc"] = proc
        task["start_time"] = time.time()
        task["done"] = False

    print(f"\nAll {len(tasks)} generation jobs launched simultaneously!")
    print("-" * 80)

    # Monitoring loop
    last_print = 0
    all_done = False

    try:
        while not all_done:
            time.sleep(5)
            now = time.time()
            all_done = True
            statuses = []

            for task in tasks:
                proc = task["proc"]
                ret = proc.poll()
                out_dir = task["output_dir"]

                if ret is None:
                    all_done = False
                    frame, state = get_current_frame(out_dir)
                    if frame > 0:
                        pct = min(100.0, (frame / task["est_frames"]) * 100)
                        statuses.append(f"{task['name']}: {frame}/{task['est_frames']} ({pct:.1f}%)")
                    else:
                        statuses.append(f"{task['name']}: {state}")
                else:
                    if not task["done"]:
                        task["done"] = True
                        task["exit_code"] = ret
                        elapsed = time.time() - task["start_time"]
                        status_str = "SUCCESS" if ret == 0 else f"FAILED (code {ret})"
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] JOB FINISHED: {task['name']} -> {status_str} in {elapsed:.1f}s")
                    statuses.append(f"{task['name']}: {'DONE' if ret == 0 else 'ERR'}")

            if now - last_print >= 10 or all_done:
                last_print = now
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] " + " | ".join(statuses))

    except KeyboardInterrupt:
        print("\nTermination requested! Stopping all jobs...")
        for task in tasks:
            proc = task["proc"]
            if proc.poll() is None:
                proc.terminate()
        for task in tasks:
            proc = task["proc"]
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(1)
    finally:
        for f in log_handles.values():
            try:
                f.close()
            except Exception:
                pass

    print("=" * 80)
    print("ALL JOBS FINISHED. Summary:")
    all_success = True
    for task in tasks:
        code = task.get("exit_code", -1)
        manifest_path = task["output_dir"] / "manifest.json"
        status_info = f"exit code {code}"
        if manifest_path.is_file():
            try:
                import json
                m = json.loads(manifest_path.read_text(encoding="utf-8"))
                status_info += f", status={m.get('status')}, frames={m.get('frame_count')}"
            except Exception:
                pass
        print(f"  - {task['name']}: {status_info}")
        if code != 0:
            all_success = False

    print("=" * 80)
    if not all_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
