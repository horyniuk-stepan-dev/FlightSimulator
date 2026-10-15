"""
Record the Bochkivtsi reference layers (500 / 1000 / 2000 m) in parallel with
real-time progress monitoring.

Without arguments it records the best measured recipe, preset ``max`` — the
layers of the testtopboch_max test bench (2026-09-29), the most accurate maps
of all simulator projects:

    layer   folder                    speed     zoom
    1000 m  bochkivtsi_max_1000m      100 m/s   18
    2000 m  bochkivtsi_max_2000m      150 m/s   17
     500 m  bochkivtsi_max25_500m      25 m/s   18

with heading-hold 0 (every leg keeps one camera orientation, so neighbouring
legs match each other) and a GT anchor every 500 m along the legs. Single
options override the preset, e.g. ``--only 1000`` or ``--speed 500=50``.
``--preset original`` reproduces the first recordings (150 m/s, nose follows
the velocity vector, output/bochkivtsi*, default anchors). With ``--tag`` the
folders are output/bochkivtsi_<tag>_<alt>m; the testtopboch_hh layers were

    python run_parallel_bochkivtsi.py --preset original --tag hh --heading-hold-deg 0 \
        --speed 500=50 --speed 1000=100 --speed 2000=150

A folder that already holds a recording (manifest.json) is never overwritten
unless --force is given.

Every 30 s (--status-every) it prints one line per layer: map loading; flight
progress against the route the simulator planned (frames, video time, render
fps, ETA); the keyframe scan percentage; anchors / ground truth; and at the end
frames, keyframes and anchors of each recording. The planned frame count is the
route length over the speed; the autopilot cuts corners, so a flight ends at
about 93-98 % of it.

--watch shows the same lines for recordings started earlier or from another
console (same --preset/--tag/--only selects the folders) and launches nothing:

    python run_parallel_bochkivtsi.py --tag relief --watch
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = str(ROOT_DIR / ".venv" / "Scripts" / "python.exe")

# Layer altitudes and the first recordings' folders (preset original).
TASKS = [
    {
        "name": "bochkivtsi",
        "altitude": 1000,
        "speed": 150.0,
        "output_dir": ROOT_DIR / "output" / "bochkivtsi",
    },
    {
        "name": "bochkivtsi_500m",
        "altitude": 500,
        "speed": 150.0,
        "output_dir": ROOT_DIR / "output" / "bochkivtsi_500m",
    },
    {
        "name": "bochkivtsi_2000m",
        "altitude": 2000,
        "speed": 150.0,
        "output_dir": ROOT_DIR / "output" / "bochkivtsi_2000m",
    },
]


PRESETS = {
    # testtopboch_max test bench: non-anchor keyframe error 2.7 / 6.0 / 9.2 m on
    # main, 1.95 / 4.5 / 6.6 m on layer 2; the slow 500 m layer keeps the most
    # keyframe overlap the overlap selector allows.
    "max": {
        "heading_hold_deg": 0.0,
        "anchor_spacing_m": 500.0,
        "layers": {
            1000: {"name": "bochkivtsi_max_1000m", "speed": 100.0, "zoom": 18},
            2000: {"name": "bochkivtsi_max_2000m", "speed": 150.0, "zoom": 17},
            500: {"name": "bochkivtsi_max25_500m", "speed": 25.0, "zoom": 18},
        },
    },
    # First recordings (2026-09-27): nose follows the velocity vector, 180 deg
    # between neighbouring legs, cross-leg loop closures rejected.
    "original": {"heading_hold_deg": None, "anchor_spacing_m": None, "layers": {}},
}


PLAN_RE = re.compile(
    r"Planned route: (?P<waypoints>\d+) waypoints, (?P<km>[\d.]+) km, "
    r"(?P<seconds>\d+) s of flight, ~(?P<frames>\d+) video frames"
)
SCAN_RE = re.compile(r"Keyframe scan: (\d+)%")
KEPT_RE = re.compile(r"Image selector kept (\d+)/(\d+) slots")
FRAMES_SAVED_RE = re.compile(r"\[FrameGroundTruthLogger\] Saved (\d+) frame records")
FRAME_STEP = 30  # simulator --frame-step = database.frame_step: one DB slot per 30 video frames
RATE_WINDOW_S = 120.0


def fmt_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def clock_after(seconds: float) -> str:
    return (datetime.now() + timedelta(seconds=seconds)).strftime("%H:%M")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def last_telemetry(path: Path) -> tuple[int, float] | None:
    """(frame index, flight time in s) of the last telemetry row."""
    try:
        with open(path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 2048))
            lines = stream.read().decode("utf-8", errors="replace").strip().splitlines()
    except OSError:
        return None
    for line in reversed(lines[1:] if len(lines) > 1 else []):
        cells = line.split(",")
        try:
            return int(cells[0]), float(cells[3])
        except (IndexError, ValueError):
            continue
    return None


def finished_summary(output_dir: Path) -> str:
    """Frames, keyframes and anchors of a finished recording."""
    parts = []
    try:
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        parts.append(f"{manifest.get('frame_count')} frames, status {manifest.get('status')}")
    except (OSError, ValueError):
        pass
    try:
        keyframes = json.loads((output_dir / "video.keyframes.json").read_text(encoding="utf-8"))
        parts.append(f"{len(keyframes['selected_slots'])}/{keyframes['total_slots']} keyframes")
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        calibration = json.loads((output_dir / "calibration.json").read_text(encoding="utf-8"))
        parts.append(f"{len(calibration.get('anchors') or [])} anchors")
    except (OSError, ValueError):
        pass
    return ", ".join(parts) or "no manifest"


def earlier_scan_rate(output_root: Path, exclude: Path) -> float | None:
    """Slots per second of earlier keyframe scans in this output folder.

    Measured from video.mp4 (written when the flight ends) to video.keyframes.json
    (written when the scan ends). Large recordings first: model loading
    dominates the short scans.
    """
    rates = []
    for keyframes in output_root.glob("*/video.keyframes.json"):
        folder = keyframes.parent
        if folder.resolve() == exclude.resolve():
            continue
        try:
            slots = int(json.loads(keyframes.read_text(encoding="utf-8"))["total_slots"])
            seconds = keyframes.stat().st_mtime - (folder / "video.mp4").stat().st_mtime
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if seconds > 0:
            rates.append((slots, slots / seconds))
    large = [rate for slots, rate in rates if slots >= 2000]
    pool = large or [rate for _slots, rate in rates]
    return statistics.median(pool) if pool else None


class JobMonitor:
    """One status line per layer: what the job does now and when it should end."""

    def __init__(self, task: dict):
        self.task = task
        self.output_dir = Path(task["output_dir"])
        self.samples: deque[tuple[float, int]] = deque()
        self.plan: dict | None = None
        self.flight_end: float | None = None
        self.scan_rate: float | None = None
        self.scan_rate_checked = False

    def plan_from_log(self, log: str) -> dict | None:
        if self.plan is None:
            match = PLAN_RE.search(log)
            if match:
                self.plan = {
                    "waypoints": int(match["waypoints"]),
                    "km": float(match["km"]),
                    "seconds": int(match["seconds"]),
                    "frames": int(match["frames"]),
                }
        return self.plan

    def render_fps(self, now: float, frame: int) -> float | None:
        """Recorded video frames per wall-clock second over the last two minutes."""
        if not self.samples or self.samples[-1][1] != frame:
            self.samples.append((now, frame))
        while len(self.samples) > 2 and now - self.samples[1][0] >= RATE_WINDOW_S:
            self.samples.popleft()
        (t0, f0), (t1, f1) = self.samples[0], self.samples[-1]
        if t1 - t0 < 15.0 or f1 <= f0:
            return None
        return (f1 - f0) / (t1 - t0)

    def running_status(self, now: float) -> str:
        log_path = self.output_dir / "generation.log"
        if not log_path.exists():
            return "no recording in this folder"
        log = read_text(log_path)
        plan = self.plan_from_log(log)
        if "Starting simulation loop" not in log:
            if self.task.get("start_time") is None:
                return "preparing map and route"
            return f"preparing map and route ({fmt_duration(now - self.task['start_time'])})"

        if "[VideoWriterSink] Video saved to" not in log:
            if "[FrameGroundTruthLogger] Saved" in log:
                return "flight finished, closing the video"
            telemetry = last_telemetry(self.output_dir / "telemetry.csv")
            if telemetry is None:
                return "flying, no telemetry yet"
            frame, flight_s = telemetry
            fps = self.render_fps(now, frame)
            if plan is None:
                # A simulator without the planned-route line: no total to compare with.
                parts = [f"flying {frame} frames", f"video {fmt_duration(flight_s)}"]
                if fps:
                    parts.append(f"{fps:.1f} fps")
                return ", ".join(parts)
            total = plan["frames"]
            parts = [
                f"flying {frame}/~{total} frames ({100.0 * frame / total:.1f}%)",
                f"video {fmt_duration(flight_s)} of ~{fmt_duration(plan['seconds'])}",
            ]
            if fps:
                parts.append(f"{fps:.1f} fps")
                if frame < total:
                    eta = (total - frame) / fps
                    parts.append(f"ETA ~{fmt_duration(eta)} ({clock_after(eta)}) + keyframe scan")
                else:
                    parts.append("past the planned frames, ending soon")
            return ", ".join(parts)

        if self.flight_end is None:
            try:  # the video file is closed when the flight ends
                self.flight_end = (self.output_dir / "video.mp4").stat().st_mtime
            except OSError:
                self.flight_end = now
        since = now - self.flight_end
        kept = KEPT_RE.search(log)
        if kept:
            return (
                f"anchors, ground truth, manifest ({kept[1]}/{kept[2]} keyframes; "
                f"{fmt_duration(since)} since the flight)"
            )
        scans = SCAN_RE.findall(log)
        if not scans:
            return self.scan_estimate(log, since)
        percent = int(scans[-1])
        status = f"keyframe scan {percent}% ({fmt_duration(since)}"
        if percent > 0:
            eta = since * (100 - percent) / percent
            status += f", ~{fmt_duration(eta)} left ({clock_after(eta)})"
        return status + ")"

    def scan_estimate(self, log: str, since: float) -> str:
        """Keyframe scan without progress lines (a simulator older than this launcher)."""
        saved = FRAMES_SAVED_RE.search(log)
        status = f"keyframe scan running {fmt_duration(since)}"
        if not saved:
            return status
        slots = math.ceil(int(saved[1]) / FRAME_STEP)
        status = f"keyframe scan of {slots} slots running {fmt_duration(since)}"
        if not self.scan_rate_checked:
            self.scan_rate_checked = True
            self.scan_rate = earlier_scan_rate(self.output_dir.parent, self.output_dir)
        if not self.scan_rate:
            return status
        left = slots / self.scan_rate - since
        if left <= 0:
            return status + f"; earlier scans ran {self.scan_rate:.1f} slots/s, should end any moment"
        return (
            status + f"; earlier scans ran {self.scan_rate:.1f} slots/s -> "
            f"~{fmt_duration(left)} left ({clock_after(left)})"
        )

    def watched_exit_code(self) -> int | None:
        """Exit state of a job this process did not launch (--watch), from its files."""
        log_path = self.output_dir / "generation.log"
        log = read_text(log_path)
        if "Simulation ended." in log:
            return 0
        try:
            stale = time.time() - log_path.stat().st_mtime > 120
        except OSError:
            return None
        if stale and "Traceback (most recent call last)" in log:
            return 1
        return None

    def elapsed(self, now: float) -> str:
        start = self.task.get("start_time")
        if start is None:
            return "?"
        end = self.task.get("end_time")
        if end is None and self.task.get("proc") is None:
            try:  # a watched recording: the manifest is its last file
                end = (self.output_dir / "manifest.json").stat().st_mtime
            except OSError:
                end = now
        return fmt_duration((end or now) - start)

    def status(self, now: float) -> str:
        proc = self.task.get("proc")
        code = proc.poll() if proc is not None else self.watched_exit_code()
        if code is None:
            return self.running_status(now)
        if code == 0:
            return f"DONE in {self.elapsed(now)}: {finished_summary(self.output_dir)}"
        lines = [line for line in read_text(self.output_dir / "generation.log").splitlines() if line.strip()]
        return f"FAILED (exit {code}) after {self.elapsed(now)}: {lines[-1][:150] if lines else 'empty log'}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Record the Bochkivtsi reference layers in parallel.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="max",
        help="max (default): the testtopboch_max bench recipe; original: the first recordings",
    )
    parser.add_argument("--tag", default="", help="output/bochkivtsi_<tag>_<alt>m instead of the preset folders")
    parser.add_argument(
        "--heading-hold-deg",
        type=float,
        default=None,
        help="constant camera heading (survey gimbal); legs keep one orientation "
        "(preset max: 0; a negative value turns heading-hold off)",
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
        "(1 slot = 1 s at the default frame_step 30 / 30 fps); preset max: 500; "
        "0 = the simulator's default (60 slots)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite folders with a recording")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="only show the status of recordings already running or finished in the selected "
        "folders (same --preset/--tag/--only); launches nothing, Ctrl+C stops only the display",
    )
    parser.add_argument(
        "--status-every",
        type=float,
        default=30.0,
        metavar="S",
        help="seconds between status blocks (default 30)",
    )
    return parser.parse_args(argv)


def resolve_preset(args):
    """Fill heading-hold and anchor spacing from the preset unless given explicitly."""
    preset = PRESETS[args.preset]
    if args.heading_hold_deg is None:
        args.heading_hold_deg = preset["heading_hold_deg"]
    elif args.heading_hold_deg < 0:
        args.heading_hold_deg = None
    if args.anchor_spacing_m is None:
        args.anchor_spacing_m = preset["anchor_spacing_m"]
    elif args.anchor_spacing_m <= 0:
        args.anchor_spacing_m = None
    return preset


def build_tasks(args):
    preset = resolve_preset(args)
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
        layer = preset["layers"].get(base["altitude"], {})
        if layer:
            task["name"] = layer["name"]
            task["output_dir"] = ROOT_DIR / "output" / layer["name"]
        task["speed"] = speeds.get(task["altitude"], layer.get("speed", base["speed"]))
        task["zoom"] = zooms.get(task["altitude"], layer.get("zoom"))
        task["anchor_slots"] = (
            max(1, round(args.anchor_spacing_m / task["speed"]))
            if args.anchor_spacing_m is not None
            else None
        )
        if args.tag:
            task["name"] = f"bochkivtsi_{args.tag}_{task['altitude']}m"
            task["output_dir"] = ROOT_DIR / "output" / task["name"]
        tasks.append(task)
    return tasks


def first_telemetry_time(output_dir: Path) -> float | None:
    """Wall-clock time of the first telemetry row (the start of the flight)."""
    try:
        with open(output_dir / "telemetry.csv", encoding="utf-8", errors="replace") as stream:
            stream.readline()
            return float(stream.readline().split(",")[2])
    except (OSError, ValueError, IndexError):
        return None


FINAL_STATES = ("DONE", "FAILED", "no recording")


def watch(tasks: list[dict], args) -> None:
    """Status of recordings started elsewhere; never touches the jobs."""
    for task in tasks:
        task["proc"] = None
        task["start_time"] = first_telemetry_time(task["output_dir"])
    monitors = [JobMonitor(task) for task in tasks]
    width = max(len(task["name"]) for task in tasks)
    print(f"Watching {len(tasks)} recording folder(s); Ctrl+C stops only this display.")
    try:
        while True:
            now = time.time()
            statuses = [monitor.status(now) for monitor in monitors]
            print(f"[{datetime.now().strftime('%H:%M:%S')}]")
            for monitor, status in zip(monitors, statuses):
                print(f"  {monitor.task['name']:<{width}}  {status}")
            if all(status.startswith(FINAL_STATES) for status in statuses):
                return
            time.sleep(args.status_every)
    except KeyboardInterrupt:
        print("\nStopped watching; the recordings keep running.")


def main(argv=None):
    args = parse_args(argv)
    tasks = build_tasks(args)
    if args.watch:
        watch(tasks, args)
        return
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

    # Monitoring loop: a status block every --status-every seconds
    monitors = [JobMonitor(task) for task in tasks]
    width = max(len(task["name"]) for task in tasks)
    launch_time = time.time()
    last_print = 0.0
    all_done = False

    try:
        while not all_done:
            time.sleep(5)
            now = time.time()
            all_done = True
            lines = []
            for monitor in monitors:
                task = monitor.task
                ret = task["proc"].poll()
                if ret is None:
                    all_done = False
                elif not task["done"]:
                    task["done"] = True
                    task["exit_code"] = ret
                    task["end_time"] = now
                    stamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{stamp}] {task['name']}: {monitor.status(now)}")
                    if ret != 0:
                        tail = read_text(task["output_dir"] / "generation.log").splitlines()[-8:]
                        print("    " + "\n    ".join(tail))
                had_plan = monitor.plan is not None
                lines.append(f"  {task['name']:<{width}}  {monitor.status(now)}")
                if not had_plan and monitor.plan is not None:
                    plan = monitor.plan
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] {task['name']}: route "
                        f"{plan['km']:.1f} km, {plan['waypoints']} waypoints -> ~{plan['frames']} "
                        f"frames, {fmt_duration(plan['seconds'])} of video"
                    )

            if now - last_print >= args.status_every or all_done:
                last_print = now
                stamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{stamp}] {fmt_duration(now - launch_time)} since launch")
                print("\n".join(lines))

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
        print(f"  - {task['name']}: exit code {code}, {finished_summary(task['output_dir'])}")
        if code != 0:
            all_success = False

    print("=" * 80)
    if not all_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
