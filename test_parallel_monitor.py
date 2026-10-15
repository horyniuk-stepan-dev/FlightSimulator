"""Progress lines of run_parallel_bochkivtsi.py and the simulator lines they read."""

import json
import os
import sys

import pytest

import run_parallel_bochkivtsi as rpb
from simulator import main as simulator_main
from simulator.planning.survey_planner import Waypoint


class _Proc:
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


def _monitor(tmp_path, code=None, start=0.0):
    task = {"name": "layer", "output_dir": tmp_path, "proc": _Proc(code), "start_time": start}
    return rpb.JobMonitor(task)


def _write_log(tmp_path, *lines):
    (tmp_path / "generation.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_telemetry(tmp_path, frame):
    header = "frame_index,timestamp,sys_time,sim_time,pos_x,pos_y,alt_z,vel_x,vel_y,vel_z,yaw_rad,speed_3d"
    row = f"{frame},{frame / 30:.9f},0,{frame / 30:.3f},0,0,500,0,25,0,0,25"
    (tmp_path / "telemetry.csv").write_text(f"{header}\n0,0,0,0,0,0,500,0,0,0,0,0\n{row}\n", encoding="utf-8")


def test_planned_route_line_is_what_the_monitor_parses():
    waypoints = [Waypoint(0, 0, 500), Waypoint(0, 3000, 500), Waypoint(100, 3000, 500), Waypoint(100, 0, 500)]
    line = simulator_main.planned_route_line(waypoints, 25.0, 30)

    assert line == "Planned route: 4 waypoints, 6.1 km, 244 s of flight, ~7320 video frames"
    match = rpb.PLAN_RE.search(line)
    assert (match["waypoints"], match["seconds"], match["frames"]) == ("4", "244", "7320")


def test_planned_route_of_a_scenario_and_frame_cap():
    waypoints = [Waypoint(0, 0, 500), Waypoint(0, 100, 500)]

    assert "180 s of flight, ~5400 video frames" in simulator_main.planned_route_line(
        waypoints, 25.0, 30, duration_s=180.0
    )
    assert "~900 video frames" in simulator_main.planned_route_line(waypoints, 1.0, 30, max_frames=900)


def test_selector_progress_is_echoed_and_json_returned(tmp_path, capsys):
    script = (
        "import sys\n"
        "print('model noise', file=sys.stderr)\n"
        "print('Keyframe scan: 40%', file=sys.stderr)\n"
        "print('{\"ok\": 1}')\n"
    )
    result = simulator_main._run_selector_process([sys.executable, "-c", script], tmp_path, None)

    assert json.loads(result.stdout) == {"ok": 1}
    assert capsys.readouterr().out == "Keyframe scan: 40%\n"


def test_selector_failure_keeps_the_error_tail(tmp_path):
    script = "import sys\nprint('boom: no model', file=sys.stderr)\nsys.exit(3)\n"

    with pytest.raises(RuntimeError, match="exit 3.*boom: no model"):
        simulator_main._run_selector_process([sys.executable, "-c", script], tmp_path, None)


def test_preparing_before_the_flight(tmp_path):
    _write_log(tmp_path, "Downloading tiles...")

    assert _monitor(tmp_path).status(65.0) == "preparing map and route (1:05)"


def test_flight_progress_against_the_plan_with_rate_and_eta(tmp_path):
    _write_log(
        tmp_path,
        "Planned route: 4 waypoints, 6.1 km, 300 s of flight, ~9000 video frames",
        "Starting simulation loop (MAX SPEED; video written @ 30 FPS)...",
    )
    monitor = _monitor(tmp_path)
    _write_telemetry(tmp_path, 3000)
    first = monitor.status(100.0)
    _write_telemetry(tmp_path, 3600)
    second = monitor.status(120.0)

    assert first == "flying 3000/~9000 frames (33.3%), video 1:40 of ~5:00"
    assert second.startswith("flying 3600/~9000 frames (40.0%), video 2:00 of ~5:00, 30.0 fps, ETA ~3:00 (")


def test_flight_of_an_older_simulator_has_no_total(tmp_path):
    _write_log(tmp_path, "Starting simulation loop (MAX SPEED; video written @ 30 FPS)...")
    _write_telemetry(tmp_path, 250)

    assert _monitor(tmp_path).status(10.0) == "flying 250 frames, video 0:08"


def test_flight_finished_before_the_video_is_closed(tmp_path):
    _write_log(tmp_path, "Starting simulation loop", "[FrameGroundTruthLogger] Saved 42047 frame records to x")

    assert _monitor(tmp_path).status(10.0) == "flight finished, closing the video"


def test_keyframe_scan_and_calibration_phases(tmp_path):
    lines = [
        "Starting simulation loop (MAX SPEED; video written @ 30 FPS)...",
        "[VideoWriterSink] Video saved to video.mp4",
        "Keyframe scan: 0%",
        "Keyframe scan: 40%",
    ]
    _write_log(tmp_path, *lines)
    monitor = _monitor(tmp_path)
    monitor.status(1000.0)
    scanning = monitor.status(1400.0)
    _write_log(tmp_path, *lines, "Image selector kept 672/1402 slots (672 with local features).")

    assert scanning.startswith("keyframe scan 40% (6:40, ~10:00 left (")
    assert monitor.status(1500.0) == "anchors, ground truth, manifest (672/1402 keyframes; 8:20 since the flight)"


def test_finished_and_failed_jobs(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"frame_count": 42047, "status": "complete"}), encoding="utf-8")
    (tmp_path / "video.keyframes.json").write_text(
        json.dumps({"selected_slots": [0, 5, 9], "total_slots": 12}), encoding="utf-8"
    )
    (tmp_path / "calibration.json").write_text(json.dumps({"anchors": [{}, {}]}), encoding="utf-8")
    done = _monitor(tmp_path, code=0)
    done.task["end_time"] = 3725.0
    failed = _monitor(tmp_path, code=1)
    failed.task["end_time"] = 30.0
    _write_log(tmp_path, "Starting simulation loop", "TerrainClearanceError: too low", "")

    assert done.status(4000.0) == "DONE in 1:02:05: 42047 frames, status complete, 3/12 keyframes, 2 anchors"
    assert failed.status(40.0) == "FAILED (exit 1) after 0:30: TerrainClearanceError: too low"


def test_status_every_option():
    assert rpb.parse_args([]).status_every == 30.0
    assert rpb.parse_args(["--status-every", "60"]).status_every == 60.0


def test_launcher_loop_prints_blocks(tmp_path, monkeypatch, capsys):
    """The monitoring loop runs end to end with a fake simulator process."""
    out = tmp_path / "bochkivtsi_max_2000m"
    procs = []

    def fake_popen(cmd, cwd, env, stdout, stderr):
        out.mkdir(exist_ok=True)
        (out / "manifest.json").write_text(json.dumps({"frame_count": 10, "status": "complete"}), encoding="utf-8")
        proc = _Proc(code=0)
        procs.append(cmd)
        return proc

    monkeypatch.setattr(rpb, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(rpb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(rpb.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sys, "argv", ["run_parallel_bochkivtsi.py", "--only", "2000"])
    tasks = rpb.build_tasks(rpb.parse_args(["--only", "2000"]))
    for task in tasks:
        task["output_dir"] = out
    monkeypatch.setattr(rpb, "build_tasks", lambda _args: tasks)

    rpb.main(["--only", "2000"])

    printed = capsys.readouterr().out
    assert "bochkivtsi_max_2000m: DONE in" in printed
    assert "10 frames, status complete" in printed
    assert len(procs) == 1


def _old_recording(root, name, slots, scan_seconds):
    folder = root / name
    folder.mkdir()
    video = folder / "video.mp4"
    video.write_bytes(b"")
    keyframes = folder / "video.keyframes.json"
    keyframes.write_text(json.dumps({"selected_slots": [0], "total_slots": slots}), encoding="utf-8")
    os.utime(video, (1000.0, 1000.0))
    os.utime(keyframes, (1000.0 + scan_seconds, 1000.0 + scan_seconds))


def test_scan_without_progress_lines_is_timed_from_earlier_scans(tmp_path):
    _old_recording(tmp_path, "max25_500m", 8666, 597)
    _old_recording(tmp_path, "max_500m", 4344, 309)
    _old_recording(tmp_path, "max_2000m", 342, 49)  # short scan: model loading dominates, ignored
    current = tmp_path / "relief_500m"
    current.mkdir()
    _write_log(
        current,
        "Starting simulation loop",
        "[FrameGroundTruthLogger] Saved 525288 frame records to video.frames.jsonl",
        "[VideoWriterSink] Video saved to video.mp4",
    )
    (current / "video.mp4").write_bytes(b"")
    os.utime(current / "video.mp4", (5000.0, 5000.0))

    status = _monitor(current).status(5180.0)

    # 17510 slots at the median 14.3 slots/s of the two large scans: 1224 s, 180 s gone
    assert status.startswith("keyframe scan of 17510 slots running 3:00; earlier scans ran 14.3 slots/s -> ~17:2")


def test_watch_reads_state_from_the_files(tmp_path, capsys):
    done = tmp_path / "bochkivtsi_relief_2000m"
    done.mkdir()
    _write_log(done, "Starting simulation loop", "Simulation ended.")
    (done / "telemetry.csv").write_text("frame_index,timestamp,sys_time\n0,0,1000.0\n", encoding="utf-8")
    (done / "manifest.json").write_text(json.dumps({"frame_count": 11622, "status": "complete"}), encoding="utf-8")
    os.utime(done / "manifest.json", (1000.0 + 1300, 1000.0 + 1300))
    tasks = [
        {"name": done.name, "output_dir": done},
        {"name": "bochkivtsi_relief_500m", "output_dir": tmp_path / "missing"},
    ]

    rpb.watch(tasks, rpb.parse_args(["--watch"]))

    printed = capsys.readouterr().out
    assert "bochkivtsi_relief_2000m  DONE in 21:40: 11622 frames, status complete" in printed
    assert "bochkivtsi_relief_500m   no recording in this folder" in printed
