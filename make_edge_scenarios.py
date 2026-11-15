"""Edge-of-map test routes for DroneLocalization (vision-only plan 0.4).

The reference layers of testtopboch_max were surveyed over the default
Bochkivtsi map, so their coverage ends near that map's edge. The scenario route
check refuses points whose footprint would leave the *simulator* map, so these
query flights are recorded on the same map enlarged to the east and north;
routes are designed in reference-map metres and converted to the enlarged map.

Writes scenarios/edge_{exit_return,along_border,start_outside,control}.json and
prints the record commands (same imagery + one Esri Wayback variant).

    python make_edge_scenarios.py [--east-m 3000] [--north-m 1500] [--zoom 18]
        [--speed 50] [--altitude 1000] [--wayback 2019-04-03]

Route coordinates are relative to the bounding-box centre; the renderer centres
on its tile mosaic, which may shift the actual route by up to a tile
(inference: <= ~150 m at z18). The localizer scores against simulator ground
truth, so the shift does not affect the evaluation, only the exact geometry.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

R = 6378137.0
# simulator/main.py defaults = the map the reference layers were surveyed over
REF_BBOX = (48.39950223106055, 26.102003186115134, 48.44349548338706, 26.257266910369705)
FPS = 30


def merc(lat, lon):
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def inv_merc(x, y):
    return math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2), math.degrees(x / R)


class Frame:
    """Local metres (x east, y north) around a bbox centre, Mercator scale at centre."""

    def __init__(self, bbox):
        lat_min, lon_min, lat_max, lon_max = bbox
        x0, y0 = merc(lat_min, lon_min)
        x1, y1 = merc(lat_max, lon_max)
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.gs = math.cos(math.radians(inv_merc(self.cx, self.cy)[0]))
        self.half = ((x1 - x0) / 2 * self.gs, (y1 - y0) / 2 * self.gs)

    def to_ll(self, x, y):
        return inv_merc(self.cx + x / self.gs, self.cy + y / self.gs)

    def from_ll(self, lat, lon):
        mx, my = merc(lat, lon)
        return (mx - self.cx) * self.gs, (my - self.cy) * self.gs


def enlarged_bbox(east_m, north_m):
    lat_min, lon_min, lat_max, lon_max = REF_BBOX
    ref = Frame(REF_BBOX)
    hx, hy = ref.half
    lat_n, lon_e = ref.to_ll(hx + east_m, hy + north_m)
    return (lat_min, lon_min, lat_n, lon_e)


def routes(hx, hy):
    """Routes in reference-map metres; hx, hy = half size of the reference map."""
    e = hx  # east edge of the reference map
    n = hy  # north edge
    return {
        # (a) straight exit east, ~60 s outside, return
        "edge_exit_return": [
            (e - 2700, -500),
            (e + 1300, -500),
            (e + 1300, 500),
            (e - 3200, 500),
        ],
        # (b) along the north border, footprint half in / half out. The
        # reference coverage ends ~300 m inside the map edge (first edge
        # replay 2026-11-10: 404 m to the border at y = n - 700).
        "edge_along_border": [
            (-4500, n - 900),
            (-3500, n - 300),
            (4500, n - 300),
            (4500, n - 900),
        ],
        # (c) start outside (north-east) and enter
        "edge_start_outside": [
            (e + 2300, n + 600),
            (e - 2700, 0),
            (0, 0),
        ],
        # (d) control: stays inside
        "edge_control": [
            (-3000, -1000),
            (3000, -1000),
            (3000, 1000),
            (-3000, 1000),
        ],
    }


DESCRIPTIONS = {
    "edge_exit_return": "straight exit across the east border of the reference coverage, "
    "~60 s outside, return",
    "edge_along_border": "flight along the north border, camera footprint half inside / half outside",
    "edge_start_outside": "cold start outside the reference coverage (north-east), then enter",
    "edge_control": "control loop that stays inside the reference coverage (no alert expected)",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--east-m", type=float, default=3000.0)
    ap.add_argument("--north-m", type=float, default=1500.0)
    ap.add_argument("--zoom", type=int, default=18)
    ap.add_argument("--speed", type=float, default=50.0)
    ap.add_argument("--altitude", type=float, default=1000.0)
    ap.add_argument("--wayback", default="2019-04-03", help="map date of the wayback variant")
    ap.add_argument("--out-dir", default="scenarios")
    args = ap.parse_args()

    ref = Frame(REF_BBOX)
    bbox = enlarged_bbox(args.east_m, args.north_m)
    query = Frame(bbox)
    hx, hy = ref.half
    # same margin rule as FlightScenario.route_waypoints (nadir, constant altitude)
    fw = args.altitude * 8.8 / 13.2
    fh = fw * 720 / 1280
    margin = math.hypot(fw, fh) / 2
    qhx, qhy = query.half
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for name, pts_ref in routes(hx, hy).items():
        pts = [query.from_ll(*ref.to_ll(x, y)) for x, y in pts_ref]
        for x, y in pts:
            if not (-qhx + margin <= x <= qhx - margin and -qhy + margin <= y <= qhy - margin):
                raise SystemExit(
                    f"{name}: point ({x:.0f}, {y:.0f}) too close to the enlarged map edge; "
                    "increase --east-m / --north-m"
                )
        length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        duration = math.ceil(length / args.speed) + 10
        scenario = {
            "version": 1,
            "name": name,
            "description": (
                f"{DESCRIPTIONS[name]}. Vision-only plan 0.4 edge test, generated by "
                f"make_edge_scenarios.py. Map bbox (lat_min, lon_min, lat_max, lon_max) = "
                f"{[round(v, 9) for v in bbox]} (default Bochkivtsi map + {args.east_m:.0f} m "
                f"east, + {args.north_m:.0f} m north); route in local metres of that map; "
                f"{args.altitude:.0f} m, nadir, yaw 0, {length / 1000:.1f} km at "
                f"{args.speed:.0f} m/s."
            ),
            "duration_s": float(duration),
            "reference_route_m": [{"x_m": round(x, 1), "y_m": round(y, 1)} for x, y in pts_ref],
            "route": [{"x_m": round(x, 1), "y_m": round(y, 1)} for x, y in pts],
            "profiles": {
                "altitude_m": [{"time_s": 0.0, "value": args.altitude}],
                "pitch_deg": [{"time_s": 0.0, "value": 0.0}],
                "roll_deg": [{"time_s": 0.0, "value": 0.0}],
                "yaw_deg": [{"time_s": 0.0, "value": 0.0}],
            },
        }
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(scenario, indent=1), encoding="utf-8")
        print(f"wrote {path}: {length / 1000:.1f} km, {duration} s")
        variants = [(name, args.zoom, "")]
        if name == "edge_exit_return" and args.wayback:
            variants.append((f"{name}_wayback_{args.wayback}", 17, args.wayback))
        for out_name, zoom, date in variants:
            o = f"output\\{out_name}"
            cmd = (
                f"python -m simulator.main --mode record --scenario {path} "
                f"--speed {args.speed:g} --zoom {zoom} "
                f"--lat_min {bbox[0]!r} --lon_min {bbox[1]!r} "
                f"--lat_max {bbox[2]!r} --lon_max {bbox[3]!r} "
                + (f"--map-date {date} " if date else "")
                + f"--max-frames {duration * FPS} --fast --no-display --yes "
                f"--video-file {o}\\video.mp4 --frame-gt-file {o}\\video.frames.jsonl "
                f"--gt-file {o}\\ground_truth.json --calib-file {o}\\calibration.json "
                f"--telemetry-file {o}\\telemetry.csv --manifest-file {o}\\manifest.json"
            )
            commands.append((o, cmd))
    print("\nRecord (from the FlightSimulator folder, one at a time):")
    for o, cmd in commands:
        print(f"mkdir {o}\n{cmd}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
