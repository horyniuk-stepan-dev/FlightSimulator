"""Layer-choice test flights for DroneLocalization, for any map and layer set.

Each scenario isolates one behaviour of the multi-layer localizer that the
single route3min flight mixes together (DroneLocalization project docs
claude/handoff-2026-11-14-layer-both-directions.md):

  layer_steps       long holds exactly at each layer altitude, up and down: the
                    layer in use must be the one at the drone's altitude.
  layer_boundaries  oscillation around every boundary between two layers
                    (geometric mean of their altitudes): hysteresis, no
                    back-and-forth switching.
  low_hold          long flight at and below the lowest layer: the track must
                    pass from frame to frame on the densest layer.
  sawtooth          fastest climb / descent through all layers, three times:
                    scale changes fast, the worst case for a frozen scale.
  between_layers    long holds at the boundary altitudes, farthest from every
                    layer: the altitudes where fixes are hardest.

Routes are zigzags of their own (different start side and leg count per
scenario), so the ground path is not the one any threshold was tuned on. All
flights are nadir with yaw 0: tilt and rotation are tested by route3min /
yawsweep. The scenario format has no camera zoom; a zoom change scales the
image like an altitude change, so `sawtooth` is the closest available test.

    python make_layer_scenarios.py                                # Bochkivtsi, 500/1000/2000 m
    python make_layer_scenarios.py --prefix steppe --layers 600,1200 ^
        --bbox 48.161373,32.842047,48.238627,32.957953 --wayback 2019-04-03

Writes scenarios/<prefix>_<name>.json and prints the record commands (current
imagery, and one Esri Wayback date per --wayback). Altitudes are above the
simulator's base plane (lowest DEM point of the map), as in every scenario: over
relief the height above ground is lower; the simulator stops a flight that would
come within --min-clearance of the terrain (use --altitude-offset then).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from make_edge_scenarios import FPS, REF_BBOX, Frame

# simulator/config.py camera: 8.8 mm sensor width, 13.2 mm focal length, 16:9
SENSOR_OVER_FOCAL = 8.8 / 13.2


def footprint(altitude_m: float) -> tuple[float, float]:
    width = altitude_m * SENSOR_OVER_FOCAL
    return width, width * 9.0 / 16.0


def nearest_layer(altitude: float, layers: list[float]) -> float:
    return min(layers, key=lambda a: abs(math.log(altitude / a)))


class Profile:
    """Altitude keyframes built from holds and constant-rate moves."""

    def __init__(self, start: float, rate: float, layers: list[float]) -> None:
        self.rate = float(rate)
        self.layers = layers
        self.t = 0.0
        self.alt = float(start)
        self.keys = [(0.0, float(start))]
        self.phases: list[dict] = []

    def hold(self, seconds: float, label: str) -> None:
        self._phase(seconds, label, nearest_layer(self.alt, self.layers))
        self.t += seconds
        self.keys.append((self.t, self.alt))

    def move(self, target: float, label: str = "") -> None:
        if abs(target - self.alt) < 1e-6:
            return
        seconds = max(1.0, round(abs(target - self.alt) / self.rate))
        self._phase(seconds, label or f"to_{target:.0f}", None)
        self.t += seconds
        self.alt = float(target)
        self.keys.append((self.t, self.alt))

    def _phase(self, seconds, label, expected) -> None:
        phase = {"from_s": self.t, "to_s": self.t + seconds, "label": label}
        if expected is not None:
            # layer altitude nearest in scale; the evaluator maps it to a layer id
            phase["expected_layer_altitude_m"] = expected
        self.phases.append(phase)


def profiles(layers: list[float], rate: float) -> dict[str, tuple[Profile, str]]:
    low, high = layers[0], layers[-1]
    mid = layers[len(layers) // 2]
    bounds = [math.sqrt(a * b) for a, b in zip(layers, layers[1:])]
    out: dict[str, tuple[Profile, str]] = {}

    p = Profile(mid, rate, layers)
    p.hold(15, f"hold_{mid:.0f}")
    for target in [a for a in layers if a > mid] + layers[::-1][1:] + [mid]:
        if target == p.alt:
            continue
        p.move(target)
        p.hold(20, f"hold_{target:.0f}")
    out["layer_steps"] = (p, "holds exactly at each layer altitude, up and down")

    p = Profile(bounds[0] * 0.85, rate, layers)
    p.hold(10, "start")
    for b in bounds:
        lo, hi = b * 0.85, b * 1.18
        p.move(lo)
        for _ in range(3):
            p.move(hi, f"up_through_{b:.0f}")
            p.hold(4, f"above_{b:.0f}")
            p.move(lo, f"down_through_{b:.0f}")
            p.hold(4, f"below_{b:.0f}")
    out["layer_boundaries"] = (p, "oscillation around every boundary between two layers")

    p = Profile(low, rate, layers)
    p.hold(60, f"hold_{low:.0f}")
    p.move(low * 0.8)
    p.hold(30, f"below_lowest_{low * 0.8:.0f}")
    p.move(low * 1.2)
    p.hold(30, f"above_lowest_{low * 1.2:.0f}")
    p.move(low)
    p.hold(20, f"hold_{low:.0f}_again")
    out["low_hold"] = (p, "long flight at and just around the lowest layer")

    p = Profile(high, rate, layers)
    p.hold(10, f"hold_{high:.0f}")
    for target in (low, high, low):
        p.move(target, f"fast_to_{target:.0f}")
        p.hold(5, f"hold_{target:.0f}")
    out["sawtooth"] = (p, "fastest climb / descent through all layers")

    p = Profile(bounds[0], rate, layers)
    for b in bounds + bounds[::-1][1:]:
        if b != p.alt:
            p.move(b)
        p.hold(40, f"hold_between_{b:.0f}")
    out["between_layers"] = (p, "long holds at the altitudes farthest from every layer")
    return out


def zigzag(half_x, half_y, margin, length_m, variant: int) -> list[tuple[float, float]]:
    """A zigzag across the usable map, at least ``length_m`` long."""
    x_max = half_x - margin
    y_max = (half_y - margin) * 0.85
    if x_max <= 0 or y_max <= 0:
        raise SystemExit(
            f"map too small for the camera footprint at the highest altitude "
            f"(margin {margin:.0f} m, half size {half_x:.0f} x {half_y:.0f} m)"
        )
    legs = 2
    while legs * math.hypot(2 * x_max / legs, 2 * y_max) < length_m and legs < 40:
        legs += 1
    x_sign = 1 if variant % 2 == 0 else -1  # start in the west or in the east
    y_sign = 1 if (variant // 2) % 2 == 0 else -1
    points = []
    for i in range(legs + 1):
        x = x_sign * (-x_max + 2 * x_max * i / legs)
        y = y_sign * y_max * (1 if i % 2 == 0 else -1)
        points.append((round(x, 1), round(y, 1)))
    return points


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prefix", default="testtopboch", help="scenario / output name prefix")
    ap.add_argument("--layers", default="500,1000,2000", help="layer altitudes, metres")
    ap.add_argument(
        "--bbox",
        default=",".join(repr(v) for v in REF_BBOX),
        help="lat_min,lon_min,lat_max,lon_max of the simulator map (default: Bochkivtsi)",
    )
    ap.add_argument("--speed", type=float, default=50.0)
    ap.add_argument("--rate", type=float, default=20.0, help="climb / descent, m/s")
    ap.add_argument("--zoom", type=int, default=18)
    ap.add_argument("--wayback", action="append", default=[], help="Esri Wayback date(s)")
    ap.add_argument("--wayback-zoom", type=int, default=17)
    ap.add_argument(
        "--edge-keepout",
        type=float,
        default=900.0,
        help="metres the route stays away from the map edge: the reference layers' "
        "coverage ends a few hundred metres inside it (Bochkivtsi: ~300-400 m)",
    )
    ap.add_argument("--out", default="scenarios")
    args = ap.parse_args()

    layers = sorted(float(v) for v in args.layers.split(","))
    if len(layers) < 2:
        raise SystemExit("--layers needs at least two altitudes")
    bbox = tuple(float(v) for v in args.bbox.split(","))
    half_x, half_y = Frame(bbox).half
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    commands = []
    for variant, (name, (profile, what)) in enumerate(profiles(layers, args.rate).items()):
        duration = int(math.ceil(profile.t))
        top = max(a for _, a in profile.keys)
        margin = max(math.hypot(*footprint(top)) / 2 + 30.0, args.edge_keepout)
        route = zigzag(half_x, half_y, margin, args.speed * duration * 1.03, variant)
        length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
        scenario = {
            "version": 1,
            "name": f"{args.prefix}_{name}",
            "description": (
                f"Layer-choice test ({what}), generated by make_layer_scenarios.py for "
                f"layers {[int(a) for a in layers]} m. Map bbox (lat_min, lon_min, lat_max, "
                f"lon_max) = {[round(v, 9) for v in bbox]}; route in local metres of that "
                f"map, {length / 1000:.1f} km at {args.speed:g} m/s; nadir, yaw 0; vertical "
                f"rate <= {args.rate:g} m/s. phases[].expected_layer_altitude_m is the layer "
                f"nearest in scale during a hold (extra keys are ignored by the loader)."
            ),
            "duration_s": float(duration),
            "layers_m": layers,
            "phases": profile.phases,
            "route": [{"x_m": x, "y_m": y} for x, y in route],
            "profiles": {
                "altitude_m": [{"time_s": t, "value": round(a, 1)} for t, a in profile.keys],
                "pitch_deg": [{"time_s": 0.0, "value": 0.0}],
                "roll_deg": [{"time_s": 0.0, "value": 0.0}],
                "yaw_deg": [{"time_s": 0.0, "value": 0.0}],
            },
        }
        path = out_dir / f"{args.prefix}_{name}.json"
        path.write_text(json.dumps(scenario, indent=1), encoding="utf-8")
        low, high = min(a for _, a in profile.keys), top
        print(f"wrote {path}: {duration} s, {low:.0f}-{high:.0f} m, route {length / 1000:.1f} km")
        for date, zoom in [("", args.zoom)] + [(d, args.wayback_zoom) for d in args.wayback]:
            out_name = f"{args.prefix}_{name}" + (f"_wayback_{date}" if date else "")
            o = f"output\\{out_name}"
            commands.append(
                (
                    o,
                    f"python -m simulator.main --mode record --scenario {path} "
                    f"--speed {args.speed:g} --zoom {zoom} "
                    f"--lat_min {bbox[0]!r} --lon_min {bbox[1]!r} "
                    f"--lat_max {bbox[2]!r} --lon_max {bbox[3]!r} "
                    + (f"--map-date {date} " if date else "")
                    + f"--max-frames {duration * FPS} --fast --no-display --yes "
                    f"--video-file {o}\\video.mp4 --frame-gt-file {o}\\video.frames.jsonl "
                    f"--gt-file {o}\\ground_truth.json --calib-file {o}\\calibration.json "
                    f"--telemetry-file {o}\\telemetry.csv --manifest-file {o}\\manifest.json",
                )
            )
    print("\nRecord (from the FlightSimulator folder, one at a time):")
    for o, cmd in commands:
        print(f"mkdir {o}\n{cmd}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
