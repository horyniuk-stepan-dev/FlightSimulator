"""
FPV drone builder CLI.

  python -m simulator.fpv.cli --list
  python -m simulator.fpv.cli --preset freestyle_5in_6s
  python -m simulator.fpv.cli --preset freestyle_5in_6s --battery 4s_1500 --prop 5x3_tri
  python -m simulator.fpv.cli --compare freestyle_5in_6s race_5in_6s longrange_7in_6s
  python -m simulator.fpv.cli --preset longrange_7in_6s --payload 300 --save my_build.json
  python -m simulator.fpv.cli --load my_build.json --json
"""

from __future__ import annotations

import argparse
import json
import math

from .components import CATALOG, PRESETS, build, build_preset
from .report import full_report

ROWS = [
    ("mass_g", "Mass, g", "{:.0f}"),
    ("thrust_to_weight", "Thrust/weight", "{:.1f}"),
    ("max_thrust_g", "Max thrust, g", "{:.0f}"),
    ("full_throttle_rpm", "Full-throttle RPM", "{:.0f}"),
    ("full_throttle_current_a", "Full-throttle current, A", "{:.0f}"),
    ("battery_c_limit_a", "Battery C-limit, A", "{:.0f}"),
    ("full_throttle_sag_v", "Static sag at full, V", "{:.1f}"),
    ("hover_throttle_pct", "Hover throttle, %", "{:.0f}"),
    ("hover_current_a", "Hover current, A", "{:.1f}"),
    ("hover_time_min", "Hover-only time (80%), min", "{:.1f}"),
    ("top_speed_kmh_est", "Top speed est., km/h", "{:.0f}"),
    ("roll_rise_ms", "Roll 0→670°/s rise, ms", "{:.0f}"),
    ("roll_overshoot_pct", "Roll overshoot, %", "{:.0f}"),
    ("punch_climb_m_1s", "Punch-out climb in 1 s, m", "{:.1f}"),
    ("punch_min_vbat", "Punch-out min Vbat, V", "{:.1f}"),
    ("angle_fwd_speed_kmh_3s", "Angle mode fwd speed @3 s, km/h", "{:.0f}"),
]


def _fmt(v, f):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f.format(v)


def print_table(names, reports):
    w0 = max(len(r[1]) for r in ROWS) + 2
    w = max(14, *(len(n) + 2 for n in names))
    print("".ljust(w0) + "".join(n.rjust(w) for n in names))
    for key, label, f in ROWS:
        vals = [{**r["static"], **r["dynamic"]}.get(key) for r in reports]
        print(label.ljust(w0) + "".join(_fmt(v, f).rjust(w) for v in vals))
    for n, r in zip(names, reports):
        for wmsg in r["warnings"]:
            print(f"! {n}: {wmsg}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="FPV drone builder")
    ap.add_argument("--list", action="store_true", help="list catalog parts and presets")
    ap.add_argument("--preset", choices=list(PRESETS))
    for slot in CATALOG:
        ap.add_argument(f"--{slot}", choices=list(CATALOG[slot]))
    ap.add_argument("--payload", type=float, default=0.0, help="extra payload, g")
    ap.add_argument("--compare", nargs="+", metavar="PRESET")
    ap.add_argument("--save", help="save the build parts as JSON")
    ap.add_argument("--load", help="load a build JSON saved with --save")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.list:
        for slot, items in CATALOG.items():
            print(f"[{slot}]")
            for k, v in items.items():
                print(f"  {k:18s} {v.name}")
        print("[presets]")
        for k, v in PRESETS.items():
            print(f"  {k:18s} {v}")
        return

    if a.compare:
        names = a.compare
        reports = [full_report(build_preset(n)) for n in names]
        print_table(names, reports)
        return

    if a.load:
        with open(a.load, encoding="utf-8") as f:
            parts = json.load(f)
    else:
        parts = dict(PRESETS[a.preset or "freestyle_5in_6s"])
        parts["payload_g"] = a.payload
    for slot in CATALOG:
        if getattr(a, slot):
            parts[slot] = getattr(a, slot)
    if a.payload and a.load:
        parts["payload_g"] = a.payload
    b = build(**parts)
    if a.save:
        with open(a.save, "w", encoding="utf-8") as f:
            json.dump(parts, f, indent=2)
    rep = full_report(b)
    if a.json:
        rep["static"]["inertia_kgm2"] = list(rep["static"]["inertia_kgm2"])
        print(json.dumps({"parts": parts, **rep}, indent=2))
    else:
        print(" + ".join(f"{getattr(b, s).name}" for s in CATALOG))
        print_table(["build"], [rep])


if __name__ == "__main__":
    main()
