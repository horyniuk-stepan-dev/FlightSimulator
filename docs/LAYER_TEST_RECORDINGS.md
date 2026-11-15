# Layer-choice test recordings (for any map)

How to record query flights that test whether DroneLocalization uses the layer
that matches the drone's altitude, and how to score them. Written 2026-11-14;
status of the localizer side: DroneLocalization project docs
`claude/handoff-2026-11-14-layer-both-directions.md`.

## Why separate scenarios

Everything tuned on 2026-11-13/14 was measured on one flight (`route3min`) in
nine imagery variants over one area. These scenarios isolate single behaviours,
fly other ground paths, and can be generated for any other map, so thresholds
are checked on data they were not chosen on.

| Scenario | Flight | What it shows |
|---|---|---|
| `layer_steps` | holds exactly at every layer altitude, up and down (245 s for 500/1000/2000 m) | the layer in use is the one at the drone's altitude, in both directions |
| `layer_boundaries` | oscillation around each boundary between two layers (707 m and 1414 m), three times each (298 s) | hysteresis; back-and-forth switching between two layers |
| `low_hold` | 60 s at the lowest layer, then 20 % below and 20 % above it (160 s) | the track passes from frame to frame on the densest layer (the main <-> 500 m flapping of 2026-11-14) |
| `sawtooth` | highest -> lowest -> highest -> lowest at the full vertical rate (250 s) | fast scale change: the lost stretch of the 2000 -> 1000 m descent; closest available stand-in for a camera zoom |
| `between_layers` | 40 s holds at the boundary altitudes (190 s) | the altitudes farthest from every layer, where fixes are weakest |

All are nadir, yaw 0, 50 m/s, vertical rate <= 20 m/s. Tilt and rotation stay
with `route3min`, `route3min_heading` and `route3min_yawsweep`. The scenario
format has no focal-length profile, so a real zoom cannot be recorded yet.

## 1. Generate the scenarios

From the FlightSimulator folder:

```powershell
.\.venv\Scripts\python.exe make_layer_scenarios.py                 # Bochkivtsi map, layers 500/1000/2000 m
```

writes `scenarios/testtopboch_<scenario>.json` and prints the record commands.
Each file carries `phases[]` with `expected_layer_altitude_m` for the holds.

Another map / layer set:

```powershell
.\.venv\Scripts\python.exe make_layer_scenarios.py --prefix steppe --layers 600,1200 `
    --bbox 48.161373,32.842047,48.238627,32.957953
```

- `--bbox` is the simulator map (lat_min, lon_min, lat_max, lon_max) the
  reference layers were surveyed over.
- `--edge-keepout` (900 m) keeps the route away from the map edge: reference
  coverage ends a few hundred metres inside it (Bochkivtsi: ~300-400 m).
- Altitudes are above the map's base plane (lowest DEM point). Over relief the
  height above ground is lower; if the simulator stops with a clearance error,
  record with the `--altitude-offset` it suggests.
- The route is a zigzag across the usable map; the script stops if the map is
  too small for the camera footprint at the highest altitude.

## 2. Record

```powershell
powershell -ExecutionPolicy Bypass -File record_layer_tests.ps1                              # current imagery
powershell -ExecutionPolicy Bypass -File record_layer_tests.ps1 -Wayback "2014-02-20,2018-01-18"
powershell -ExecutionPolicy Bypass -File record_layer_tests.ps1 -Prefix steppe `
    -Bbox "48.161373,32.842047,48.238627,32.957953" -Wayback "2019-04-03"
```

One recording after another (the renderer keeps the whole map on the GPU).
Output `output\<prefix>_<scenario>[_wayback_<date>]\` with `video.mp4`,
`video.frames.jsonl` (per-frame ground truth incl. height above ground),
`telemetry.csv`, `manifest.json`. A folder with `manifest.json` is skipped.
Old imagery uses zoom 17, as the existing wayback recordings do.

Check each `manifest.json`: `status` must be `complete`, `validation_errors`
empty.

## 3. A new area from scratch

1. Pick a bbox away from the areas a VLAD vocabulary was built from
   (`run_vocab_regions.py` lists them); ~8-12 km on the long side.
2. Record the reference layers at the altitudes you want as layers — recipe of
   the best maps so far: `run_parallel_bochkivtsi.py` (heading-hold 0, a ground
   truth anchor every 500 m, speeds 25 / 100 / 150 m/s for 500 / 1000 / 2000 m,
   zoom 18 for the two lower layers). For another bbox pass the same
   `--lat_min/--lon_min/--lat_max/--lon_max` to `simulator.main` per layer.
3. Build the project in DroneLocalization:
   `python scripts/build_multilayer_project.py --project "D:/My Projects/TEST/<name>" --layer main 1000 <rec> --layer 2 2000 <rec> --layer 3 500 <rec>`
   and read the map-error report it prints (a layer with a bad map makes every
   test on it meaningless).
4. Generate and record the layer tests with that bbox and those layer altitudes
   (steps 1-2), on current imagery and at least one Wayback date.

Imagery of another provider (not another date) is not available: it needs a
second provider in `simulator/terrain/tile_loader.py`.

## 4. Replay and score (DroneLocalization folder)

```powershell
powershell -ExecutionPolicy Bypass -File scratch\testtopboch_analysis\lost_layer_ab.ps1 `
    -Prefix "testtopboch_" -Suffix "_layers" -Variants "current" -Baseline "current_layers" `
    -Routes "layer_steps,layer_boundaries,low_hold,sawtooth,between_layers,layer_steps_wayback_2014-02-20,sawtooth_wayback_2014-02-20,low_hold_wayback_2018-01-18"
```

- `-Variants "current"` replays with `user_config.json` as it is; other variants
  are config overrides defined at the top of the script.
- `-Maps` selects the project (default `D:\My Projects\TEST\testtopboch_max_dino336`).
- Reports go to `<project>\reports\<variant><Suffix>\`.

Two tables are printed:

- `lost_layer_table.py`: confirmed rows, dead-reckoning rows, keyframe fixes /
  attempts, fixes > 25 m, p95, longest gap, fixes per layer.
- `scripts\layer_match_report.py <reports dir> <tag> ... --layers 3=500,main=1000,2=2000`:
  share of fixes on the layer nearest to the true height above ground
  (`nearest`), share on it or inside the switch hysteresis (`ok`), the longest
  stretch on a wrong layer, layer switches and flaps (A -> B -> A within 5 s).

Reference numbers on `route3min` (vision runs, `ok` share): before the layer
switch 52-66 % on current imagery, with both directions enabled 87-90 %; flaps
0 -> 6 on `base`.

## What to look for

| Scenario | Pass looks like |
|---|---|
| `layer_steps` | `ok` near 100 % during holds; one switch per altitude change |
| `layer_boundaries` | switches about one per crossing (12 crossings), flaps 0 |
| `low_hold` | fixes on the lowest layer through the 60 s hold, no return to the middle layer |
| `sawtooth` | no gap longer than a few seconds, no fix > 25 m |
| `between_layers` | fixes continue; either neighbouring layer is acceptable |
