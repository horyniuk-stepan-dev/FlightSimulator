# Simulator ground-truth audit — 2026-08-01

Triggered by a real propagation run of the localization project on a recording
from this simulator (77 anchors, 571 DB slots, 1000 m altitude). The graph
converged, but its diagnostics were hostile: 18 of 76 anchor gaps flagged
inconsistent or broken, 60 anchors with stress above 2x the median, and
leave-one-out disagreement of 15-27 m between anchors one slot apart, rising to
207 m for distant ones. Ground-truth anchors should not disagree with each
other. Two simulator defects explain it; a third is a recording parameter.

## 1. Anchors were built on the flat Z=0 plane, the image was not — FIXED

`_render_gpu` displaces every sample towards the nadir point in proportion to
the terrain height under it (parallax displacement mapping), which is what
makes relief look like relief. `CalibrationLogger` meanwhile mapped its five
reference pixels through `H1 = P[:, [0, 1, 3]]` — the homography of the flat
Z=0 plane, with no elevation term. The two describe different images.

The error is zero at the nadir point and grows linearly with radius:

    error = r * h_rel / altitude

Measured against the cached DEM for the recorded area (`elevation_408ba699e784_z15.tif`,
relief 178 m, median 51 m above base) at 1000 m altitude and 0.5208 m/px:

| position in frame | r | median relief | p90 relief | max relief |
|---|---|---|---|---|
| centre | 0 m | 0.0 m | 0.0 m | 0.0 m |
| mid-frame | 191 m | 9.7 m | 19.6 m | 34.0 m |
| corner | 382 m | 19.4 m | 39.1 m | 68.0 m |

That is the same order as the observed anchor-to-anchor disagreement.

What made it hard to notice: all five reference points share the same wrong
assumption, so the anchor's own LSQ residual stays at ~6e-06 m. Every anchor in
`calibration.json` reports itself as perfect while being tens of metres off.
Elevation is on by default — `main.py` only skips `download_elevation` when an
explicit `--geotiff` is passed.

**Fix.** The displacement formula now lives once in
`simulator/terrain/parallax.py`. The renderer injects a CuPy sampler over the
full pixel grid; `CalibrationLogger._apply_terrain` injects a NumPy sampler over
its five points. A map with no DEM returns the inputs unchanged, so flat scenes
are bit-for-bit as before.

**Consequence to expect:** anchor `rmse_m` will no longer be ~1e-06. Over broken
terrain an affine cannot describe the true pixel-to-ground mapping, so the
residual becomes genuinely non-zero — that number is now information rather
than decoration, and it feeds the localizer's `soft_anchors` sigma.

## 2. CPU render path leaked the half-resolution projection matrix — FIXED

`_compute_homography` assigned `self.last_P` on every call. `_render_cpu` calls
it twice — full resolution, then half resolution for speed — so `last_P` ended
up holding the **half-res** matrix, after the explicit full-res assignment.
`CalibrationLogger` then combined that matrix with full-resolution pixel
coordinates: every anchor recorded on a machine without CuPy came out at half
scale, silently.

The GPU path assigns `last_P` once and was unaffected, which is why existing
recordings are fine — verified: the observed anchor scale of 0.5208 m/px matches
the full-resolution camera model exactly (8.8 mm sensor, 13.2 mm focal,
1280 px, 1000 m), not twice it.

**Fix.** The helper no longer touches `last_P`; each render path assigns it once
from the full-resolution matrix.

## 3. Inter-slot overlap is marginal at the recorded speed — NOT a code bug

Measured from `calibration.json`: 149.6 m of travel per DB slot against a frame
footprint of 667 x 375 m (1280 x 720 px at 0.5208 m/px).

* along the short axis: (375 - 149.6) / 375 = **60% overlap**
* along the long axis: (667 - 149.6) / 667 = 78% overlap

60% is enough for a match but not for a *well-spread* one: the shared region is
a strip, so inliers concentrate in it. That is visible in the localizer log as
`Spatial collapse: 380/555 temporal edges got reduced weight (inliers
clustered, ref=0.15)` — 68% of the temporal chain downweighted. It also makes
matching fail outright on turn arcs, which is where the flagged "chain broken"
gaps are.

Not a defect in the code — a consequence of flying ~100 m/s with
`frame_step=30` at 30 fps (one slot per second). Any of these fixes it:

1. lower the flight speed;
2. lower `frame_step` in both projects together (the localizer's
   `database.frame_step` must match — see the parent project's CLAUDE.md);
3. fly higher, which enlarges the footprint.

Option 2 is the cheapest test: at `frame_step=15` the overlap along the short
axis rises to 80%.

## Verification

`test_parallax_calibration.py` (new):

1. `apply_parallax` — zero relief is the identity; the nadir point is a fixed
   point; the displacement equals `frac * r` exactly.
2. `sample_bilinear` matches `scipy.ndimage.map_coordinates(order=1,
   mode="nearest")` to 1.1e-13, so the five-point CPU path and the GPU grid
   path compute the same thing.
3. `CalibrationLogger` — without a DEM the anchors are unchanged; with a
   constant-relief DEM the reference points shift by exactly the predicted
   amount (39.2 m at r = 220 m, relief 178 m, altitude 1000 m).

`test_calibration_logger.py` and `test_ground_truth_export.py` pass unchanged.
`ruff check` is clean on all touched files.

## Not done

Re-recording. Every existing `flight.mp4` / `calibration.json` pair was produced
with flat-plane anchors over elevated terrain, so their anchors carry the error
in the table above. They remain usable for smoke tests, but not as ground truth
for accuracy benchmarks.
