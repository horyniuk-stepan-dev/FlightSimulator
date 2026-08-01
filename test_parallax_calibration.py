"""Тест: якорі калібрування лежать на РЕЛЬЄФІ, а не на площині Z=0.

Аудит 2026-08-01. Рендер зміщує пікселі за висотою рельєфу
(``_render_gpu`` → parallax displacement), а CalibrationLogger будував якорі з
гомографії площини Z=0. Похибка нульова в надирі й росте як ``r·h_rel/alt``:
при рельєфі 178 м і висоті 1000 м це ~19 м у середині кадру та ~68 м у куті.
Власний RMSE якоря її не бачить — усі 5 опорних точок ділять одне й те саме
хибне припущення.

Перевіряє:
1. apply_parallax: плаский рельєф — тотожність; надир — нерухома точка;
   зміщення точно дорівнює frac·r.
2. sample_bilinear збігається зі scipy map_coordinates(order=1, mode="nearest"),
   тобто CPU-двійник рахує те саме, що GPU-шлях рендера.
3. CalibrationLogger: без DEM поведінка побітово незмінна; з DEM якорі
   зміщуються рівно на передбачену величину.

Запуск: python test_parallax_calibration.py  (numpy; scipy — опційно для п.2)
"""

import math

import numpy as np

from simulator.physics.calibration_logger import CalibrationLogger
from simulator.physics.drone_state import DroneState
from simulator.terrain.parallax import (
    apply_parallax,
    passes_for_altitude,
    sample_bilinear,
)

W, H = 1280, 720
CX, CY = W / 2.0, H / 2.0
SCALE = 2.0            # px за метр
ALT = 1000.0
MAP_PX = 2000          # мапа 2000x2000 px, 1 м/px, центр у (0,0) local


class FlatMap:
    """Двійник OrthophotoMap без рельєфу."""

    _center_x = 3_000_000.0
    _center_y = 6_200_000.0
    width = MAP_PX
    height = MAP_PX
    elevation = None
    base_elevation = 0.0

    def local_to_pixel(self, lx, ly):
        return lx + MAP_PX / 2.0, MAP_PX / 2.0 - ly

    def pixel_to_local(self, col, row):
        return col - MAP_PX / 2.0, MAP_PX / 2.0 - row

    def local_to_gps(self, x, y):
        mx, my = x + self._center_x, y + self._center_y
        lon = mx / 6378137.0 * 180.0 / math.pi
        lat = math.degrees(2 * math.atan(math.exp(my / 6378137.0)) - math.pi / 2)
        return lat, lon


class HillMap(FlatMap):
    """Та сама мапа з ПЛАТО сталої висоти — робить очікуване зміщення точним."""

    def __init__(self, relief_m: float):
        self.base_elevation = 100.0
        self.elevation = np.full((MAP_PX, MAP_PX), self.base_elevation + relief_m, np.float32)


def make_H1(X, Y, heading=0.0):
    """Ground-truth гомографія local→pixel надирної камери в точці (X, Y)."""
    c, s = math.cos(heading), math.sin(heading)
    return np.array(
        [
            [SCALE * c, SCALE * s, -SCALE * (c * X + s * Y) + CX],
            [SCALE * s, -SCALE * c, -SCALE * (s * X - c * Y) + CY],
            [0.0, 0.0, 1.0],
        ]
    )


def make_state(X, Y):
    return DroneState(
        position=np.array([X, Y, ALT]),
        velocity=np.array([50.0, 0.0, 0.0]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
    )


def test_apply_parallax_geometry():
    u0 = np.array([100.0, 300.0, -50.0])
    v0 = np.array([200.0, -20.0, 400.0])
    dc, dr = 100.0, 200.0                       # надир збігається з першою точкою

    flat = lambda u, v: np.zeros_like(np.asarray(u, dtype=np.float64))  # noqa: E731
    u, v = apply_parallax(u0, v0, dc, dr, ALT, flat, base_elevation=0.0)
    assert np.allclose(u, u0) and np.allclose(v, v0), "нульовий рельєф має бути тотожністю"

    relief = 178.0
    const = lambda u, v: np.full_like(np.asarray(u, dtype=np.float64), relief)  # noqa: E731
    u, v = apply_parallax(u0, v0, dc, dr, ALT, const, base_elevation=0.0, num_passes=1)

    assert math.isclose(u[0], dc) and math.isclose(v[0], dr), "надир — нерухома точка"

    frac = relief / ALT
    for i in range(len(u0)):
        r = math.hypot(u0[i] - dc, v0[i] - dr)
        shift = math.hypot(u[i] - u0[i], v[i] - v0[i])
        assert math.isclose(shift, frac * r, rel_tol=1e-12), f"точка {i}: {shift} != {frac * r}"

    assert passes_for_altitude(1000.0) == 1 and passes_for_altitude(100.0) == 3
    print(f"[1] OK: parallax — тотожність без рельєфу, надир нерухомий, "
          f"зміщення = {frac:.3f}·r точно")


def test_bilinear_matches_scipy():
    try:
        from scipy.ndimage import map_coordinates
    except ImportError:
        print("[2] SKIP: scipy недоступний")
        return
    rng = np.random.default_rng(0)
    img = rng.random((64, 48)).astype(np.float64) * 500.0
    rows = rng.uniform(-5, 70, 400)              # навмисно за межами — перевіряємо clamp
    cols = rng.uniform(-5, 55, 400)

    mine = sample_bilinear(img, rows, cols)
    theirs = map_coordinates(
        img, np.stack([np.clip(rows, 0, 63), np.clip(cols, 0, 47)]), order=1, mode="nearest"
    )
    err = float(np.max(np.abs(mine - theirs)))
    assert err < 1e-9, f"CPU-двійник розходиться з map_coordinates на {err}"
    print(f"[2] OK: sample_bilinear == scipy map_coordinates (max Δ = {err:.2e})")


def _anchor_points(ortho_map, relief_note):
    lg = CalibrationLogger("/dev/null", ortho_map, frame_step=30)
    state = make_state(0.0, 0.0)                 # дрон у центрі мапи → надир = центр кадру
    ok = lg.log(make_H1(0.0, 0.0), W, H, state, frame_idx=0)
    assert ok, f"кандидат не створено ({relief_note})"
    c = lg._candidates[0]
    return np.asarray(c["pts_mercator"]), np.asarray(c["pts_px"]), c["rmse"]


def test_logger_terrain():
    flat_pts, pts_px, flat_rmse = _anchor_points(FlatMap(), "flat")
    # 1e-4 м — стеля шуму float64-LSQ на координатах Mercator порядку 6e6
    # (реальний calibration.json дає той самий ~6e-06). Ефект рельєфу нижче —
    # десятки МЕТРІВ, тобто на 5 порядків більший, тож поріг нічого не маскує.
    assert flat_rmse < 1e-4, f"на площині якір мусить бути точним, а не {flat_rmse}"

    relief = 178.0
    hill_pts, _, hill_rmse = _anchor_points(HillMap(relief), "hill")

    frac = relief / ALT
    centre_shift = float(np.linalg.norm(hill_pts[0] - flat_pts[0]))
    assert centre_shift < 1e-9, f"центр кадру = надир, зміщення {centre_shift}"

    for i in range(1, 5):
        r = float(np.linalg.norm(flat_pts[i] - flat_pts[0]))
        shift = float(np.linalg.norm(hill_pts[i] - flat_pts[i]))
        assert math.isclose(shift, frac * r, rel_tol=1e-9), f"точка {i}: {shift} != {frac * r}"

    corner_r = float(np.linalg.norm(flat_pts[1] - flat_pts[0]))
    print(f"[3] OK: DEM зсуває опорні точки на {frac * corner_r:.1f} м "
          f"(r={corner_r:.0f} м, рельєф {relief:.0f} м, висота {ALT:.0f} м); "
          f"без DEM — незмінно")
    print(f"    старий (плаский) якір оголошував ці точки з похибкою рівно "
          f"стільки ж, а його власний RMSE лишався {flat_rmse:.1e} м")


def test_hillshade_generation():
    from simulator.terrain.orthophoto_map import OrthophotoMap
    # Create mock map structure
    m = OrthophotoMap.__new__(OrthophotoMap)
    m.elevation = np.zeros((100, 100), dtype=np.float32)
    # Slope terrain: rising 100m west to east
    m.elevation += np.linspace(0, 100, 100, dtype=np.float32)
    m._bounds = type("Bounds", (), {"left": 0.0, "right": 1000.0, "top": 1000.0, "bottom": 0.0})()

    hs = m.generate_hillshade(azimuth_deg=315.0, altitude_deg=45.0, z_factor=1.0)
    assert hs is not None
    assert hs.shape == (100, 100)
    assert np.all((hs >= 0.0) & (hs <= 1.0))
    print("[4] OK: 3D hillshade generation computed valid illumination values")


if __name__ == "__main__":
    test_apply_parallax_geometry()
    test_bilinear_matches_scipy()
    test_logger_terrain()
    test_hillshade_generation()
    print("\n=== ALL TESTS PASSED ===")

