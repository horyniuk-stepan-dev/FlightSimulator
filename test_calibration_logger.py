"""
Тест синхронізації CalibrationLogger ↔ DatabaseBuilder (система локалізації).

Перевіряє:
1. Семплінг: кандидати якорів лягають РІВНО на кадри, які DatabaseBuilder
   кладе у слоти БД (range(0, total_frames, frame_step), слот = кадр // step).
2. Відбір якорів: якорі ставляться на межах змін напрямку руху (розворотах),
   на першому/останньому слоті, а прямі ділянки добиваються fill-якорями.
3. Формат calibration.json = v2.4 системи локалізації (det<0, 5 точок, QA,
   блок keyframe_selection зі стратегією відбору кадрів).
5. Якорі лежать РІВНО на слотах, які локалізатор залишить keyframe-ами
   (criterion=overlap) — інакше пропагація снепить їх сама або падає на
   колізії двох якорів в одному слоті.
4. Точність: афінна матриця якоря відтворює ground-truth позицію центру кадру.

Запуск: python test_calibration_logger.py  (потрібен лише numpy)
"""
import json
import math
import os
import tempfile

import numpy as np

from simulator.physics.calibration_logger import CalibrationLogger
from simulator.physics.drone_state import DroneState

W, H = 1280, 720
CX, CY = W / 2.0, H / 2.0
SCALE = 2.0  # px за метр
FRAME_STEP = 30
FPS = 30.0
SPEED = 150.0


class FakeOrthoMap:
    """Мінімальний двійник OrthophotoMap: local == Mercator - center."""
    _center_x = 3_000_000.0
    _center_y = 6_200_000.0

    def local_to_gps(self, x: float, y: float):
        mx, my = x + self._center_x, y + self._center_y
        lon = mx / 6378137.0 * 180.0 / math.pi
        lat = math.degrees(2 * math.atan(math.exp(my / 6378137.0)) - math.pi / 2)
        return lat, lon


def make_H1(X, Y, heading):
    """Ground-truth гомографія local→pixel для дрона в (X, Y) з курсом heading."""
    c, s = math.cos(heading), math.sin(heading)
    return np.array(
        [
            [SCALE * c, SCALE * s, -SCALE * (c * X + s * Y) + CX],
            [SCALE * s, -SCALE * c, -SCALE * (s * X - c * Y) + CY],
            [0.0, 0.0, 1.0],
        ]
    )


def make_state(X, Y, heading):
    q = np.array([0.0, 0.0, math.sin(heading / 2), math.cos(heading / 2)])
    v = np.array([SPEED * math.cos(heading), SPEED * math.sin(heading), 0.0])
    return DroneState(position=np.array([X, Y, 1000.0]), velocity=v, quaternion=q)


def simulate_lawnmower(n_legs=4, leg_frames=600, turn_frames=60):
    """Serpentine: прямі по ±x, розвороти на 180° за turn_frames кадрів.
    Повертає (frames, turn_frame_ranges): frames = [(X, Y, heading)]."""
    frames, turns = [], []
    X, Y, heading = 0.0, 0.0, 0.0
    dt = 1.0 / FPS
    for leg in range(n_legs):
        for _ in range(leg_frames):
            frames.append((X, Y, heading))
            X += SPEED * math.cos(heading) * dt
            Y += SPEED * math.sin(heading) * dt
        if leg < n_legs - 1:
            target = math.pi - heading if abs(heading) < 1e-9 else 0.0
            t0 = len(frames)
            dh = (target - heading + math.pi) % (2 * math.pi) - math.pi
            step_h = dh / turn_frames
            for _ in range(turn_frames):
                heading += step_h
                frames.append((X, Y, heading))
                X += SPEED * math.cos(heading) * dt
                Y += SPEED * math.sin(heading) * dt
            heading = target
            turns.append((t0, len(frames) - 1))
    return frames, turns


def database_builder_sampled_frames(total_frames, frame_step):
    """Точна копія семплінгу DatabaseBuilder (decord- і cv2-гілки збігаються):
    prefetch_frames: indices = list(range(0, total_frames, frame_step)),
    слот = індекс // frame_step."""
    return list(range(0, total_frames, frame_step))


def main():
    ok = True
    frames, turns = simulate_lawnmower()
    total = len(frames)
    tmpdir = tempfile.mkdtemp()
    calib_path = os.path.join(tmpdir, "calibration.json")

    logger = CalibrationLogger(
        calib_path, FakeOrthoMap(), frame_step=FRAME_STEP,
        turn_rate_deg_per_slot=3.0, min_anchor_spacing_slots=15,
        max_anchor_spacing_slots=60,
    )

    buffered_frames = []
    for idx, (X, Y, heading) in enumerate(frames):
        if logger.log(make_H1(X, Y, heading), W, H, make_state(X, Y, heading), idx):
            buffered_frames.append(idx)
    logger.close()

    # [1] Семплінг ідентичний DatabaseBuilder
    db_frames = database_builder_sampled_frames(total, FRAME_STEP)
    assert buffered_frames == db_frames, (
        f"sampling mismatch: logger={buffered_frames[:5]}..., db={db_frames[:5]}...")
    slots = [c["slot"] for c in logger._candidates]
    assert slots == [f // FRAME_STEP for f in db_frames]
    assert slots == list(range(len(db_frames)))
    print(f"[1] OK: семплінг збігається з DatabaseBuilder "
          f"({len(db_frames)} слотів із {total} кадрів, step={FRAME_STEP})")

    # [2] Формат файлу v2.4 (2.3 + блок keyframe_selection)
    with open(calib_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == "2.4"
    ks = data["keyframe_selection"]
    assert ks["criterion"] in ("overlap", "step")
    assert ks["frame_step"] == FRAME_STEP
    assert data["projection"]["mode"] == "WEB_MERCATOR"
    assert data["frame_size"] == [W, H]
    anchors = data["anchors"]
    assert anchors, "no anchors saved"
    ids = [a["frame_id"] for a in anchors]
    assert ids == sorted(set(ids)), "anchor ids not sorted/unique"
    for a in anchors:
        M = np.array(a["affine_matrix"])
        assert M.shape == (2, 3)
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        assert det < 0, f"anchor {a['frame_id']}: det={det} >= 0"
        qa = a["qa_data"]
        assert len(qa["points_2d"]) == 5 and len(qa["points_gps"]) == 5
        assert qa["rmse_m"] < 1.0, f"rmse too high: {qa['rmse_m']}"
    print(f"[2] OK: calibration.json v2.4, {len(anchors)} якорів, всі det<0, RMSE<1м")

    # [3] Якорі на межах розворотів
    reasons = {a["frame_id"]: a["qa_data"]["notes"] for a in anchors}
    turn_slot_ranges = [(t0 // FRAME_STEP, (t1 // FRAME_STEP) + 1) for t0, t1 in turns]
    for s0, s1 in turn_slot_ranges:
        near_start = [i for i in ids if abs(i - s0) <= 1]
        near_end = [i for i in ids if abs(i - s1) <= 1]
        assert near_start, f"no anchor near turn start slot {s0}"
        assert near_end, f"no anchor near turn end slot {s1}"
    n_turn_anchors = sum("turn_" in reasons[i] for i in ids)
    assert n_turn_anchors >= 2 * len(turns), f"too few turn anchors: {n_turn_anchors}"
    assert "[first]" in reasons[ids[0]] and ids[0] == slots[0]
    assert "[last]" in reasons[ids[-1]] and ids[-1] == slots[-1]
    print(f"[3] OK: якорі на межах усіх {len(turns)} розворотів "
          f"(turn-якорів: {n_turn_anchors}); перший/останній слоти покриті")

    # [4] Максимальний інтервал між якорями
    gaps = np.diff(ids)
    assert gaps.max() <= 60, f"max anchor gap {gaps.max()} > 60 slots"
    print(f"[4] OK: макс. інтервал між якорями = {int(gaps.max())} слотів (≤ 60)")

    # [5] Точність: центр кадру → ground-truth Mercator-позиція дрона
    for a in anchors:
        fid = a["frame_id"]
        X, Y, _ = frames[fid * FRAME_STEP]
        M = np.array(a["affine_matrix"])
        mx = M[0, 0] * CX + M[0, 1] * CY + M[0, 2]
        my = M[1, 0] * CX + M[1, 1] * CY + M[1, 2]
        gt = (X + FakeOrthoMap._center_x, Y + FakeOrthoMap._center_y)
        err = math.hypot(mx - gt[0], my - gt[1])
        assert err < 1e-3, f"anchor {fid}: center error {err} m"
    print("[5] OK: афінні матриці відтворюють ground-truth позицію центру (<1 мм)")

    # [6] Юніт: select_anchor_indices на короткому профілі
    sel = CalibrationLogger.select_anchor_indices(
        slots=list(range(10)), headings_rad=[0.0] * 10,
        turn_rate_deg_per_slot=3.0, min_spacing_slots=2, max_spacing_slots=4)
    assert sel[0] == "first" and sel[9] == "last"
    assert all(np.diff(sorted(sel)) <= 4)
    hs = [0.0] * 5 + [math.radians(30 * i) for i in range(1, 5)] + [math.radians(120)] * 5
    sel2 = CalibrationLogger.select_anchor_indices(
        slots=list(range(len(hs))), headings_rad=hs,
        turn_rate_deg_per_slot=3.0, min_spacing_slots=2, max_spacing_slots=100)
    vals = set(sel2.values())
    assert "turn_start" in vals and "turn_end" in vals, vals
    print("[6] OK: select_anchor_indices — straight fill та межі розвороту")

    print("\n=== ALL TESTS PASSED ===")
    return ok


if __name__ == "__main__":
    main()
