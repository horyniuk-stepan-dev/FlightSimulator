"""Стратегія відбору кадрів симулятора = стратегія локалізатора.

Контракт, який тут закріплюється: якорі calibration.json лягають РІВНО на ті
слоти, які DatabaseBuilder залишить keyframe-ами. Інакше пропагація
локалізатора або снепить якір до чужого слота (при overlap-порозі 0.5 це до
півкадру руху), або зупиняється з помилкою на двох якорях в одному слоті.

Формула перекриття продубльована з ``keyframe_selector.overlap_fraction``
системи локалізації (спільної бібліотеки між проєктами немає). Числові
значення нижче — ті самі, що й у тестах DroneLocalization, тож розходження
реалізацій ловиться тут.

Запуск: python test_keyframe_predictor.py  (потрібен лише numpy)
"""

import numpy as np

from simulator.physics.calibration_logger import CalibrationLogger
from simulator.physics.keyframe_predictor import overlap_fraction, predict_keyframe_slots

W, H = 1280, 720


def _shift(dx, dy=0.0, s=1.0, deg=0.0):
    a = np.deg2rad(deg)
    c, si = np.cos(a), np.sin(a)
    return np.array([[s * c, -s * si, dx], [s * si, s * c, dy], [0.0, 0.0, 1.0]])


def test_overlap_values():
    assert abs(overlap_fraction(np.eye(3), W, H) - 1.0) < 1e-9
    assert abs(overlap_fraction(_shift(W * 0.5), W, H) - 0.5) < 1e-3
    assert abs(overlap_fraction(_shift(W * 0.25), W, H) - 0.75) < 1e-3
    assert abs(overlap_fraction(_shift(W), W, H)) < 1e-6
    # 50% по обох осях → лишається чверть
    assert abs(overlap_fraction(_shift(W * 0.5, H * 0.5), W, H) - 0.25) < 1e-3
    # вироджені входи → 0.0 (безпечний напрям: змушує взяти keyframe)
    assert overlap_fraction(np.zeros((3, 3)), W, H) == 0.0
    assert overlap_fraction(np.full((3, 3), np.nan), W, H) == 0.0
    assert overlap_fraction(None, W, H) == 0.0
    print("[1] OK: overlap_fraction збігається з еталонними значеннями локалізатора")


def test_prediction_matches_expected_cadence():
    # GSD 0.5 м/px, 32 м між слотами → 64 px зсуву; поріг 0.5 спрацьовує на 640 px
    affines = [np.array([[0.5, 0.0, i * 32.0], [0.0, -0.5, 0.0]]) for i in range(41)]
    kept = predict_keyframe_slots(affines, W, H, max_overlap=0.5, max_gap_frames=0)
    assert kept == [0, 10, 20, 30, 40], kept

    kept25 = predict_keyframe_slots(affines, W, H, max_overlap=0.25, max_gap_frames=0)
    assert kept25 == [0, 15, 30], kept25
    print(f"[2] OK: 64px/слот, поріг 50% → keyframe кожні 10 слотів {kept}")


def test_max_gap_forces_keyframe():
    # Нерухома камера: перекриття завжди 1.0, рятує лише запобіжник
    affines = [np.array([[0.5, 0.0, 0.0], [0.0, -0.5, 0.0]]) for _ in range(21)]
    assert predict_keyframe_slots(affines, W, H, 0.5, max_gap_frames=0) == [0]
    kept = predict_keyframe_slots(affines, W, H, 0.5, max_gap_frames=5)
    assert kept == [0, 6, 12, 18], kept
    print(f"[3] OK: без руху max_gap_frames=5 тримає керування {kept}")


def test_scale_change_is_seen():
    # Зниження (кадр захоплює менше землі) зменшує перекриття зі збереженим видом
    assert overlap_fraction(_shift(0, 0, s=0.5), W, H) < 0.5
    # Набір висоти лишає стару площу видимою — навмисна асиметрія метрики
    assert overlap_fraction(_shift(0, 0, s=2.0), W, H) == 1.0
    print("[4] OK: масштаб враховано (асиметрія по висоті — за побудовою)")


class _FakeMap:
    _center_x = 0.0
    _center_y = 0.0
    elevation = None

    def local_to_gps(self, x, y):
        return (50.0 + y * 1e-5, 30.0 + x * 1e-5)


def test_anchors_land_on_keyframes():
    """Головний контракт: жоден якір не стоїть на не-keyframe слоті."""
    from simulator.physics.drone_state import DroneState

    logger = CalibrationLogger(
        "/dev/null",
        _FakeMap(),
        frame_step=30,
        keyframe_criterion="overlap",
        keyframe_max_overlap=0.5,
        keyframe_max_gap_frames=60,
    )
    logger._frame_size = (W, H)

    # Прямий проліт: 32 м на слот, GSD 0.5 м/px
    n = 60
    for i in range(n):
        M = np.array([[0.5, 0.0, i * 32.0], [0.0, -0.5, 0.0]])
        logger._candidates.append(
            {
                "slot": i,
                "video_frame": i * 30,
                "heading": 0.0,
                "yaw_deg": 0.0,
                "alt": 500.0,
                "M": M,
                "rmse": 0.0,
                "median": 0.0,
                "max": 0.0,
                "pts_px": [],
                "pts_gps": [],
                "pts_mercator": [],
            }
        )
        _ = DroneState  # модуль імпортується — структура кандидата та сама

    kf = set(logger._predicted_keyframe_indices())
    raw = logger.select_anchor_indices(
        slots=[c["slot"] for c in logger._candidates],
        headings_rad=[c["heading"] for c in logger._candidates],
        turn_rate_deg_per_slot=3.0,
        min_spacing_slots=15,
        max_spacing_slots=60,
    )
    restricted = logger._restrict_to_keyframes(raw)

    assert restricted, "усі якорі зникли"
    off = [i for i in restricted if i not in kf]
    assert not off, f"якорі поза keyframe-слотами: {off}"
    # Межі запису мають лишитися: без них інтерполятор не покриє все відео
    assert min(restricted) == min(kf)
    assert max(restricted) == max(kf)
    print(
        f"[5] OK: {len(restricted)} якорів, усі на keyframe-слотах "
        f"({len(kf)}/{n} слотів — keyframe-и)"
    )


if __name__ == "__main__":
    test_overlap_values()
    test_prediction_matches_expected_cadence()
    test_max_gap_forces_keyframe()
    test_scale_change_is_seen()
    test_anchors_land_on_keyframes()
    print("\n=== ALL TESTS PASSED ===")
