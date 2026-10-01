"""Геометричний прогноз keyframe-ів за перекриттям кадрів.

Система локалізації відбирає keyframe-и адаптивно: кадр зберігається,
коли перекриття з ОСТАННІМ ЗБЕРЕЖЕНИМ keyframe падає до
``database.keyframe_max_overlap`` (див. ``src/database/keyframe_selector.py``
і ``frame_processor.py`` у DroneLocalization). Пропагація будує вузли графа
ЛИШЕ на слотах із фічами, тож якір, що потрапив на не-keyframe слот, там
приснеплюється до найближчого keyframe — а при 50%-му порозі «найближчий»
може бути за півкадру руху. Тому симулятор мусить ставити якорі рівно на ті
слоти, які локалізатор залишить keyframe-ами.

Цей прогноз НЕ є еквівалентом відбору DatabaseBuilder: база оцінює рух із
ознак зображення та може зберегти кадр після невдалого матчингу. При записі
відео з калібруванням симулятор запускає справжній селектор локалізатора
на закодованому MP4; цей модуль лишається для геометричних тестів і записів
без візуального селектора.

ДУБЛЮВАННЯ. Формула перекриття повторює
``keyframe_selector.overlap_fraction`` з DroneLocalization. Спільної бібліотеки
між проєктами немає, тому дублювання свідоме; воно закріплене тестом, що звіряє
обидві реалізації чисельно. Перетин многокутників тут — Сазерленд–Ходжман на
numpy, щоб модуль фізики не тягнув OpenCV.
"""

from __future__ import annotations

import numpy as np


def _clip_convex(subject: np.ndarray, w: float, h: float) -> np.ndarray:
    """Відсікає опуклий многокутник прямокутником [0,w]×[0,h] (Сазерленд–Ходжман)."""
    poly = subject
    # Межі прямокутника як (нормаль, зсув): точка всередині, якщо n·p + d >= 0
    edges = ((1.0, 0.0, 0.0), (-1.0, 0.0, w), (0.0, 1.0, 0.0), (0.0, -1.0, h))
    for nx, ny, d in edges:
        if len(poly) == 0:
            return poly
        out: list[np.ndarray] = []
        n_pts = len(poly)
        for i in range(n_pts):
            cur = poly[i]
            prv = poly[i - 1]
            cur_in = nx * cur[0] + ny * cur[1] + d >= 0.0
            prv_in = nx * prv[0] + ny * prv[1] + d >= 0.0
            if cur_in != prv_in:
                dc = nx * cur[0] + ny * cur[1] + d
                dp = nx * prv[0] + ny * prv[1] + d
                denom = dp - dc
                if abs(denom) > 1e-12:
                    out.append(prv + (cur - prv) * (dp / denom))
            if cur_in:
                out.append(cur)
        poly = np.array(out, dtype=np.float64) if out else np.empty((0, 2))
    return poly


def _polygon_area(poly: np.ndarray) -> float:
    """Площа многокутника за формулою шнурівки (модуль, тож орієнтація байдужа)."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def overlap_fraction(H: np.ndarray | None, frame_w: int, frame_h: int) -> float:
    """Частка площі keyframe-а, яку ще видно в поточному кадрі, ∈ [0, 1].

    ``H`` переводить координати ПОТОЧНОГО кадру в координати keyframe-а.
    Вироджені випадки (не-скінченна H, |det| < 1e-9, точка за площиною
    зображення) → 0.0: нуль змушує взяти keyframe, а не пропустити його.
    """
    if H is None:
        return 0.0
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return 0.0
    if abs(np.linalg.det(H)) < 1e-9:
        return 0.0

    w, h = float(frame_w), float(frame_h)
    if w <= 0 or h <= 0:
        return 0.0

    corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float64)
    projected = np.hstack([corners, np.ones((4, 1))]) @ H.T
    if np.any(projected[:, 2] <= 1e-12) or not np.all(np.isfinite(projected)):
        return 0.0
    projected = projected[:, :2] / projected[:, 2:3]
    if not np.all(np.isfinite(projected)):
        return 0.0

    inter = _polygon_area(_clip_convex(projected, w, h))
    return float(np.clip(inter / (w * h), 0.0, 1.0))


def _to_3x3(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (3, 3):
        return M
    out = np.eye(3, dtype=np.float64)
    out[:2, :3] = M
    return out


def predict_keyframe_slots(
    affines: list[np.ndarray],
    frame_w: int,
    frame_h: int,
    max_overlap: float = 0.5,
    max_gap_frames: int = 60,
) -> list[int]:
    """Індекси кандидатів, які локалізатор залишить keyframe-ами.

    Повторює цикл ``FrameProcessor``: перший слот зберігається завжди
    (``keyframe_always_save_first``), далі слот стає keyframe-ом, коли
    перекриття з останнім збереженим ≤ ``max_overlap`` або коли підряд
    пропущено ``max_gap_frames`` слотів (запобіжник, 0 = вимкнено).

    ``affines`` — матриці 2×3 або 3×3 «пікселі → метри» по одній на слот,
    у порядку зростання слота.
    """
    if not affines:
        return []

    kept = [0]
    ref_inv = np.linalg.inv(_to_3x3(affines[0]))
    since = 0

    for i in range(1, len(affines)):
        cur = _to_3x3(affines[i])
        # H_rel: пікселі поточного кадру → пікселі keyframe-а
        h_rel = ref_inv @ cur
        take = overlap_fraction(h_rel, frame_w, frame_h) <= float(max_overlap)
        if not take and max_gap_frames > 0 and since >= max_gap_frames:
            take = True
        if take:
            kept.append(i)
            ref_inv = np.linalg.inv(cur)
            since = 0
        else:
            since += 1

    return kept
